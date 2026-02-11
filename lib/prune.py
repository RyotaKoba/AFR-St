import torch
import torch.nn as nn
from .data import get_loaders, get_mm_loaders
from tqdm import tqdm
import sys
from .model import rm_modules, get_vision_rm_modules
from .gmm import gmm_edge_outlier_removal, select_K_by_bic
from .gesd import gesd_outlier_cleaning_torch
from .kde import kde_edge_outlier_removal
from .dpm import dpm_edge_outlier_removal
from .bmm import bmm_edge_outlier_removal

import torch.nn.utils.prune as prune

SCORE = None

class Pruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        print("amount:",self.amount)
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        print('number of parameters:', tensor_size)
        print('number of parameters to prune:', nparams_toprune)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        global SCORE
        print(SCORE.shape)
        if nparams_toprune != 0:
            topk = torch.topk(SCORE, k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0

        return mask

def unstructured_compress(layer, weight_mask, device):
    gate_mask, up_mask, down_mask = [m.to(device) for m in weight_mask]
    with torch.no_grad():
        layer.mlp.gate_proj.weight.data.mul_(gate_mask)
        layer.mlp.up_proj.weight.data.mul_(up_mask)
        layer.mlp.down_proj.weight.data.mul_(down_mask)
    torch.cuda.empty_cache()

def compress(layer, mlp_mask, device):
    mlp_mask = mlp_mask.to(device)

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    print(layer.mlp.intermediate_size)

    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]
    layer.mlp.down_proj.weight.data = output_weight
    layer.mlp.down_proj.in_features = mlp_mask.sum().item()
    torch.cuda.empty_cache()

def compress_vision(layer, mlp_mask, device):
    mlp_mask = mlp_mask.to(device)

    layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.fc2.weight.data = layer.mlp.fc2.weight.data[:,torch.where(mlp_mask)[0]]
    if layer.mlp.fc1.bias is not None:
        layer.mlp.fc1.bias.data = layer.mlp.fc1.bias.data[torch.where(mlp_mask)[0]]
    layer.mlp.fc1.out_features = mlp_mask.sum().item()
    layer.mlp.fc2.in_features = mlp_mask.sum().item()

    torch.cuda.empty_cache()

def snip(args, model, tokenizer, device):
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]
    num_layers = len(model.model.layers)
    rm_module = None
    del rm_module

    with torch.no_grad():
        accum_score = [torch.zeros_like(w).to("cpu") for w in rm_weights]

    it = iter(dataloader)
    for i in tqdm(range(args.nsamples), desc="snip"):
        inp, tar = next(it)
        inp = inp.to("cuda:0")
        tar = tar.to("cuda:0")
        model.zero_grad(set_to_none=True)
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        loss = nn.CrossEntropyLoss()(outputs, tar)
        grads = list(torch.autograd.grad(loss, rm_weights))
        with torch.no_grad():
            for k, (weight, grad) in enumerate(zip(rm_weights, grads)):
                accum_score[k] += (weight.cpu() * grad.cpu()).abs()
        grads = None
        del grads
    rm_weights = None
    dataloader = None
    del inp, tar, outputs, loss, it, dataloader, rm_weights
    model.eval()
    model = model.half()
    model.zero_grad()
    torch.cuda.empty_cache()

    score = torch.cat([s.view(-1) for s in accum_score])
    score = score.half()
    print("score: ", score.shape)

    score, _ = torch.sort(score, descending=True)
    threshold = score[int(score.shape[0] * (1 - args.pruning_ratio))]
    score = None
    del score

    print("go!")
    # mlp_mask = []
    # mlp_mask = (accum_score.t() >= threshold).t()
    # del accum_score

    # # accum_score layout: gate(0..L-1), up(L..2L-1), down(2L..3L-1)
    for k in range(num_layers):
        gate_mask = accum_score[k] >= threshold
        up_mask = accum_score[k + num_layers] >= threshold
        down_mask = accum_score[k + num_layers * 2] >= threshold
        print("aaaaaaa")
    print("Mask prepared.")
    for k in range(num_layers):
        unstructured_compress(model.model.layers[k], [gate_mask, up_mask, down_mask], device)
    model.zero_grad()

def structured_snip(args, model, tokenizer, device=torch.device("cuda:0")):
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    for i, (inp, tar) in enumerate(dataloader):
        inp = inp.to("cuda:0")
        tar = tar.to("cuda:0")
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        tar = tar.to("cuda:0")
        loss = nn.CrossEntropyLoss()(outputs, tar)
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    layers = model.model.layers

    mlp_metric_list = []
    mlp_mask = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        W_down = rm_weights[i+64] * grads[i+64]
        W_up = (rm_weights[i+32] * grads[i+32]).t()
        W_gate = (rm_weights[i] * grads[i]).t()
        W_metric = W_down + W_up + W_gate
        W_metric = W_metric.mean(axis=0)
        mlp_metric_list.append(W_metric)
        print("W_metric:",W_metric)
    print("mlp_metric_list: ", mlp_metric_list[0])
    mlp_metric = torch.stack(mlp_metric_list)
    print("mlp_metric: ", mlp_metric.shape)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven

    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    print(f"thresholds: {thresholds}")
    print(thresholds.shape)
    thresholds = thresholds.to(mlp_metric.device)

    mlp_mask = (mlp_metric.t() >= thresholds).t()

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.zero_grad()

    torch.cuda.empty_cache()

P_SVD_loss = torch.zeros(1)
def AFR(args, model, tokenizer, device):
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    global P_SVD_loss
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    def store_feature(module, input, output):
        global P_SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return

        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")
        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)
        P_SVD_loss = P_SVD_loss + singular_value_mean

    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook)

    with torch.no_grad():
        fo_grads = [torch.zeros_like(w) for w in rm_weights]

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to("cuda:0")
        targets = targets.to("cuda:0")
        outputs = model(inputs)
        break
    del inputs
    del dataloader

    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights,retain_graph=True))
    for hook in hooks:
        hook.remove()
    
    P_SVD_loss = torch.zeros(1)
    del hook
    # FOスコアの計算
    print("before calc fo_score")
    with torch.no_grad():
        fo_score = [(weight * grad).view(-1).abs() for weight, grad in zip(rm_weights, fo_grads)]
        fo_score = torch.cat(fo_score)
    print("after calc fo_score")
    del fo_grads
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    loss = nn.CrossEntropyLoss()(outputs, targets) # CE Loss
    del outputs
    del targets
    with torch.no_grad():
        snip_grads = [torch.zeros_like(w) for w in rm_weights]
    snip_grads = list(torch.autograd.grad(loss, rm_weights))
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        snip_score = [(weight * grad).view(-1).abs() for weight, grad in zip(rm_weights, snip_grads)]
        snip_score = torch.cat(snip_score)
    del snip_grads
    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    score = fo_score_standardized + snip_score_standardized
    model.zero_grad()  # 勾配をリセット
    del P_SVD_loss  # グローバル変数をリセット
    del fo_score_standardized
    del snip_score_standardized
    del fo_score
    del snip_score
    del rm_module
    del rm_weights
    gc.collect()
    P_SVD_loss = torch.zeros(1)  # グローバル変数の再初期化
    return score  # スコアを返す

SVD_loss = torch.zeros(1)
def ReFer_SVD(args, model, tokenizer, device):
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    global SVD_loss
    def store_feature(module, input, output):
        global SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return

        output = output.reshape(output.size(0), -1)  # バッチサイズ次元以外をフラットにする
        U, S, Vh = torch.svd(output)  # SVDを計算
        singular_value_mean = S.mean()
        SVD_loss = SVD_loss + singular_value_mean

    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]

    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        grads = list(torch.autograd.grad(SVD_loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)

    model.zero_grad()
    loss = torch.zeros(1)

    return score

SVD_loss = torch.zeros(1)
def Structured_ReFer_SVD(args, model, tokenizer, device):
    dataloader= get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    def calculate_neuron_score_v2(W_metric):
        """
        方法2: Signal-to-Noise比的なアプローチ
        平均の絶対値 / (標準偏差 + epsilon)
        """

        """トリム平均を使用してニューロンスコアを計算"""
        trim_percent = 2
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]  # 4096
        trim_count = int(n_rows * trim_percent / 100)
        trimmed_W = sorted_W[trim_count:-trim_count, :]
        mean_scores = trimmed_W.mean(axis=0)
        std_scores = trimmed_W.std(axis=0)
        snr_scores = torch.abs(mean_scores) / (std_scores + 1e-8)
        return snr_scores

    global SVD_loss
    SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")

    def store_feature(module, input, output):
        global SVD_loss
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return

        output = output.reshape(output.size(0), -1).to(dtype=torch.float32)
        U, S, Vh = torch.svd(output)  # SVDを計算
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)
        SVD_loss = SVD_loss + singular_value_mean

    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]

    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        break

    grads = list(torch.autograd.grad(SVD_loss, rm_weights))
    for i, grad in enumerate(grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"fo_grads[{i}] contains inf or nan")
        else:
            print(f"fo_grads[{i}] is clean")
    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)
    layers = model.model.layers
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * grads[i+64]
            W_up = (rm_weights[i+32] * grads[i+32]).t()
            W_gate = (rm_weights[i] * grads[i]).t()
            W_metric = W_down + W_up + W_gate
            print(f"SVD_W_metric {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"SVD_W_metric {i} have inf:", torch.isinf(W_metric).any().item())
            W_metric = calculate_neuron_score_v2(W_metric)
            # W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())
    mlp_metric = torch.stack(mlp_metric_list)
    print("score contains nan:", torch.isnan(mlp_metric).any().item())
    print("score contains inf:", torch.isinf(mlp_metric).any().item())
    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven
    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    mlp_mask = (mlp_metric.t() >= thresholds).t()

    print('*'*30)
    for idx in range(len(layers)):
        compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    torch.cuda.empty_cache()

def Structured_AFR(args, model, tokenizer, device):
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    def calculate_neuron_score(W_metric):
        # ============================================================
        # Method 1: Trim x%
        # ============================================================
        trim_percent = 2
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]
        trim_count = int(n_rows * trim_percent / 100)
        cleaned_scores = sorted_W[trim_count:-trim_count, :]

        # ============================================================
        # Method 2: GMM Trim
        # ============================================================
        # cleaned_scores, _, _= gmm_edge_outlier_removal(W_metric, K=3, alpha_edge=0.05,q_tail=0.02, use_density=True, check_bic=True, K_range=(1, 5)))

        # ===========================================================
        # Method 3: KDE Trim
        # ===========================================================
        # cleaned_scores = kde_edge_outlier_removal(W_metric)

        # ============================================================
        # Method 4: GESD Trim
        # ============================================================
        # cleaned_scores = gesd_outlier_cleaning_torch(W_metric)

        # ===========================================================
        # Method 5: DPM Trim
        # ===========================================================
        # cleaned_scores = dpm_edge_outlier_removal(W_metric)

        # ===========================================================
        # Method 6: BMM Trim
        # ===========================================================
        # cleaned_scores = bmm_edge_outlier_removal(W_metric)

        return weight_wise_to_neuron_wise(cleaned_scores)

    def weight_wise_to_neuron_wise(scores):
        # ===========================================================
        # Method 1: MeanAbs
        # ==========================================================~
        mean_scores = scores.mean(axis=0)
        return torch.abs(mean_scores)

    global P_SVD_loss
    P_SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")

    def store_feature(module, input, output):
        global P_SVD_loss

        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return
    
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")

        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to("cpu", dtype=torch.float32)
        if torch.isnan(singular_value_mean).any().item():
            print("singular_value_mean contains nan is in module:", module.__class__.__name__)
        P_SVD_loss = P_SVD_loss + singular_value_mean
 
    hooks = []
    for name, module in model.named_modules():
        # gate_proj, up_proj, down_projのみ
        if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            hook = module.register_forward_hook(store_feature)
            hooks.append(hook)

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]
    fo_score = {}
    snip_score = {}

    it = iter(dataloader)
    num_layers = len(model.model.layers)
    for i in tqdm(range(args.nsamples), desc="Processing layers"):
        inputs, targets = next(it)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        inputs = inputs.to("cuda:0")
        outputs = model(inputs)

        if torch.isnan(P_SVD_loss).any().item() or torch.isinf(P_SVD_loss).any().item():
            print("P_SVD_loss contains nan or inf:",i,"layer")

        fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights,retain_graph=True))
        for k, grad in enumerate(fo_grads):
            has_inf = torch.isinf(grad).any().item()
            has_nan = torch.isnan(grad).any().item()
            if has_inf or has_nan:
                print(f"fo_grads[{k}] contains inf or nan")
        with torch.no_grad():
            for k in range(num_layers):
                W_down = rm_weights[k+num_layers*2] * fo_grads[k+num_layers*2]
                W_up = (rm_weights[k+num_layers] * fo_grads[k+num_layers]).t()
                W_gate = (rm_weights[k] * fo_grads[k]).t()
                W_metric = W_down + W_up + W_gate
                W_metric = calculate_neuron_score(W_metric)
                W_metric = W_metric.mean(axis=0)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(num_layers):
                        fo_score[m] = torch.zeros_like(W_metric)
                        snip_score[m] = torch.zeros_like(W_metric)
                W_metric = torch.abs(W_metric)
                has_inf = torch.isinf(W_metric).any().item()
                has_nan = torch.isnan(W_metric).any().item()
                if has_inf or has_nan:
                    print(f"W_metric[{k}] contains inf or nan")
                fo_score[k].add_(W_metric)

        P_SVD_loss = torch.zeros(1)
        del fo_grads
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
        targets = targets.reshape(-1).to(dtype=torch.long, device="cuda:0")
        loss = nn.CrossEntropyLoss()(outputs, targets)
        snip_grads = list(torch.autograd.grad(loss, rm_weights))
        for k, grad in enumerate(snip_grads):
            has_inf = torch.isinf(grad).any().item()
            has_nan = torch.isnan(grad).any().item()
            if has_inf or has_nan:
                print(f"snip_grads[{k}] contains inf or nan")
        with torch.no_grad():
            for k in range(num_layers):
                W_down = rm_weights[k+num_layers*2] * snip_grads[k+num_layers*2]
                W_up = (rm_weights[k+num_layers] * snip_grads[k+num_layers]).t()
                W_gate = (rm_weights[k] * snip_grads[k]).t()
                W_metric = W_down + W_up + W_gate
                W_metric = calculate_neuron_score(W_metric)
                snip_score[k].add_(W_metric)
        del snip_grads

    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)
    del P_SVD_loss
    model.zero_grad()
    for i in range(num_layers):
        snip_score[i] = snip_score[i]*snip_score[i]

    fo_score = torch.stack(list(fo_score.values()), dim=0)
    snip_score = torch.stack(list(snip_score.values()), dim=0)

    del inputs
    del W_down
    del W_up
    del W_gate
    del W_metric

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    score = fo_score_standardized + snip_score_standardized

    score = score.to("cpu")
    if args.global_pruning:
        sorted_mlp_metric, _ = torch.sort(score.view(-1), descending=True)
        limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]
        thresholds = torch.tensor([limit for i in range(num_layers)])
    else:
        sorted_mlp_metric, _ = torch.sort(score, descending=True)
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(num_layers)])
    mlp_mask = (score.t() >= thresholds).t()
    print('*'*30)
    for idx in range(num_layers):
        compress(model.model.layers[idx], mlp_mask[idx], device)

    model.zero_grad()  # 勾配をリセット


def Structured_AFR_LLaVA(args, model, tokenizer, device, image_processor):
    dataloader = get_mm_loaders()

    def calculate_neuron_score(W_metric):
        # ============================================================
        # Method 1: Trim x%
        # ============================================================
        trim_percent = 2
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]
        trim_count = int(n_rows * trim_percent / 100)
        cleaned_scores = sorted_W[trim_count:-trim_count, :]

        # ============================================================
        # Method 2: GMM Trim
        # ============================================================
        # cleaned_scores, _, _= gmm_edge_outlier_removal(W_metric, K=3, alpha_edge=0.05,q_tail=0.02, use_density=True, check_bic=True, K_range=(1, 5)))

        # ===========================================================
        # Method 3: KDE Trim
        # ===========================================================
        # cleaned_scores = kde_edge_outlier_removal(W_metric)

        return weight_wise_to_neuron_wise(cleaned_scores)

    def weight_wise_to_neuron_wise(scores):
        # ===========================================================
        # Method 1: MeanAbs
        # ==========================================================~
        mean_scores = scores.mean(axis=0)
        return torch.abs(mean_scores)

    global P_SVD_loss
    global P_SVD_loss_vision
    P_SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")
    P_SVD_loss_vision = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")

    def store_feature(module, input, output):
        global P_SVD_loss

        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return

        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")

        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to("cpu", dtype=torch.float32)
        if torch.isnan(singular_value_mean).any().item():
            print("singular_value_mean contains nan is in module:", module.__class__.__name__)
        P_SVD_loss = P_SVD_loss + singular_value_mean

    def store_feature_vision(module, input, output):
        global P_SVD_loss_vision

        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        if output is None:
            return

        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")

        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to(dtype=torch.float32, device="cpu")
        if torch.isnan(singular_value_mean).any().item():
            print("singular_value_mean contains nan is in module:", module.__class__.__name__)
        P_SVD_loss_vision = P_SVD_loss_vision + singular_value_mean

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_model = vision_tower.vision_tower
    vision_model.requires_grad_(True)
    model.requires_grad_(True)
    hooks = []
    for name, module in model.named_modules():
        # gate_proj, up_proj, down_projのみ
        if 'vision_tower' not in name and 'mm_projector' not in name:
            hook = module.register_forward_hook(store_feature)
            hooks.append(hook)
    hooks_vision = []
    for name, module in vision_model.named_modules():
        if any(x in name for x in ['fc1', 'fc2']):
            hook = module.register_forward_hook(store_feature_vision)
            hooks_vision.append(hook)

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]
    rm_module = get_vision_rm_modules(vision_model)
    rm_weights_vision = [module.weight for module, _ in rm_module]
    model.model.mm_projector.requires_grad_(True)
    for param in model.model.mm_projector.parameters():
        param.requires_grad_(True)
    for w in rm_weights_vision:
        w.requires_grad_(True)

    fo_score = {}
    snip_score = {}
    fo_score_vision = {}
    snip_score_vision = {}

    num_layers = len(model.model.layers)
    it = iter(dataloader)
    for i in range(args.nsamples):
        batch = next(it)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        image = batch['image'][0]
        text = batch['text'][0]
        target_text = batch['target'][0]
        # 画像処理
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(dtype=model.dtype)
        # 入力
        inputs = tokenizer(text, return_tensors='pt').input_ids.cuda()
        inputs = inputs.to(model.device, dtype=torch.long)
        # ターゲット
        targets = tokenizer(target_text, return_tensors='pt').input_ids.cuda()
        vision_model.requires_grad_(True)
        model.model.vision_tower.vision_tower.requires_grad_(True)
        outputs = model(input_ids=inputs, images=image_tensor)

        if torch.isnan(P_SVD_loss).any().item() or torch.isinf(P_SVD_loss).any().item():
            print("P_SVD_loss contains nan or inf:",i,"layer")

        fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights,retain_graph=True))
        for k, grad in enumerate(fo_grads):
            has_inf = torch.isinf(grad).any().item()
            has_nan = torch.isnan(grad).any().item()
            if has_inf or has_nan:
                print(f"fo_grads[{k}] contains inf or nan")
        with torch.no_grad():
            for k in range(num_layers):
                W_down = rm_weights[k+num_layers*2] * fo_grads[k+num_layers*2]
                W_up = (rm_weights[k+num_layers] * fo_grads[k+num_layers]).t()
                W_gate = (rm_weights[k] * fo_grads[k]).t()
                W_metric = W_down + W_up + W_gate
                W_metric = calculate_neuron_score(W_metric)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(num_layers):
                        fo_score[m] = torch.zeros_like(W_metric)
                        snip_score[m] = torch.zeros_like(W_metric)
                fo_score[k].add_(W_metric)

        P_SVD_loss = torch.zeros(1)
        del fo_grads
        outputs = outputs.logits
        target_len = targets.shape[1]
        outputs = outputs[:, -target_len:, :]
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
        targets = targets.reshape(-1).to(dtype=torch.long, device="cuda:0")
        loss = nn.CrossEntropyLoss()(outputs, targets)
        snip_grads = list(torch.autograd.grad(loss, rm_weights,retain_graph=True))
        for k, grad in enumerate(snip_grads):
            has_inf = torch.isinf(grad).any().item()
            has_nan = torch.isnan(grad).any().item()
            if has_inf or has_nan:
                print(f"snip_grads[{k}] contains inf or nan")
        with torch.no_grad():
            for k in range(num_layers):
                W_down = rm_weights[k+num_layers*2] * snip_grads[k+num_layers*2]
                W_up = (rm_weights[k+num_layers] * snip_grads[k+num_layers]).t()
                W_gate = (rm_weights[k] * snip_grads[k]).t()
                W_metric = W_down + W_up + W_gate
                W_metric = calculate_neuron_score(W_metric)
                snip_score[k].add_(W_metric)

        del snip_grads

###############vision part##################
        fo_grads = list(torch.autograd.grad(P_SVD_loss_vision, rm_weights_vision,retain_graph=True))
        for k, grad in enumerate(fo_grads):
            has_inf = torch.isinf(grad).any().item()
            has_nan = torch.isnan(grad).any().item()
            if has_inf or has_nan:
                print(f"fo_grads[{k}] contains inf or nan")
        with torch.no_grad():
            for k in range(len(vision_model.vision_model.encoder.layers)):
                W_fc2 = (rm_weights_vision[k*2+1] * fo_grads[k*2+1])
                W_fc1 = (rm_weights_vision[k*2] * fo_grads[k*2]).t()
                W_metric = W_fc2 + W_fc1
                W_metric = calculate_neuron_score(W_metric)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(len(vision_model.vision_model.encoder.layers)):
                        fo_score_vision[m] = torch.zeros_like(W_metric)
                        snip_score_vision[m] = torch.zeros_like(W_metric)
                fo_score_vision[k].add_(W_metric)

        P_SVD_loss_vision = torch.zeros(1)
        del fo_grads
        # outputs = outputs.logits
        # target_len = targets.shape[1]
        # outputs = outputs[:, -target_len:, :]
        # outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
        # targets = targets.reshape(-1).to(dtype=torch.long, device="cuda:0")
        # loss = nn.CrossEntropyLoss()(outputs, targets)
        # print(loss.requires_grad, loss.grad_fn)
        # for name, param in vision_model.named_parameters():
        #     if param.grad is None:
        #         print(f"{name}: grad is None")
        # snip_grads = list(torch.autograd.grad(loss, rm_weights_vision))
        # for k, grad in enumerate(snip_grads):
        #     has_inf = torch.isinf(grad).any().item()
        #     has_nan = torch.isnan(grad).any().item()
        #     if has_inf or has_nan:
        #         print(f"snip_grads[{k}] contains inf or nan")
        # with torch.no_grad():
        #     for k in range(len(vision_model.vision_model.encoder.layers)):
        #         W_fc2 = (rm_weights_vision[k*2+1] * snip_grads[k*2+1])
        #         W_fc1 = (rm_weights_vision[k*2] * snip_grads[k*2]).t()
        #         W_metric = W_fc2 + W_fc1
        #         W_metric = calculate_neuron_score(W_metric)
        #         W_metric = W_metric.mean(axis=0)
        #         # snip_sign_vision[k].add_(torch.sign(W_metric))
        #         W_metric = torch.abs(W_metric)
        #         snip_score_vision[k].add_(W_metric)

        # del snip_grads

    for hook in hooks:
        hook.remove()
    for hook in hooks_vision:
        hook.remove()
    P_SVD_loss = torch.zeros(1)
    P_SVD_loss_vision = torch.zeros(1)
    del P_SVD_loss
    del P_SVD_loss_vision
    model.zero_grad()
    for i in range(num_layers):
        snip_score[i] = snip_score[i]*snip_score[i]

    fo_score = torch.stack(list(fo_score.values()), dim=0)
    snip_score = torch.stack(list(snip_score.values()), dim=0)
    fo_score_vision = torch.stack(list(fo_score_vision.values()), dim=0)
    # snip_score_vision = torch.stack(list(snip_score_vision.values()), dim=0)

    del inputs
    del W_down
    del W_up
    del W_gate
    del W_metric
    torch.cuda.empty_cache()

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    fo_score_vision_standardized = (fo_score_vision - fo_score_vision.mean()) / fo_score_vision.std()
    score = fo_score_standardized + snip_score_standardized
    # score_vision = fo_score_vision_standardized + snip_score_vision_standardized
    score_vision = fo_score_vision_standardized

    print(score.shape)
    score = score.to("cpu")
    score_vision = score_vision.to("cpu")
    if args.global_pruning:
        sorted_mlp_metric, _ = torch.sort(score.view(-1), descending=True)
        sorted_mlp_metric_vision, _ = torch.sort(score_vision.view(-1), descending=True)
        limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]
        limit_vision = sorted_mlp_metric_vision[int(sorted_mlp_metric_vision.shape[0]*(1-args.pruning_ratio))]
        thresholds = torch.tensor([limit for i in range(num_layers)])
        thresholds_vision = torch.tensor([limit_vision for i in range(len(vision_model.vision_model.encoder.layers))])
    else:
        sorted_mlp_metric, _ = torch.sort(score, descending=True)
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(num_layers)])
        sorted_mlp_metric_vision, _ = torch.sort(score_vision, descending=True)
        thresholds_vision = torch.tensor([sorted_mlp_metric_vision[i][int(sorted_mlp_metric_vision.shape[1]*(1-args.pruning_ratio))] for i in range(len(vision_model.vision_model.encoder.layers))])
    mlp_mask = (score.t() >= thresholds).t()
    mlp_mask_vision = (score_vision.t() >= thresholds_vision).t()
    print('*'*30)
    for idx in range(num_layers):
        compress(model.model.layers[idx], mlp_mask[idx], device)
    for idx in range(len(vision_model.vision_model.encoder.layers)):
        compress_vision(vision_model.vision_model.encoder.layers[idx], mlp_mask_vision[idx], device)

    model.zero_grad()
