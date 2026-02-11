import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT

from transformers import AutoModelForCausalLM

import json
import os
from datetime import datetime
from .data import get_loaders, get_mm_loaders
import math
from tqdm import tqdm
import sys
from .block_metrics import block_influence
import numpy as np
from .model import rm_modules, all_rm_modules, get_vision_rm_modules
from .gmm import gmm_edge_outlier_removal, select_K_by_bic
from .gesd import gesd_outlier_cleaning_torch
from .kde import kde_edge_outlier_removal
from .dpm import dpm_edge_outlier_removal
from .bmm import bmm_edge_outlier_removal

from sklearn.mixture import GaussianMixture

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

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    check_sparsity for llama3

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    # print(intermediate_size)
    # print(hidden_size)

    count = 0.0
    total_params = 0.0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0

        for name in subset:
            # print(name)
            W = subset[name].weight.data
            sub_count += W.numel()
            # print(W.numel())

            count += W.numel()

            if name == 'self_attn.q_proj' or name == 'self_attn.o_proj':

                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

            elif name == 'self_attn.k_proj' or name == 'self_attn.v_proj':

                total_params += (hidden_size * hidden_size / 8)
                sub_params += (hidden_size * hidden_size / 8)

            else:

                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    print(total_params)
    model.config.use_cache = use_cache
    return float(count)/total_params

def check_sparsity_refer(model):#, save_path):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    count = 0 
    total_params = 0
    sparsity_data =[]
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            # total_params += W.numel()

            sub_count += (W==0).sum().item()
            # sub_params += W.numel()
            
            if name == 'self_attn.q_proj' or name == 'self_attn.o_proj':

                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

            elif name == 'self_attn.k_proj' or name == 'self_attn.v_proj':

                total_params += (hidden_size * hidden_size / 4)
                sub_params += (hidden_size * hidden_size / 4)
            else:

                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

        sparsity = float(sub_count) / sub_params if sub_params > 0 else 0.0
        sparsity_data.append({"Layer": i, "Sparsity": sparsity})
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    # df = pd.DataFrame(sparsity_data)
    # df.to_csv(save_path, index=False)
    print(total_params)
    return float(count)/total_params, float(count)



def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)  # 2048 is the upper limit.

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compress(layer, mlp_mask, device):#形状を変更

    mlp_mask = mlp_mask.to(device)

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    print(layer.mlp.intermediate_size)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight
    layer.mlp.down_proj.in_features = mlp_mask.sum().item()

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()

def compress_vision(layer, mlp_mask, device):#形状を変更

    mlp_mask = mlp_mask.to(device)

    layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.fc2.weight.data = layer.mlp.fc2.weight.data[:,torch.where(mlp_mask)[0]]

    if layer.mlp.fc1.bias is not None:
        layer.mlp.fc1.bias.data = layer.mlp.fc1.bias.data[torch.where(mlp_mask)[0]]
    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.fc1.out_features = mlp_mask.sum().item()
    # layer.mlp.fc2.out_features = mlp_mask.sum().item()
    layer.mlp.fc2.in_features = mlp_mask.sum().item()

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()

# def compress(layer, mlp_mask, device): #形状は維持　０置換
#     mlp_mask = mlp_mask.to(device)
    
#     # 重みを直接削除する代わりに、マスクを使用して不要な重みを0に設定
#     # up_projの処理
#     masked_up_weight = torch.zeros_like(layer.mlp.up_proj.weight.data)
#     masked_up_weight[torch.where(mlp_mask)[0]] = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
#     layer.mlp.up_proj.weight.data = masked_up_weight
    
#     # gate_projの処理
#     masked_gate_weight = torch.zeros_like(layer.mlp.gate_proj.weight.data)
#     masked_gate_weight[torch.where(mlp_mask)[0]] = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
#     layer.mlp.gate_proj.weight.data = masked_gate_weight
    
#     # down_projの処理
#     masked_down_weight = torch.zeros_like(layer.mlp.down_proj.weight.data)
#     masked_down_weight[:, torch.where(mlp_mask)[0]] = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]
#     layer.mlp.down_proj.weight.data = masked_down_weight
    
#     # 元のサイズ情報を維持（テンソル形状は変更せず）
#     # layer.mlp.intermediate_size = 14336 # 元のサイズを維持
    
#     # スパース性を記録（オプション）
#     effective_neurons = mlp_mask.sum().item()
#     print(f"有効なニューロン数: {effective_neurons}/{len(mlp_mask)}")
    
#     # メモリの解放
#     torch.cuda.empty_cache()

# for flap
def compress_bias(layer, mlp_mask, mlp_mean_inp, device):

    bias = True

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    if bias:
        # Add the additional bias to compensate for the loss
        output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    if bias:
        # Re-initialize the Linear layer with new shape and bias
        layer.mlp.down_proj.in_features = mlp_mask.sum().item()
        # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
        layer.mlp.down_proj.bias.data = output_bias

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()




def prune_cfsp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    our method
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader= get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers


    mlp_metric_list = []
    mlp_mask = []

    layer_importances = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_importance = 0.0

        for j in range(args.nsamples):
            with torch.no_grad():
                rotary_impl = None
                for m in model.modules():
                    if hasattr(m, "rotary_emb"):
                        rotary_impl = m.rotary_emb
                        break
                if rotary_impl is None:
                    raise RuntimeError("rotary_impl not found on model")

                # prepare tensors
                x = inps[j].unsqueeze(0)                      # (1, seq_len, hidden)
                seq_len = x.shape[1]
                position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

                # call rotary_impl with the discovered signature
                try:
                    cos, sin = rotary_impl(x, position_ids)
                except Exception as e:
                    print("rotary_impl call failed:", repr(e))
                    raise

                # sanity checks
                # print("cos type/shape:", type(cos), getattr(cos, "shape", None))
                # print("sin type/shape:", type(sin), getattr(sin, "shape", None))
                if cos is None or sin is None:
                    raise RuntimeError("rotary_impl returned None for cos/sin")

                # attention_mask: make it match model expectation (bool works for many versions)
                attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=x.device)

                # final call
                outs[j] = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin)
                )[0]
                # --- patch end ---
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                if args.global_metrics == 'angular':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='angular').sum().cpu().item()
                elif args.global_metrics == 'cosine':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='cosine').sum().cpu().item()
                elif args.global_metrics == 'mse':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mse').sum().cpu().item()
                elif args.global_metrics == 'mae':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mae').sum().cpu().item()
                else:
                    layer_importance += 100
            # print("korenani, j:",j)
            



        layer_importances.append(layer_importance)
        for h in handles:
            h.remove()

        for name in subset:
            if args.local_metrics == "wanda_base":
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            elif args.local_metrics == "mag_base":
                W_metric = torch.norm(subset[name].weight.data, dim=0)

            elif args.local_metrics == "one_a":
                W = subset[name].weight.data
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**0.5

            elif args.local_metrics == "three_w_one_a":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W)/torch.sum(torch.abs(W), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "three_w_one_wa":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "one_wa":
                W = subset[name].weight.data
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_metric = (torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            # print("W_metric: ", W_metric.shape)
            W_metric = W_metric.mean(axis=0)
            # print("W_metric: ", W_metric.shape)
            mlp_metric_list.append(W_metric.cpu())

            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                rotary_impl = None
                for m in model.modules():
                    if hasattr(m, "rotary_emb"):
                        rotary_impl = m.rotary_emb
                        break
                if rotary_impl is None:
                    raise RuntimeError("rotary_impl not found on model")

                # prepare tensors
                x = inps[j].unsqueeze(0)                      # (1, seq_len, hidden)
                seq_len = x.shape[1]
                position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

                # call rotary_impl with the discovered signature
                try:
                    cos, sin = rotary_impl(x, position_ids)
                except Exception as e:
                    print("rotary_impl call failed:", repr(e))
                    raise

                # sanity checks
                # print("cos type/shape:", type(cos), getattr(cos, "shape", None))
                # print("sin type/shape:", type(sin), getattr(sin, "shape", None))
                if cos is None or sin is None:
                    raise RuntimeError("rotary_impl returned None for cos/sin")

                # attention_mask: make it match model expectation (bool works for many versions)
                attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=x.device)

                # final call
                outs[j] = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin)
                )[0]
                # --- patch end ---
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer

        torch.cuda.empty_cache()


    layer_importances_sorted = sorted(enumerate(layer_importances), key=lambda x: x[1], reverse=True)

    for i in range(len(layer_importances_sorted)):
        index2 = layer_importances_sorted[i][0]
        number2 = layer_importances_sorted[i][1]
        print(f"layer: {index2} , importance: {number2} ")

    print(f"{args.global_metrics} layer_importances_sorted: {layer_importances_sorted}")


    def sigmoid(x):
        return 1 / (1 + np.exp(-x*args.a))

    layer_importances_mid = sum(layer_importances) / len(layer_importances)

    layer_importances = [(i-layer_importances_mid)/1e4 for i in layer_importances]
    layer_importances = [sigmoid(i) for i in layer_importances]


    avg = sum(layer_importances) / len(layer_importances)
    max_score = max(layer_importances)
    if max_score / avg * (1-args.pruning_ratio) >= 1:
        #
        scale_factor = (avg * (1 / (1-args.pruning_ratio) - 1)) /  (max_score - avg) / 1.05
        for i in range(len(layer_importances)):
            if layer_importances[i] > avg:
                layer_importances[i] = avg + (layer_importances[i] - avg) * scale_factor
            else:
                layer_importances[i] = avg - (avg - layer_importances[i]) * scale_factor
        avg = sum(layer_importances) / len(layer_importances)

    print("mlp_metric_list:", len(mlp_metric_list))
    print("mlp_metric_list:", len(mlp_metric_list[0]))
    mlp_metric = torch.stack(mlp_metric_list)
    print("mlp_metric: ", mlp_metric.shape)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)
    # print(sorted_mlp_metric.shape)

    every_pruning_ratios = [i/avg*(1-args.pruning_ratio) for i in layer_importances]
    print(f"every_pruning_ratios: {every_pruning_ratios}")



    if args.cuda_friendly:
        thresholds = torch.tensor([
            sorted_mlp_metric[i][int(((sorted_mlp_metric.shape[1]*every_pruning_ratios[i])+64)/128)*128-1]
                                   for i in range(len(every_pruning_ratios))
                                   ])
        print(f"thresholds: {thresholds}")

    else:
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*every_pruning_ratios[i])] for i in range(len(every_pruning_ratios))])
        print(f"thresholds: {thresholds}")


    if len(every_pruning_ratios) == len(layers):
        print("そのまま使える〜〜")

    mlp_mask = (mlp_metric.t() >= thresholds).t()
    print(mlp_mask.shape)
    print("mlp_mask: ", mlp_mask)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def snip(args, model, tokenizer, device):
    device = [i + 1 for i in range(device - 1)]
    # model = nn.DataParallel(model, device_ids=device).to('cuda:1')
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inp, tar) in enumerate(dataloader):
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        loss = nn.CrossEntropyLoss()(outputs, tar)
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        print("score: ", len(score))
        score = torch.cat(score)
    print("score: ", len(score))
    
    model.zero_grad()
    return score

def structured_snip(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    
    print("loading calibdation data")
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    # inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    # print(f"calied inp:{inps[0].unsqueeze(0).shape}")
    
    rm_module = rm_modules(model)
    
    rm_weights = [module.weight for module, _ in rm_module]
    
    # input_batch = torch.stack([item[0] for item in dataloader]).squeeze(1)  # バッチの最初の要素を取得
    for i, (inp, tar) in enumerate(dataloader):
        print(f"no calied inp:{inp.shape}")
        # inp = inp.to(device)
        # tar = tar.to(device)]
        inp = inp.to("cuda:0")
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
    # mlp_mask = (mlp_metric.t() <= thresholds).t() #reverse
    
    print(mlp_mask)
    print(mlp_mask.shape)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.zero_grad()  # 勾配をリセット

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

loss = torch.zeros(1)
def ReFer_L1(args, model,tokenizer, device):
    device = [i + 1 for i in range(device - 1)]
    
    print("loading calibdation data")
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]


    
    global loss
    def store_feature(module, input, output):
        global loss
        # if 'LlamaMLP' in module.__class__.__name__:
            # loss = loss + output.abs().sum().to('cuda:1')
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        loss = loss + output.abs().sum()

    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)
    # model = nn.DataParallel(model, device_ids=device)
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        outputs = model(inputs)  # フォワードパス
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    # for batch in dataloader:
    #     outputs = model(batch[0])
    #     print(loss.shape)
    #     print(rm_weights[0].shape)

    #     grads = list(torch.autograd.grad(loss, rm_weights))
    #     # print(grads)
    #     break
    
    with torch.no_grad():
        score=[(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    # loss = torch.zeros(1).to('cuda:1')

    return score
    
P_SVD_loss = torch.zeros(1)
def AFR(args, model, tokenizer, device):
    device = [i for i in range(device)]
    print("loading calibdation data")
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    # 特徴空間の損失を取る関数．hookで呼ばれる．
    # def store_feature(module, input, output):
    #     global P_SVD_loss
        
    #     # if 'LlamaMLP' in module.__class__.__name__:


    #     if hasattr(output, 'last_hidden_state'):
    #         output = output.last_hidden_state
    #         return
    #     elif hasattr(output, 'hidden_states'):
    #         output = output.hidden_states
    #         return
    #     elif isinstance(output, tuple):
    #         output = output[0]
       

    #     if output is None:
    #         print("output is None")
    #         return
        
    #     # # 出力テンソルの特異値分解を実行
    #     # print("before output: ", output.shape)
    #     # output = output.reshape(output.size(0), -1)  # バッチサイズ次元以外をフラットにする
    #     # print("after output: ", output.shape)
    #     # U, S, Vh = torch.svd(output)  # SVDを計算
    #     # # print("S: ", S.shape)
    #     # # print("S: ", S)1sannpuru
    #     # # 特異値の平均を計算してP_SVD_lossに加算
    #     # singular_value_mean = S.mean()
    #     # # print("singular_value_mean: ", singular_value_mean)
    #     # # P_SVD_loss += singular_value_mean
    #     # P_SVD_loss = P_SVD_loss + singular_value_mean

    #      # [1, seq, dim] → [seq, dim] - これが重要！
    #     if output.dim() == 3 and output.size(0) == 1:
    #         output = output.squeeze(0)  # [1024, 4096]
            
    #     print(f"True SVD shape: {output.shape}")
        
    #     # 真の特異値分解
    #     U, S, Vh = torch.svd(output)  # [1024, 4096]のSVD
    #     print(f"Singular values shape: {S.shape}")  # [1024]
    #     print(f"Number of singular values: {len(S)}")
        
    #     # 特異値の平均（複数の特異値から）
    #     singular_value_mean = S.mean()
    #     P_SVD_loss = P_SVD_loss + singular_value_mean
        
    #     print(f"True SVD mean: {singular_value_mean.item():.6f}")
    def store_feature(module, input, output):
        global P_SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:

        if hasattr(output, 'last_hidden_state'):
        # transformers の出力オブジェクトから hidden_states を取得
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
        # hidden_states が利用可能な場合
            output = output.hidden_states
        elif isinstance(output, tuple):
        # タプルの場合は最初の要素を使用
            output = output[0]

        if output is None:
            return

        # 出力テンソルの特異値分解を実行
        # output = output.reshape(output.size(0), -1).to(dtype=torch.float32)  # バッチサイズ次元以外をフラットにする
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")  # バッチサイズ次元以外をフラットにする
        # U, S, Vh = torch.svd(output)  # SVDを計算
        S = torch.linalg.svdvals(output)
        # 特異値の平均を計算してSVD_lossに加算
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)
        # P_SVD_loss += singular_value_mean
        P_SVD_loss = P_SVD_loss + singular_value_mean

    # 全てのモジュールに対してhookをかける．
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    # model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)


    rm_weights = [module.weight for module, _ in rm_module]  # FOが枝刈りを担当する重み

    # 勾配初期化
    with torch.no_grad():
        fo_grads = [torch.zeros_like(w) for w in rm_weights]

    # 1バッチのみ処理して勾配を計算
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        inputs = inputs.to("cuda:0")
        targets = targets.to("cuda:0")
        outputs = model(inputs)  # フォワードパス
        break
    print("P_SVD_loss: ", P_SVD_loss)
    print("P_SVD_loss: ", P_SVD_loss.shape)
    del inputs
    del dataloader

    # fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))  # FOの勾配を計算
    print("before calc fo_grads")
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights,retain_graph=True))
    print("after calc fo_grads")
    for hook in hooks:
        hook.remove() # メモリ消費えぐいのでhookを外す
    
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

    # outputs = model(inputs)
    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    print("before calc snip_loss")
    loss = nn.CrossEntropyLoss()(outputs, targets) # CE Loss
    print("after calc snip_loss")
    del outputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        snip_grads = [torch.zeros_like(w) for w in rm_weights]
    print("before calc snip_grads")
    snip_grads = list(torch.autograd.grad(loss, rm_weights)) #SNIPの勾配を計算
    print("after calc snip_grads")    
    # SNIPスコアの計算
    print("before calc snip_score")
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        snip_score = [(weight * grad).view(-1).abs() for weight, grad in zip(rm_weights, snip_grads)]
        snip_score = torch.cat(snip_score)
    print("after calc snip_score")
    del snip_grads
    
    # fo_score と snip_score を標準化
    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    
    # FOスコアとSNIPスコアの結合
    score = fo_score_standardized + snip_score_standardized
    # score = snip_score_standardized
    # score = fo_score_standardized
    

    
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
    print("Start ReFer_SVD")
    device = [i + 1 for i in range(device - 1)]
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    global SVD_loss
    # 特異値分解のフックを定義
    def store_feature(module, input, output):
        global SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:

        if hasattr(output, 'last_hidden_state'):
        # transformers の出力オブジェクトから hidden_states を取得
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
        # hidden_states が利用可能な場合
            output = output.hidden_states
        elif isinstance(output, tuple):
        # タプルの場合は最初の要素を使用
            output = output[0]

        if output is None:
            return


        # 出力テンソルの特異値分解を実行
        output = output.reshape(output.size(0), -1)  # バッチサイズ次元以外をフラットにする
        U, S, Vh = torch.svd(output)  # SVDを計算
        # 特異値の平均を計算してSVD_lossに加算
        singular_value_mean = S.mean()
        # SVD_loss += singular_value_mean
        SVD_loss = SVD_loss + singular_value_mean
        
    # モデル内の各モジュールにフックを追加
    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)

    # model = nn.DataParallel(model, device_ids=device).to('cuda:1')

    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(1), targets.cuda(1)
        outputs = model(inputs)

        grads = list(torch.autograd.grad(SVD_loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    #del loss
    loss = torch.zeros(1)

    return score

SVD_loss = torch.zeros(1)
def Structured_ReFer_SVD(args, model, tokenizer, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("Start ReFer_SVD")
    # device = [i + 1 for i in range(device - 1)]
    print("loading calibdation data")
    dataloader= get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    class WeightScoreLogger:
        def __init__(self, save_dir="weight_scores"):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # FOとSNIPのスコアを分けて保存
            self.fo_weight_scores = []
            self.snip_weight_scores = []
            
            # メタデータ
            self.metadata = {
                'timestamp': datetime.now().isoformat(),
                'layers_processed': 0,
                'score_shape': None
            }
        
        def save_fo_layer_scores(self, layer_idx, W_metric):
            """FOの重み単位スコアを保存"""
            # [hidden_dim, intermediate_size] の形状で保存
            self.fo_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
            
        def save_snip_layer_scores(self, layer_idx, W_metric):
            """SNIPの重み単位スコアを保存"""
            self.snip_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
        
        def save_to_files(self):
            """ファイルに保存"""
            # FOスコア保存
            torch.save(self.fo_weight_scores, 
                    os.path.join(self.save_dir, 'fo_weight_scores.pt'))
            
            # SNIPスコア保存
            torch.save(self.snip_weight_scores, 
                    os.path.join(self.save_dir, 'snip_weight_scores.pt'))
            
            # メタデータ保存
            with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Saved weight scores to {self.save_dir}")
            print(f"FO layers: {len(self.fo_weight_scores)}")
            print(f"SNIP layers: {len(self.snip_weight_scores)}")

    def calculate_neuron_score_v2(W_metric):
        """
        方法2: Signal-to-Noise比的なアプローチ
        平均の絶対値 / (標準偏差 + epsilon)
        """

        """トリム平均を使用してニューロンスコアを計算"""
        # 上位5%と下位5%を除外する設定
        trim_percent = 2
        
        # 各列をソートして上位・下位を除外
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]  # 4096
        
        # 除外する要素数を計算
        trim_count = int(n_rows * trim_percent / 100)
        
        # 中央部分を抽出して平均を計算
        trimmed_W = sorted_W[trim_count:-trim_count, :]
        mean_scores = trimmed_W.mean(axis=0)
        std_scores = trimmed_W.std(axis=0)
        snr_scores = torch.abs(mean_scores) / (std_scores + 1e-8)
        return snr_scores

    global SVD_loss
    SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")


    # logger = WeightScoreLogger(save_dir=f"weight_scores_{args.pruning_ratio}")

    # 特異値分解のフックを定義
    def store_feature(module, input, output):
        global SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:

        if hasattr(output, 'last_hidden_state'):
        # transformers の出力オブジェクトから hidden_states を取得
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
        # hidden_states が利用可能な場合
            output = output.hidden_states
        elif isinstance(output, tuple):
        # タプルの場合は最初の要素を使用
            output = output[0]

        if output is None:
            return

        # 出力テンソルの特異値分解を実行
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32)  # バッチサイズ次元以外をフラットにする
        U, S, Vh = torch.svd(output)  # SVDを計算
        # 特異値の平均を計算してSVD_lossに加算
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)
        # SVD_loss += singular_value_mean
        SVD_loss = SVD_loss + singular_value_mean
        
    # モデル内の各モジュールにフックを追加
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(1), targets.cuda(1)
        outputs = model(inputs)
        break
    print("P_SVD_loss contains nan:", torch.isnan(SVD_loss).any().item())
    print("P_SVD_loss contains inf:", torch.isinf(SVD_loss).any().item())
    print("P_SVD_loss tensor dtype:", SVD_loss.dtype)
    print("P_SVD_loss shape:", SVD_loss.shape)
    grads = list(torch.autograd.grad(SVD_loss, rm_weights))
    for i, grad in enumerate(grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"fo_grads[{i}] contains inf or nan")
        else:
            print(f"fo_grads[{i}] is clean")
    for hook in hooks:
        hook.remove() # メモリ消費えぐいのでhookを外す
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
            logger.save_fo_layer_scores(i, W_metric)
            W_metric = calculate_neuron_score_v2(W_metric)
            # W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())
    mlp_metric = torch.stack(mlp_metric_list)
    print("score contains nan:", torch.isnan(mlp_metric).any().item())
    print("score contains inf:", torch.isinf(mlp_metric).any().item())
    print("score: ", mlp_metric.shape)
    print("score: ", mlp_metric)
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
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logger.save_to_files()


loss = torch.zeros(1)
def Structured_ReFer_L1(args, model,tokenizer, device):
    # device = [i + 1 for i in range(device - 1)]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("loading calibdation data")
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    
    global loss
    def store_feature(module, input, output):
        global loss
        # if 'LlamaMLP' in module.__class__.__name__:
            # loss = loss + output.abs().sum().to('cuda:1')
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return

        loss = loss + output.abs().sum().to("cpu")

    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        inputs = inputs.to("cuda:0")
        outputs = model(inputs)  # フォワードパス
        grads = list(torch.autograd.grad(loss, rm_weights))
        break
    layers = model.model.layers
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * grads[i+64]
            W_up = (rm_weights[i+32] * grads[i+32]).t()
            W_gate = (rm_weights[i] * grads[i]).t()
            W_metric = W_down + W_up + W_gate
            W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())

    mlp_metric = torch.stack(mlp_metric_list)
    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven
    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    thresholds = thresholds.to(mlp_metric.device)
    mlp_mask = (mlp_metric.t() >= thresholds).t()
    

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    model.zero_grad()


P_SVD_loss = torch.zeros(1)

def Structured_AFR(args, model, tokenizer, device, image_processor):

    use_cache = model.config.use_cache
    print("use_cache:", use_cache)
    model.config.use_cache = False
    if args.model != "meta-llama/Meta-Llama-3-8B" and args.model != "lmsys/vicuna-13b-v1.5":
        dataloader = get_mm_loaders()
    else:
        dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    
    def calculate_neuron_score_v2(W_metric):
        """
        Neuron score aggregation methods

        Current best: Simple mean + abs (baseline)
        Various alternative aggregation methods are provided below.
        Comment/uncomment to switch between methods.
        """

        # ============================================================
        # Method 0: Simple mean + abs (BASELINE - currently best)
        # ============================================================
        # mean_scores = W_metric.mean(axis=0)
        # return torch.abs(mean_scores)
        
        # ============================================================
        # Method 1: Trimmed + MeanAbs (alternative trimmed mean)
        # # ============================================================
        # trim_percent = 2
        # sorted_W, _ = torch.sort(W_metric, dim=0)
        # n_rows = W_metric.shape[0]
        # trim_count = int(n_rows * trim_percent / 100)
        # trimmed_W = sorted_W[trim_count:-trim_count, :]
        # mean_scores = trimmed_W.mean(axis=0)
        # return torch.abs(mean_scores)
    
        # ============================================================
        # Method 2: GMM Trim + MeanAbs (experimental)
        # ============================================================
        # cleaned_scores, _, _= gmm_edge_outlier_removal(W_metric, K=3, alpha_edge=0.05,q_tail=0.02, use_density=True, check_bic=True, K_range=(1, 5))
        # mean_scores = cleaned_scores.mean(axis=0)
        # return torch.abs(mean_scores)

        # ============================================================
        # Method 3: GESD Trim + MeanAbs (experimental)
        # ============================================================
        # cleaned_scores = gesd_outlier_cleaning_torch(W_metric)
        # return cleaned_scores
        
        # ===========================================================
        # Method 4: KDE Trim + MeanAbs (experimental)
        # ===========================================================
        # cleaned_scores = kde_edge_outlier_removal(W_metric)
        # mean_scores = cleaned_scores.mean(axis=0)
        # return torch.abs(mean_scores)
        
        # ===========================================================
        # Method 5: DPM Trim + MeanAbs (experimental)
        # ===========================================================
        # cleaned_scores = dpm_edge_outlier_removal(W_metric)
        # mean_scores = cleaned_scores.mean(axis=0)
        # return torch.abs(mean_scores)
        
        # ===========================================================
        # Method 6: BMM Trim + MeanAbs (experimental)
        # ===========================================================
        # cleaned_scores = bmm_edge_outlier_removal(W_metric)
        # mean_scores = cleaned_scores.mean(axis=0)
        # return torch.abs(mean_scores)

    import torch

    def weightwise_outlier_then_aggregate(array, threshold=3.5):
        """
        ウェイトレベルで外れ値を除去してから集約
        
        Args:
            array: (4096, 14336)
            threshold: 外れ値閾値
        """
        # 全体をフラットにして外れ値除去
        flat = array.reshape(-1)
        
        median = flat.median()
        mad = (flat - median).abs().median()
        
        if mad < 1e-8:
            return array.mean(dim=0)
        
        z_scores = torch.abs(flat - median) / (1.4826 * mad)
        mask = z_scores <= threshold
        
        # マスクを元の形状に戻して列ごとに平均
        mask_2d = mask.reshape(array.shape)
        
        result = torch.zeros(array.shape[1], device=array.device, dtype=array.dtype)
        for col_idx in range(array.shape[1]):
            col_mask = mask_2d[:, col_idx]
            if col_mask.any():
                result[col_idx] = array[:, col_idx][col_mask].mean()
            else:
                result[col_idx] = array[:, col_idx].mean()
        
        return torch.abs(result)
        # return result

    class WeightScoreLogger:
        def __init__(self, save_dir="weight_scores"):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # FOとSNIPのスコアを分けて保存
            self.fo_weight_scores = []
            self.snip_weight_scores = []
            
            # メタデータ
            self.metadata = {
                'timestamp': datetime.now().isoformat(),
                'layers_processed': 0,
                'score_shape': None
            }
        
        def save_fo_layer_scores(self, layer_idx, W_metric):
            """FOの重み単位スコアを保存"""
            # [hidden_dim, intermediate_size] の形状で保存
            self.fo_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
            
        def save_snip_layer_scores(self, layer_idx, W_metric):
            """SNIPの重み単位スコアを保存"""
            self.snip_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
        
        def save_to_files(self):
            """ファイルに保存"""
            # FOスコア保存
            torch.save(self.fo_weight_scores, 
                    os.path.join(self.save_dir, 'fo_weight_scores.pt'))
            
            # SNIPスコア保存
            torch.save(self.snip_weight_scores, 
                    os.path.join(self.save_dir, 'snip_weight_scores.pt'))
            
            # メタデータ保存
            with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Saved weight scores to {self.save_dir}")
            print(f"FO layers: {len(self.fo_weight_scores)}")
            print(f"SNIP layers: {len(self.snip_weight_scores)}")

    global P_SVD_loss
    P_SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")
    print("P_SVD_loss shape:", P_SVD_loss.shape)

    # logger = WeightScoreLogger(save_dir=f"weight_scores_{args.pruning_ratio}")

    def store_feature(module, input, output):
        global P_SVD_loss

        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]
        # print("module:", module)

        if output is None:
            return
    
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")  # バッチサイズ次元以外をフラットにする
        # print("reshaped outputs contains nan:", torch.isnan(output).any().item())
        # print("reshaped outputs contains inf:", torch.isinf(output).any().item())
        # print(f"Shape: {output.shape}, dtype: {output.dtype}")
        # print(f"Min: {output.min().item():.6f}, Max: {output.max().item():.6f}")

        S = torch.linalg.svdvals(output)
        # _,S,_ = torch.svd(output)
        singular_value_mean = S.mean().to("cpu", dtype=torch.float32)
        if torch.isnan(singular_value_mean).any().item():
            print("singular_value_mean contains nan is in module:", module.__class__.__name__)
        P_SVD_loss = P_SVD_loss + singular_value_mean
        
        del S, output
        torch.cuda.empty_cache()

    if args.vision:
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_model = vision_tower.vision_tower
        vision_model.requires_grad_(True)
        


    if args.model != "meta-llama/Meta-Llama-3-8B" and args.model != "lmsys/vicuna-13b-v1.5":
        hooks = []
        if args.vision:
            for name, module in vision_model.named_modules():
                hook = module.register_forward_hook(store_feature)
                hooks.append(hook)
        else:
            for name, module in model.model.named_modules():
            # vision_towerとmm_projectorを除外
                if 'vision_tower' not in name and 'mm_projector' not in name:
                    hook = module.register_forward_hook(store_feature)
                    hooks.append(hook)
    elif args.model == "meta-llama/Meta-Llama-3-8B" or args.model == "lmsys/vicuna-13b-v1.5":
        hooks = []
        for name, module in model.named_modules():
            hook = module.register_forward_hook(store_feature)
            hooks.append(hook)

        
    # hooks = []
    # for name, module in model.named_modules():
    #     # gate_proj, up_proj, down_projのみ
    #     if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
    #         hook = module.register_forward_hook(store_feature)
    #         hooks.append(hook)
    #         print(f"Hooked: {name}")
    if args.vision:
        rm_module = get_vision_rm_modules(vision_model)
        rm_weights = [module.weight for module, _ in rm_module]
    else:
        rm_module = rm_modules(model)
        rm_weights = [module.weight for module, _ in rm_module]  # FOが枝刈りを担当する重み

    # FOフェーズ: SVDベースの勾配計算
    if args.model != "meta-llama/Meta-Llama-3-8B" and args.model != "lmsys/vicuna-13b-v1.5":
        for batch in dataloader:
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
            if args.vision:
                outputs = vision_model(image_tensor)
            else:
                outputs = model(input_ids=inputs, images=image_tensor)
            break
    else:
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to("cuda:0")
            outputs = model(inputs)
            break
    

    print("P_SVD_loss contains nan:", torch.isnan(P_SVD_loss).any().item())
    print("P_SVD_loss contains inf:", torch.isinf(P_SVD_loss).any().item())

    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights,retain_graph=True))
    for i, grad in enumerate(fo_grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"fo_grads[{i}] contains inf or nan")
        else:
            print(f"fo_grads[{i}] is clean")

    for hook in hooks:
        hook.remove()

    P_SVD_loss = torch.zeros(1)
    model.zero_grad()
    if args.vision:
        layers = vision_model.vision_model.encoder.layers
    else:
        layers = model.model.layers
    mlp_metric_list = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            if (args.model == "lmsys/vicuna-13b-v1.5" or args.model == "liuhaotian/llava-v1.5-13b") and not args.vision:
                W_down = rm_weights[i+80] * fo_grads[i+80]
                W_up = (rm_weights[i+40] * fo_grads[i+40]).t()
                W_gate = (rm_weights[i] * fo_grads[i]).t()
            elif args.vision:
                W_fc1 = (rm_weights[i*2] * fo_grads[i*2]).t()  # [intermediate_size, hidden_size]
                W_fc2 = rm_weights[i*2 + 1] * fo_grads[i*2 + 1]  # [hidden_size, intermediate_size]
            else:
                W_down = rm_weights[i+64] * fo_grads[i+64]
                W_up = (rm_weights[i+32] * fo_grads[i+32]).t()
                W_gate = (rm_weights[i] * fo_grads[i]).t()
            # print("W_down shape:", W_down.shape)
            # print("W_up shape:", W_up.shape)
            # print("W_gate shape:", W_gate.shape)
            W_down = torch.abs(W_down)
            W_up = torch.abs(W_up)
            W_gate = torch.abs(W_gate)
            if args.vision:
                W_metric = W_fc1 + W_fc2  # [intermediate_size, intermediate_size]
            else:
                W_metric = W_down + W_up + W_gate
            # W_metric = torch.abs(W_metric)
            W_metric = torch.mean(W_metric, axis=0)
            # W_metric = calculate_neuron_score_v2(W_metric)
            # W_metric = weightwise_outlier_then_aggregate(W_metric)
            mlp_metric_list.append(W_metric.cpu())
            print(f"SVD_W_metric3 {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"SVD_W_metric3 {i} have inf:", torch.isinf(W_metric).any().item())
    fo_score = torch.stack(mlp_metric_list)

    # SNIPフェーズ: 勾配ベースの重要度計算
    # for i, (inputs, targets) in enumerate(dataloader):
    #     outputs = model(inputs)
    #     break
    del inputs
    if args.vision:
        del image_tensor
        del W_fc1
        del W_fc2
    else:
        del W_down
        del W_up
        del W_gate
    del W_metric
    del fo_grads
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    outputs = outputs.logits
    if args.model == "liuhaotian/llava-v1.5-13b":
        target_len = targets.shape[1]
        outputs = outputs[:, -target_len:, :]
    outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
    targets = targets.reshape(-1).to(dtype=torch.long, device="cuda:0")
    loss = nn.CrossEntropyLoss()(outputs, targets)
    del outputs
    del targets
    torch.cuda.empty_cache()
    snip_grads = list(torch.autograd.grad(loss, rm_weights))
    for i, grad in enumerate(snip_grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"snip_grads[{i}] contains inf or nan")
        else:
            print(f"snip_grads[{i}] is clean")

    mlp_metric_list = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            if args.model == "lmsys/vicuna-13b-v1.5" or args.model == "liuhaotian/llava-v1.5-13b":
                W_down = rm_weights[i+80] * snip_grads[i+80]
                W_up = (rm_weights[i+40] * snip_grads[i+40]).t()
                W_gate = (rm_weights[i] * snip_grads[i]).t()
            else:
                W_down = rm_weights[i+64] * snip_grads[i+64]
                W_up = (rm_weights[i+32] * snip_grads[i+32]).t()
                W_gate = (rm_weights[i] * snip_grads[i]).t()
            W_down = torch.abs(W_down)
            W_up = torch.abs(W_up)
            W_gate = torch.abs(W_gate)
            W_metric = W_down + W_up + W_gate
            # W_metric = torch.abs(W_metric)
            W_metric = torch.mean(W_metric, axis=0)
            # W_metric = calculate_neuron_score_v2(W_metric)
            # W_metric = weightwise_outlier_then_aggregate(W_metric)
            mlp_metric_list.append(W_metric.cpu())
            print(f"snip_W_metric3 {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"snip_W_metric3 {i} have inf:", torch.isinf(W_metric).any().item())
    snip_score = torch.stack(mlp_metric_list)
    del snip_grads
    torch.cuda.empty_cache()

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    score = fo_score_standardized + snip_score_standardized
    # score_means =  []
    # for i in range(len(layers)):
    #     print(f"score[{i}]: ", score[i].shape)
    #     score_mean = calculate_neuron_score_v2(score[i]) #ここを変えて集約方法変更
    #     score_means.append(score_mean)
    # score = torch.stack(score_means)
    # print("score after mean: ", score.shape)
    if args.global_pruning:
        sorted_mlp_metric, _ = torch.sort(score.view(-1), descending=True)
        limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]
        thresholds = torch.tensor([limit for i in range(len(layers))])    
    else:
        sorted_mlp_metric, _ = torch.sort(score, descending=True)
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])
    mlp_mask = (score.t() >= thresholds).t()
    print('*'*30)
    for idx in range(len(layers)):
        compress(model.model.layers[idx], mlp_mask[idx], device)
    
    model.zero_grad()  # 勾配をリセット
    del P_SVD_loss  # グローバル変数をリセット
    P_SVD_loss = torch.zeros(1)  # グローバル変数の再初期化
    # logger.save_to_files()
    
    
def Structured_AFR2(args, model, tokenizer, device, image_processor):
    use_cache = model.config.use_cache
    print("use_cache:", use_cache)
    model.config.use_cache = False
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

        # return weight_wise_to_neuron_wise(cleaned_scores)
        return cleaned_scores

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
    
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")  # バッチサイズ次元以外をフラットにする

        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to("cpu", dtype=torch.float32)
        if torch.isnan(singular_value_mean).any().item():
            print("singular_value_mean contains nan is in module:", module.__class__.__name__)
        P_SVD_loss = P_SVD_loss + singular_value_mean
 
    hooks = []
    # for name, module in model.named_modules():
    #     hook = module.register_forward_hook(store_feature)
    #     hooks.append(hook)

    for name, module in model.named_modules():
        # gate_proj, up_proj, down_projのみ
        if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            hook = module.register_forward_hook(store_feature)
            hooks.append(hook)
            # print(f"Hooked: {name}")

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module] 
    fo_score = {}
    snip_score = {}
    fo_sign = {}
    snip_sign  = {}
    
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

                # weights_sign = torch.sign(W_down)
                # weights_sign.add_(torch.sign(W_up))
                # weights_sign.add_(torch.sign(W_gate))
                # torch.save(weights_sign,"../fo_weights_sign.pt")
                # import sys
                # sys.exit()
                
                W_metric = calculate_neuron_score(W_metric)
                W_metric = W_metric.mean(axis=0)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(num_layers):
                        fo_score[m] = torch.zeros_like(W_metric)
                        snip_score[m] = torch.zeros_like(W_metric)
                        fo_sign[m] = torch.zeros_like(W_metric)
                        snip_sign[m] = torch.zeros_like(W_metric)
                fo_sign[k].add_(torch.sign(W_metric))
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
                W_metric = W_metric.mean(axis=0)
                snip_sign[k].add_(torch.sign(W_metric))
                W_metric = torch.abs(W_metric)
                snip_score[k].add_(W_metric)
                
        del snip_grads
                
            
    for hook in hooks:
        hook.remove()
    P_SVD_loss = torch.zeros(1)
    del P_SVD_loss  # グローバル変数をリセット
    model.zero_grad()
    eps = 1e-12
    # for i in range(num_layers):
        # K = args.nsamples
        # mu = fo_score[i]/float(K)
        # m = torch.abs(fo_sign[i])/float(K)
        # fo_score[i] = mu * m + eps
        # mu = snip_score[i]/float(K)
        # m = torch.abs(snip_sign[i])/float(K)
        # snip_score[i] = mu * m + eps
        # fo_score[i] = fo_score[i]*fo_score[i]
        # snip_score[i] = snip_score[i]*snip_score[i]

    # fo_sign = torch.stack(list(fo_sign.values()), dim=0)
    # snip_sign = torch.stack(list(snip_sign.values()), dim=0)
    # torch.save(fo_sign,"../fo_sign.pt")
    # torch.save(snip_sign,"../snip_sign.pt")
    # del fo_sign,snip_sign
 
    fo_score = torch.stack(list(fo_score.values()), dim=0)
    snip_score = torch.stack(list(snip_score.values()), dim=0)

    del inputs
    del W_down
    del W_up
    del W_gate
    del W_metric
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    # fo_score_standardized = (fo_score - fo_score.min()) / (fo_score.max() - fo_score.min())
    # snip_score_standardized = (snip_score - snip_score.min()) / (snip_score.max() - snip_score.min())
    print("snip max:",torch.max(snip_score_standardized))
    print("snip min:",torch.min(snip_score_standardized))
    print("fo max:",torch.max(fo_score_standardized))
    print("fo min:",torch.min(fo_score_standardized))
    # torch.save(fo_score_standardized,"./fo_norm_score.pt")
    # torch.save(snip_score_standardized,"./snip_norm_score.pt")
    score = fo_score_standardized + snip_score_standardized

    print(score.shape)
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
    use_cache = model.config.use_cache
    print("use_cache:", use_cache)
    model.config.use_cache = False
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

        # return weight_wise_to_neuron_wise(cleaned_scores)
        return cleaned_scores

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
    
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")  # バッチサイズ次元以外をフラットにする

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

        # print("Vision module:", module.__class__.__name__)
        # print(output.requires_grad)
    
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32, device="cpu")  # バッチサイズ次元以外をフラットにする

        S = torch.linalg.svdvals(output)
        singular_value_mean = S.mean().to(dtype=torch.float32, device="cpu")
        # print("singular_value_mean in vision module:", singular_value_mean.item())
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
            # print(f"Hooked: {name}")
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
        # print(w.requires_grad, w.is_leaf)

    fo_score = {}
    snip_score = {}
    fo_score_vision = {}
    snip_score_vision = {}
    # fo_sign = {}
    # snip_sign  = {}
    
    num_layers = len(model.model.layers)
    i=0
    for batch in tqdm(dataloader):
        image = batch['image'][0]
        text = batch['text'][0]
        target_text = batch['target'][0]
        # 画像処理
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda().to(dtype=model.dtype)
        # image_tensor.requires_grad_(True)
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
        # print(P_SVD_loss_vision.grad_fn)

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

                # weights_sign = torch.sign(W_down)
                # weights_sign.add_(torch.sign(W_up))
                # weights_sign.add_(torch.sign(W_gate))
                # torch.save(weights_sign,"../fo_weights_sign.pt")
                # import sys
                # sys.exit()
                
                W_metric = calculate_neuron_score(W_metric)
                W_metric = W_metric.mean(axis=0)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(num_layers):
                        fo_score[m] = torch.zeros_like(W_metric)
                        snip_score[m] = torch.zeros_like(W_metric)
                        # fo_sign[m] = torch.zeros_like(W_metric)
                        # snip_sign[m] = torch.zeros_like(W_metric)
                # fo_sign[k].add_(torch.sign(W_metric))
                W_metric = torch.abs(W_metric)
                has_inf = torch.isinf(W_metric).any().item()
                has_nan = torch.isnan(W_metric).any().item()
                if has_inf or has_nan:
                    print(f"W_metric[{k}] contains inf or nan")
                fo_score[k].add_(W_metric)
                
        P_SVD_loss = torch.zeros(1)
        del fo_grads
        outputs = outputs.logits
        target_len = targets.shape[1]
        outputs = outputs[:, -target_len:, :]
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
        targets = targets.reshape(-1).to(dtype=torch.long, device="cuda:0")
        loss = nn.CrossEntropyLoss()(outputs, targets)
        # print(loss.requires_grad, loss.grad_fn)
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
                W_metric = W_metric.mean(axis=0)
                # snip_sign[k].add_(torch.sign(W_metric))
                W_metric = torch.abs(W_metric)
                snip_score[k].add_(W_metric)
                
        del snip_grads

###############vision part
        # P_SVD_loss_vision = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")
        # _ = vision_model(image_tensor)
        # for w in rm_weights_vision:
            # print(w.requires_grad, w.is_leaf)
        # P_SVD_loss_vision.requires_grad_(True)
        # print(P_SVD_loss_vision.requires_grad)
        # print(P_SVD_loss_vision.grad_fn)
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
                # print(W_fc1.shape,W_fc2.shape)
                W_metric = W_fc2 + W_fc1

                # weights_sign = torch.sign(W_down)
                # weights_sign.add_(torch.sign(W_up))
                # weights_sign.add_(torch.sign(W_gate))
                # torch.save(weights_sign,"../fo_weights_sign.pt")
                # import sys
                # sys.exit()
                
                W_metric = calculate_neuron_score(W_metric)
                W_metric = W_metric.mean(axis=0)
                if i == 0 and k == 0:
                    print("Initial Setting")
                    for m in range(len(vision_model.vision_model.encoder.layers)):
                        fo_score_vision[m] = torch.zeros_like(W_metric)
                        snip_score_vision[m] = torch.zeros_like(W_metric)
                        # fo_sign[m] = torch.zeros_like(W_metric)
                        # snip_sign[m] = torch.zeros_like(W_metric)
                # fo_sign[k].add_(torch.sign(W_metric))
                W_metric = torch.abs(W_metric)
                has_inf = torch.isinf(W_metric).any().item()
                has_nan = torch.isnan(W_metric).any().item()
                if has_inf or has_nan:
                    print(f"W_metric[{k}] contains inf or nan")
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
        i = i + 1
        if i >= args.nsamples:
            break
                
            
    for hook in hooks:
        hook.remove()
    for hook in hooks_vision:
        hook.remove()
    P_SVD_loss = torch.zeros(1)
    P_SVD_loss_vision = torch.zeros(1)
    del P_SVD_loss  # グローバル変数をリセット
    del P_SVD_loss_vision  # グローバル変数をリセット
    model.zero_grad()
    eps = 1e-12
    for i in range(num_layers):
        # K = args.nsamples
        # mu = fo_score[i]/float(K)
        # m = torch.abs(fo_sign[i])/float(K)
        # fo_score[i] = mu * m + eps
        # mu = snip_score[i]/float(K)
        # m = torch.abs(snip_sign[i])/float(K)
        # snip_score[i] = mu * m + eps
        # fo_score[i] = fo_score[i]*fo_score[i]
        snip_score[i] = snip_score[i]*snip_score[i]

    # fo_sign = torch.stack(list(fo_sign.values()), dim=0)
    # snip_sign = torch.stack(list(snip_sign.values()), dim=0)
    # torch.save(fo_sign,"../fo_sign.pt")
    # torch.save(snip_sign,"../snip_sign.pt")
    # del fo_sign,snip_sign
 
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
    import gc
    gc.collect()

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    fo_score_vision_standardized = (fo_score_vision - fo_score_vision.mean()) / fo_score_vision.std()
    # snip_score_vision_standardized = (snip_score_vision - snip_score_vision.mean()) / snip_score_vision.std()
    # fo_score_standardized = (fo_score - fo_score.min()) / (fo_score.max() - fo_score.min())
    # snip_score_standardized = (snip_score - snip_score.min()) / (snip_score.max() - snip_score.min())
    print("snip max:",torch.max(snip_score_standardized))
    print("snip min:",torch.min(snip_score_standardized))
    print("fo max:",torch.max(fo_score_standardized))
    print("fo min:",torch.min(fo_score_standardized))
    # torch.save(fo_score_standardized,"./fo_norm_score.pt")
    # torch.save(snip_score_standardized,"./snip_norm_score.pt")
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
    
    model.zero_grad()  # 勾配をリセット
