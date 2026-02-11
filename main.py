import sys
import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from importlib.metadata import version

from lib.prune import prune_cfsp, ReFer_L1,ReFer_SVD,check_sparsity,check_sparsity_refer, snip, AFR, structured_snip, Structured_ReFer_SVD, Structured_ReFer_L1, Structured_AFR, Structured_AFR2, Structured_AFR_LLaVA
from lib.eval import eval_ppl, show_model_input_output
import lib.prune as pruner
from lib.model import rm_modules, all_rm_modules
import torch.nn.utils.prune as prunee
from transformers.models.llama.modeling_llama import LlamaMLP
from lib.builder import load_pretrained_model


print('torch', version('torch'))  # 2.1.0
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

def get_llm_gpu(args):
    if args.model == "liuhaotian/llava-v1.5-13b":
        llava_model_args = {}
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model,None,'llava-v1.5-13b',device_map="auto",**llava_model_args)
        return model, tokenizer, torch.device("cuda:0"), image_processor
    elif args.model == "meta-llama/Meta-Llama-3-8B" or args.model == "lmsys/vicuna-13b-v1.5":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            # dtype=torch.float32,
            # trust_remote_code=True,
            # cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        device = torch.device("cuda:0")
        model.seqlen = 1024
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        return model, tokenizer, device, None
    else:
        model = None
        checkpoint = torch.load(args.model, map_location='cuda', weights_only=False)
        # 辞書からモデルを取り出す
        model = checkpoint['model']
        model.to(dtype=torch.float32)
        tokenizer = checkpoint['tokenizer']

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map="auto")
        image_processor = vision_tower.image_processor
        del checkpoint
        del vision_tower
        del mm_use_im_start_end
        del mm_use_im_patch_token
        torch.cuda.empty_cache()
        if hasattr(model.config, "max_sequence_length"):
            model._max_length = model.config.max_sequence_length
        else:
            model._max_length = 2048
        return model, tokenizer, torch.device("cuda:0"), image_processor

def get_llm_cpu(args):
    print("Loading model on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    print("Model loaded on CPU")
    model.seqlen = 1024
    device = torch.device("cuda:0")
    model.eval()
    print(f"args.model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument('--a', type=float, default=1, help='global control')
    parser.add_argument('--b', type=float, default=1, help='local control')
    parser.add_argument('--c', type=float, default=1, help='local control')
    parser.add_argument('--global_metrics', type=str, default="angular", help='angular, cosine, mse, mae, avg_base')
    parser.add_argument('--local_metrics', type=str, default="three_w_one_wa", help='one_wa, one_a, three_w_one_a, three_w_one_wa, wanda_base, mag_base')

    parser.add_argument('--cuda_friendly', action="store_true")
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument("--prune_method", type=str, default="cfsp", choices=["cfsp","refer_l1","refer_svd","snip","structured_snip","structured_refer_svd","structured_refer_l1","structured_afr","afr","structured_afr_llava","none","done"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--vision', action="store_true")
    parser.add_argument('--sample', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument('--global_pruning', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if torch.cuda.is_available():
        print(" ---- CUDA is available! ------")
    else:
        print(" ---- no cuda! ------")

    # Prune the model
    print("pruning starts")
    
    if args.cuda:
        model, tokenizer, device, image_processor = get_llm_gpu(args)
    else:
        model, tokenizer, device, image_processor = get_llm_cpu(args)
    
    print(f"loading llm model {args.model}, pruning method: {args.prune_method}")
    
    if args.prune_method == "cfsp":
        prune_cfsp(args, model, tokenizer, device)#(1,788,2427)
    elif args.prune_method == "structured_snip":
        structured_snip(args, model, tokenizer, device)
    elif args.prune_method == "structured_refer_svd":
        Structured_ReFer_SVD(args, model, tokenizer, device)
    elif args.prune_method == "structured_refer_l1":
        Structured_ReFer_L1(args, model, tokenizer, device)
    elif args.prune_method == "structured_afr":
        # Structured_AFR(args, model, tokenizer, device, image_processor)
        Structured_AFR2(args, model, tokenizer, device, image_processor)
    elif args.prune_method == "structured_afr_llava":
        Structured_AFR_LLaVA(args, model, tokenizer, device, image_processor)
    elif args.prune_method == "refer_l1":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = ReFer_L1(args, model, tokenizer,device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "refer_svd":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = ReFer_SVD(args, model, tokenizer,device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "snip":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = snip(args, model, tokenizer, device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "afr":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = AFR(args, model, tokenizer, device)
        del model
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,cache_dir=args.cache_dir,device_map="auto")
        model.load_state_dict(init_data)
        model.seqlen = 1024
        import gc
        gc.collect()
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        # pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "none":
        print(f"loading llm model {args.model} without pruning")
        model.eval()
    elif args.prune_method == "done":
        print(f"loading llm model {args.model} with pruned model")
        model_path = "./pruned_model/Llama3-8B_AFR-St_0.5p_onlyFFN_sizeChanged"
        # model_path = "./llm_weights/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2" #vicuna-model
        # model_path = "./llm_weights/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920" #llama-model
        print("evaluate :", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
        model.seqlen = 1024
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("*"*30)
    # if args.prune_method == "refer_l1" or args.prune_method == "snip" or args.prune_method == "afr" or args.prune_method == "refer_svd" or args.prune_method == "done":
    sparsity_ratio, pruned_model_param = check_sparsity_refer(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {pruned_model_param}B")
    print("*"*30)
    
    if args.eval:
        print("Start evaluation")
        ppl = eval_ppl(args, model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")
    
    if args.sample:
        print("Start show sample")
        show_model_input_output(model, tokenizer, device)

    if args.save_model and args.prune_method != "none" and args.prune_method != "done":
        if args.prune_method != "cfsp" and args.prune_method != "structured_snip" and args.prune_method != "structured_refer_svd" and args.prune_method != "structured_refer_l1" and args.prune_method != "structured_afr" and args.prune_method != "structured_afr_llava":
            for module in model.modules():
                if isinstance(module, LlamaMLP):
                    prunee.remove(module.gate_proj, 'weight') 
                    prunee.remove(module.up_proj, 'weight') 
                    prunee.remove(module.down_proj, 'weight') 
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)

        # intermediate_sizes = [layer.mlp.gate_proj.out_features for layer in model.model.layers]
        if args.global_pruning or args.prune_method == "cfsp":
            save_name = os.path.join(args.save_model, f"model.bin")
            torch.save({'model': model, 'tokenizer': tokenizer,}, save_name)
        else:
            for layer in model.model.layers:
                interm_size = layer.mlp.gate_proj.out_features
                break
            # configを更新
            model.config.intermediate_size = interm_size
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
