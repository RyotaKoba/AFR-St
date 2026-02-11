import sys
import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from importlib.metadata import version

from lib.prune import ReFer_SVD, snip, AFR, structured_snip, Structured_ReFer_SVD, Structured_AFR, Structured_AFR_LLaVA
import lib.prune as pruner
from lib.model import rm_modules
import torch.nn.utils.prune as prunee
from transformers.models.llama.modeling_llama import LlamaMLP
from lib.builder import load_pretrained_model


print('torch', version('torch'))  # 2.1.0
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm_gpu(args):
    if args.model == "liuhaotian/llava-v1.5-13b":
        llava_model_args = {}
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model,None,'llava-v1.5-13b',device_map="auto",**llava_model_args)
        return model, tokenizer, torch.device("cuda:0"), image_processor
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            # dtype=torch.float32,
            # cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        device = torch.device("cuda:0")
        model.seqlen = 1024
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        return model, tokenizer, device, None

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

    parser.add_argument('--cuda_friendly', action="store_true")
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument("--prune_method", type=str, default="structured_afr", choices=["refer_svd","snip","structured_snip","structured_refer_svd","structured_afr","afr","structured_afr_llava"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument('--cuda', action="store_true")
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

    if args.prune_method == "structured_snip":
        structured_snip(args, model, tokenizer, device)
    elif args.prune_method == "structured_refer_svd":
        Structured_ReFer_SVD(args, model, tokenizer, device)
    elif args.prune_method == "structured_afr":
        Structured_AFR(args, model, tokenizer, device)
    elif args.prune_method == "structured_afr_llava":
        Structured_AFR_LLaVA(args, model, tokenizer, device, image_processor)
    elif args.prune_method == "refer_svd":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = ReFer_SVD(args, model, tokenizer,device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
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
        rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "afr":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = AFR(args, model, tokenizer, device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,cache_dir=args.cache_dir,device_map="auto")
        model.load_state_dict(init_data)
        model.seqlen = 1024
        rm_module = rm_modules(model)
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)

    if args.save_model and args.prune_method != "none" and args.prune_method != "done":
        if args.prune_method != "structured_snip" and args.prune_method != "structured_refer_svd" and args.prune_method != "structured_afr" and args.prune_method != "structured_afr_llava":
            for module in model.modules():
                if isinstance(module, LlamaMLP):
                    prunee.remove(module.gate_proj, 'weight')
                    prunee.remove(module.up_proj, 'weight')
                    prunee.remove(module.down_proj, 'weight')
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)

        # intermediate_sizes = [layer.mlp.gate_proj.out_features for layer in model.model.layers]
        if args.global_pruning:
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
