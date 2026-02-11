# Set common variables
# model="meta-llama/Meta-Llama-3-8B"
#deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# model="lmsys/vicuna-13b-v1.5"
# CUDA_LAUNCH_BLOCKING=1 python analyzer.py
model="liuhaotian/llava-v1.5-13b"

CUDA_LAUNCH_BLOCKING=1 python3 main.py \
--model $model \
--prune_method "structured_afr_llava" \
--pruning_ratio 0.2 \
--nsamples 128 \
--a 1  \
--b 1  \
--c 1  \
--cuda \
--global_metrics angular \
--local_metrics three_w_one_wa \
--save_model "./pruned_model/LLaVA/AFR-St_0.2p_trimmed2%_global_snipdouble_vision-and-lang" \
--global_pruning \
# --vision \
# --protect_sw
# --pruning_ration : 枝刈り率

CUDA_LAUNCH_BLOCKING=1 python3 main.py \
--model $model \
--prune_method "structured_afr_llava" \
--pruning_ratio 0.5 \
--nsamples 128 \
--a 1  \
--b 1  \
--c 1  \
--cuda \
--global_metrics angular \
--local_metrics three_w_one_wa \
--save_model "./pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_snipdouble_vision-and-lang" \
--global_pruning \