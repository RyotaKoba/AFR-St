# Set common variables
model="meta-llama/Meta-Llama-3-8B"
# model="lmsys/vicuna-13b-v1.5"
# model="liuhaotian/llava-v1.5-13b"

CUDA_LAUNCH_BLOCKING=1 python3 main.py \
--model $model \
--prune_method "snip" \
--pruning_ratio 0.2 \
--nsamples 2 \
--cuda \
--save_model "./pruned_model/trash" \
# --global_pruning

# CUDA_LAUNCH_BLOCKING=1 python3 main.py \
# --model $model \
# --prune_method "structured_afr_llava" \
# --pruning_ratio 0.5 \
# --nsamples 128 \
# --cuda \
# --save_model "./pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_snipdouble_vision-and-lang" \
# --global_pruning