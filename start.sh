# Set common variables
model="meta-llama/Meta-Llama-3-8B"
# model="lmsys/vicuna-13b-v1.5"
# model="liuhaotian/llava-v1.5-13b"

CUDA_LAUNCH_BLOCKING=1 python3 main.py \
--model $model \
--prune_method "afr" \
--pruning_ratio 0.2 \
--nsamples 2 \
--cuda \
--save_model "./pruned_model/hoge" \
# --global_pruning \
# --dataset "wikitext" \
