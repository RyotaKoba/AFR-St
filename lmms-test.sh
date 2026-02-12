# MPOLayerのインポートパスを設定
export PYTHONPATH="${PWD}/MPO_pruning_static:${PYTHONPATH}"

# accelerate launch --num_processes=1 -m lmms_eval --model llava   \
# --model_args pretrained="liuhaotian/llava-v1.5-13b,pruned="../Structured_AFR/pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_vision-and-lang/model.bin",device_map=auto"  \
# --tasks vizwiz_vqa_val,gqa \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix reproduce \
# --output_path ./logs/both_AFR_FullyLLaVApruned20%

# accelerate launch --num_processes=1 -m lmms_eval --model llava   \
# --model_args pretrained="liuhaotian/llava-v1.5-13b,pruned="../Structured_AFR/pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_vision-and-lang/model.bin",device_map=auto"  \
# --tasks scienceqa_img \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix reproduce \
# --output_path ./logs/both_AFR_FullyLLaVApruned20%
accelerate launch --num_processes=1 -m lmms_eval --model llava   \
--model_args pretrained="liuhaotian/llava-v1.5-13b,pruned="../Structured_AFR/pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_vision-and-lang/model.bin",device_map=auto"  \
--tasks vizwiz_vqa_val,gqa,scienceqa_img \
--batch_size 1 \
--log_samples \
--log_samples_suffix reproduce \
--output_path ./logs/both_AFR_FullyLLaVApruned20%


# accelerate launch --num_processes=1 -m lmms_eval --model llava   \
# --model_args pretrained="liuhaotian/llava-v1.5-13b,device_map=auto"  \
# --tasks scienceqa_img,vizwiz_vqa_val,gqa \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix reproduce \
# --output_path ./logs/pruned0%
