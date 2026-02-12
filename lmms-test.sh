export PYTHONPATH="${PWD}/MPO_pruning_static:${PYTHONPATH}" #大原くんのコードのパスを通す　使用不可

accelerate launch --num_processes=1 -m lmms_eval --model llava   \
--model_args pretrained="liuhaotian/llava-v1.5-13b,pruned="./pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_vision-and-lang/model.bin",device_map=auto"  \
--tasks vizwiz_vqa_val,gqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix reproduce \
--output_path ./logs/hoge

# accelerate launch --num_processes=1 -m lmms_eval --model llava   \
# --model_args pretrained="liuhaotian/llava-v1.5-13b,pruned="./pruned_model/LLaVA/AFR-St_0.5p_trimmed2%_global_vision-and-lang/model.bin",device_map=auto"  \
# --tasks scienceqa_img \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix reproduce \
# --output_path ./logs/hoge

# accelerate launch --num_processes=1 -m lmms_eval --model llava   \
# --model_args pretrained="liuhaotian/llava-v1.5-13b,device_map=auto"  \
# --tasks scienceqa_img,vizwiz_vqa_val,gqa \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix reproduce \
# --output_path ./logs/original
