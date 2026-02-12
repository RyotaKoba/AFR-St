######## HF format ###############
# lm-eval \
# --model hf \
# --batch_size 64 \
# --model_args device_map=auto,dtype=float16,pretrained=../pruned_model/hoge \
# --task winogrande,hellaswag,arc_easy,arc_challenge,mmlu \

######### model.bin format ###############
lm-eval \
--model custom_checkpoint \
--model_args pretrained=meta-llama/Meta-Llama-3-8B,device_map=auto,dtype=float16,checkpoint=../pruned_model/hoge/model.bin \
--tasks arc_easy,winogrande,hellaswag,arc_challenge,mmlu \
--batch_size 64
