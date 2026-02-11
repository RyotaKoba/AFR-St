# lm-eval \
# --model hf \
# --batch_size 64 \
# --model_args device_map=auto,dtype=float16,pretrained=../pruned_model/Llama3-8B/AFR-St_0.2_Mean \
# --task winogrande,hellaswag,arc_easy,arc_challenge,mmlu \

#########Llama3-8B Global ###############
# lm-eval \
# --model custom_checkpoint \
# --model_args pretrained=meta-llama/Meta-Llama-3-8B,device_map=auto,dtype=float16,checkpoint=../pruned_model/Llama3-8B/AFR-St_0.2p_trimmed2%_global_arc-easy-sample128/model.bin \
# --tasks arc_easy,winogrande,hellaswag,arc_challenge,mmlu \
# --batch_size 64 

# ########Vicuna-13B Global ###############
# lm-eval \
# --model custom_checkpoint \
# --model_args pretrained=lmsys/vicuna-13b-v1.5,device_map=auto,dtype=float16,checkpoint=../pruned_model/Vicuna-13B/AFR-St_0.5p_trimmed2%_global_sample128_snipdouble/model.bin \
# --tasks arc_easy,winogrande,hellaswag,arc_challenge,mmlu \
# --batch_size 64 

#########Llama3-8B Global Perplexity ###############
lm-eval \
--model custom_checkpoint \
--model_args pretrained=meta-llama/Meta-Llama-3-8B,device_map=auto,dtype=float16,checkpoint=../pruned_model/Llama3-8B/AFR-St_0.2p_trimmed2%_global_sample128_snipdouble/model.bin \
--tasks wikitext \
--batch_size 16 \
--limit 100