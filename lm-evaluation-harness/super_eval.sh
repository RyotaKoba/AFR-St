##########Vicuna-13B Global ###############
lm-eval \
--model hf \
--batch_size 64 \
--model_args device_map=auto,dtype=float16,pretrained=../pruned_model/Vicuna-13B/AFR-St_0.2p_SNR_trimmed2p \
--task winogrande,hellaswag,arc_easy,arc_challenge,mmlu 