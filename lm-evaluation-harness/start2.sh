python -m lm_eval \
--model hf \
--batch_size 64 \
--model_args device_map=auto,dtype=float16,parallelize=True,pretrained=../snip-refer_to_llm/pruned_model/Llama3-8B_AFR-St_0.5p_AveVar17,peft=../Recovery/checkpoints/recovery_afr17/checkpoint-1000 \
--task winogrande \
