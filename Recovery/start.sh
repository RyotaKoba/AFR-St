# Set common variables

# CUDA_LAUNCH_BLOCKING=1 python recovery.py 
# CUDA_LAUNCH_BLOCKING=1 python merge_lora.py 
# CUDA_LAUNCH_BLOCKING=1 python lora-tuning.py
# CUDA_LAUNCH_BLOCKING=1 python generate.py 
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_BLOCKING_WAIT=1
# export TORCH_ELASTIC_TIMEOUT=16000




# llamafactory-cli train lora.yaml
accelerate launch $(which llamafactory-cli) train lora.yaml
