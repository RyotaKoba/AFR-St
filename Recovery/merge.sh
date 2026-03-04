#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=40:00:00
#SBATCH --output=merge_result.txt
#SBATCH --error=merge_error.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error


set -e

singularity exec --nv pro6000_llamafactory.sif llamafactory-cli export llama3_merge.yaml
