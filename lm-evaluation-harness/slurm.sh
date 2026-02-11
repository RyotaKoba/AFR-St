#!/bin/bash
#SBATCH --job-name=eval_pruned
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH --output=result.txt
#SBATCH --error=error.txt
#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

singularity exec --nv eval.sif sh start.sh
