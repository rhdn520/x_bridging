#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=1

#SBATCH --mem=24G

# Pass all arguments to the python script
srun python vllm_models.py