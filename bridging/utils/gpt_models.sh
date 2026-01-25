#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=1

#SBATCH --mem=8G

# Pass all arguments to the python script
srun python gpt_models.py