#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --nodelist=n03
#SBATCH --time=1-23:59:59
#SBATCH --mem=24000MB
#SBATCH --cpus-per-task=1


# Pass all arguments to the python script
torchrun --nproc_per_node=3 --master_port=12233 train.py 