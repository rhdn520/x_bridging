#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --nodelist=n04
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=1

#SBATCH --mem=24G

# Pass all arguments to the python script
srun python sent_intp.py