#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n03
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

srun python test.py
