#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

source /data3/seungwoochoi/.bashrc
source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
conda activate x_bridging


srun python test.py
