#!/bin/bash
#SBATCH --job-name=x_bridging_pairs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# Default timestep value (can be overridden by command line arg)
TIMESTEP=0

# Pass relevant arguments to the python script
srun python get_latent_path.py \
    --noise_t $TIMESTEP \
    --model_path "../model_outputs/conv_1024_1_8_1000_k5.pth" \
    --interpolation_type "lerp"
