#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=11:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1


# Pass all arguments to the python script
srun python visualize_analysis.py ../inference/inference_result/diffusion_intps_transformer_699_lerp_analysis.json
