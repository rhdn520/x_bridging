#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1


PROJ_METHOD="umap"
INTP_METHOD="slerp"

python visualize_interpolation.py \
    --model_path ../train/model_outputs/transformer_1024_1_8_1000_td1024_dtypetinystories.pth \
    --n_samples 1000 \
    --method $PROJ_METHOD \
    --intp_method $INTP_METHOD \
    --output_plot plots/${INTP_METHOD}/interpolation_viz_${PROJ_METHOD} \