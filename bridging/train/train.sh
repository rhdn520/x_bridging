#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --mem=24000MB
#SBATCH --cpus-per-task=1


# Pass all arguments to the python script
torchrun --nproc_per_node=1 --master_port=12233 train.py \
    --model_name bert-base-uncased \
    --max_len 128 \
    --batch_size 16 \
    --epochs 11 \
    --lr 1e-4 \
    --latent_channels 1 \
    --latent_width 1024 \
    --timesteps 1000 \
    --kernel_size 5 \
    --num_diffu_layers 8 \
    --time_bias 0.3 \
    --model_type transformer \
    --transformer_d_model 1024 \
    --transformer_nhead 8 \
    --transformer_num_layers 6 \
    --transformer_dim_feedforward 4096 \
    --transformer_dropout 0.1 \
    --train_samples 300 \
    --val_samples 10 \
    --test_samples 10 \
    --dataset_type c4 \
    --interpolation_data_path ../dataset/vllm_interpolation_outputs.json \
    # --resume \
    # --resume_path /home/seungwoochoi2/data/x_bridging/bridging/train/model_outputs/transformer_1024_1_8_1000_d1024.pth \
    # --save_every_epoch \