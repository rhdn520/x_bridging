#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=1

MASTER_PORT=$((12000 + RANDOM % 1000))

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} sent_interpolation.py \
    --model_path /home/seungwoochoi/data/x_bridging/DiffuSeq/diffusion_models/diffuseq_qqp_h768_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20260113-21:25:54/ema_0.9999_030000.pt \
    --step 2000 \
    --batch_size 50 \
    --seed2 123 \
    --split test \
    --out_dir generation_outputs \
    --top_p -1 
