#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=master,n01,n02
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

torchrun --nproc_per_node=1 --master_port=12236 run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 50000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 768 \
--dataset qqp \
--data_dir /home/seungwoochoi/data/x_bridging/DiffuSeq/datasets/qqp \
--vocab bert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes test-qqp \
--use_plm_init bert \
--bsz 512 \
--microbatch 512 \
# --resume_checkpoint /home/seungwoochoi/data/x_bridging/DiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20251007-22:09:08/ema_0.9999_030000.pt\


