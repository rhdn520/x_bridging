#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --nodelist=n03
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

source /data3/seungwoochoi/.bashrc
source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
conda activate diffu_seq

python -m torch.distributed.launch --nproc_per_node=2 --master_port=12233 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 50000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 1024 \
--dataset qqp \
--data_dir /home/seungwoochoi/data/x_bridging/DiffuSeq/datasets/qqp \
--vocab bert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes test-qqp \
--resume_checkpoint /home/seungwoochoi/data/x_bridging/DiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20251007-22:09:08/ema_0.9999_030000.pt\


