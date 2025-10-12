#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --nodelist=n02
#SBATCH --time=1-23:59:59
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=1

source /data3/seungwoochoi/.bashrc
source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
conda activate diffu_seq

OPENAI_LOGDIR=diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20251007-13:26:41  TOKENIZERS_PARALLELISM=false python train.py   --checkpoint_path diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20251007-13:26:41 --dataset qqp --data_dir /home/seungwoochoi/data/x_bridging/DiffuSeq/datasets/qqp --vocab bert --use_plm_init no --lr 0.0001 --batch_size 1024 --microbatch 64 --diffusion_steps 2000 --noise_schedule sqrt --schedule_sampler lossaware --resume_checkpoint none --seq_len 128 --hidden_t_dim 128 --seed 102 --hidden_dim 128 --learning_steps 50000 --save_interval 10000 --config_name bert-base-uncased --notes test-qqp20251007-13:26:41 

##WhaT!!!!!