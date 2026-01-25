#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=master,n01,n02
#SBATCH --time=1-23:59:59
#SBATCH --mem=24000MB
#SBATCH --cpus-per-task=4

# --num_train_sentences: 학습에 사용할 'train' split 문장 개수
# --val_ratio: 학습 데이터 대비 'validation' split에서 가져올 문장 비율 (예: 100개 학습이면 20개 검증)

torchrun --nproc_per_node=1 --master_port=12239 train_putt.py \
    --dlm_path "/home/seungwoochoi2/data/x_bridging/bridging/train/model_outputs/transformer_1024_1_8_1000_td1024_dtypeinterpolation.pth" \
    --sent_refiner_model_id "meta-llama/Llama-3.2-3B-Instruct" \
    --save_dir "./checkpoints_v2" \
    --num_train_sentences 500 \
    --val_ratio 0.2 \
    --batch_size 128 \
    --epochs 10 \
    --lr 1e-4 \
    --putter_hidden_dim 2048 \
    --putter_layers 3 \
    --max_seq_len 128 \
    --seed 42