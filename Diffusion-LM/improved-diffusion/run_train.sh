#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# source /data3/seungwoochoi/.bashrc
# source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
# conda activate x_bridging

export LOGDIR="diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e"
export TOKENIZERS_PARALLELISM=false
export OPENAI_LOGDIR=$LOGDIR

python scripts/train.py \
    --checkpoint_path $LOGDIR \
    --model_arch transformer \
    --modality e2e-tgt \
    --save_interval 50000 \
    --lr 0.0001 \
    --batch_size 64 \
    --diffusion_steps 2000 \
    --noise_schedule sqrt \
    --use_kl False \
    --learn_sigma False \
    --image_size 8 \
    --num_channels 128 \
    --seed 102 \
    --dropout 0.1 \
    --in_channel 16 \
    --out_channel 16 \
    --padding_mode block \
    --experiment random \
    --lr_anneal_steps 200000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 821 \
    --e2e_train ../datasets/e2e_data
