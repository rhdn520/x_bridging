#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=master
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# source /data3/seungwoochoi/.bashrc
# source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
# conda activate x_bridging


# srun python scripts/run_train.py \
#     --diff_steps 2000 \
#     --model_arch transformer \
#     --lr 0.0001 \
#     --lr_anneal_steps 200000  \
#     --seed 102 \
#     --noise_schedule sqrt \
#     --in_channel 16 \
#     --modality e2e-tgt \
#     --submit no \
#     --padding_mode block \
#     --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " \
#     --notes xstart_e2e

# python scripts/batch_decode.py /home/seungwoochoi/data/x_bridging/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e -1.0 ema
# python scripts/ppl_under_ar.py --model_path /home/seungwoochoi/data/x_bridging/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e --modality e2e-tgt  --experiment random --model_name_or_path predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None --input_text generation_outputs/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e.ema_0.9999_000000.pt.samples_-1.0.json  --mode eval
python scripts/text_sample.py \
    --model_path /home/seungwoochoi/data/x_bridging/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/ema_0.9999_200000.pt \
    --batch_size 100 \
    --num_samples 100 \
    --top_p -1.0 \
    --out_dir generation_outputs 


python run_interpolation.py \
    --model_path /home/seungwoochoi/data/x_bridging/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/ema_0.9999_200000.pt \
    --sentence1 "This is the first sentence for interpolation." \
    --sentence2 "This is the second one."\
    --model_arch transformer \
    --batch_size 100

