#!/bin/bash
#SBATCH --job-name=x_bridging_pairs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n04
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# Default timestep value (can be overridden by command line arg)
INTP_TIMESTEPS=599
TIMESTEP=799

# Usage: sbatch inference_pairs.sh [TIMESTEP]
if [ ! -z "$1" ]; then
  TIMESTEP=$1
fi

# Model Hyperparameters (Must match the saved model filename)
LATENT_WIDTH=1024
LATENT_CHANNELS=1
NUM_DIFFU_LAYERS=8
DIFFU_TIMESTEPS=1000
KERNEL_SIZE=5
TRANSFORMER_D_MODEL=1024
MODEL_TYPE="transformer"

OUTPUT_FILE="inference_result/diffusion_intps_${MODEL_TYPE}_${TIMESTEP}.json"

echo "Running batch inference pairs with timestep: $TIMESTEP"
echo "Model config: Width=$LATENT_WIDTH, Channels=$LATENT_CHANNELS, Layers=$NUM_DIFFU_LAYERS"

# Pass relevant arguments to the python script
srun python inference_pairs.py \
    --intp_noise_t $INTP_TIMESTEPS \
    --noise_t $TIMESTEP \
    --latent_width $LATENT_WIDTH \
    --latent_channels $LATENT_CHANNELS \
    --num_diffu_layers $NUM_DIFFU_LAYERS \
    --diffu_timesteps $DIFFU_TIMESTEPS \
    --kernel_size $KERNEL_SIZE \
    --transformer_d_model $TRANSFORMER_D_MODEL\
    --interpolation_type "lerp" \
    --output_file "$OUTPUT_FILE" \
    --model_type "$MODEL_TYPE"
