#!/bin/bash
#SBATCH --job-name=x_bridging_pairs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# Default timestep value (can be overridden by command line arg)
TIMESTEP=499

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

OUTPUT_FILE="inference_result/diffusion_intps_conv_$TIMESTEP.json"

echo "Running batch inference pairs with timestep: $TIMESTEP"
echo "Model config: Width=$LATENT_WIDTH, Channels=$LATENT_CHANNELS, Layers=$NUM_DIFFU_LAYERS"

# Pass relevant arguments to the python script
# Note: text1/text2 are not needed as the script generates pairs internally
srun python get_latent_path.py \
    --noise_t $TIMESTEP \
    --latent_width $LATENT_WIDTH \
    --latent_channels $LATENT_CHANNELS \
    --num_diffu_layers $NUM_DIFFU_LAYERS \
    --diffu_timesteps $DIFFU_TIMESTEPS \
    --kernel_size $KERNEL_SIZE \
    --transformer_d_model $TRANSFORMER_D_MODEL\
    --model_type "transformer" \
    --interpolation_type "lerp"
