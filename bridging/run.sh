#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n03
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# Default timestep value (can be overridden by command line arg)
TIMESTEP=499

# Usage: sbatch run_inference.sh [TIMESTEP]
if [ ! -z "$1" ]; then
  TIMESTEP=$1
fi

# Input Sentences
TEXT1="The aroma of fresh coffee filled the small kitchen."
TEXT2="My computer is broken, so I went to the repair shop."

# Model Hyperparameters (Must match the saved model filename)
LATENT_WIDTH=512
LATENT_CHANNELS=3
NUM_DIFFU_LAYERS=128
DIFFU_TIMESTEPS=1000

echo "Running inference with timestep: $TIMESTEP"
echo "Interpolating between:"
echo "  1: $TEXT1"
echo "  2: $TEXT2"

# Pass all arguments to the python script
srun python inference.py \
    --noise_t $TIMESTEP \
    --text1 "$TEXT1" \
    --text2 "$TEXT2" \
    --latent_width $LATENT_WIDTH \
    --latent_channels $LATENT_CHANNELS \
    --num_diffu_layers $NUM_DIFFU_LAYERS \
    --diffu_timesteps $DIFFU_TIMESTEPS \
    --interpolation_type "lerp" 