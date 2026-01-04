#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

# Default timestep value (can be overridden by command line arg)
TIMESTEP=699

# Usage: sbatch run_inference.sh [TIMESTEP]
if [ ! -z "$1" ]; then
  TIMESTEP=$1
fi

# Input Sentences
TEXT1="It is easy to say."
TEXT2="It is hard to do."

# Model Hyperparameters (Must match the saved model filename)
LATENT_WIDTH=1024
LATENT_CHANNELS=1
NUM_DIFFU_LAYERS=8
DIFFU_TIMESTEPS=1000
KERNEL_SIZE=5
TRANSFORMER_D_MODEL=512
MODEL_TYPE="transformer" 
VECTORDB_PATH="../vectorDB/saved_db/faiss_diffusion_embeddings.index"

echo "Running inference with timestep: $TIMESTEP"
echo "Interpolating between:"
echo "  1: $TEXT1"
echo "  2: $TEXT2"

# Pass all arguments to the python script
srun python inference_bezier.py \
    --noise_t $TIMESTEP \
    --text1 "$TEXT1" \
    --text2 "$TEXT2" \
    --latent_width $LATENT_WIDTH \
    --latent_channels $LATENT_CHANNELS \
    --num_diffu_layers $NUM_DIFFU_LAYERS \
    --diffu_timesteps $DIFFU_TIMESTEPS \
    --kernel_size $KERNEL_SIZE \
    --transformer_d_model $TRANSFORMER_D_MODEL\
    --model_type $MODEL_TYPE \
    --vectordb_path $VECTORDB_PATH
