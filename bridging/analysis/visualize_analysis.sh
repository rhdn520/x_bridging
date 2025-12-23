#!/bin/bash
#SBATCH --job-name=x_bridging
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1


# Pass all arguments to the python script
srun python visualize_analysis.py \
    --grammar-analysis-file "grammar_results/diffusion_intps_699.json" \
    --progress-analysis-file "analysis_result/progress_analysis_699.json"