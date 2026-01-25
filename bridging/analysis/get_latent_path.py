from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import os
import argparse  # Added for command line arguments
import numpy as np

# Import the model class from your training script
# Ensure diffusion_lm.py is in the same directory
import sys
sys.path.append("../")
sys.path.append("../utils")
sys.path.append("../train")
from model import DiffusionLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from latent_intp import linear_interpolate, slerp_channel_wise
import json
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def get_latent_from_sent(sent:str, model, tokenizer, device):
    tmp = tokenizer(
                    sent,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
    
    token_ids = tmp['input_ids'].to(device)
    attention_mask = tmp['attention_mask'].to(device)
    
    model.eval()
    latent_vector = model.get_latents(token_ids, attention_mask)

    # print(latent_vector.squeeze())
    # print(latent_vector.squeeze().shape)
    return latent_vector.squeeze()

if __name__ == "__main__":

    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Run diffusion inference with interpolation.")
    
    # Text Inputs
    parser.add_argument("--text1", type=str, default="She wanted to play sports with her friends.", help="First sentence for interpolation")
    parser.add_argument("--text2", type=str, default="Please stay behind the yellow line.", help="Second sentence for interpolation")
    
    # Model Configuration (defaults match your provided snippet)
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--interpolation_type", type=str, default="lerp", help="Interpolation type: lerp or slerp")
    
    # Inference specific args
    parser.add_argument("--noise_t", type=int, default=800, help="Timestep to start denoising/interpolation from. Set to -1 for direct Autoencoder reconstruction.")
    args = parser.parse_args()

    MODEL_PATH = args.model_path

    device = torch.device("cuda")
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
    
    print(f"Loading model from: {MODEL_PATH}", flush=True)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Logic to handle both New (Config+Weights) and Old (Weights Only) checkpoints
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print(">> Found configuration in checkpoint. Using saved architecture parameters.")
        config = checkpoint['config']
        
        # Initialize model using the SAVED config
        model = DiffusionLM(
            bert_model_name=config.get('bert_model_name'),
            max_seq_len=config.get('max_seq_len'),
            latent_channels=config.get('latent_channels'),
            latent_width=config.get('latent_width'),
            timesteps=config.get('timesteps'),
            num_diffu_layers=config.get('num_diffu_layers'),
            kernel_size=config.get('kernel_size', 3), # Default to 3 if missing
            model_type=config.get('model_type', 'conv'),
            transformer_config=config.get('transformer_config', None),
        )
        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        
        # Use the tokenizer from the config
        tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name'))
        
    else:
        print(">> No config found in checkpoint. Using command-line arguments for architecture.", flush=True)
        raise NotImplementedError("Loading from old checkpoints without config is not implemented in this version.")
        
    model.to(device)
    model.eval()
    
    with open("inference_result/gpt_intps.json", 'r') as f:
        gpt_intps = json.load(f)

    latent_vectors = []
    point_types = []
    group_indices = []
    current_idx = 0

    for gpt_intp in tqdm(gpt_intps):
        print(gpt_intp) 
        x_0_latent = get_latent_from_sent(gpt_intp[0], model, tokenizer, device)
        # latent_vectors.append(x_0_latent)
        x_1_latent = get_latent_from_sent(gpt_intp[-1], model, tokenizer, device)
        # latent_vectors.append(x_1_latent)

        t = torch.full((1,), args.noise_t, dtype=torch.long, device=device)

        noise = torch.randn_like(x_0_latent)
        x_0_noised, used_noise = model.q_sample(x_0_latent, t, noise=noise)

        noise = torch.randn_like(x_0_latent)
        x_1_noised, used_noise = model.q_sample(x_1_latent, t, noise=noise)

        x_0_noised = x_0_noised.squeeze().detach().cpu().numpy()
        x_1_noised = x_1_noised.squeeze().detach().cpu().numpy()

        latent_vectors.append(x_0_noised)
        latent_vectors.append(x_1_noised)
        point_types.extend(["Real", "Real"])
        # print(f"x_0_noised.shape: {x_0_noised.shape}")
        # print(f"x_1_noised.shape: {x_1_noised.shape}")

        for sent in gpt_intp[1:-1]:
            noise = torch.randn_like(x_0_latent)
            gpt_latent_vector = get_latent_from_sent(sent, model, tokenizer, device)
            gpt_latent_vector, used_noise = model.q_sample(gpt_latent_vector, t, noise=noise)
            gpt_latent_vector = gpt_latent_vector.squeeze().detach().cpu().numpy()
            latent_vectors.append(gpt_latent_vector) 
            point_types.append("Real") 
            # print(f"latent_vector.shape: {gpt_latent_vector.shape}")

        for i in [0.25, 0.5, 0.75]:
            intp_latent_vector = linear_interpolate(x_0_noised, x_1_noised, i)
            latent_vectors.append(intp_latent_vector)
            point_types.append("Interpolated")
            # print(f"intp_latent_vector.shape: {intp_latent_vector.shape}")
        
        group_len = len(latent_vectors) - current_idx
        group_indices.append((current_idx, group_len))
        current_idx = len(latent_vectors)
        
    # Fit PCA on unique vectors to avoid bias from duplicates
    all_vectors = np.stack(latent_vectors)
    unique_vectors = np.unique(all_vectors, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(unique_vectors)
    pca_result = pca.transform(all_vectors)

    labels = ["n"] * len(latent_vectors)

    for i in tqdm(range(len(gpt_intps))): 
        start_idx, length = group_indices[i]
        
        # X_0 is always first in the group
        labels[start_idx] = "X_0"
        # X_1 is always second in the group
        labels[start_idx+1] = "X_1"
        
        # GPT sentences are after X_0, X_1 (indices 2 to end-3)
        # Assumes last 3 are always Diff interpolation
        gpt_count = length - 2 - 3 
        for j in range(gpt_count):
            labels[start_idx + 2 + j] = "GPT"
            
        # Diff interpolation (last 3 items)
        for j in range(3):
            labels[start_idx + length - 3 + j] = "Diff"

        # scaler.fit_transform(latent_vectors) # Unused
        # pca = PCA(n_components=2)
        # pca_result = pca.fit_transform(latent_vectors)

        df_plot = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        df_plot["Label"] = labels
        df_plot["Type"] = point_types

        # Differentiate background Real points
        mask_n_real = (df_plot["Label"] == "n") & (df_plot["Type"] == "Real")
        df_plot.loc[mask_n_real, "Label"] = "n_Real"
        
        # Sort data so that important labels are plotted last (on top)
        df_plot['zorder'] = df_plot['Label'].apply(lambda x: 1 if x in ['X_0', 'X_1', 'GPT', 'Diff'] else 0)
        df_plot = df_plot.sort_values(by='zorder')

        # Define custom palette
        palette = {
            "n": "lightgray",
            "n_Real": "lightskyblue", # Light color for real background points
            "X_0": "tab:blue",
            "X_1": "tab:orange",
            "GPT": "tab:green",
            "Diff": "tab:red"
        }

        # Define custom markers
        markers = {
            "Real": "o",
            "Interpolated": "X" 
        }

        # Create mapping for markers based on available labels if needed, 
        # but sns.scatterplot style="Type" handles markers independently of hue="Label".
        # However, we must ensure style and hue don't conflict. 
        # Here hue="Label", style="Type". 
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Label", style="Type", markers=markers, palette=palette, s=50)
        # Move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/latent_path_{i}.png')
        plt.close()        

        #reset labels
        labels = ["n"] * len(latent_vectors)

        # break


    
        







