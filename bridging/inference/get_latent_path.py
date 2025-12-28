import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import os
import argparse  # Added for command line arguments

# Import the model class from your training script
# Ensure diffusion_lm.py is in the same directory
import sys
sys.path.append("../")
from model import DiffusionLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from interpolation import linear_interpolate, slerp_channel_wise
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
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--latent_width", type=int, default=512, help="Width of latent space")
    parser.add_argument("--latent_channels", type=int, default=1, help="Number of latent channels")
    parser.add_argument("--num_diffu_layers", type=int, default=8, help="Number of diffusion layers")
    parser.add_argument("--diffu_timesteps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--model_type", type=str, default="conv", help="Model type: conv or transformer")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for conv model")
    parser.add_argument("--transformer_d_model", type=int, default=512, help="D model size for transformer model")
    parser.add_argument("--interpolation_type", type=str, default="lerp", help="Interpolation type: lerp or slerp")
    
    # Inference specific args
    parser.add_argument("--noise_t", type=int, default=800, help="Timestep to start denoising/interpolation from. Set to -1 for direct Autoencoder reconstruction.")
    args = parser.parse_args()

    # Construct Model Path based on args
    if args.model_type == "conv":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_k{args.kernel_size}.pth"
    elif args.model_type == "transformer":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_{args.transformer_d_model}.pth"
    MODEL_PATH = os.path.join("../model_outputs", model_filename)


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
            bert_model_name=config.get('bert_model_name', args.model_name),
            max_seq_len=config.get('max_seq_len', args.max_len),
            latent_channels=config.get('latent_channels', args.latent_channels),
            latent_width=config.get('latent_width', args.latent_width),
            timesteps=config.get('timesteps', args.diffu_timesteps),
            num_diffu_layers=config.get('num_diffu_layers', args.num_diffu_layers),
            kernel_size=config.get('kernel_size', 3), # Default to 3 if missing
            model_type=config.get('model_type', 'conv'),
            transformer_config=config.get('transformer_config', None),
        )
        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        
        # Use the tokenizer from the config
        tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name', args.model_name))
        
    else:
        print(">> No config found in checkpoint. Using command-line arguments for architecture.", flush=True)
        raise NotImplementedError("Loading from old checkpoints without config is not implemented in this version.")
        
    model.to(device)
    model.eval()
    
    with open("inference_result/gpt_intps.json", 'r') as f:
        gpt_intps = json.load(f)

    latent_vectors = []

    for gpt_intp in tqdm(gpt_intps):
        print(gpt_intp) 
        x_0_latent = get_latent_from_sent(gpt_intp[0], model, tokenizer, device)
        # latent_vectors.append(x_0_latent)
        x_1_latent = get_latent_from_sent(gpt_intp[-1], model, tokenizer, device)
        # latent_vectors.append(x_1_latent)

        t = torch.full((1,), args.noise_t, dtype=torch.long, device=device)

        noise = torch.randn_like(x_0_latent)
        x_0_noised, used_noise = model.q_sample(x_0_latent, t, noise=noise)
        x_1_noised, used_noise = model.q_sample(x_1_latent, t, noise=noise)

        x_0_noised = x_0_noised.squeeze().detach().cpu().numpy()
        x_1_noised = x_1_noised.squeeze().detach().cpu().numpy()

        latent_vectors.append(x_0_noised)
        latent_vectors.append(x_1_noised)
        print(f"x_0_noised.shape: {x_0_noised.shape}")
        print(f"x_1_noised.shape: {x_1_noised.shape}")

        for sent in gpt_intp[1:-1]:
            latent_vector = get_latent_from_sent(sent, model, tokenizer, device)
            latent_vector = latent_vector.detach().cpu().numpy()
            latent_vectors.append(latent_vector) 
            print(f"latent_vector.shape: {latent_vector.shape}")

        for i in [0.25, 0.5, 0.75]:
            intp_latent_vector = linear_interpolate(x_0_noised, x_1_noised, i)
            latent_vectors.append(intp_latent_vector)
            print(f"intp_latent_vector.shape: {intp_latent_vector.shape}")
        
    labels = ["n"] * len(latent_vectors)

    for i in range(len(gpt_intps)): 
        labels[i*8] = "X_0"
        labels[i*8+1] = "X_1"
        labels[i*8+2] = "GPT"
        labels[i*8+3] = "GPT"
        labels[i*8+4] = "GPT"
        labels[i*8+5] = "Diff"
        labels[i*8+6] = "Diff"
        labels[i*8+7] = "Diff"

        scaler.fit_transform(latent_vectors)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(latent_vectors)

        df_plot = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        df_plot["Label"] = labels
        
        # Sort data so that important labels are plotted last (on top)
        df_plot['zorder'] = df_plot['Label'].apply(lambda x: 1 if x in ['X_0', 'X_1', 'GPT', 'Diff'] else 0)
        df_plot = df_plot.sort_values(by='zorder')

        # Define custom palette
        palette = {
            "n": "lightgray",
            "X_0": "tab:blue",
            "X_1": "tab:orange",
            "GPT": "tab:green",
            "Diff": "tab:red"
        }

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Label", palette=palette)
        # Move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/latent_path_{i}.png')
        plt.close()        

        #reset labels
        labels = ["n"] * len(latent_vectors)

        # break


    
        







