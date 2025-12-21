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
from interpolation import linear_interpolate, slerp_channel_wise

class DiffusionTracer:
    def __init__(self, model_path, args, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
            
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Logic to handle both New (Config+Weights) and Old (Weights Only) checkpoints
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            print(">> Found configuration in checkpoint. Using saved architecture parameters.")
            config = checkpoint['config']
            
            # Initialize model using the SAVED config
            self.model = DiffusionLM(
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
            self.model.load_state_dict(checkpoint['state_dict'])
            
            # Use the tokenizer from the config
            self.tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name', args.model_name))
            
        else:
            print(">> No config found in checkpoint. Using command-line arguments for architecture.")
            # Fallback: Initialize using CLI args
            self.model = DiffusionLM(
                bert_model_name=args.model_name,
                max_seq_len=args.max_len,
                latent_channels=args.latent_channels,
                latent_width=args.latent_width,
                timesteps=args.diffu_timesteps,
                num_diffu_layers=args.num_diffu_layers,
                kernel_size=3,
                model_type=args.model_type, # Use CLI arg if no config
            )
            
            # Handle standard state_dict or nested state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
            
        self.model.to(self.device)
        self.model.eval()

    def decode_token_ids(self, token_ids):
        """Helper to decode IDs to text, stripping special tokens."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if self.tokenizer.sep_token_id in token_ids:
            sep_idx = token_ids.index(self.tokenizer.sep_token_id)
            token_ids = token_ids[:sep_idx]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def trace_projection(self, text):
        """Step 1: Tractable Projection."""
        enc = self.tokenizer(
            text, 
            max_length=self.model.max_seq_len, 
            padding="max_length", 
            truncation=True, 
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        
        bert_out = self.model.bert(input_ids, attention_mask=attention_mask)
        last_hidden = bert_out.last_hidden_state
        
        batch_size = last_hidden.shape[0]
        flat_hidden = last_hidden.view(batch_size, -1)
        flat_latent = self.model.encoder_proj(flat_hidden)
        
        latent_x0 = flat_latent.view(batch_size, self.model.latent_channels, self.model.latent_width)
        
        return {
            "input_ids": input_ids,
            "bert_hidden": last_hidden, 
            "latent_x0": latent_x0      
        }

    @torch.no_grad()
    def trace_noising(self, latent_x0, t_val):
        """Step 2: Tractable Noising."""
        batch_size = latent_x0.shape[0]
        t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
        
        noise = torch.randn_like(latent_x0)
        noisy_latent, used_noise = self.model.q_sample(latent_x0, t, noise)
        
        return {
            "t": t_val,
            "noise": used_noise,
            "noisy_latent": noisy_latent
        }

    @torch.no_grad()
    def trace_generation(self, starting_noise=None, start_step=None, callback=None):
        """Step 3: Tractable Generation (Reverse Diffusion)."""
        batch_size = 1
        history = []
        
        if start_step is None:
            start_step = self.model.timesteps - 1

        if starting_noise is None:
            x = torch.randn(
                (batch_size, self.model.latent_channels, self.model.latent_width), 
                device=self.device
            )
            start_step = self.model.timesteps - 1
        else:
            x = starting_noise.clone()
            
        for i in reversed(range(start_step + 1)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            predicted_noise = self.model.denoise_model(x, t)
            x_0_pred = self.model.predict_x0_from_noise(x, t, predicted_noise)
            
            alpha_t = self.model.alpha[t][:, None, None]
            alpha_bar_t = self.model.alpha_bar[t][:, None, None]
            beta_t = self.model.beta[t][:, None, None]
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x_prev = mean + sigma * noise
            else:
                x_prev = mean 
            
            logits = self.model.decode_latents(x_0_pred)
            pred_ids = torch.argmax(logits, dim=-1)
            text_estimate = self.decode_token_ids(pred_ids[0])
            
            step_data = {
                "step": i,
                "x_t": x.clone(),
                "pred_noise": predicted_noise.clone(),
                "x_0_pred": x_0_pred.clone(),
                "text_estimate": text_estimate
            }
            
            history.append(step_data)
            if callback:
                callback(step_data)
            x = x_prev

        return history

def progress_callback(data):
    return

# ==========================================
# Main Execution Block
# ==========================================
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
    parser.add_argument("--latent_channels", type=int, default=3, help="Number of latent channels")
    parser.add_argument("--num_diffu_layers", type=int, default=128, help="Number of diffusion layers")
    parser.add_argument("--diffu_timesteps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--model_type", type=str, default="conv", help="Model type: conv or transformer")
    
    # Inference specific args
    parser.add_argument("--noise_t", type=int, default=800, help="Timestep to start denoising/interpolation from. Set to -1 for direct Autoencoder reconstruction.")
    parser.add_argument("--interpolation_type", type=str, default="lerp")

    args = parser.parse_args()
    
    # Construct Model Path based on args
    MODEL_PATH = f"../model_outputs/diffusion_lm_{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}.pth"
    
    # Initialize Tracer with args
    tracer = DiffusionTracer(MODEL_PATH, args)

    # --- Projection ---
    proj_data_1 = tracer.trace_projection(args.text1)    
    latent_vector_1 = proj_data_1['latent_x0']

    proj_data_2 = tracer.trace_projection(args.text2)
    latent_vector_2 = proj_data_2['latent_x0'] 

    interpolater = linear_interpolate if args.interpolation_type == "lerp" else slerp_channel_wise

    # ==========================================
    # BRANCH: Zero-Noising (Autoencoder Check)
    # ==========================================
    if args.noise_t < 0:
        print(f"--- Running in Zero-Noise Mode (Autoencoder Reconstruction) ---")
        
        # 1. Direct Decode (Skip Diffusion)
        logits_1 = tracer.model.decode_latents(latent_vector_1)
        pred_ids_1 = torch.argmax(logits_1, dim=-1)
        rec_text_1 = tracer.decode_token_ids(pred_ids_1[0])
        
        logits_2 = tracer.model.decode_latents(latent_vector_2)
        pred_ids_2 = torch.argmax(logits_2, dim=-1)
        rec_text_2 = tracer.decode_token_ids(pred_ids_2[0])
        
        print(f"\nOriginal 1: {args.text1}")
        print(f"Reconstructed 1: {rec_text_1}")
        
        print(f"\nOriginal 2: {args.text2}")
        print(f"Reconstructed 2: {rec_text_2}")
        
        print("\n--- Interpolation (Clean Latent Space) ---")
        # In this mode, we interpolate the CLEAN latents directly
        for i in range(1, 10):
            intp_latent = interpolater(latent_vector_1, latent_vector_2, i / 10)
            
            logits_intp = tracer.model.decode_latents(intp_latent)
            pred_ids_intp = torch.argmax(logits_intp, dim=-1)
            text_intp = tracer.decode_token_ids(pred_ids_intp[0])
            
            print(f"{text_intp}")
            
    # ==========================================
    # BRANCH: Standard Diffusion
    # ==========================================
    else:
        print(f"--- Running Inference with noise_t = {args.noise_t} ---")
        print(f"--- Interpolating between: ---")
        print(f"    1: '{args.text1}'")
        print(f"    2: '{args.text2}'")

        # 1. Noise the original latents individually
        noise_data_1 = tracer.trace_noising(latent_vector_1, t_val=args.noise_t)
        noise_data_2 = tracer.trace_noising(latent_vector_2, t_val=args.noise_t)
        noisy_input_1 = noise_data_1['noisy_latent']
        noisy_input_2 = noise_data_2['noisy_latent']

        # 2. Reconstruct the originals
        history_repair_1 = tracer.trace_generation(
            starting_noise=noisy_input_1, 
            start_step=args.noise_t, 
            callback=progress_callback
        )
        history_repair_2 = tracer.trace_generation(
            starting_noise=noisy_input_2, 
            start_step=args.noise_t, 
            callback=progress_callback
        )

        print(f"\nX_0 Sentence (Original): {args.text1}")
        print(f"X_0 Sentence (Reconstructed): {history_repair_1[-1]['text_estimate']}")
        
        print(f"\nX_1 Sentence (Original): {args.text2}")
        print(f"X_1 Sentence (Reconstructed): {history_repair_2[-1]['text_estimate']}")

        print("\n--- Interpolation Results ---")
        # 3. Interpolation Loop
        alphas = [0.25, 0.5, 0.75]
        for alpha in alphas:
            # Interpolate between the NOISY vectors (Channel-wise SLERP)
            intp_noised_latent = slerp_channel_wise(noisy_input_1, noisy_input_2, alpha)

            history_repair = tracer.trace_generation(
                starting_noise=intp_noised_latent, 
                start_step=args.noise_t, 
                callback=progress_callback
            )

            print(f"Alpha {alpha:.2f}: {history_repair[-1]['text_estimate']}")