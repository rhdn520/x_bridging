import sys
sys.path.append("../")
import argparse
import json
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

# Import functionality from existing project files
# We assume this script is run from the same directory where 'bridging' package is reachable or inside bridging dir
# Adjusting python path might be needed if run from root, but assuming standard behavior relative to these files.
from train import TinyStoriesDataset
from inference import DiffusionTracer
from interpolation import linear_interpolate, slerp_channel_wise

# --- Configuration (Copied from gpt.py to ensure identical data setup) ---
BERT_MODEL_NAME = "bert-base-uncased"
DATA_SPLIT = "validation"
MAX_SEQ_LEN = 128
SKIP_SAMPLES = 10000
BATCH_SIZE = 1
NUM_SENTENCES_TO_PROCESS = 20

def load_data(tokenizer):
    """
    Loads the TinyStories dataset and extracts sentences.
    Logic copied from gpt.py to ensure identical data retrieval.
    """
    print("Loading dataset...", flush=True)
    test_dataset = TinyStoriesDataset(
        tokenizer,
        split=DATA_SPLIT,
        max_seq_len=MAX_SEQ_LEN,
        skip_samples=SKIP_SAMPLES
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    sent_list = []
    print("Extracting sentences...", flush=True)
    for batch_idx, batch in enumerate(test_loader):
        sent_list.extend(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
        
    return sent_list

def generate_sentence_pairs(sent_list):
    """
    Selects a random subset of sentences and generates unique pairs.
    Logic copied from gpt.py (including seed) to ensure identical pairs.
    """
    # shuffle sent_list
    random.seed(42)
    random.shuffle(sent_list)
    subset = sent_list[:NUM_SENTENCES_TO_PROCESS]
    print(f"Selected {len(subset)} sentences for pairing: {subset}")

    products = []
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            sent1 = subset[i]
            sent2 = subset[j]
            products.append([sent1, sent2])
            
    print(f"Generated {len(products)} pairs.")
    return products

def interpolate_pair(tracer, sent1, sent2, args):
    """
    Performs interpolation between two sentences using the diffusion model.
    Returns a list of sentences: [sent1, intermediate_1, ..., sent2]
    """
    # 1. Projection: Text -> Latent
    proj_1 = tracer.trace_projection(sent1)
    latent_1 = proj_1['latent_x0']
    
    proj_2 = tracer.trace_projection(sent2)
    latent_2 = proj_2['latent_x0']
    
    # Define interpolation steps (alphas)
    alphas = [x/100 for x in list(range(5,100,5))]
    
    result_sequence = [sent1]
    
    # 2. Interpolation Logic
    interpolater = linear_interpolate if args.interpolation_type == "lerp" else slerp_channel_wise
    if args.noise_t >= 0:
        # Standard Diffusion Interpolation (Noisy Space)
        # Noise the latents to intp noise timestep
        noise_1 = tracer.trace_noising(latent_1, t=args.intp_noise_t)['noisy_latent']
        noise_2 = tracer.trace_noising(latent_2, t=args.intp_noise_t)['noisy_latent']
        
        for alpha in alphas:
            # Spherical Linear Interpolation in noisy space
            intp_noisy = interpolater(noise_1, noise_2, alpha)

            intp_noisy = tracer.trace_noising(intp_noisy, t=args.intp_noise_t, start_t=args.intp_noise_t)['noisy_latent']
            
            # Denoise (Generate) from the interpolated noisy state
            # We use the tracer's generation function but only need the final text
            history = tracer.trace_generation(
                starting_noise=intp_noisy,
                start_step=args.intp_noise_t
            )
            # failure handling could be added here, but trace_generation is robust
            text_est = history[-1]['text_estimate']
            result_sequence.append(text_est)
            
    else:
        # Zero-Noise / Autoencoder Interpolation (Clean Latent Space)

        for alpha in alphas:
            intp_latent = interpolater(latent_1, latent_2, alpha)
            
            # Decode directly
            logits = tracer.model.decode_latents(intp_latent)
            pred_ids = torch.argmax(logits, dim=-1)
            text_est = tracer.decode_token_ids(pred_ids[0])
            result_sequence.append(text_est)

    result_sequence.append(sent2)
    return result_sequence

def main():
    parser = argparse.ArgumentParser(description="Run diffusion inference on fixed GPT sentence pairs.")
    
    # Model Configuration Args (Defaults match bridging/inference.sh)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--latent_width", type=int, default=512, help="Width of latent space")
    parser.add_argument("--latent_channels", type=int, default=1, help="Number of latent channels")
    parser.add_argument("--num_diffu_layers", type=int, default=8, help="Number of diffusion layers")
    parser.add_argument("--diffu_timesteps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--model_type", type=str, default="conv", help="Model type: conv or transformer")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for conv model")
    parser.add_argument("--transformer_d_model", type=int, default=512, help="D model size for transformer model")
    
    # Inference Args
    parser.add_argument("--intp_noise_t", type=int, default=1000, help="Noise timestep when interpolation happens")
    parser.add_argument("--noise_t", type=int, default=790, help="Timestep to start denoising from (-1 for autoencoder mode)")
    parser.add_argument("--interpolation_type", type=str, default="lerp", choices=["lerp", "slerp"])
    parser.add_argument("--output_file", type=str, default="bridging/diffusion_intps.json", help="Path to save JSON results")
    
    args = parser.parse_args()
    
    print("--- Starting Diffusion Interpolation on GPT Pairs ---")
    
    # 1. Initialize Model
    # Construct default model path if not provided (following inference.py convention)
    # Assumes running from root of repo where model_outputs/ exists
    if args.model_type == "conv":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_k{args.kernel_size}.pth"
    elif args.model_type == "transformer":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_{args.transformer_d_model}.pth"
    model_path = os.path.join("../model_outputs", model_filename)
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    tracer = DiffusionTracer(model_path, args)
    
    # 2. Data Loading & Pair Generation
    print("Preparing data...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    all_sentences = load_data(tokenizer)
    sentence_pairs = generate_sentence_pairs(all_sentences)

    print(sentence_pairs, flush=True)
    
    # 3. Processing Pairs
    all_responses = []
    print(f"\nProcessing {len(sentence_pairs)} pairs...")
    
    for i, (sent1, sent2) in enumerate(tqdm(sentence_pairs)):
        try:
            # Run interpolation
            res = interpolate_pair(tracer, sent1, sent2, args)
            all_responses.append(res)
            
            # Optional: Print first result to verify it's working
            if i == 0:
                print(f"\n[Example Result Pair 0]", flush=True)
                print(f"Start: {res[0]}", flush=True)
                print(f"Mid: {res[len(res)//2]}", flush=True)
                print(f"End: {res[-1]}\n", flush=True)
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            # Append failed partial or skip? 
            # We'll append just start/end to keep index alignment if needed, or handle gracefully.
            all_responses.append([sent1, "ERROR", sent2])
            
    # 4. Save Results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=4)
        print("Success.")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()
