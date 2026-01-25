import sys
# sys.path.append("../")
sys.path.append("../train/")
sys.path.append("../utils/")
sys.path.append("../analysis/")
import argparse
import json
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from custom_dataset import TinyStoriesDataset
from inference import DiffusionTracer
import numpy as np
from latent_intp import linear_interpolate, slerp_channel_wise, bezier_2nd_order, bezier_3rd_order
from langchain_community.vectorstores import FAISS
from get_latent_path import get_latent_from_sent

# --- Configuration (Copied from gpt.py to ensure identical data setup) ---
BERT_MODEL_NAME = "bert-base-uncased"
DATA_SPLIT = "validation"
MAX_SEQ_LEN = 128
SKIP_SAMPLES = 10000
BATCH_SIZE = 1
NUM_SENTENCES_TO_PROCESS = 20

# ==========================================
# Helper Class: DiffusionEmbeddings (DB Load)
# ==========================================
class DiffusionEmbeddings:
    def __init__(self, model, tokenizer, device, t=499):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.t = t

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            with torch.no_grad():
                latent = get_latent_from_sent(text, self.model, self.tokenizer, self.device)
                t_tensor = torch.tensor([self.t]).to(self.device)
                latent = self.model.q_sample_no_stochastic(latent, t_tensor)
                latent = latent.squeeze(0).cpu().numpy()
                embeddings.append(latent.flatten())
        return embeddings        

    def embed_query(self, text):
        return self.embed_documents([text])[0]

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

def interpolate_pair(tracer, sent1, sent2, args, vector_store=None):
    """
    Performs interpolation between two sentences.
    Supports: Linear (lerp), SLERP (slerp), Bezier (bezier).
    """
    # 1. Projection: Text -> Latent
    proj_1 = tracer.trace_projection(sent1)
    latent_1 = proj_1['latent_x0']
    
    proj_2 = tracer.trace_projection(sent2)
    latent_2 = proj_2['latent_x0']
    
    # Define interpolation steps (alphas)
    alphas = [x/100 for x in list(range(5,100,5))]
    result_sequence = [sent1]

    # --- BEZIER INTERPOLATION LOGIC ---
    if "bezier" in args.interpolation_type:
        if vector_store is None:
            raise ValueError("VectorDB is required for Bezier interpolation")
            
        if args.interpolation_type == "bezier_2nd":
            # 2nd Order: Mean(V0, V2) -> Search DB -> V1
            avg_latent = (latent_1 + latent_2) / 2.0
            query_vector = avg_latent.squeeze(0).cpu().numpy().flatten().tolist()
            
            results = vector_store.similarity_search_with_score_by_vector(query_vector, k=1)
            control_text = results[0][0].page_content
            
            proj_control = tracer.trace_projection(control_text)
            latent_v1 = proj_control['latent_x0']

            if args.noise_t >= 0:
                noise_1 = tracer.trace_noising(latent_1, t_val=args.noise_t)['noisy_latent']
                noise_v1 = tracer.trace_noising(latent_v1, t_val=args.noise_t)['noisy_latent']
                noise_2 = tracer.trace_noising(latent_2, t_val=args.noise_t)['noisy_latent']
                
                for alpha in alphas:
                    intp_noisy = bezier_2nd_order(noise_1, noise_v1, noise_2, alpha)
                    history = tracer.trace_generation(starting_noise=intp_noisy, start_step=args.noise_t)
                    result_sequence.append(history[-1]['text_estimate'][0])
            else:
                for alpha in alphas:
                    intp_latent = bezier_2nd_order(latent_1, latent_v1, latent_2, alpha)
                    logits = tracer.model.decode_latents(intp_latent)
                    pred_ids = torch.argmax(logits, dim=-1)
                    result_sequence.append(tracer.decode_token_ids(pred_ids[0]))

        elif args.interpolation_type == "bezier_3rd":
             # 3rd Order: 1/3 and 2/3 points
            t1 = 1.0 / 3.0
            p1_latent = (1 - t1) * latent_1 + t1 * latent_2
            query_p1 = p1_latent.squeeze(0).cpu().numpy().flatten().tolist()
            
            results_p1 = vector_store.similarity_search_with_score_by_vector(query_p1, k=1)
            cp1_text = results_p1[0][0].page_content
            proj_cp1 = tracer.trace_projection(cp1_text)
            latent_cp1 = proj_cp1['latent_x0']
            
            t2 = 2.0 / 3.0
            p2_latent = (1 - t2) * latent_1 + t2 * latent_2
            query_p2 = p2_latent.squeeze(0).cpu().numpy().flatten().tolist()
            
            results_p2 = vector_store.similarity_search_with_score_by_vector(query_p2, k=1)
            cp2_text = results_p2[0][0].page_content
            proj_cp2 = tracer.trace_projection(cp2_text)
            latent_cp2 = proj_cp2['latent_x0']

            if args.noise_t >= 0:
                noise_1 = tracer.trace_noising(latent_1, t_val=args.noise_t)['noisy_latent']
                noise_cp1 = tracer.trace_noising(latent_cp1, t_val=args.noise_t)['noisy_latent']
                noise_cp2 = tracer.trace_noising(latent_cp2, t_val=args.noise_t)['noisy_latent']
                noise_2 = tracer.trace_noising(latent_2, t_val=args.noise_t)['noisy_latent']
                
                for alpha in alphas:
                    intp_noisy = bezier_3rd_order(noise_1, noise_cp1, noise_cp2, noise_2, alpha)
                    history = tracer.trace_generation(starting_noise=intp_noisy, start_step=args.noise_t)
                    result_sequence.append(history[-1]['text_estimate'][0])
            else:
                 for alpha in alphas:
                    intp_latent = bezier_3rd_order(latent_1, latent_cp1, latent_cp2, latent_2, alpha)
                    logits = tracer.model.decode_latents(intp_latent)
                    pred_ids = torch.argmax(logits, dim=-1)
                    result_sequence.append(tracer.decode_token_ids(pred_ids[0]))

    # --- LERP / SLERP INTERPOLATION LOGIC ---
    else:
        interpolater = linear_interpolate if args.interpolation_type == "lerp" else slerp_channel_wise
        if args.noise_t >= 0:
            # Standard Diffusion Interpolation (Noisy Space)
            noise_1 = tracer.trace_noising(latent_1, t=args.intp_noise_t)['noisy_latent']
            noise_2 = tracer.trace_noising(latent_2, t=args.intp_noise_t)['noisy_latent']
            
            for alpha in alphas:
                intp_noisy = interpolater(noise_1, noise_2, alpha)
                intp_noisy = tracer.trace_noising(intp_noisy, t=args.intp_noise_t, start_t=args.intp_noise_t)['noisy_latent']
                history = tracer.trace_generation(
                    starting_noise=intp_noisy,
                    start_step=args.intp_noise_t
                )
                text_est = history[-1]['text_estimate'][0]
                result_sequence.append(text_est)
        else:
            # Autoencoder Interpolation
            for alpha in alphas:
                intp_latent = interpolater(latent_1, latent_2, alpha)
                logits = tracer.model.decode_latents(intp_latent)
                pred_ids = torch.argmax(logits, dim=-1)
                text_est = tracer.decode_token_ids(pred_ids[0])
                result_sequence.append(text_est)

    result_sequence.append(sent2)
    return result_sequence

def main():
    parser = argparse.ArgumentParser(description="Run diffusion inference on fixed GPT sentence pairs.")
    
    # Model Configuration Args
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--latent_width", type=int, default=512, help="Width of latent space")
    parser.add_argument("--latent_channels", type=int, default=1, help="Number of latent channels")
    parser.add_argument("--num_diffu_layers", type=int, default=8, help="Number of diffusion layers")
    parser.add_argument("--diffu_timesteps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--model_type", type=str, default="conv", help="Model type: conv or transformer")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for conv model")
    parser.add_argument("--transformer_d_model", type=int, default=512, help="D model size for transformer model")
    parser.add_argument("--putter_path", type=str, default="", help="Path to putter checkpoint")

    # Inference Args
    parser.add_argument("--intp_noise_t", type=int, default=1000, help="Noise timestep when interpolation happens (for Lerp/Slerp)")
    parser.add_argument("--noise_t", type=int, default=790, help="Timestep to start denoising from (-1 for autoencoder/clean mode)")
    parser.add_argument("--interpolation_type", type=str, default="lerp", choices=["lerp", "slerp", "bezier_2nd", "bezier_3rd"])
    parser.add_argument("--output_file", type=str, default="bridging/diffusion_intps.json", help="Path to save JSON results")
    
    # Bezier Specific Args
    parser.add_argument("--vectordb_path", type=str, default="./saved_db/faiss_diffusion_embeddings.index", help="Path to VectorDB for Bezier")
    # parser.add_argument("--bezier_order", type=int, default=2, choices=[2, 3], help="Order of Bezier interpolation (2 or 3)")

    args = parser.parse_args()
    
    print(f"--- Starting Diffusion Interpolation: {args.interpolation_type.upper()} ---")
    
    # 1. Initialize Model
    if args.model_type == "conv":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_k{args.kernel_size}.pth"
    elif args.model_type == "transformer":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_td{args.transformer_d_model}_dtypetinystories.pth"
    
    # Check current dir or parent dir for model_outputs
    possible_paths = [
        os.path.join("../model_outputs", model_filename),
        os.path.join("../train/model_outputs", model_filename),
        os.path.join("model_outputs", model_filename)
    ]
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        print(f"Error: Model file not found. Checked: {possible_paths}")
        # Construct a default fallback to avoid crash if user knows what they are doing with paths
        model_path = os.path.join("../train/model_outputs", model_filename)

    print(f"Loading model from: {model_path}")
    tracer = DiffusionTracer(model_path, args.putter_path, args)


    
    # 2. Initialize VectorDB (If Bezier)
    vector_store = None
    if "bezier" in args.interpolation_type:
        print(f"Loading VectorDB from {args.vectordb_path}...")
        try:
            embedder_wrapper = DiffusionEmbeddings(tracer.model, tracer.tokenizer, tracer.device, t=499)
            vector_store = FAISS.load_local(
                folder_path=args.vectordb_path,
                embeddings=embedder_wrapper,
                allow_dangerous_deserialization=True
            )
            print(">> VectorDB loaded successfully.")
        except Exception as e:
            print(f">> Error loading DB: {e}")
            return

    # 3. Data Loading & Pair Generation
    print("Preparing data...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    all_sentences = load_data(tokenizer)
    sentence_pairs = generate_sentence_pairs(all_sentences)
    print(sentence_pairs, flush=True)
    
    # 4. Processing Pairs
    all_responses = []
    print(f"\nProcessing {len(sentence_pairs)} pairs...")
    
    for i, (sent1, sent2) in enumerate(tqdm(sentence_pairs)):
        try:
            # Run interpolation
            res = interpolate_pair(tracer, sent1, sent2, args, vector_store=vector_store)
            all_responses.append(res)
            
            # Optional: Print first result
            if i == 0:
                print(f"\n[Example Result Pair 0]", flush=True)
                print(f"Start: {res[0]}", flush=True)
                print(f"Mid: {res[len(res)//2]}", flush=True)
                print(f"End: {res[-1]}\n", flush=True)
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            all_responses.append([sent1, "ERROR", sent2])
            
    # 5. Save Results
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
