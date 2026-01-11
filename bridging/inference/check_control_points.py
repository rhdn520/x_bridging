import sys
sys.path.append("../")
import argparse
import json
import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from langchain_community.vectorstores import FAISS

# Import functionality from existing project files
from train import TinyStoriesDataset
from inference import DiffusionTracer
from get_latent_path import get_latent_from_sent
import faiss

# --- Configuration (Copied from gpt.py) ---
BERT_MODEL_NAME = "bert-base-uncased"
DATA_SPLIT = "validation"
MAX_SAMPLES = 1000
MAX_SEQ_LEN = 128
SKIP_SAMPLES = 10000
BATCH_SIZE = 1
NUM_SENTENCES_TO_PROCESS = 20

# ==========================================
# Helper Class: DiffusionEmbeddings (DB 로드용)
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

# ==========================================
# Data Loading Functions
# ==========================================
def load_data(tokenizer):
    print("Loading dataset...", flush=True)
    test_dataset = TinyStoriesDataset(
        tokenizer,
        split=DATA_SPLIT,
        max_seq_len=MAX_SEQ_LEN,
        skip_samples=SKIP_SAMPLES,
        dataset_size=MAX_SAMPLES
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    sent_list = []
    print("Extracting sentences...", flush=True)
    for batch_idx, batch in enumerate(test_loader):
        sent_list.extend(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
        
    return sent_list

def generate_sentence_pairs(sent_list):
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

# ==========================================
# Core Logic: Check Control Points
# ==========================================
def process_pair_control_points(tracer, vector_store, sent1, sent2):
    """
    Finds and returns control points for both 2nd and 3rd order strategies.
    
    Returns:
        dict: {
            "sent1": str,
            "sent2": str,
            "cp_2nd_order": str,
            "cp_3rd_order_1": str,
            "cp_3rd_order_2": str
        }
    """
    
    # 1. Projection: Get V0 and V_end
    proj_1 = tracer.trace_projection(sent1)
    latent_v0 = proj_1['latent_x0']
    
    proj_2 = tracer.trace_projection(sent2)
    latent_v_end = proj_2['latent_x0']
    
    result_entry = {
        "sent1": sent1,
        "sent2": sent2,
    }

    # --- 2nd Order Strategy (Midpoint) ---
    avg_latent = (latent_v0 + latent_v_end) / 2.0
    query_mid = avg_latent.squeeze(0).cpu().numpy().flatten().tolist()
    
    results_mid = vector_store.similarity_search_with_score_by_vector(query_mid, k=1)
    result_entry["cp_2nd_order"] = results_mid[0][0].page_content

    # --- 3rd Order Strategy (1/3 and 2/3 points) ---
    # Point 1/3
    t1 = 1.0 / 3.0
    p1_latent = (1 - t1) * latent_v0 + t1 * latent_v_end
    query_p1 = p1_latent.squeeze(0).cpu().numpy().flatten().tolist()
    
    results_p1 = vector_store.similarity_search_with_score_by_vector(query_p1, k=1)
    result_entry["cp_3rd_order_1"] = results_p1[0][0].page_content
    
    # Point 2/3
    t2 = 2.0 / 3.0
    p2_latent = (1 - t2) * latent_v0 + t2 * latent_v_end
    query_p2 = p2_latent.squeeze(0).cpu().numpy().flatten().tolist()
    
    results_p2 = vector_store.similarity_search_with_score_by_vector(query_p2, k=1)
    result_entry["cp_3rd_order_2"] = results_p2[0][0].page_content
    
    return result_entry

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Check control points for Bezier strategies.")
    
    # Model Configuration
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--latent_width", type=int, default=512)
    parser.add_argument("--latent_channels", type=int, default=1)
    parser.add_argument("--num_diffu_layers", type=int, default=8)
    parser.add_argument("--diffu_timesteps", type=int, default=1000)
    parser.add_argument("--model_type", type=str, default="conv")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--transformer_d_model", type=int, default=512)
    
    # Script Args
    parser.add_argument("--output_file", type=str, default="./check_control_points.json")
    parser.add_argument("--vectordb_path", type=str, default="./saved_db/faiss_diffusion_embeddings.index", help="Path to VectorDB")
    
    args = parser.parse_args()
    
    print("--- Starting Control Point Check ---")
    
    # 1. Initialize Model
    if args.model_type == "conv":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_k{args.kernel_size}.pth"
    else:
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_d{args.transformer_d_model}.pth"
    model_path = os.path.join("../model_outputs", model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    tracer = DiffusionTracer(model_path, args)
    
    # 2. Initialize VectorDB
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
        
    # 3. Data Loading
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    all_sentences = load_data(tokenizer)
    sentence_pairs = generate_sentence_pairs(all_sentences)
    
    # 4. Processing
    results = []
    print(f"\nChecking control points for {len(sentence_pairs)} pairs...")
    
    for i, (sent1, sent2) in enumerate(tqdm(sentence_pairs)):
        try:
            entry = process_pair_control_points(tracer, vector_store, sent1, sent2)
            results.append(entry)
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            results.append({
                "sent1": sent1, 
                "sent2": sent2, 
                "error": str(e)
            })
            
    # 5. Save
    print(f"Saving results to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Success.")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()
