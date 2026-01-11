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
from interpolation import bezier_2nd_order, bezier_3rd_order  # Bezier 함수 임포트
import faiss  # FAISS GPU 상태 확인용

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
# Core Logic: Bezier Interpolation
# ==========================================
def interpolate_pair(tracer, vector_store, sent1, sent2, args):
    """
    Performs Bezier interpolation:
    1. Get V0 (sent1), V2/V3 (sent2)
    2. Calc Control Points
       - Order 2: Mean(V0, V2) -> Search DB -> V1
       - Order 3: Search(V0) -> V1, Search(V3) -> V2
    3. Bezier Interpolate
    """
    
    # 1. Projection: Get V0 and V_end
    proj_1 = tracer.trace_projection(sent1)
    latent_v0 = proj_1['latent_x0']
    
    proj_2 = tracer.trace_projection(sent2)
    latent_v_end = proj_2['latent_x0']
    
    alphas = [x/100 for x in list(range(5, 100, 5))]
    result_sequence = [sent1] # Start with V0 text
    
    if args.bezier_order == 2:
        # 2nd Order Logic
        # Calculate Average Latent (Clean Space)
        avg_latent = (latent_v0 + latent_v_end) / 2.0
        query_vector = avg_latent.squeeze(0).cpu().numpy().flatten().tolist()
        
        # Search
        results = vector_store.similarity_search_with_score_by_vector(query_vector, k=1)
        control_text = results[0][0].page_content
        
        # Get V1 Latent
        proj_control = tracer.trace_projection(control_text)
        latent_v1 = proj_control['latent_x0']

        # Interpolation
        if args.noise_t >= 0:
            noise_v0 = tracer.trace_noising(latent_v0, t_val=args.noise_t)['noisy_latent']
            noise_v1 = tracer.trace_noising(latent_v1, t_val=args.noise_t)['noisy_latent']
            noise_v2 = tracer.trace_noising(latent_v_end, t_val=args.noise_t)['noisy_latent']
            
            for alpha in alphas:
                intp_noisy = bezier_2nd_order(noise_v0, noise_v1, noise_v2, alpha)
                history = tracer.trace_generation(starting_noise=intp_noisy, start_step=args.noise_t)
                result_sequence.append(history[-1]['text_estimate'])
        else:
            for alpha in alphas:
                intp_latent = bezier_2nd_order(latent_v0, latent_v1, latent_v_end, alpha)
                logits = tracer.model.decode_latents(intp_latent)
                pred_ids = torch.argmax(logits, dim=-1)
                result_sequence.append(tracer.decode_token_ids(pred_ids[0]))

    elif args.bezier_order == 3:
        # 3rd Order Logic
        # Calculate 1/3 and 2/3 points (Clean Space)
        # L(t) = (1-t) * V0 + t * V_end
        
        # Point 1/3
        t1 = 1.0 / 3.0
        p1_latent = (1 - t1) * latent_v0 + t1 * latent_v_end
        query_p1 = p1_latent.squeeze(0).cpu().numpy().flatten().tolist()
        
        results_p1 = vector_store.similarity_search_with_score_by_vector(query_p1, k=1)
        cp1_text = results_p1[0][0].page_content
        proj_cp1 = tracer.trace_projection(cp1_text)
        latent_cp1 = proj_cp1['latent_x0']
        
        # Point 2/3
        t2 = 2.0 / 3.0
        p2_latent = (1 - t2) * latent_v0 + t2 * latent_v_end
        query_p2 = p2_latent.squeeze(0).cpu().numpy().flatten().tolist()
        
        results_p2 = vector_store.similarity_search_with_score_by_vector(query_p2, k=1)
        cp2_text = results_p2[0][0].page_content
        proj_cp2 = tracer.trace_projection(cp2_text)
        latent_cp2 = proj_cp2['latent_x0']
        
        # Interpolation
        if args.noise_t >= 0:
            noise_v0 = tracer.trace_noising(latent_v0, t_val=args.noise_t)['noisy_latent']
            noise_cp1 = tracer.trace_noising(latent_cp1, t_val=args.noise_t)['noisy_latent']
            noise_cp2 = tracer.trace_noising(latent_cp2, t_val=args.noise_t)['noisy_latent']
            noise_vend = tracer.trace_noising(latent_v_end, t_val=args.noise_t)['noisy_latent']
            
            for alpha in alphas:
                intp_noisy = bezier_3rd_order(noise_v0, noise_cp1, noise_cp2, noise_vend, alpha)
                history = tracer.trace_generation(starting_noise=intp_noisy, start_step=args.noise_t)
                result_sequence.append(history[-1]['text_estimate'])
        else:
            for alpha in alphas:
                intp_latent = bezier_3rd_order(latent_v0, latent_cp1, latent_cp2, latent_v_end, alpha)
                logits = tracer.model.decode_latents(intp_latent)
                pred_ids = torch.argmax(logits, dim=-1)
                result_sequence.append(tracer.decode_token_ids(pred_ids[0]))

    result_sequence.append(sent2) # End text
    return result_sequence

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run diffusion interpolation (Bezier) on pairs.")
    
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
    
    # Inference Args
    parser.add_argument("--noise_t", type=int, default=790, help="Timestep to start denoising (-1 for AE)")
    parser.add_argument("--output_file", type=str, default="./inference_result/diffusion_bezier_intps.json")
    parser.add_argument("--vectordb_path", type=str, default="./saved_db/faiss_diffusion_embeddings.index", help="Path to VectorDB")
    parser.add_argument("--bezier_order", type=int, default=2, choices=[2, 3], help="Order of Bezier interpolation (2 or 3)")
    
    args = parser.parse_args()
    
    print("--- Starting Diffusion Bezier Interpolation ---")
    
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
        # DB 로드를 위해 Embedding Class 인스턴스화
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
        
    # if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
    #     print(">> Moving FAISS index to GPU...")
    #     # GPU 리소스 생성
    #     res = faiss.StandardGpuResources()
        
    #     # CPU 인덱스를 GPU 인덱스로 변환
    #     # (주의: LangChain 래퍼 내부의 index를 교체해줍니다)
    #     gpu_index = faiss.index_cpu_to_gpu(res, 0, vector_store.index)
    #     vector_store.index = gpu_index
    #     print(">> FAISS is now running on GPU.")
    # else:
    #     print(">> GPU not found or faiss-cpu installed. Running on CPU.")


    # 3. Data Loading
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    all_sentences = load_data(tokenizer)
    sentence_pairs = generate_sentence_pairs(all_sentences)
    
    # 4. Processing Pairs
    all_responses = []
    print(f"\nProcessing {len(sentence_pairs)} pairs with Bezier Interpolation...")
    
    for i, (sent1, sent2) in enumerate(tqdm(sentence_pairs)):
        try:
            # Pass vector_store to the interpolation function
            res = interpolate_pair(tracer, vector_store, sent1, sent2, args)
            all_responses.append(res)
            
            if i == 0:
                print(f"\n[Example Pair 0]", flush=True)
                print(f"Start: {res[0]}", flush=True)
                print(f"Mid (~0.5): {res[len(res)//2]}", flush=True)
                print(f"End: {res[-1]}\n", flush=True)
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            all_responses.append([sent1, "ERROR", sent2])
            
    # 5. Save
    output_path = args.output_file + f"_o{args.bezier_order}_t{args.noise_t}.json"
    print(f"Saving results to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=4)
        print("Success.")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()