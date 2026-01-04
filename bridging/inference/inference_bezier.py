import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import os
import argparse
import sys
import numpy as np
import textwrap

# 경로 설정
sys.path.append("../")
from model import DiffusionLM
# [수정 1] interpolation.py에서 베지에 곡선 함수 임포트
from interpolation import linear_interpolate, slerp_channel_wise, bezier_2nd_order
# [수정 2] Vector DB 검색을 위한 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from get_latent_path import get_latent_from_sent

# ==========================================
# Helper Class: DiffusionEmbeddings (DB 로드용)
# ==========================================
class DiffusionEmbeddings:
    """
    Vector DB를 로드하기 위해 필요한 래퍼 클래스입니다.
    """
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
# Main Tracer Class
# ==========================================
class DiffusionTracer:
    def __init__(self, model_path, args, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
            
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            print(">> Found configuration in checkpoint.")
            config = checkpoint['config']
            self.model = DiffusionLM(
                bert_model_name=config.get('bert_model_name', args.model_name),
                max_seq_len=config.get('max_seq_len', args.max_len),
                latent_channels=config.get('latent_channels', args.latent_channels),
                latent_width=config.get('latent_width', args.latent_width),
                timesteps=config.get('timesteps', args.diffu_timesteps),
                num_diffu_layers=config.get('num_diffu_layers', args.num_diffu_layers),
                kernel_size=config.get('kernel_size', 3),
                model_type=config.get('model_type', 'conv'),
                transformer_config=config.get('transformer_config', None),
            )
            self.model.load_state_dict(checkpoint['state_dict'])
            self.tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name', args.model_name))
        else:
            # Fallback logic omitted for brevity, same as before
            pass
            
        self.model.to(self.device)
        self.model.eval()

    def decode_token_ids(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if self.tokenizer.sep_token_id in token_ids:
            sep_idx = token_ids.index(self.tokenizer.sep_token_id)
            token_ids = token_ids[:sep_idx]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def trace_projection(self, text):
        """Step 1: Tractable Projection (Text -> Latent X0)."""
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
        
        if start_step is None:
            start_step = self.model.timesteps - 1

        if starting_noise is None:
            x = torch.randn((batch_size, self.model.latent_channels, self.model.latent_width), device=self.device)
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
            
            x = x_prev

        # Final decode
        logits = self.model.decode_latents(x_0_pred)
        pred_ids = torch.argmax(logits, dim=-1)
        text_estimate = self.decode_token_ids(pred_ids[0])
            
        return {"text_estimate": text_estimate}

def progress_callback(data):
    return

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments setup
    parser.add_argument("--text1", type=str, default="She wanted to play sports with her friends.")
    parser.add_argument("--text2", type=str, default="Please stay behind the yellow line.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--latent_width", type=int, default=512)
    parser.add_argument("--latent_channels", type=int, default=1)
    parser.add_argument("--num_diffu_layers", type=int, default=8)
    parser.add_argument("--diffu_timesteps", type=int, default=1000)
    parser.add_argument("--model_type", type=str, default="conv")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--transformer_d_model", type=int, default=512)
    
    # Inference specific args
    parser.add_argument("--noise_t", type=int, default=800, help="Timestep to start. -1 for Autoencoder.")
    parser.add_argument("--vectordb_path", type=str, default="./saved_db/faiss_diffusion_embeddings.index")

    args = parser.parse_args()
    
    # 1. Model Load
    if args.model_type == "conv":
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_k{args.kernel_size}.pth"
    else:
        model_filename = f"{args.model_type}_{args.latent_width}_{args.latent_channels}_{args.num_diffu_layers}_{args.diffu_timesteps}_d{args.transformer_d_model}.pth"
    
    MODEL_PATH = os.path.join("../model_outputs", model_filename)
    tracer = DiffusionTracer(MODEL_PATH, args)

    # 2. Vector DB Load
    print(f"\n>> Loading VectorDB from {args.vectordb_path} for Control Point Search...")
    try:
        # Load wrapper for FAISS
        embedder_wrapper = DiffusionEmbeddings(tracer.model, tracer.tokenizer, tracer.device, t=499)
        vector_store = FAISS.load_local(
            folder_path=args.vectordb_path,
            embeddings=embedder_wrapper,
            allow_dangerous_deserialization=True
        )
        print(">> VectorDB loaded successfully.")
    except Exception as e:
        print(f">> Error loading DB: {e}")
        sys.exit(1)

    # 3. Projection of Input Texts (V0, V2)
    proj_data_1 = tracer.trace_projection(args.text1)    
    latent_vector_1 = proj_data_1['latent_x0'] # V0

    proj_data_2 = tracer.trace_projection(args.text2)
    latent_vector_2 = proj_data_2['latent_x0'] # V2

    # ==========================================
    # Logic: Find Control Point (V1)
    # ==========================================
    print(f"\n[Search Strategy] Calculating Average(V0, V2) -> Searching DB...")
    
    # 1. Calculate Average Latent in Clean Space
    avg_latent = (latent_vector_1 + latent_vector_2) / 2.0
    
    # 2. Add diffusion noise to the average latent to match DB storage condition?
    #    (If DB was stored with t=499 noise, we should apply noise before searching.
    #     The DiffusionEmbeddings class handles 't' internally. 
    #     However, here we have a latent tensor. We need to format it for search.)
    
    # DB에 저장된 벡터는 특정 t(예: 499)에서의 latent입니다.
    # 우리가 구한 avg_latent는 t=0 상태입니다. 
    # 따라서 일관성을 위해:
    # A. avg_latent를 바로 t=499로 noising해서 검색에 사용하거나
    # B. DB가 t=0(Clean) 상태로 저장되었다면 그냥 사용.
    
    # [중요] 이전 코드의 `DiffusionEmbeddings`를 보면, DB 저장 시 t=499로 Noising을 해서 저장했습니다.
    # 따라서 검색할 때도 avg_latent를 t=499로 Noising한 뒤 검색해야 가장 정확합니다.
    
    t_search = 499 # DB 생성시 사용한 t값과 맞춰야 함
    t_tensor = torch.tensor([t_search]).to(tracer.device)
    
    # Deterministic noise for stable search (or random)
    search_noise = torch.zeros_like(avg_latent) # Or torch.randn_like if strict match desired
    # Note: q_sample_no_stochastic is better if available in your model to remove randomness variable
    # If not available, we use q_sample. Here assuming q_sample_no_stochastic exists as per previous context
    
    if hasattr(tracer.model, 'q_sample_no_stochastic'):
        query_latent_t = tracer.model.q_sample_no_stochastic(avg_latent, t_tensor)
    else:
        # Fallback if method doesn't exist
        query_latent_t, _ = tracer.model.q_sample(avg_latent, t_tensor, torch.randn_like(avg_latent))
    
    # Flatten for FAISS search
    query_vector = query_latent_t.squeeze(0).cpu().numpy().flatten().tolist()
    
    # 3. Search for Control Point
    results = vector_store.similarity_search_with_score_by_vector(query_vector, k=1)
    control_doc, score = results[0]
    control_text = control_doc.page_content
    
    print(f">> Found Control Point Text (Dist: {score:.4f}):")
    print(f"   '{control_text}'")
    
    # 4. Get Latent of Control Point (V1)
    proj_data_control = tracer.trace_projection(control_text)
    latent_vector_control = proj_data_control['latent_x0'] # V1

    # ==========================================
    # Interpolation Execution
    # ==========================================
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    if args.noise_t < 0:
        print(f"\n--- Zero-Noise Mode (Clean Latent Bezier) ---")
        # Curve in clean space: Bezier(V0, V1, V2) -> Decode
        for alpha in alphas:
            # Bezier Interpolation
            intp_latent = bezier_2nd_order(latent_vector_1, latent_vector_control, latent_vector_2, alpha)
            
            # Decode
            logits = tracer.model.decode_latents(intp_latent)
            pred_ids = torch.argmax(logits, dim=-1)
            text_res = tracer.decode_token_ids(pred_ids[0])
            print(f"[t={alpha:.1f}] {text_res}")
            
    else:
        print(f"\n--- Diffusion Mode (Noisy Latent Bezier at t={args.noise_t}) ---")
        # 1. Noise all three vectors to target t
        # Note: Using distinct noise for each might distort geometry, but using same noise preserves relative structure better.
        # Let's use independent noise as standard diffusion usually assumes independent states.
        
        n1 = tracer.trace_noising(latent_vector_1, args.noise_t)['noisy_latent']      # Noisy V0
        n_c = tracer.trace_noising(latent_vector_control, args.noise_t)['noisy_latent'] # Noisy V1
        n2 = tracer.trace_noising(latent_vector_2, args.noise_t)['noisy_latent']      # Noisy V2
        
        for alpha in alphas:
            # 2. Bezier Interpolation on Noisy Latents
            # Curve: (1-t)^2 * N0 + 2t(1-t) * N1 + t^2 * N2
            intp_noisy_latent = bezier_2nd_order(n1, n_c, n2, alpha)
            
            # 3. Denoise from interpolated noisy state
            result = tracer.trace_generation(
                starting_noise=intp_noisy_latent,
                start_step=args.noise_t
            )
            print(f"[t={alpha:.1f}] {result['text_estimate']}")