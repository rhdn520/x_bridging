import sys
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Add paths to sys.path to import modules from other directories
sys.path.append("../")
sys.path.append("../train")
sys.path.append("../utils")

from model import DiffusionLM
from custom_dataset import TinyStoriesDataset
# Import get_latent_from_sent from the local directory
from get_latent_path import get_latent_from_sent
from latent_intp import linear_interpolate, slerp_channel_wise, bezier_2nd_order, bezier_3rd_order

# Try importing UMAP
try:
    import umap
except ImportError:
    print("UMAP not found. Please install it using `pip install umap-learn` if you want to use it.")
    umap = None
    
# Try importing FAISS
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("FAISS or LangChain not found. Bezier interpolation requiring VectorDB will fail.")
    FAISS = None

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


def load_model(model_path, device):
    """
    Loads the DiffusionLM model from a checkpoint.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    print(f"Loading model from: {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        model = DiffusionLM(
            bert_model_name=config.get('bert_model_name'),
            max_seq_len=config.get('max_seq_len'),
            latent_channels=config.get('latent_channels'),
            latent_width=config.get('latent_width'),
            timesteps=config.get('timesteps'),
            num_diffu_layers=config.get('num_diffu_layers'),
            kernel_size=config.get('kernel_size', 3),
            model_type=config.get('model_type', 'conv'),
            transformer_config=config.get('transformer_config', None),
        )
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name'))
    else:
        # Fallback for old checkpoints (assuming default config if not present, though likely will fail if structure differs)
        raise NotImplementedError("Loading from old checkpoints without config is not fully supported in this script.")

    model.to(device)
    model.eval()
    return model, tokenizer

def get_data_loader(tokenizer, split, limit=1000):
    """
    Loads the TinyStories dataset and returns a list of sentences.
    """
    print(f"Loading {split} dataset...", flush=True)
    dataset = TinyStoriesDataset(
        tokenizer,
        split=split,
        max_seq_len=128,
        skip_samples=0 # Start from beginning
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sentences = []
    count = 0
    for batch in tqdm(loader, total=limit, desc=f"Loading {split} sentences"):
        if count >= limit:
            break
        text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        sentences.append(text)
        count += 1
        
    return sentences

def extract_latents(model, tokenizer, sentences, device):
    """
    Extracts latent vectors for a list of sentences.
    """
    latents = []
    print("Extracting latents...", flush=True)
    for sent in tqdm(sentences):
        vec = get_latent_from_sent(sent, model, tokenizer, device)
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        latents.append(vec.flatten()) 
                                      
    return np.array(latents)

def interpolate_path(model, tokenizer, sent1, sent2, device, method='lerp', steps=10, vector_store=None):
    """
    Generates latent vectors for the interpolation path.
    """
    z1 = get_latent_from_sent(sent1, model, tokenizer, device)
    z2 = get_latent_from_sent(sent2, model, tokenizer, device)
    
    # Ensure z1, z2 are tensors on device for interpolation math
    if not isinstance(z1, torch.Tensor):
        z1 = torch.tensor(z1).to(device)
    if not isinstance(z2, torch.Tensor):
        z2 = torch.tensor(z2).to(device)
    
    # Reshape to (B, C, W) for slerp_channel_wise (requires 3 dims: b, c, w)
    # get_latent_from_sent returns .squeeze(), so it could be (W,) or (C, W)
    if z1.ndim == 1:
        z1 = z1.unsqueeze(0).unsqueeze(0) # (W) -> (1, 1, W)
    elif z1.ndim == 2:
        z1 = z1.unsqueeze(0) # (C, W) -> (1, C, W)
        
    if z2.ndim == 1:
        z2 = z2.unsqueeze(0).unsqueeze(0)
    elif z2.ndim == 2:
        z2 = z2.unsqueeze(0)

    path_latents = []
    alphas = np.linspace(0, 1, steps)
    
    if method == "bezier_2nd":
        if vector_store is None:
            raise ValueError("VectorDB is required for Bezier interpolation")
            
        # 2nd Order: Mean(V0, V2) -> Search DB -> V1
        avg_latent = (z1 + z2) / 2.0
        query_vector = avg_latent.squeeze().cpu().numpy().flatten().tolist()
        
        results = vector_store.similarity_search_with_score_by_vector(query_vector, k=1)
        control_text = results[0][0].page_content
        print(f"Bezier 2nd Control Point: {control_text}")
        
        z_v1 = get_latent_from_sent(control_text, model, tokenizer, device)
        # Ensure z_v1 shape
        if z_v1.ndim == 1:
            z_v1 = z_v1.unsqueeze(0).unsqueeze(0)
        elif z_v1.ndim == 2:
            z_v1 = z_v1.unsqueeze(0)
        
        for alpha in alphas:
            z_intp = bezier_2nd_order(z1, z_v1, z2, alpha)
            if isinstance(z_intp, torch.Tensor):
                z_intp = z_intp.detach().cpu().numpy()
            path_latents.append(z_intp.flatten())
            
    elif method == "bezier_3rd":
        if vector_store is None:
            raise ValueError("VectorDB is required for Bezier interpolation")
            
        # 3rd Order: 1/3 and 2/3 points
        t1 = 1.0 / 3.0
        p1_latent = (1 - t1) * z1 + t1 * z2
        query_p1 = p1_latent.squeeze().cpu().numpy().flatten().tolist()
        
        results_p1 = vector_store.similarity_search_with_score_by_vector(query_p1, k=1)
        cp1_text = results_p1[0][0].page_content
        print(f"Bezier 3rd Control Point 1: {cp1_text}")
        z_cp1 = get_latent_from_sent(cp1_text, model, tokenizer, device)
        if z_cp1.ndim == 1: z_cp1 = z_cp1.unsqueeze(0).unsqueeze(0)
        elif z_cp1.ndim == 2: z_cp1 = z_cp1.unsqueeze(0)
        
        t2 = 2.0 / 3.0
        p2_latent = (1 - t2) * z1 + t2 * z2
        query_p2 = p2_latent.squeeze().cpu().numpy().flatten().tolist()
        
        results_p2 = vector_store.similarity_search_with_score_by_vector(query_p2, k=1)
        cp2_text = results_p2[0][0].page_content
        print(f"Bezier 3rd Control Point 2: {cp2_text}")
        z_cp2 = get_latent_from_sent(cp2_text, model, tokenizer, device)
        if z_cp2.ndim == 1: z_cp2 = z_cp2.unsqueeze(0).unsqueeze(0)
        elif z_cp2.ndim == 2: z_cp2 = z_cp2.unsqueeze(0)
        
        for alpha in alphas:
            z_intp = bezier_3rd_order(z1, z_cp1, z_cp2, z2, alpha)
            if isinstance(z_intp, torch.Tensor):
                z_intp = z_intp.detach().cpu().numpy()
            path_latents.append(z_intp.flatten())

    else:
        # LERP / SLERP
        for alpha in alphas:
            if method == 'lerp':
                z_intp = linear_interpolate(z1, z2, alpha)
            elif method == 'slerp':
                z_intp = slerp_channel_wise(z1, z2, alpha)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
                
            if isinstance(z_intp, torch.Tensor):
                z_intp = z_intp.detach().cpu().numpy()
            
            path_latents.append(z_intp.flatten())
        
    return np.array(path_latents)

def main():
    parser = argparse.ArgumentParser(description="Visualize interpolation path with PCA/UMAP.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "umap"], help="Dimensionality reduction method")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for training/validation data")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--intp_method", type=str, default="lerp", choices=["lerp", "slerp", "bezier_2nd", "bezier_3rd"], help="Interpolation method")
    parser.add_argument("--output_plot", type=str, default="plots/interpolation_viz.png", help="Output path for the plot")
    parser.add_argument("--vectordb_path", type=str, default="../inference/saved_db/faiss_diffusion_embeddings.index", help="Path to VectorDB for Bezier")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model, tokenizer = load_model(args.model_path, device)
    
    # 2. Load VectorDB if needed
    vector_store = None
    if "bezier" in args.intp_method:
        if FAISS is None:
            raise ImportError("FAISS is required for bezier interpolation but not installed/imported.")
            
        print(f"Loading VectorDB from {args.vectordb_path}...")
        try:
            # We need a dummy wrapper for embeddings if FAISS.load_local expects one.
            # In inference_pairs.py, they pass embedder_wrapper which implements embed_documents and embed_query
            embedder_wrapper = DiffusionEmbeddings(model, tokenizer, device, t=499)
            vector_store = FAISS.load_local(
                folder_path=args.vectordb_path,
                embeddings=embedder_wrapper,
                allow_dangerous_deserialization=True
            )
            print(">> VectorDB loaded successfully.")
        except Exception as e:
            print(f">> Error loading DB: {e}")
            sys.exit(1)
    
    # 3. Load Data
    # Training Data
    train_sentences = get_data_loader(tokenizer, split="train", limit=args.n_samples)
    # Validation Data
    val_sentences = get_data_loader(tokenizer, split="validation", limit=args.n_samples)
    
    # 4. Extract Latents
    print("Processing Training Data...")
    train_latents = extract_latents(model, tokenizer, train_sentences, device)
    
    print("Processing Validation Data...")
    val_latents = extract_latents(model, tokenizer, val_sentences, device)
    
    # 5. Generate Interpolation Paths
    import random
    random.seed(42)
    
    # Logic from inference_pairs.py to generate pairs
    # Use a subset of validation sentences to make pairs
    NUM_SENTENCES_TO_PROCESS = 20
    # Ensure we have enough sentences
    if len(val_sentences) < NUM_SENTENCES_TO_PROCESS:
         subset = val_sentences
    else:
         # Shuffle and take subset
         shuffled_sent = val_sentences[:]
         random.shuffle(shuffled_sent)
         subset = shuffled_sent[:NUM_SENTENCES_TO_PROCESS]
    
    print(f"Selected {len(subset)} sentences for pairing.")
    
    sentence_pairs = []
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
             sentence_pairs.append((subset[i], subset[j]))
             
    print(f"Generated {len(sentence_pairs)} pairs to visualize.")
    
    # 6. Dimensionality Reduction
    print(f"Fitting {args.method.upper()} on Training Latents...")
    if args.method == "pca":
        reducer = PCA(n_components=2)
    elif args.method == "umap":
        if umap is None:
            raise ImportError("UMAP is not installed.")
        reducer = umap.UMAP(n_components=2, random_state=42)
        
    # Fit on training data
    reducer.fit(train_latents)
    
    # Transform background data once
    train_proj = reducer.transform(train_latents)
    val_proj = reducer.transform(val_latents)
    
    # 7. Loop and Plot
    print(f"Plotting {len(sentence_pairs)} pairs...")
    
    base_output_path = args.output_plot
    # Remove extension if present to append index
    if base_output_path.endswith(".png"):
        base_output_path = base_output_path[:-4]
        
    os.makedirs(os.path.dirname(base_output_path) if os.path.dirname(base_output_path) else ".", exist_ok=True)

    for i, (s1, s2) in enumerate(tqdm(sentence_pairs)):
        # Calculate path
        try:
            intp_latents = interpolate_path(model, tokenizer, s1, s2, device, method=args.intp_method, steps=args.steps, vector_store=vector_store)
        except Exception as e:
            print(f"Skipping pair {i} due to error: {e}")
            continue

        # Transform path
        intp_proj = reducer.transform(intp_latents)
        
        plt.figure(figsize=(12, 10))
        
        # Plot Training Data
        plt.scatter(train_proj[:, 0], train_proj[:, 1], c='lightgray', alpha=0.3, label='Training Data', s=10)
        
        # Plot Validation Data
        plt.scatter(val_proj[:, 0], val_proj[:, 1], c='lightblue', alpha=0.5, label='Validation Data', s=10)
        
        # Plot Interpolation Path
        plt.plot(intp_proj[:, 0], intp_proj[:, 1], c='red', linestyle='-', linewidth=2, label=f'Interpolation Path ({args.intp_method})', alpha=0.9)
        plt.scatter(intp_proj[:, 0], intp_proj[:, 1], c='red', s=30, marker='o')
        
        # Mark Start and End
        plt.scatter(intp_proj[0, 0], intp_proj[0, 1], c='green', s=150, marker='*', label='Start', edgecolors='k', zorder=10)
        plt.scatter(intp_proj[-1, 0], intp_proj[-1, 1], c='orange', s=150, marker='*', label='End', edgecolors='k', zorder=10)
        
        plt.title(f"Latent Manifold Projection ({args.method.upper()}) - {args.intp_method} - Pair {i}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{base_output_path}_{i}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"All plots saved to {base_output_path}_*.png")

if __name__ == "__main__":
    main()
