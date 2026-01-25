from random import sample
import sys

from numpy import diff
sys.path.append("../utils")
sys.path.append("../train")
sys.path.append("../inference")
import os

from model import DiffusionLM
import torch

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from train import TinyStoriesDataset

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from get_latent_path import get_latent_from_sent
from tqdm import tqdm

class DiffusionEmbeddings():
    def __init__(self, model, tokenizer, device, t=499):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.t = t

    def embed_documents(self, texts):
        embeddings = []
        for text in tqdm(texts):
            with torch.no_grad():
                latent = get_latent_from_sent(
                    text,
                    self.model,
                    self.tokenizer,
                    self.device)
                
                # latent = self.model.q_sample_no_stochastic(latent, torch.tensor([self.t]).to(self.device))
                latent = latent.squeeze(0).cpu().numpy()  # (latent_channels, latent_width)
                embeddings.append(latent.flatten())  # Flatten to 1D array
        return embeddings        


# LOAD DiffusionLM for embedding
MODEL_PATH = "../train/model_outputs/transformer_1024_1_8_1000_td1024_dtypetinystories.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

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
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.get('bert_model_name'))
else:
    raise ValueError(">> No configuration found in checkpoint. Unable to build model.")

# Load Train Database 
print("\n--- Loading Datasets ---", flush=True)

sample_size = 5000
train_dataset = TinyStoriesDataset(
                    tokenizer, 
                    split="train", 
                    dataset_size=sample_size, 
                    max_seq_len=128
                )

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)

# Build Embedding

print("\n--- Generating Embeddings ---", flush=True)
texts = []
embeddings = []

diffusion_embedder = DiffusionEmbeddings(model, tokenizer, device)

for batch in tqdm(train_loader):
    # input_ids = batch['input_ids'].squeeze(0)  # (seq_len)
    # attention_mask = batch['attention_mask'].squeeze(0)  # (seq_len)
    print(batch['text'], flush=True)
    with torch.no_grad():
        texts.extend(batch['text'])

# Build VectorDB
vector_store = FAISS.from_texts(texts, embedding=diffusion_embedder)

# Save VectorDB
VECTOR_DB_PATH = "./saved_db/faiss_diffusion_embeddings.index"
vector_store.save_local(VECTOR_DB_PATH)
print(f"\n--- VectorDB saved to {VECTOR_DB_PATH} ---", flush=True)
