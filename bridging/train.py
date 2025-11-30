import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from model import DiffusionLM
from tqdm import tqdm
from datasets import load_dataset

# --- New Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lines = []
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Dataset will be empty.")
        else:
            print(f"Loading data from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line: # Skip empty lines
                        self.lines.append(line)
            print(f"Loaded {len(self.lines)} lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Important: Remove the batch dimension (1, seq_len) -> (seq_len)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


# --- Streaming Dataset Class ---
class StreamC4Dataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_samples=None, skip_samples=0, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        
        print(f"Loading streaming C4 dataset ({split})...")
        # Load C4 in streaming mode.
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        
        # Shuffle with a buffer. This ensures randomness within the buffer window.
        # We seed it so that the order is deterministic (allowing correct skipping).
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        
        # Skip samples if requested (Crucial for splitting streams)
        if skip_samples > 0:
            print(f"Skipping first {skip_samples} samples.")
            self.dataset = self.dataset.skip(skip_samples)

        # If max_samples is set, we limit the iterator to that many items.
        if max_samples is not None:
            print(f"Taking next {max_samples} examples.")
            self.dataset = self.dataset.take(max_samples)

    def __iter__(self):
        for example in self.dataset:
            text = example['text']
            
            # Tokenize
            enc = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            yield {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            }

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = 'bert-base-uncased' 
    MAX_LEN = 512
    BATCH_SIZE = 256
    EPOCHS = 500
    LR = 5e-5
    LATENT_WIDTH = 1024
    LATENT_CHANNELS = 1
    NUM_DIFFU_LAYERS = 64
    DIFFU_TIMESTEPS = 2000
    
    # File Paths
    TRAIN_FILE = "dataset/train.txt"
    VAL_FILE = "dataset/valid.txt"
    TEST_FILE = "dataset/test.txt"
    SAVE_PATH = f"diffusion_lm_{LATENT_WIDTH}.pth"

    # Sampling Limits (since C4 is huge)
    TRAIN_SAMPLES = 200000
    VAL_SAMPLES = 10000
    TEST_SAMPLES = 10000


    # --- 2. Setup Device & Tokenizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # --- 3. Data Loading (Separate Datasets) ---
    print("\n--- Loading Datasets ---" ,flush=True)
    # Train is separate, just take first N
    train_dataset = StreamC4Dataset(tokenizer, split="train", max_samples=TRAIN_SAMPLES, max_seq_len=MAX_LEN)
    
    # Val: Take first 1000 of validation split
    val_dataset = StreamC4Dataset(
        tokenizer, 
        split="validation", 
        max_samples=VAL_SAMPLES, 
        skip_samples=0, 
        max_seq_len=MAX_LEN
    )
    
    # Test: Skip the 1000 used for Val, then take next 500
    test_dataset = StreamC4Dataset(
        tokenizer, 
        split="validation", 
        max_samples=TEST_SAMPLES, 
        skip_samples=VAL_SAMPLES, # Skip what we used for Val
        max_seq_len=MAX_LEN
    )


    # Check if data exists
    # if len(train_dataset) == 0:
    #     raise ValueError("Training dataset is empty!")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- 4. Model Initialization ---
    model = DiffusionLM(
        bert_model_name=MODEL_NAME, 
        max_seq_len=MAX_LEN, 
        latent_channels=LATENT_CHANNELS, 
        latent_width=LATENT_WIDTH, 
        timesteps=DIFFU_TIMESTEPS,
        num_diffu_layers=NUM_DIFFU_LAYERS
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---",flush=True)
    for epoch in tqdm(range(EPOCHS)):
        # Training Phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Train Loss: {loss.item():.4f}",flush=True)
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        
        # We capture one batch to do the visual check
        sample_val_batch = None
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                loss = model(input_ids, attention_mask)
                total_val_loss += loss.item()
                if sample_val_batch is None:
                    sample_val_batch = (input_ids, attention_mask)
        
        # Handle case where val_loader might be empty
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",flush=True)

        # --- VISUALIZATION: Noisy Prediction Check ---
        if sample_val_batch is not None:
            print("\n--- Visualizing Predictions on Val Set (Noise -> Denoise) ---")
            inp, mask = sample_val_batch
            # Limit to 5 samples
            num_show = min(5, inp.shape[0])
            pred_ids, t_vals = model.check_prediction(inp, mask, num_samples=num_show)
            
            for i in range(num_show):
                original = tokenizer.decode(inp[i], skip_special_tokens=True)
                predicted = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
                timestep = t_vals[i].item()
                print(f"Sample {i+1} | t={timestep:03d} | Orig: {original[:]} -> Pred: {predicted[:]}",flush=True)
            print("-------------------------------------------------------------\n")

    # --- 6. Saving Model ---
    print(f"\nSaving model to {SAVE_PATH}...")
    torch.save(model.state_dict(), SAVE_PATH)
    print("Model saved successfully.")

    # --- 7. Final Test Set Evaluation ---
    print("\n--- Final Test Set Evaluation ---")
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss = model(input_ids, attention_mask)
            total_test_loss += loss.item()
            
    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    # --- 8. Quick Inference Test ---
    print("\n--- Inference Check (Pure Generation) ---")
    generated_ids = model.sample(batch_size=1)
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated from noise: '{gen_text}'")