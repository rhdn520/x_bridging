import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import nltk
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer
from tqdm import tqdm
from datasets import load_dataset

# Import your custom model components
# Ensure 'model.py' is in the same directory or python path
from model import DiffusionLM, decode_token_ids

# --- DDP Helper Functions ---
def setup_ddp():
    """Sets up the distributed process group."""
    if 'LOCAL_RANK' not in os.environ:
        # Default for single-GPU debugging
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroys the process group."""
    dist.destroy_process_group()

def print_ddp(msg):
    """Prints only from Rank 0."""
    if dist.is_initialized() and dist.get_rank() == 0:
        print(msg, flush=True)
    elif not dist.is_initialized():
        print(msg, flush=True)

def download_nltk_resources():
    """Safe NLTK download for DDP (Rank 0 downloads, others wait)."""
    if dist.get_rank() == 0:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
    # Barrier ensures all processes wait until Rank 0 is done
    dist.barrier()

# --- Dataset Class ---
class TinyStoriesDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 split="train",
                 skip_samples=0, 
                 max_seq_len=128,
                 dataset_size=300000):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # We load the streaming dataset on all ranks.
        # DistributedSampler will handle the splitting of indices later.
        # It is safer to load identical data on all ranks than to manually shard here.
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        ds = ds.shuffle(seed=42) 

        self.data = []
        
        # Only print progress on Rank 0
        iterator = ds
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Loading {dataset_size} samples for {split}...")
            # iterator = tqdm(ds, total=dataset_size, desc=f"Loading {split}") # Optional visual

        # Collect valid sentences
        count_needed = dataset_size + skip_samples
        
        for example in iterator:
            text = example['text']
            sentences = nltk.sent_tokenize(text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 5: 
                    continue
                
                self.data.append(sentence.strip())

                if len(self.data) >= count_needed:
                    break
            
            if len(self.data) >= count_needed:
                break

        # Apply skip
        if skip_samples > 0:
            self.data = self.data[skip_samples:]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sentence = self.data[idx]
        
        enc = self.tokenizer(
            sentence,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        return {
                    'text': sentence,
                    'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0)
                }

# --- Main Execution ---
if __name__ == "__main__":
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    print_ddp(f"World size: {dist.get_world_size()}")

    # 1. Setup Resources
    download_nltk_resources()
    
    # 2. Configuration
    MODEL_NAME = 'bert-base-uncased' 
    MAX_LEN = 128
    BATCH_SIZE = 128    # Batch size per GPU
    EPOCHS = 10
    LR = 1e-4
    RESUME_TRAINING = False

    # Hyperparameters
    LATENT_CHANNELS = 1
    LATENT_WIDTH = 1024
    TIMESTEPS = 1000
    KERNEL_SIZE = 5
    NUM_DIFFU_LAYERS = 8
    TIME_BIAS = 0.3
    
    MODEL_TYPE = 'transformer' 
    TRANSFORMER_CONFIG = {
        'd_model': 1024,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.1
    }

    # Paths
    os.makedirs("model_outputs", exist_ok=True)
    SAVE_PATH = f"model_outputs/{MODEL_TYPE}_{LATENT_WIDTH}_{LATENT_CHANNELS}_{NUM_DIFFU_LAYERS}_{TIMESTEPS}_d{TRANSFORMER_CONFIG['d_model']}.pth"

    # Data Limits
    TRAIN_SAMPLES = 3000000
    VAL_SAMPLES = 100000
    TEST_SAMPLES = 100000

    # 3. Tokenizer & Data
    print_ddp("\n--- Initializing Tokenizer & Data ---")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Load Data (Identical on all ranks)
    train_dataset = TinyStoriesDataset(tokenizer, split="train", dataset_size=TRAIN_SAMPLES, max_seq_len=MAX_LEN)
    val_dataset = TinyStoriesDataset(tokenizer, split="validation", dataset_size=VAL_SAMPLES, skip_samples=0, max_seq_len=MAX_LEN)
    test_dataset = TinyStoriesDataset(tokenizer, split="validation", dataset_size=TEST_SAMPLES, skip_samples=VAL_SAMPLES, max_seq_len=MAX_LEN)

    # Samplers handles the splitting
    dist_sampler_train = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    dist_sampler_val = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    dist_sampler_test = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    # DataLoaders (num_workers > 0 is important for speed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=dist_sampler_train, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=dist_sampler_val, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=dist_sampler_test, num_workers=1, pin_memory=True)

    # 4. Model Initialization
    model = DiffusionLM(
        bert_model_name=MODEL_NAME, 
        max_seq_len=MAX_LEN, 
        latent_channels=LATENT_CHANNELS, 
        latent_width=LATENT_WIDTH, 
        timesteps=TIMESTEPS,
        model_type=MODEL_TYPE,
        transformer_config=TRANSFORMER_CONFIG,
        num_diffu_layers=NUM_DIFFU_LAYERS,
        kernel_size=KERNEL_SIZE,
        time_bias=TIME_BIAS
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Resume Logic (Before DDP Wrap) ---
    start_epoch = 0
    min_val_loss = float('inf')

    if RESUME_TRAINING and os.path.exists(SAVE_PATH):
        print_ddp(f"Resuming training from {SAVE_PATH}...")
        # Map location is critical for DDP resume
        checkpoint = torch.load(SAVE_PATH, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        min_val_loss = checkpoint.get('val_loss', float('inf'))
        print_ddp(f"Resumed from epoch {start_epoch}, Min Val Loss: {min_val_loss:.4f}")

    # Wrap Model in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. Training Loop
    print_ddp("\n--- Starting Training ---")
    
    for epoch in range(start_epoch, EPOCHS):
        # Crucial: Set epoch for sampler so shuffling is different every epoch
        dist_sampler_train.set_epoch(epoch)
        
        # ================= TRAIN =================
        model.train()
        total_train_loss = torch.tensor(0.0, device=device)
        train_batch_count = torch.tensor(0.0, device=device)
        
        # Only show progress bar on rank 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", disable=(dist.get_rank() != 0))
        
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            loss, info = model(input_ids, attention_mask)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batch_count += 1
            
            # Simple logging on progress bar
            if dist.get_rank() == 0:
                iterator.set_postfix({'loss': f"{loss.item():.4f}"})

        # Aggregate Train Metrics
        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_batch_count, op=dist.ReduceOp.SUM)
        avg_train_loss = total_train_loss / train_batch_count

        # ================= VALIDATION =================
        model.eval()
        total_val_loss = torch.tensor(0.0, device=device)
        val_batch_count = torch.tensor(0.0, device=device)
        sample_val_batch = None
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                loss, _ = model(input_ids, attention_mask)
                total_val_loss += loss.item()
                val_batch_count += 1
                
                # Save one batch for visualization (Rank 0 only)
                if dist.get_rank() == 0 and sample_val_batch is None:
                    sample_val_batch = (input_ids, attention_mask)
        
        # Aggregate Val Metrics
        dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batch_count, op=dist.ReduceOp.SUM)
        avg_val_loss = total_val_loss / val_batch_count
        
        print_ddp(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ================= SAVE & VISUALIZE =================
        if dist.get_rank() == 0:
            # Save if better
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                print_ddp(f"Saving new best model ({min_val_loss:.4f})...")
                checkpoint = {
                    'config': {
                        'bert_model_name': MODEL_NAME,
                        'max_seq_len': MAX_LEN,
                        'latent_channels': LATENT_CHANNELS,
                        'latent_width': LATENT_WIDTH,
                        'timesteps': TIMESTEPS,
                        'model_type': MODEL_TYPE,
                        'transformer_config': TRANSFORMER_CONFIG,
                        'num_diffu_layers': NUM_DIFFU_LAYERS,
                        'kernel_size': KERNEL_SIZE,
                    },
                    # Save module.state_dict() to unwrap DDP
                    'state_dict': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }
                torch.save(checkpoint, SAVE_PATH)

            # Visualization
            if sample_val_batch is not None:
                print_ddp("\n--- Val Visualization ---")
                inp, mask = sample_val_batch
                num_show = min(3, inp.shape[0])
                
                # Use model.module for custom methods not in forward
                pred_ids, t_vals = model.module.check_prediction(inp, mask, num_samples=num_show)
                
                for i in range(num_show):
                    original = tokenizer.decode(inp[i], skip_special_tokens=True)
                    predicted = decode_token_ids(pred_ids[i], tokenizer)
                    print_ddp(f"Orig: {original[:60]}... -> Pred: {predicted[:60]}...")
                print_ddp("-------------------------\n")

    # 6. Final Test
    print_ddp("\n--- Final Test Set Evaluation ---")
    model.eval()
    total_test_loss = torch.tensor(0.0, device=device)
    test_batch_count = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss, _ = model(input_ids, attention_mask)
            total_test_loss += loss.item()
            test_batch_count += 1

    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_batch_count, op=dist.ReduceOp.SUM)
    avg_test_loss = total_test_loss / test_batch_count

    print_ddp(f"Final Test Loss: {avg_test_loss:.4f}")

    # 7. Inference Test (Rank 0 only)
    if dist.get_rank() == 0:
        print_ddp("\n--- Generation Check ---")
        generated_ids = model.module.sample(batch_size=1)
        gen_text = decode_token_ids(generated_ids[0], tokenizer)
        print_ddp(f"Generated: '{gen_text}'")

    cleanup_ddp()