from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
from custom_dataset import TinyStoriesDataset, InterpolationDataset

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


# --- Main Execution ---
if __name__ == "__main__":
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    print_ddp(f"World size: {dist.get_world_size()}")

    # 1. Setup Resources
    download_nltk_resources()
    
    # 2. Configuration via Argparse
    import argparse
    parser = argparse.ArgumentParser(description="Train DiffusionLM")

    # Basic Config
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume training if checkpoint exists')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume training from')

    # Hyperparameters
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels')
    parser.add_argument('--latent_width', type=int, default=1024, help='Latent width')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for diffusion')
    parser.add_argument('--num_diffu_layers', type=int, default=8, help='Number of diffusion layers')
    parser.add_argument('--time_bias', type=float, default=0.3, help='Time bias')

    # Model Type
    parser.add_argument('--model_type', type=str, default='transformer', help='Model type')

    # Transformer Config
    parser.add_argument('--transformer_d_model', type=int, default=1024, help='Transformer d_model')
    parser.add_argument('--transformer_nhead', type=int, default=8, help='Transformer nhead')
    parser.add_argument('--transformer_num_layers', type=int, default=6, help='Transformer num_layers')
    parser.add_argument('--transformer_dim_feedforward', type=int, default=4096, help='Transformer dim_feedforward')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Transformer dropout')

    # Data Limits
    parser.add_argument('--train_samples', type=int, default=3000000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=100000, help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=100000, help='Number of test samples')

    # Dataset Config
    parser.add_argument('--interpolation_data_path', type=str, default='bridging/dataset/vllm_interpolation_outputs.json', help='Path to interpolation dataset JSON')

    # Save Policy
    parser.add_argument('--save_every_epoch', action='store_true', help='Save model after every epoch.')

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    RESUME_TRAINING = args.resume
    SAVE_EVERY_EPOCH = args.save_every_epoch
    LATENT_CHANNELS = args.latent_channels
    LATENT_WIDTH = args.latent_width
    TIMESTEPS = args.timesteps
    KERNEL_SIZE = args.kernel_size
    NUM_DIFFU_LAYERS = args.num_diffu_layers
    TIME_BIAS = args.time_bias
    
    MODEL_TYPE = args.model_type
    TRANSFORMER_CONFIG = {
        'd_model': args.transformer_d_model,
        'nhead': args.transformer_nhead,
        'num_layers': args.transformer_num_layers,
        'dim_feedforward': args.transformer_dim_feedforward,
        'dropout': args.transformer_dropout
    }

    # Paths
    os.makedirs("model_outputs", exist_ok=True)
    SAVE_PATH = f"model_outputs/{MODEL_TYPE}_{LATENT_WIDTH}_{LATENT_CHANNELS}_{NUM_DIFFU_LAYERS}_{TIMESTEPS}_td{TRANSFORMER_CONFIG['d_model']}_intp.pth"

    # Data Limits
    TRAIN_SAMPLES = args.train_samples
    VAL_SAMPLES = args.val_samples
    TEST_SAMPLES = args.test_samples

    # 3. Tokenizer & Data
    print_ddp("\n--- Initializing Tokenizer & Data ---")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        
    train_dataset = InterpolationDataset(tokenizer, data_path=args.interpolation_data_path, dataset_size=TRAIN_SAMPLES, max_seq_len=MAX_LEN, return_all=True)
    val_dataset = InterpolationDataset(tokenizer, data_path=args.interpolation_data_path, dataset_size=VAL_SAMPLES, skip_samples=TRAIN_SAMPLES, max_seq_len=MAX_LEN, return_all=True)
    test_dataset = InterpolationDataset(tokenizer, data_path=args.interpolation_data_path, dataset_size=TEST_SAMPLES, skip_samples=TRAIN_SAMPLES + VAL_SAMPLES, max_seq_len=MAX_LEN, return_all=True)

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

    if RESUME_TRAINING and args.resume_path is not None and os.path.exists(args.resume_path):
        print_ddp(f"Resuming training from {args.resume_path}...")
        # Map location is critical for DDP resume
        checkpoint = torch.load(args.resume_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        min_val_loss = checkpoint.get('val_loss', float('inf'))
        print_ddp(f"Resumed from epoch {start_epoch}, Min Val Loss: {min_val_loss:.4f}")
    
    else:
        print_ddp("No resume path provided or resume path does not exist. Starting from scratch.")

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
            optimizer.zero_grad()
            loss, _ = model.module.forward_interpolation(batch, device)
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

        info = None
        
        with torch.no_grad():
            for batch in val_loader:
                loss, info = model.module.forward_interpolation(batch, device)
                total_val_loss += loss.item()
                val_batch_count += 1

        if dist.get_rank() == 0 and info is not None:
            print(info['text_intp_hat'][:3])  # Print first 3 predictions for inspection

        dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batch_count, op=dist.ReduceOp.SUM)
        avg_val_loss = total_val_loss / val_batch_count

        print_ddp(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


    #     # ================= SAVE & VISUALIZE =================
        if dist.get_rank() == 0:
            # Save if better or if save_every_epoch is True
            if (avg_val_loss < min_val_loss) or SAVE_EVERY_EPOCH:
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

    # 6. Final Test
    print_ddp("\n--- Final Test Set Evaluation ---")
    model.eval()
    total_test_loss = torch.tensor(0.0, device=device)
    test_batch_count = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for batch in test_loader:
            loss, _ = model.module.forward_interpolation(batch, device)
            total_test_loss += loss.item()
            test_batch_count += 1

    dist.all_reduce(total_test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_batch_count, op=dist.ReduceOp.SUM)
    avg_test_loss = total_test_loss / test_batch_count

    print_ddp(f"Final Test Loss: {avg_test_loss:.4f}")

    # # 7. Inference Test (Rank 0 only)
    if dist.get_rank() == 0:
        print_ddp("\n--- Generation Check ---")
        generated_ids = model.module.sample(batch_size=1)
        gen_text = decode_token_ids(generated_ids[0], tokenizer)
        print_ddp(f"Generated: '{gen_text}'")

    cleanup_ddp()