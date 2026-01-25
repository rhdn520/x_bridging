import sys
# sys.path.append("../")
sys.path.append("../inference/")
sys.path.append("../utils/")
import argparse
import json
import os
import random
import torch
import torch.nn as nn
from datasets import Dataset   
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F
from custom_dataset import TinyStoriesDataset
from model import DiffusionLM, Putter 
from latent_intp import linear_interpolate, slerp_channel_wise #type:ignore
from inference import DiffusionTracer  # type: ignore
from vllm_models import VllmSentenceRefiner # type: ignore

# --- Helper Function: Set Seed ---
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(tokenizer, args, split_name, num_samples):
    """
    특정 split에서 num_samples만큼 데이터를 로드하는 함수
    """
    print(f"Loading '{split_name}' dataset (Target Size: {num_samples})...", flush=True)
    
    # TinyStoriesDataset 초기화 시 split과 size를 직접 지정
    ds = TinyStoriesDataset(
        tokenizer,
        split=split_name,
        max_seq_len=args.max_seq_len,
        skip_samples=args.skip_samples,
        dataset_size=num_samples
    )

    # num_samples가 0이거나 데이터가 없으면 빈 리스트 반환
    if len(ds) == 0:
        return []

    ds_loader = DataLoader(ds, batch_size=1)
    
    sent_list = []
    print(f"Extracting sentences from {split_name}...", flush=True)
    for batch in ds_loader:
        tmp = {
            "text": batch["text"][0],
            "input_ids": batch["input_ids"][0].tolist(),
            "attention_mask": batch["attention_mask"][0].tolist()
        }
        sent_list.append(tmp)
        
    return sent_list

def generate_sentence_pairs(sent_list, seed=42):
    random.seed(seed)
    products = []
    for i in range(len(sent_list)):
        for j in range(i+1, len(sent_list)):
            products.append({
                "sent0_ids": sent_list[i]["input_ids"],
                "sent1_ids": sent_list[j]["input_ids"],
                "sent0_attention_mask": sent_list[i]["attention_mask"],
                "sent1_attention_mask": sent_list[j]["attention_mask"],
            })     
    return products

# --- Core Logic for Training and Validation ---
def compute_loss(batch, putter, diffusionTracer, refiner, tokenizer, max_seq_len):
    sent0_ids = batch['sent0_ids'].to(diffusionTracer.device)
    sent1_ids = batch['sent1_ids'].to(diffusionTracer.device)
    sent0_attention_mask = batch['sent0_attention_mask'].to(diffusionTracer.device)
    sent1_attention_mask = batch['sent1_attention_mask'].to(diffusionTracer.device)
    
    # 1. Latent Projections
    z_0_0 = diffusionTracer.model.bert(sent0_ids, attention_mask=sent0_attention_mask).last_hidden_state
    batch_size = z_0_0.size(0)
    z_0_0 = diffusionTracer.model.encoder_proj(z_0_0.view(batch_size, -1))
    z_0_0 = z_0_0.view(batch_size, diffusionTracer.model.latent_channels, diffusionTracer.model.latent_width)
    
    z_1_0 = diffusionTracer.model.bert(sent1_ids, attention_mask=sent1_attention_mask).last_hidden_state
    z_1_0 = diffusionTracer.model.encoder_proj(z_1_0.view(batch_size, -1))
    z_1_0 = z_1_0.view(batch_size, diffusionTracer.model.latent_channels, diffusionTracer.model.latent_width)

    # 2. Add Noise
    t = random.randint(0, diffusionTracer.model.timesteps - 1)
    z_0_t = diffusionTracer.trace_noising(z_0_0, t)['noisy_latent']
    z_1_t = diffusionTracer.trace_noising(z_1_0, t)['noisy_latent']
    
    # 3. Interpolate
    z_alpha_t = linear_interpolate(z_0_t, z_1_t, 0.5)
    
    # 4. Generate (Prediction)
    z_alpha_0_pred_trace = diffusionTracer.trace_generation(z_alpha_t, t)
    x_0_pred = z_alpha_0_pred_trace[-1]['x_0_pred'] 
    text_estimate = z_alpha_0_pred_trace[-1]['text_estimate']

    # 5. Refine (Target)
    refined_text = refiner.refine_batch(text_estimate)
    refined_text = [text.strip().lower() for text in refined_text]
    
    refined_text_encode = [tokenizer(text, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True) for text in refined_text]
    refined_text_labels = torch.cat([x['input_ids'] for x in refined_text_encode], dim=0).to(diffusionTracer.device)

    # 6. Putter Forward
    z_alpha_0_putt = putter(x_0_pred.view(batch_size, -1))
    z_alpha_0_putt = z_alpha_0_putt.view(batch_size, diffusionTracer.model.latent_channels, diffusionTracer.model.latent_width)
    
    # 7. Decode & Loss
    z_alpha_logits = diffusionTracer.model.decode_latents(z_alpha_0_putt)
    
    loss = F.cross_entropy(
        z_alpha_logits.view(-1, z_alpha_logits.size(-1)), 
        refined_text_labels.view(-1), 
        ignore_index=tokenizer.pad_token_id
    )
    
    return loss, refined_text

def main():    
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--dlm_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # Putter Model
    parser.add_argument('--putter_hidden_dim', type=int, default=2048)
    parser.add_argument('--putter_layers', type=int, default=3)
    
    # External Models
    parser.add_argument('--sent_refiner_model_id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    
    # Data Config
    parser.add_argument('--num_train_sentences', type=int, default=40, help='Number of sentences for Training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio to determine Validation set size based on Train size')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--skip_samples', type=int, default=0)
    
    # Training Config
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load Models
    print(f"Loading Diffusion Model...", flush=True)
    diffusionTracer = DiffusionTracer(model_path=args.dlm_path, args=None)
    diffusionTracer.model.eval() 
    tokenizer = diffusionTracer.tokenizer
    
    putter = Putter(input_dim = diffusionTracer.model.latent_channels * diffusionTracer.model.latent_width,
                    hidden_dim = args.putter_hidden_dim,
                    output_dim = diffusionTracer.model.latent_channels * diffusionTracer.model.latent_width,
                    layer_num = args.putter_layers)
    putter.to(diffusionTracer.device)
    
    refiner = VllmSentenceRefiner(model_id=args.sent_refiner_model_id)

    # 2. Data Preparation (Modified)
    # Train Data Load
    train_sent_list = load_data(
        tokenizer, 
        args, 
        split_name="train", 
        num_samples=args.num_train_sentences
    )
    
    # Validation Data Load
    # Validation size is calculated based on ratio, or you can add a separate arg
    num_val_sentences = int(args.num_train_sentences * args.val_ratio)
    val_sent_list = load_data(
        tokenizer, 
        args, 
        split_name="validation", 
        num_samples=num_val_sentences
    )
    
    print(f"Loaded Sentences - Train: {len(train_sent_list)}, Val: {len(val_sent_list)}")
    
    # Generate Pairs
    train_pairs = generate_sentence_pairs(train_sent_list, seed=args.seed)
    val_pairs = generate_sentence_pairs(val_sent_list, seed=args.seed)
    
    print(f"Generated Pairs - Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Loaders
    def create_loader(pairs, bs, shuffle):
        if len(pairs) == 0: return None
        ds = Dataset.from_list(pairs)
        ds.set_format(type='torch', columns=['sent0_ids', 'sent1_ids', 'sent0_attention_mask', 'sent1_attention_mask'])
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = create_loader(train_pairs, args.batch_size, shuffle=True)
    val_loader = create_loader(val_pairs, args.batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(putter.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    print(f"\nStarting Training Loop...", flush=True)
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Train
        putter.train()
        train_loss_sum = 0
        train_steps = 0
        if train_loader:
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                optimizer.zero_grad()
                loss, _ = compute_loss(batch, putter, diffusionTracer, refiner, tokenizer, args.max_seq_len)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_steps += 1
                pbar.set_postfix({'loss': loss.item()})
            print(f"Epoch {epoch+1} Train Loss: {train_loss_sum/train_steps:.4f}")

        # Validation
        putter.eval() 
        val_loss_sum = 0
        val_steps = 0
        if val_loader:
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    loss, refined_text = compute_loss(batch, putter, diffusionTracer, refiner, tokenizer, args.max_seq_len)
                    val_loss_sum += loss.item()
                    val_steps += 1
            avg_val_loss = val_loss_sum / val_steps
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved. Saving model...")
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.save_dir, 'best_putter.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': putter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'args': vars(args)
                }, save_path)
        else:
            print("Skipping validation (no data).")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()