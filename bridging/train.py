print("train.py", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from model import DiffusionLM, decode_token_ids
from tqdm import tqdm

from datasets import load_dataset
import nltk


# NLTK의 문장 분리기 데이터 다운로드 (최초 1회만 실행됨)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') # 최신 버전 호환용

class StreamTinyStoriesDataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_samples=None, skip_samples=0, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        
        print(f"Loading streaming TinyStories dataset ({split})...")
        # TinyStories 데이터셋 로드
        self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
        # 셔플
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        
        if skip_samples > 0:
            self.dataset = self.dataset.skip(skip_samples)
        
        if max_samples is not None:
            self.dataset = self.dataset.take(max_samples)

    def __iter__(self):
        for example in self.dataset:
            text = example['text']
            
            # 1. 텍스트를 문장 단위로 분리 (List[str] 반환)
            sentences = nltk.sent_tokenize(text)
            
            # 2. 분리된 각 문장에 대해 반복
            for sentence in sentences:
                # 너무 짧은 문장(예: 공백이나 특수문자만 있는 경우)은 건너뛰기
                if len(sentence.strip()) < 5: 
                    continue

                # 3. 문장 토큰화
                enc = self.tokenizer(
                    sentence,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                # input_ids 길이를 체크할 필요가 거의 없음 (문장 하나는 128보다 짧을 확률이 높음)
                # 만약 문장이 128보다 길면 truncation=True에 의해 자동으로 잘립니다.

                yield {
                    'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0)
                }


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

class StreamC4Dataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_samples=None, skip_samples=0, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        
        print(f"Loading streaming C4 dataset ({split})...")
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        
        if skip_samples > 0:
            print(f"Skipping first {skip_samples} samples.")
            self.dataset = self.dataset.skip(skip_samples)

        if max_samples is not None:
            print(f"Taking next {max_samples} examples.")
            self.dataset = self.dataset.take(max_samples)

    def __iter__(self):
        # max_samples가 적용된 데이터셋이라도 필터링으로 인해 
        # 실제 산출되는 배치의 수는 줄어들 수 있음을 유의해야 합니다.
        for example in self.dataset:
            text = example['text']
            
            # 1. 먼저 자르지 않고(Truncation=False), 패딩 없이 토큰화만 수행하여 길이를 확인합니다.
            input_ids = self.tokenizer(
                text,
                truncation=False, # 중요: 여기서 자르지 않음
                padding=False,    # 중요: 여기서 패딩하지 않음
                add_special_tokens=True
            )['input_ids']
            
            # 2. 토큰 길이가 설정한 max_seq_len보다 길면 건너뜁니다.
            if len(input_ids) > self.max_seq_len:
                continue 
            
            # 3. 길이가 조건에 맞으면 텐서로 변환하고 패딩을 적용합니다.
            # (이미 input_ids를 가지고 있으므로 tokenizer.pad를 사용하면 효율적입니다)
            
            # tokenizer.pad는 딕셔너리 형태의 입력을 기대하므로 랩핑합니다.
            batch_encoding = self.tokenizer.pad(
                {'input_ids': input_ids},
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            yield {
                'input_ids': batch_encoding['input_ids'].squeeze(0),
                'attention_mask': batch_encoding['attention_mask'].squeeze(0)
            }

class LM1BDataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_samples=None, skip_samples=0, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        
        print(f"Loading streaming C4 dataset ({split})...")
        self.dataset = load_dataset("dvruette/lm1b", split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        
        if skip_samples > 0:
            print(f"Skipping first {skip_samples} samples.")
            self.dataset = self.dataset.skip(skip_samples)

        if max_samples is not None:
            print(f"Taking next {max_samples} examples.")
            self.dataset = self.dataset.take(max_samples)

    def __iter__(self):
        # max_samples가 적용된 데이터셋이라도 필터링으로 인해 
        # 실제 산출되는 배치의 수는 줄어들 수 있음을 유의해야 합니다.
        for example in self.dataset:
            text = example['text']
            
            # 1. 먼저 자르지 않고(Truncation=False), 패딩 없이 토큰화만 수행하여 길이를 확인합니다.
            input_ids = self.tokenizer(
                text,
                truncation=False, # 중요: 여기서 자르지 않음
                padding=False,    # 중요: 여기서 패딩하지 않음
                add_special_tokens=True
            )['input_ids']
            
            # 2. 토큰 길이가 설정한 max_seq_len보다 길면 건너뜁니다.
            if len(input_ids) > self.max_seq_len:
                continue 
            
            # 3. 길이가 조건에 맞으면 텐서로 변환하고 패딩을 적용합니다.
            # (이미 input_ids를 가지고 있으므로 tokenizer.pad를 사용하면 효율적입니다)
            
            # tokenizer.pad는 딕셔너리 형태의 입력을 기대하므로 랩핑합니다.
            batch_encoding = self.tokenizer.pad(
                {'input_ids': input_ids},
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            yield {
                'input_ids': batch_encoding['input_ids'].squeeze(0),
                'attention_mask': batch_encoding['attention_mask'].squeeze(0)
            }


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = 'bert-base-uncased' 
    MAX_LEN = 128
    BATCH_SIZE = 512
    EPOCHS = 10
    LR = 5e-5

    # --- Hyperparameters as Variables ---
    LATENT_CHANNELS = 1
    LATENT_WIDTH = 512
    TIMESTEPS = 1000
    KERNEL_SIZE = 3
    NUM_DIFFU_LAYERS = 8
    REG_WEIGHT = 1e-4 # Weight for regularization

    
    
    # File Paths
    TRAIN_FILE = "dataset/train.txt"
    VAL_FILE = "dataset/valid.txt"
    TEST_FILE = "dataset/test.txt"
    SAVE_PATH = f"model_outputs/diffusion_lm_{LATENT_WIDTH}_{LATENT_CHANNELS}_{NUM_DIFFU_LAYERS}_{TIMESTEPS}.pth"

    # Sampling Limits
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
    train_dataset = StreamTinyStoriesDataset(tokenizer, split="train", max_samples=TRAIN_SAMPLES, max_seq_len=MAX_LEN)
    
    # Val: Take first 1000 of validation split
    val_dataset = StreamTinyStoriesDataset(
        tokenizer, 
        split="validation", 
        max_samples=VAL_SAMPLES, 
        skip_samples=0, 
        max_seq_len=MAX_LEN
    )
    
    # Test: Skip the samples used for Val
    test_dataset = StreamTinyStoriesDataset(
        tokenizer, 
        split="validation", 
        max_samples=TEST_SAMPLES, 
        skip_samples=VAL_SAMPLES,
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
        timesteps=TIMESTEPS,
        num_diffu_layers=NUM_DIFFU_LAYERS,
        kernel_size=KERNEL_SIZE,
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- 5. Training Loop ---

    min_val_loss = 100

    print("\n--- Starting Training ---", flush=True)
    for epoch in tqdm(range(EPOCHS)):
        # Training Phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Train Loss: {loss.item():.4f}",flush=True)
        
        avg_train_loss = total_train_loss / batch_count

        # Validation Phase
        model.eval()
        total_val_loss = 0
        
        # We capture one batch to do the visual check
        sample_val_batch = None
        
        val_batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                val_batch_count += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                loss, _ = model(input_ids, attention_mask)
                total_val_loss += loss.item()
                if sample_val_batch is None:
                    sample_val_batch = (input_ids, attention_mask)
        
        # Handle case where val_loader might be empty
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0.0
        
        print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",flush=True)

        # --- SAVE MODEL ---
        if(avg_val_loss < min_val_loss):
            print(f"\nSaving model w/ minimum val loss to {SAVE_PATH}...")
            checkpoint = {
                'config': {
                    'bert_model_name': MODEL_NAME,
                    'max_seq_len': MAX_LEN,
                    'latent_channels': LATENT_CHANNELS,
                    'latent_width': LATENT_WIDTH,
                    'timesteps': TIMESTEPS,
                    'num_diffu_layers': NUM_DIFFU_LAYERS,
                    'kernel_size': KERNEL_SIZE,
                    # 'diversity_weight': DIVERSITY_WEIGHT
                },
                'state_dict': model.state_dict()
            }
            
            torch.save(checkpoint, SAVE_PATH)
            
            print("Model saved successfully.")

        # --- VISUALIZATION: Noisy Prediction Check ---
        if sample_val_batch is not None:
            print("\n--- Visualizing Predictions on Val Set (Noise -> Denoise) ---")
            inp, mask = sample_val_batch
            # Limit to 5 samples
            num_show = min(5, inp.shape[0])
            pred_ids, t_vals = model.check_prediction(inp, mask, num_samples=num_show)
            
            for i in range(num_show):
                original = tokenizer.decode(inp[i], skip_special_tokens=True)
                predicted = decode_token_ids(pred_ids[i], tokenizer)
                timestep = t_vals[i].item()
                print(f"Sample {i+1} | t={timestep:03d} | Orig: {original[:]} -> Pred: {predicted[:]}",flush=True)
            print("-------------------------------------------------------------\n")

    # --- 7. Final Test Set Evaluation ---
    print("\n--- Final Test Set Evaluation ---")
    model.eval()
    total_test_loss = 0
    test_batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            loss, _ = model(input_ids, attention_mask)
            total_test_loss += loss.item()
            test_batch_count += 1
            
    avg_test_loss = total_test_loss / test_batch_count if test_batch_count > 0 else 0.0
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    # --- 8. Quick Inference Test ---
    print("\n--- Inference Check (Pure Generation) ---")
    generated_ids = model.sample(batch_size=1)
    gen_text = decode_token_ids(generated_ids[0], tokenizer)
    print(f"Generated from noise: '{gen_text}'")