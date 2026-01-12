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