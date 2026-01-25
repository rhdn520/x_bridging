import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import nltk
import json
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer
from tqdm import tqdm
from datasets import load_dataset



# --- Dataset Class ---
class C4Dataset(Dataset):
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
        ds = load_dataset("allenai/c4", split=split, streaming=True)
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
        ds = load_dataset("roneneldan/TinyStories", split=split)
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

class InterpolationDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 data_path,
                 split="train", # Argument kept for compatibility but not strictly used for file selection
                 skip_samples=0, 
                 max_seq_len=128,
                 return_all=False,
                 dataset_size=None): # dataset_size optional, if None use all available
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.return_all = return_all
        
        if dist.is_initialized() and dist.get_rank() == 0:
             print(f"Loading Interpolation data from {data_path}...")

        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            # print("raw_data", len(raw_data))
            
        # Extract middle sentences
        all_paths = []
        for sents in raw_data:
            tmp = sents.split('\n')
            tmp = [x.strip() for x in tmp]
            tmp = [x.lower() for x in tmp]
            tmp = self.remove_duplicates_preserve_order(tmp)
            if(len(tmp) >= 3):
                all_paths.append(tmp)
        
        print("Total valid Intp Paths:", len(all_paths))

        all_triplets = []
        for path in all_paths:
            n = len(path)
            for i in range(1, n-1):
                triplet = [path[i-1], path[i], path[i+1]]
                all_triplets.append(triplet)
        # all_paths = all_triplets
        print("Total valid Intp Triplets:", len(all_triplets))
        del all_paths

                

        start_idx = skip_samples
        end_idx = len(all_triplets)

        #shuffle data for randomness
        import random
        random.seed(42)
        random.shuffle(all_triplets)
        
        if dataset_size is not None:
             end_idx = start_idx + dataset_size
             
        # Slice the data
        # Ensure we don't go out of bounds
        if start_idx >= len(all_triplets):
            self.data = []
        else:
            final_end = min(end_idx, len(all_triplets))
            self.data = all_triplets[start_idx:final_end]
            
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Loaded {len(self.data)} samples after skip={skip_samples} and size limit={dataset_size}")


    def __len__(self):
        return len(self.data)

    def remove_duplicates_preserve_order(self, lst):
        seen = set()
        unique = []

        for x in lst:
            if x not in seen:
                seen.add(x)
                unique.append(x)

        return unique
    
    def __getitem__(self, idx):
        sents = self.data[idx]

        res = {
            'text_start': sents[0],
            'text_intp': sents[1],
            'text_end': sents[2]
        }  

        for i, sentence in enumerate(sents):
            enc = self.tokenizer(
                sentence,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            if i == 0:
                res['input_ids_start'] = enc['input_ids'].squeeze(0)
                res['attention_mask_start'] = enc['attention_mask'].squeeze(0)
            elif i == 1:
                res['input_ids_intp'] = enc['input_ids'].squeeze(0)
                res['attention_mask_intp'] = enc['attention_mask'].squeeze(0)
            elif i == 2:
                res['input_ids_end'] = enc['input_ids'].squeeze(0)
                res['attention_mask_end'] = enc['attention_mask'].squeeze(0)

        return res