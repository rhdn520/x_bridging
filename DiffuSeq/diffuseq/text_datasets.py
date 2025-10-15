# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2

def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            sampler=sampler,
            # shuffle=not deterministic,
            num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            # sampler=sampler,
            shuffle=not deterministic,
            num_workers=1,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst): #여기서는 mask를 넣기는 하는데 일단 다 0으로 채움 뒤에 pad_function에서 제대로 하는듯
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg) #SRC랑 TGT이 하나로 합쳐지는 순간!!!!!!
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(lm_datasets['input_id_x'])
    print(lm_datasets['input_id_y'])
    print(lm_datasets['input_ids'])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            content = json.loads(row)
            sentence_lst['src'].append(content['src'].strip())
            sentence_lst['trg'].append(content['trg'].strip())

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    # print("_collate_batch_helper", flush=True)
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int16).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int16).tolist()
    # print(result, flush=True)
    # print(mask_, flush=True)
    for i, example in tqdm(enumerate(examples)):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


# def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
#     # 1) 입력을 텐서로 변환하고 max_length로 자르기 (루프는 변환에만 사용)
#     if not torch.is_tensor(examples[0]):
#         seqs = [torch.tensor(x[:max_length], dtype=torch.long) for x in examples]
#     else:
#         seqs = [x.to(dtype=torch.long)[:max_length] for x in examples]

#     # 2) pad_sequence는 C++ 구현 → 매우 빠름
#     padded = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)  # (B, L*)
#     B, Lstar = padded.shape

#     # 3) 정확히 max_length로 자르거나 더 패딩
#     if Lstar > max_length:
#         padded = padded[:, :max_length]
#     elif Lstar < max_length:
#         padded = torch.nn.functional.pad(padded, (0, max_length - Lstar), value=pad_token_id)

#     if not return_mask:
#         return padded.tolist()  # ✅ 원래와 동일하게 list 반환

#     # 4) 마스크: 원래 함수의 의미를 그대로 유지(채워진 위치=1, 패딩=pad_token_id)
#     lens = torch.tensor([min(len(x), max_length) for x in examples], device=padded.device)
#     ar = torch.arange(max_length, device=padded.device).unsqueeze(0)  # (1, L)
#     mask_bool = ar < lens.unsqueeze(1)                                # (B, L) bool
#     mask = torch.where(
#         mask_bool,
#         torch.ones_like(padded, dtype=torch.long),
#         torch.full_like(padded, pad_token_id, dtype=torch.long)
#     )

#     return padded.tolist(), mask.tolist()  # ✅ 둘 다 list로 반환

# def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
#     # 1. 모든 계산은 텐서 상태에서 빠르게 수행합니다.
#     result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64)
    
#     if return_mask:
#         # 패딩은 0, 실제 데이터는 1을 갖는 마스크 텐서 생성
#         mask = torch.zeros([len(examples), max_length], dtype=torch.int64)

#     # 2. 파이썬 for문이 아닌, 텐서에 직접 값을 채워넣습니다.
#     # 이 과정은 리스트를 다룰 때보다 훨씬 빠릅니다.
#     for i, example in enumerate(examples):
#         curr_len = min(len(example), max_length)
#         result[i, :curr_len] = torch.tensor(example[:curr_len], dtype=torch.int64)
#         if return_mask:
#             mask[i, :curr_len] = 1
            
#     # 3. 최종적으로 반환하기 직전에만 .tolist()를 호출하여
#     # 원래 함수와 동일한 형식(파이썬 리스트)으로 맞춰줍니다.
#     if return_mask:
#         return result.tolist(), mask.tolist()
        
#     return result.tolist()
