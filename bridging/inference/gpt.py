
import os
import sys
sys.path.append("../")
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from train import StreamTinyStoriesDataset
from tqdm import tqdm

# --- Configuration ---
BERT_MODEL_NAME = "bert-base-uncased"
GPT_MODEL_NAME = "gpt-4.1-nano-2025-04-14"

DATA_SPLIT = "validation"
MAX_SAMPLES = 1000
MAX_SEQ_LEN = 128
SKIP_SAMPLES = 10000
BATCH_SIZE = 1

NUM_SENTENCES_TO_PROCESS = 20
OUTPUT_FILE = "/home/seungwoochoi/data/x_bridging/bridging/gpt_intps_new.json"


def load_data(tokenizer):
    """
    Loads the TinyStories dataset and extracts sentences.
    
    Args:
        tokenizer: The tokenizer to communicate with the dataset class.
        
    Returns:
        list: A list of decoded sentences.
    """
    print("Loading dataset...", flush=True)
    test_dataset = StreamTinyStoriesDataset(
        tokenizer,
        split=DATA_SPLIT,
        max_samples=MAX_SAMPLES,
        max_seq_len=MAX_SEQ_LEN,
        skip_samples=SKIP_SAMPLES
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    sent_list = []
    print("Extracting sentences...", flush=True)
    for batch_idx, batch in enumerate(test_loader):
        sent_list.extend(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
        
    return sent_list

def generate_sentence_pairs(sent_list):
    """
    Selects a random subset of sentences and generates unique pairs.
    
    Args:
        sent_list (list): List of all available sentences.
        
    Returns:
        list: A list of [sentence1, sentence2] pairs.
    """
    # shuffle sent_list
    random.seed(42)
    random.shuffle(sent_list)
    subset = sent_list[:NUM_SENTENCES_TO_PROCESS]
    print(f"Selected {len(subset)} sentences for pairing: {subset}")

    products = []
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            sent1 = subset[i]
            sent2 = subset[j]
            products.append([sent1, sent2])
            
    print(f"Generated {len(products)} pairs.")
    return products

def get_gpt_responses(products, client):
    """
    Queries the GPT model to interpolate between sentence pairs.
    
    Args:
        products (list): List of sentence pairs.
        client (OpenAI): The OpenAI client instance.
        
    Returns:
        list: List of lists, where each inner list contains the interpolation sequence.
    """
    responses = [] 
    print("Querying GPT model...", flush=True)
    
    for sent1, sent2 in tqdm(products): 
        prompt = (
            f"I will give you two sentences. Can you gradually change the first sentence "
            f"to make it exactly the same as the second sentence? Just give me ten sentences "
            f"and don’t provide additional comments. Don't add numbering in front of sentences. "
            f"The degree of change should be same for each sentence, and at least one word should be changed."
            f"Sentence1: {sent1}\nSentence2: {sent2}"
        )

        try:
            response = client.responses.create(
                model=GPT_MODEL_NAME,
                input=prompt
            )
            
            # Parse response
            res = [sent1]
            generated_text = response.output[0].content[0].text
            generated_text = generated_text.split('\n')
            generated_text = [line.strip() for line in generated_text]
            generated_text = [line.lower() for line in generated_text]
            res.extend(generated_text)
            res.append(sent2)
            if(len(res) != 12):
                print(f"Warning: Expected 12 sentences but got {len(res)}. Pair: ({sent1[:20]}..., {sent2[:20]}...)")
            responses.append(res)
            
        except Exception as e:
            print(f"Error processing pair ({sent1[:20]}..., {sent2[:20]}...): {e}")

    return responses

def save_results(responses, filepath):
    """
    Saves the responses to a JSON file.
    
    Args:
        responses (list): The data to save.
        filepath (str): The destination file path.
    """
    print(f"Saving results to {filepath}...", flush=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Correct arguments: object first, then file pointer
            json.dump(responses, f, ensure_ascii=False, indent=4)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Client and Tokenizer
    client = OpenAI()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Process
    all_sentences = load_data(tokenizer)
    sentence_pairs = generate_sentence_pairs(all_sentences)
    responses = get_gpt_responses(sentence_pairs, client)
    save_results(responses, OUTPUT_FILE)

if __name__ == "__main__":
    main()