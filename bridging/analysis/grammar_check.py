
import os
import sys
import json
import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path to allow importing modules if needed, purely for safety
sys.path.append("../")

# Configuration
INPUT_DIR = "/home/seungwoochoi/data/x_bridging/bridging/inference/inference_result"
OUTPUT_DIR = "/home/seungwoochoi/data/x_bridging/bridging/analysis/grammar_results"
GPT_MODEL_NAME = "gpt-4.1-nano-2025-04-14"

CACHE_FILE = os.path.join(OUTPUT_DIR, "grammar_check_cache.json")
SENTENCE_CACHE = {}

def load_cache():
    global SENTENCE_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                SENTENCE_CACHE = json.load(f)
            print(f"Loaded cache with {len(SENTENCE_CACHE)} entries.")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            SENTENCE_CACHE = {}

def save_cache():
    try:
        # Create output dir if it doesn't exist (though main creates it? No, process_file does. Better safe.)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(SENTENCE_CACHE, f, ensure_ascii=False, indent=4)
        print(f"Saved cache with {len(SENTENCE_CACHE)} entries.")
    except Exception as e:
        print(f"Failed to save cache: {e}")

def check_sentence_grammar(sentence, client):
    """
    Checks the grammar of a single sentence.
    Returns 1 if there is a grammatical error, 0 otherwise.
    Uses caching to avoid redundant calls.
    """
    if not sentence or not sentence.strip():
        return 0

    # Check cache
    if sentence in SENTENCE_CACHE:
        return SENTENCE_CACHE[sentence]

    prompt = (
        "Check if the following sentence has any grammatical errors. "
        "Return specifically '1' if there is an error, and '0' if there is no error. "
        "Do not provide any other text or explanation.\n\n"
        f"Sentence: {sentence}"
    )

    result = 1 # Default to error on failure
    try:
        response = client.responses.create(
            model=GPT_MODEL_NAME,
            input=prompt
        )
        
        content = response.output[0].content[0].text.strip()
        
        if "1" in content:
            result = 1
        elif "0" in content:
            result = 0
        else:
            try:
                val = int(content)
                result = 1 if val != 0 else 0
            except:
                print(f"Warning: unexpected response for '{sentence[:20]}...': {content}")
                result = 1 

    except Exception as e:
        print(f"Error checking grammar for sentence '{sentence[:20]}...': {e}")
        result = 1
        
    # Update cache
    SENTENCE_CACHE[sentence] = result
    return result

def check_grammar(sentences, client):
    """
    Checks the grammar of a list of sentences using LLM.
    Returns a list of 0s and 1s (1 for error, 0 for no error).
    """
    results = []
    for sentence in sentences:
        results.append(check_sentence_grammar(sentence, client))
    return results

def process_file(filename, client):
    filepath = os.path.join(INPUT_DIR, filename)
    print(f"Processing {filepath}...", flush=True)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return

    all_results = []
    
    for chain in tqdm.tqdm(data, desc=f"Processing chains in {filename}"):
        chain_results = check_grammar(chain, client)
        all_results.append(chain_results)
        
    # Save results
    output_filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_filepath}")
        
        # Save cache after every file processed to prevent data loss
        save_cache()
        
    except Exception as e:
        print(f"Failed to save results for {filename}: {e}")

def main():
    load_dotenv()
    client = OpenAI()
    
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    # Initialize cache
    load_cache()

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    print(f"Found {len(files)} JSON files to process.")
    
    for filename in files:
        process_file(filename, client)

if __name__ == "__main__":
    main()
