import json
from .calc_score import get_bert_score
from .bm_retriever import SimpleBM25Retriever
import torch


def process_jsonl_file(file_path):
    """
    Reads and processes a JSONL (JSON Lines) file line by line.

    Each line in a JSONL file is a separate, valid JSON object. This function
    opens the specified file, reads it line by line, parses each line
    from a JSON string into a Python dictionary, and then prints it.

    Args:
        file_path (str): The path to the .jsonl file.

    Yields:
        dict: A dictionary representing the JSON object from a line in the file.
              Returns nothing if a line is invalid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # Remove leading/trailing whitespace
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    # Parse the JSON string from the current line
                    json_object = json.loads(line)
                    yield json_object
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line {line_number} as JSON. Skipping.")
                    print(f"--> Content: {line}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    file_name = '/home/seungwoochoi/data/x_bridging/DiffuSeq/generation_outputs/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20251008-22:34:08/ema_0.9999_050000.pt.samples/seed123_step0.json'

    print(f"Attempting to read and process '{file_name}'...")
    print("-" * 30)

    # The function returns a generator, so we iterate through it
    
    output_lst = []
    ref_lst = []
    count = 0
    for data_record in process_jsonl_file(file_name):
        # You can now work with each JSON object as a Python dictionar
        output_lst.append(data_record['recover'])
        ref_lst.append(data_record['reference'])
    
    print(len(output_lst), flush=True)
    print(len(ref_lst), flush=True)

    _, _, f1_score = get_bert_score(output_lst, ref_lst)

    print(f1_score)

    topk = 100
    topk_indices = torch.topk(f1_score, topk)[1]
    print(topk_indices)

    topk_references = [ref_lst[i] for i in topk_indices]

    with open("startpool.txt", "w", encoding="utf-8") as f:
        for line in topk_references:
            f.write(line + "\n")

    

    
