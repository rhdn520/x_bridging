import argparse
import json
import numpy as np
import os
import sys
from sacrebleu import CHRF

# Add current directory to path to allow imports from analysis.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "../utils"))

from analysis import SentenceSimilarity
from levenshtein_distance import Levenshtein
from gpt_models import GptSentenceRefiner

def main():
    parser = argparse.ArgumentParser(description="Measure interpolation quality.")
    parser.add_argument("--input_file", type=str, help="Path to the inference result file (JSON).")
    parser.add_argument("--n_sents_in_path", type=int, default=21, help="Number of sentences in each interpolation path.")
    args = parser.parse_args()

    # Initialize metrics
    # Note: SentenceSimilarity loads 'all-MiniLM-L6-v2' by default which matches analysis.py
    sbert_sim = SentenceSimilarity() 
    chrf = CHRF()
    refiner = GptSentenceRefiner()

    # Load data
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Prepare results container
    # We will store the full lists (progress) and their standard deviations
    results = {
        'sbert_from_start': [],
        'sbert_from_end': [],
        'lev_from_start': [],
        'lev_from_end': [],
        'chrf_from_start': [],
        'chrf_from_end': [],
        
        'std_sbert_from_start': [],
        'std_sbert_from_end': [],
        'std_lev_from_start': [],
        'std_lev_from_end': [],
        'std_chrf_from_start': [],
        'std_chrf_from_end': [],

        'sbert_directness': [],
        'lev_directness': [],
        'chrf_directness': [],
    
        'refined_sentences': [],
        'lev_refined': [],
        'std_lev_refined': []
    }

    # Process each interpolation path
    for intp_path in data:
        # Check for empty path to avoid errors, though unlikely in this context
        if not intp_path:
            continue
        
        if len(intp_path) != args.n_sents_in_path:
            print(f"Warning: Expected {args.n_sents_in_path} sentences but got {len(intp_path)}. Skipping this path.")
            continue

        start_sent = intp_path[0]
        end_sent = intp_path[-1]

        # Temp lists for this path
        path_sbert_start = []
        path_sbert_end = []
        path_lev_start = []
        path_lev_end = []
        path_chrf_start = []
        path_chrf_end = []

        for sent in intp_path:
            # SBERT
            # analysis.py uses similarity(sent, target)
            path_sbert_start.append(sbert_sim.similarity(sent, start_sent))
            path_sbert_end.append(sbert_sim.similarity(sent, end_sent))

            # Levenshtein
            # analysis.py uses Levenshtein(sent, target).distance()
            path_lev_start.append(Levenshtein(sent, start_sent).distance())
            path_lev_end.append(Levenshtein(sent, end_sent).distance())

            # ChrF
            # analysis.py uses chrf.sentence_score(sent, [target]).score
            path_chrf_start.append(chrf.sentence_score(sent, [start_sent]).score)
            path_chrf_end.append(chrf.sentence_score(sent, [end_sent]).score)

        # Refine and measure quality with GPT
        print(f"Refining batch of {len(intp_path)} sentences...", flush=True)
        refined_sents = refiner.refine_batch(intp_path)
        path_refined_sents = refined_sents
        path_lev_refined = []
        
        for sent, refined_sent in zip(intp_path, refined_sents):
            path_lev_refined.append(Levenshtein(sent, refined_sent).distance())

        # Calculate directness for each metric
        # Directness = (distance increase from start) / (distance decrease to end)
        path_sbert_directness = []
        path_lev_directness = []
        path_chrf_directness = []
        
        for i in range(1, len(intp_path)):
            # Levenshtein directness: move_away / move_closer
            lev_move_away = path_lev_start[i] - path_lev_start[i-1]
            lev_move_closer = path_lev_end[i-1] - path_lev_end[i]
            if lev_move_closer != 0:
                path_lev_directness.append(lev_move_away / lev_move_closer)
            else:
                path_lev_directness.append(0.0)
            
            # SBERT directness: (start_similarity decrease) / (end_similarity increase)
            sbert_move_away = path_sbert_start[i-1] - path_sbert_start[i]
            sbert_move_closer = path_sbert_end[i] - path_sbert_end[i-1]
            if sbert_move_closer != 0:
                path_sbert_directness.append(sbert_move_away / sbert_move_closer)
            else:
                path_sbert_directness.append(0.0)
            
            # ChrF directness: (start_score decrease) / (end_score increase)
            chrf_move_away = path_chrf_start[i-1] - path_chrf_start[i]
            chrf_move_closer = path_chrf_end[i] - path_chrf_end[i-1]
            if chrf_move_closer != 0:
                path_chrf_directness.append(chrf_move_away / chrf_move_closer)
            else:
                path_chrf_directness.append(0.0)

        # Append progress lists
        results['sbert_from_start'].append(path_sbert_start)
        results['sbert_from_end'].append(path_sbert_end)
        results['lev_from_start'].append(path_lev_start)
        results['lev_from_end'].append(path_lev_end)
        results['chrf_from_start'].append(path_chrf_start)
        results['chrf_from_end'].append(path_chrf_end)
        results['sbert_directness'].append(path_sbert_directness)
        results['lev_directness'].append(path_lev_directness)
        results['chrf_directness'].append(path_chrf_directness)
        results['refined_sentences'].append(path_refined_sents)
        results['lev_refined'].append(path_lev_refined)

        # Calculate and append standard deviations
        results['std_sbert_from_start'].append(float(np.std(path_sbert_start)))
        results['std_sbert_from_end'].append(float(np.std(path_sbert_end)))
        results['std_lev_from_start'].append(float(np.std(path_lev_start)))
        results['std_lev_from_end'].append(float(np.std(path_lev_end)))
        results['std_chrf_from_start'].append(float(np.std(path_chrf_start)))
        results['std_chrf_from_end'].append(float(np.std(path_chrf_end)))
        
    # Generate output filename
    base_name = os.path.basename(args.input_file)
    output_filename = base_name.replace(".json", "_analysis.json")
    output_path = os.path.join(os.path.dirname(args.input_file), output_filename)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=3)
    
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()
