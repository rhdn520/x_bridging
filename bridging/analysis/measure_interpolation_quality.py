import argparse
import json
import numpy as np
import os
import sys
from sacrebleu import CHRF

# Add current directory to path to allow imports from analysis.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from analysis import SentenceSimilarity
from levenshtein_distance import Levenshtein

def main():
    parser = argparse.ArgumentParser(description="Measure interpolation quality.")
    parser.add_argument("--input_file", type=str, help="Path to the inference result file (JSON).")
    args = parser.parse_args()

    # Initialize metrics
    # Note: SentenceSimilarity loads 'all-MiniLM-L6-v2' by default which matches analysis.py
    sbert_sim = SentenceSimilarity() 
    chrf = CHRF()

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
        'std_chrf_from_end': []
    }

    # Process each interpolation path
    for intp_path in data:
        # Check for empty path to avoid errors, though unlikely in this context
        if not intp_path:
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

        # Append progress lists
        results['sbert_from_start'].append(path_sbert_start)
        results['sbert_from_end'].append(path_sbert_end)
        results['lev_from_start'].append(path_lev_start)
        results['lev_from_end'].append(path_lev_end)
        results['chrf_from_start'].append(path_chrf_start)
        results['chrf_from_end'].append(path_chrf_end)

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
