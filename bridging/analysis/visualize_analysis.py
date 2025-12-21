#!/usr/bin/env python3
"""
Visualize `analysis_result.json` by plotting histogram of CHRF scores and
line charts for delta scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def load_analysis(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_chrf_hist(chrf_scores: Iterable[Dict[str, float]]) -> None:
    """Plot histogram of each CHRF score variant on the same axes."""
    score_names = ["score_1", "score_2", "score_3"]
    # score_names = ["First Intp","Second Intp", "Third Intp"]
    # score_names = ['corpus_score']
    series: Dict[str, List[float]] = {name: [] for name in score_names}
    for entry in chrf_scores:
        for name in score_names:
            value = entry.get(name)
            if isinstance(value, (int, float)):
                series[name].append(float(value))

    plt.figure(figsize=(10, 6))
    for name, values in series.items():
        if not values:
            continue
        print(f"Average score of {name}: {sum(values) / len(values)}")
        plt.hist(values, bins=50, range=(0,100), alpha=0.3, label=name.replace("_", " ").title(), histtype='stepfilled')

    plt.title("CHRF Score Between Intp Sents")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig('chrf.png')

def plot_delta_lines(delta_scores: Iterable[Iterable[float]], file_suffix: str) -> None:
    """Plot each delta score sequence as a line."""
    sequences = [list(seq) for seq in delta_scores]
    if not sequences:
        return

    plt.figure(figsize=(12, 6))

    score_0 = []
    score_1 = []
    score_2 = []
    score_3 = []
    score_4 = []

    for idx, sequence in enumerate(sequences):
        # label = f"Sample {idx}" if idx < 10 else None  # avoid overcrowding legend
        plt.plot(list(range(len(sequence))), sequence, alpha=0.6)
        score_0.append(sequence[0])
        score_1.append(sequence[1])
        score_2.append(sequence[2])
        score_3.append(sequence[3])
        score_4.append(sequence[4])

    print(f"Average score of 0: {sum(score_0) / len(score_0)}")
    print(f"Average score of 1: {sum(score_1) / len(score_1)}")
    print(f"Average score of 2: {sum(score_2) / len(score_2)}")
    print(f"Average score of 3: {sum(score_3) / len(score_3)}")
    print(f"Average score of 4: {sum(score_4) / len(score_4)}")
    
    plt.title("ChrF Score (Compared to Last Sent)")
    plt.xlabel("Intp Index")
    plt.ylabel("ChrF")
    if len(sequences) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'delta_{file_suffix}.png')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CHRF and delta score distributions."
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=Path("analysis_result.json"),
        help="Path to analysis_result.json (default: %(default)s)",
    )
    args = parser.parse_args()

    data = load_analysis(args.analysis_file)
    chrf_scores = data.get("chrf_scores", [])
    delta_scores_diff = data.get("delta_scores_diff", [])
    delta_scores_gpt = data.get("delta_scores_gpt", [])

    if not chrf_scores:
        raise ValueError("analysis_result.json does not contain 'chrf_scores'.")
    if not delta_scores_diff:
        raise ValueError("analysis_result.json does not contain 'delta_scores_diff'.")
    if not delta_scores_gpt:
        raise ValueError("analysis_result.json does not contain 'delta_scores_gpt'.")

    plot_chrf_hist(chrf_scores)
    plot_delta_lines(delta_scores_diff, file_suffix="diff")
    plot_delta_lines(delta_scores_gpt, file_suffix="gpt")
    # plt.show()


if __name__ == "__main__":
    main()
