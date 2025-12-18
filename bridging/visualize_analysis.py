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
        plt.hist(values, bins=50, range=(0,100), alpha=0.3, label=name.replace("_", " ").title(), histtype='stepfilled')

    plt.title("CHRF Score Between Intp Sents (Gpt to Diff)")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig('chrf.png')

def plot_delta_lines(delta_scores: Iterable[Iterable[float]]) -> None:
    """Plot each delta score sequence as a line."""
    sequences = [list(seq) for seq in delta_scores]
    if not sequences:
        return

    plt.figure(figsize=(12, 6))
    for idx, sequence in enumerate(sequences):
        # label = f"Sample {idx}" if idx < 10 else None  # avoid overcrowding legend
        plt.plot(list(range(len(sequence))), sequence, alpha=0.6)

    plt.title("ChrF Score (Compared to Last Sent)")
    plt.xlabel("Intp Index")
    plt.ylabel("ChrF")
    if len(sequences) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig('delta.png')


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
    delta_scores = data.get("delta_scores", [])

    if not chrf_scores:
        raise ValueError("analysis_result.json does not contain 'chrf_scores'.")
    if not delta_scores:
        raise ValueError("analysis_result.json does not contain 'delta_scores'.")

    plot_chrf_hist(chrf_scores)
    plot_delta_lines(delta_scores)
    # plt.show()


if __name__ == "__main__":
    main()
