#!/usr/bin/env python3
"""
Visualize interpolation quality analysis results.
Plots progress lines (distance sequences) and standard deviation distributions.
"""

from __future__ import annotations
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


def load_analysis(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_progress(
    data: List[List[float]], 
    metric_name: str, 
    direction: str, 
    output_dir: Path,
    base_filename: str
) -> None:
    """Plot the sequence of distances (progress) for each interpolation path."""
    if not data:
        print(f"No data for {metric_name} ({direction})")
        return

    plt.figure(figsize=(12, 10))
    
    # Plot each path
    # Use alpha to handle many overlapping lines
    for seq in data:
        plt.plot(range(len(seq)), seq, alpha=0.1, color='blue', linewidth=1)
        
    # Plot average trend
    max_len = max(len(s) for s in data)
    avg_seq = []
    for i in range(max_len):
        vals = [s[i] for s in data if i < len(s)]
        if vals:
            avg_seq.append(sum(vals) / len(vals))
            
    plt.plot(range(len(avg_seq)), avg_seq, color='red', linewidth=2, label='Average')

    title = f"{metric_name.upper()} Progress (from {direction})"
    if direction == '':
        title = f"{metric_name.upper()}"
    plt.title(title)
    plt.xlabel("Step Index")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = output_dir / f"{base_filename}_progress_{metric_name}_{direction}.png" if direction else output_dir / f"{base_filename}_progress_{metric_name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved progress plot to {out_path}")


def plot_std_distribution(
    data: List[float], 
    metric_name: str, 
    direction: str, 
    output_dir: Path,
    base_filename: str
) -> None:
    """Plot the distribution format standard deviations."""
    if not data:
        print(f"No std data for {metric_name} ({direction})")
        return

    plt.figure(figsize=(10, 6))
    
    plt.hist(data, bins=50, alpha=0.7, color='green', edgecolor='black')
    
    avg_std = sum(data) / len(data)
    plt.axvline(avg_std, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_std:.4f}')

    title = f"Distribution of STD: {metric_name.upper()} (from {direction})"
    plt.title(title)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = output_dir / f"{base_filename}_std_dist_{metric_name}_{direction}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved std distribution plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize interpolation analysis results."
    )
    parser.add_argument(
        "analysis_file",
        type=Path,
        help="Path to the analysis json file (e.g. *_analysis.json)",
    )
    args = parser.parse_args()

    data = load_analysis(args.analysis_file)
    output_dir = args.analysis_file.parent.joinpath("plots")
    print(f"Creating output directory at {output_dir}...")
    base_filename = args.analysis_file.stem
    print(f"Base filename for plots: {base_filename}")

    # Metrics to visualize
    metrics = ['sbert', 'lev', 'chrf']
    directions = ['start', 'end']

    for metric in metrics:
        for direction in directions:
            # Plot Progress
            key_progress = f"{metric}_from_{direction}"
            if key_progress in data:
                plot_progress(
                    data[key_progress], 
                    metric, 
                    direction, 
                    output_dir,
                    base_filename
                )

            # Plot STD Distribution
            key_std = f"std_{metric}_from_{direction}"
            if key_std in data:
                plot_std_distribution(
                    data[key_std], 
                    metric, 
                    direction, 
                    output_dir,
                    base_filename
                )

    # Visualize directness metrics (no direction, just metric-based)
    directness_metrics = ['sbert_directness', 'lev_directness', 'chrf_directness', 'lev_refined']
    
    for metric in directness_metrics:
        # Plot Progress
        key_progress = metric
        if key_progress in data:
            plot_progress(
                data[key_progress], 
                metric, 
                '', 
                output_dir,
                base_filename
            )
            
if __name__ == "__main__":
    main()
