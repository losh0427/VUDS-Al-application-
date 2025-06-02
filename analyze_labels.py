#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script analyzes label distributions and sample counts in the processed_dataset
# It generates the following statistics and plots:
# 1. Distribution bar chart for each label
# 2. Comparison of valid vs invalid report counts
# 3. Comparison of sample counts across pftg, pfus, xray, and pfs types
# Input:
#   --processed_dir: directory where processed_dataset resides (default: ../../processed_dataset)
# Output:
#   analysis/data_counts.json
#   analysis/label_counts.json
#   analysis/plots/ containing:
#     <label>_dist.png
#     valid_invalid_counts.png
#     data_counts.png
# Summary statistics are also printed to the terminal.

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Map each group to its associated labels
label_cols_map = {
    "pftg": ["Detrusor_instability"],
    "pfus": ["Flow_pattern", "EMG_ES_relaxation"],
    "xray": [
        "Trabeculation", "Diverticulum", "Cystocele", "VUR",
        "Bladder_neck_relaxation", "External_sphincter_relaxation",
        "Pelvic_floor_relaxation"
    ]
}
# Invert mapping to get a lookup from label to group
label_to_group = {lbl: grp for grp, cols in label_cols_map.items() for lbl in cols}


def count_labels(samples_csv: Path, label_cols):
    """
    Count occurrences of each label value in the CSV file.
    Returns a dict mapping each label column to a defaultdict of value counts.
    """
    counts = {col: defaultdict(int) for col in label_cols}
    with samples_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in label_cols:
                val = row.get(col)
                if val in ("", "None", "null", None):
                    continue
                counts[col][str(val)] += 1
    return counts


def merge_dict(a, b):
    """
    Merge counts from dictionary b into dictionary a.
    Both a and b map label columns to defaultdicts of counts.
    """
    for k, v in b.items():
        for sub, n in v.items():
            a[k][sub] += n


def annotate_bars(ax, bars):
    """
    Add text labels above each bar displaying its height.
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha='center',
            va='bottom'
        )


def plot_label_distributions(label_counts, data_counts, plots_dir: Path):
    """
    Generate and save distribution bar chart for each label.
    """
    for lbl, subs in label_counts.items():
        grp = label_to_group.get(lbl)
        total = data_counts.get(grp, 0)
        present = sum(subs.values())
        missing = total - present
        if missing > 0:
            subs["NA"] = missing

        categories = list(subs.keys())
        values = [subs[c] for c in categories]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(categories, values)
        annotate_bars(ax, bars)
        ax.set_title(f"{lbl} distribution")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_xticklabels(categories, rotation=45, ha="right")
        plt.tight_layout()
        safe_lbl = lbl.replace(" ", "_")
        fig.savefig(plots_dir / f"{safe_lbl}_dist.png")
        plt.close(fig)


def plot_valid_invalid(proc_dir: Path, plots_dir: Path):
    """
    Generate and save bar chart comparing valid vs invalid report counts.
    """
    valid_path = proc_dir.parent / "valid_reports.json"
    invalid_path = proc_dir.parent / "invalid_reports.json"
    valid = json.loads(valid_path.read_text(encoding="utf-8")) if valid_path.exists() else []
    invalid = json.loads(invalid_path.read_text(encoding="utf-8")) if invalid_path.exists() else []

    counts = {"valid": len(valid), "invalid": len(invalid)}
    categories = list(counts.keys())
    values = [counts[k] for k in categories]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(categories, values, color=["green", "red"])
    annotate_bars(ax, bars)
    ax.set_title("Valid vs Invalid Reports")
    ax.set_ylabel("Number of Reports")
    plt.tight_layout()
    fig.savefig(plots_dir / "valid_invalid_counts.png")
    plt.close(fig)


def plot_data_counts(data_counts, plots_dir: Path):
    """
    Generate and save bar chart of sample counts by type.
    """
    categories = list(data_counts.keys())
    values = [data_counts[k] for k in categories]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(categories, values)
    annotate_bars(ax, bars)
    ax.set_title("Data Counts by Type")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    fig.savefig(plots_dir / "data_counts.png")
    plt.close(fig)


def analyze_processed_data(processed_dir_path: str):
    """
    Core function that performs analysis on a processed_dataset directory.
    Creates:
      - analysis/data_counts.json
      - analysis/label_counts.json
      - analysis/plots/
    Prints summary statistics to the terminal.
    """
    proc = Path(processed_dir_path).resolve()
    analysis_dir = proc / "analysis"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Read sample CSV files and count samples per type
    csv_files = {
        "pftg": proc / "metadata" / "pftg_samples.csv",
        "pfus": proc / "metadata" / "pfus_samples.csv",
        "xray": proc / "metadata" / "xray_samples.csv",
        "pfs": proc / "metadata" / "pfs_samples.csv",
    }
    data_counts = {}
    for key, path in csv_files.items():
        if path.exists():
            with path.open(encoding="utf-8") as f:
                # Subtract 1 for header row
                data_counts[key] = sum(1 for _ in f) - 1
        else:
            data_counts[key] = 0

    # Calculate label distributions
    label_counts = defaultdict(lambda: defaultdict(int))
    for grp, cols in label_cols_map.items():
        path = csv_files.get(grp)
        if path and path.exists():
            counts = count_labels(path, cols)
            merge_dict(label_counts, counts)

    # Write JSON outputs
    (analysis_dir / "data_counts.json").write_text(
        json.dumps(data_counts, indent=2, ensure_ascii=False)
    )
    label_counts_json = {k: dict(v) for k, v in label_counts.items()}
    (analysis_dir / "label_counts.json").write_text(
        json.dumps(label_counts_json, indent=2, ensure_ascii=False)
    )

    # Print summary to terminal
    print("=== Data sample counts ===")
    for k, n in data_counts.items():
        print(f"{k:<5} : {n}")
    print("\n=== Label distribution ===")
    for lbl, subs in label_counts.items():
        subs_str = ", ".join(f"{cat}:{cnt}" for cat, cnt in subs.items())
        print(f"{lbl:<30} -> {subs_str}")

    # Generate plots
    plot_valid_invalid(proc, plots_dir)
    plot_data_counts(data_counts, plots_dir)
    plot_label_distributions(label_counts, data_counts, plots_dir)

    print(f"\nPlots saved in {plots_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir",
        default="../../processed_dataset",
        help="Directory containing processed_dataset outputs"
    )
    args = parser.parse_args()

    analyze_processed_data(args.processed_dir)


if __name__ == "__main__":
    main()
