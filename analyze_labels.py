#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 processed_dataset 內四份 samples.csv 的 label 分佈與樣本數量，
並產生以下統計圖，所有長條上方都會顯示其數值：
  1. 每個 label 的分佈長條圖
  2. valid vs invalid 報告數量比較
  3. pftg/pfus/xray/pfs 四類 sample 數比較

輸入：
  --processed_dir  預設 ../../processed_dataset
輸出：
  analysis/data_counts.json
  analysis/label_counts.json
  analysis/plots/
    ├─ <label>_dist.png
    ├─ valid_invalid_counts.png
    └─ data_counts.png
於終端列印統計總覽。
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ---- label→group 對照 ------------------------------------------------------
label_cols_map = {
    "pftg": ["Detrusor_instability"],
    "pfus": ["Flow_pattern", "EMG_ES_relaxation"],
    "xray": [
        "Trabeculation", "Diverticulum", "Cystocele", "VUR",
        "Bladder_neck_relaxation", "External_sphincter_relaxation",
        "Pelvic_floor_relaxation"
    ]
}
# 反轉成：label → group
label_to_group = {
    lbl: grp for grp, cols in label_cols_map.items() for lbl in cols
}


def count_labels(samples_csv: Path, label_cols):
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
    for k, v in b.items():
        for sub, n in v.items():
            a[k][sub] += n


def annotate_bars(ax, bars):
    """在每個 bar 上方標註高度"""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha='center',
            va='bottom'
        )


def plot_label_distributions(label_counts, data_counts, plots_dir):
    """為每個 label 畫分佈長條圖並存檔"""
    for lbl, subs in label_counts.items():
        grp = label_to_group.get(lbl)
        total = data_counts.get(grp, 0)
        present = sum(subs.values())
        missing = total - present
        if missing > 0:
            subs["NA"] = missing

        cats = list(subs.keys())
        vals = [subs[c] for c in cats]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(cats, vals)
        annotate_bars(ax, bars)
        ax.set_title(f"{lbl} distribution")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_xticklabels(cats, rotation=45, ha="right")
        plt.tight_layout()
        safe_lbl = lbl.replace(" ", "_")
        fig.savefig(plots_dir / f"{safe_lbl}_dist.png")
        plt.close(fig)


def plot_valid_invalid(proc_dir, plots_dir):
    """畫 valid vs invalid 報告數量比較"""
    valid_path = proc_dir.parent / "valid_reports.json"
    invalid_path = proc_dir.parent / "invalid_reports.json"
    valid = json.loads(valid_path.read_text(encoding="utf-8")) if valid_path.exists() else []
    invalid = json.loads(invalid_path.read_text(encoding="utf-8")) if invalid_path.exists() else []

    counts = {"valid": len(valid), "invalid": len(invalid)}
    cats = list(counts.keys())
    vals = [counts[k] for k in cats]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(cats, vals, color=["green", "red"])
    annotate_bars(ax, bars)
    ax.set_title("Valid vs Invalid Reports")
    ax.set_ylabel("Number of Reports")
    plt.tight_layout()
    fig.savefig(plots_dir / "valid_invalid_counts.png")
    plt.close(fig)


def plot_data_counts(data_counts, plots_dir):
    """畫四類 sample 數比較圖"""
    cats = list(data_counts.keys())
    vals = [data_counts[k] for k in cats]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(cats, vals)
    annotate_bars(ax, bars)
    ax.set_title("Data Counts by Type")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    fig.savefig(plots_dir / "data_counts.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir", default="../../processed_dataset",
        help="build_dataset 輸出資料夾"
    )
    args = parser.parse_args()

    proc = Path(args.processed_dir).resolve()
    analysis_dir = proc / "analysis"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 讀取四類 samples.csv 並計算 data_counts
    csv_files = {
        "pftg": proc / "metadata" / "pftg_samples.csv",
        "pfus": proc / "metadata" / "pfus_samples.csv",
        "xray": proc / "metadata" / "xray_samples.csv",
        "pfs":  proc / "metadata" / "pfs_samples.csv",
    }
    data_counts = {}
    for key, path in csv_files.items():
        if path.exists():
            with path.open(encoding="utf-8") as f:
                data_counts[key] = sum(1 for _ in f) - 1
        else:
            data_counts[key] = 0

    # 計算 label 分佈
    label_counts = defaultdict(lambda: defaultdict(int))
    for grp, cols in label_cols_map.items():
        path = csv_files.get(grp)
        if path and path.exists():
            c = count_labels(path, cols)
            merge_dict(label_counts, c)

    # 輸出 JSON
    (analysis_dir / "data_counts.json").write_text(
        json.dumps(data_counts, indent=2, ensure_ascii=False)
    )
    label_counts_json = {k: dict(v) for k, v in label_counts.items()}
    (analysis_dir / "label_counts.json").write_text(
        json.dumps(label_counts_json, indent=2, ensure_ascii=False)
    )

    # 終端總覽
    print("=== Data sample counts ===")
    for k, n in data_counts.items():
        print(f"{k:<5} : {n}")
    print("\n=== Label distribution ===")
    for lbl, subs in label_counts.items():
        subs_str = ", ".join(f"{cat}:{cnt}" for cat, cnt in subs.items())
        print(f"{lbl:<30} → {subs_str}")

    # 繪圖
    plot_valid_invalid(proc, plots_dir)
    plot_data_counts(data_counts, plots_dir)
    plot_label_distributions(label_counts, data_counts, plots_dir)

    print(f"\n📊 Plots saved in {plots_dir}")


if __name__ == "__main__":
    main()

