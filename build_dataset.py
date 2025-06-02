#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# build_dataset.py
# Convert valid_reports.json into four datasets: pftg, pfus, pfs, xray
# - Each image corresponds to one sample
# - Images are copied to images/<type>/
# - Generate metadata/<type>_samples.csv for each category
#   • pftg_samples.csv   → only contains Detrusor_instability
#   • pfus_samples.csv   → contains Flow_pattern, EMG_ES_relaxation
#   • xray_samples.csv   → contains the other 7 X-ray labels
#   • pfs_samples.csv    → image path + csv_path (no labels)
# - Also output label_stats.json (aggregate statistics for all 10 labels)

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# Define label groups for each dataset type
LABEL_GROUPS = {
    "pftg": ["Detrusor_instability"],
    "pfus": ["Flow_pattern", "EMG_ES_relaxation"],
    "xray": [
        "Trabeculation", "Diverticulum", "Cystocele", "VUR",
        "Bladder_neck_relaxation", "External_sphincter_relaxation", "Pelvic_floor_relaxation",
    ],
    # pfs has no labels
}
ALL_LABELS = [label for labels in LABEL_GROUPS.values() for label in labels]

# Function to parse the output.txt file and return a label dictionary
def parse_output_txt(txt_path: Path):
    # Initialize result with None for each label
    result = {label: None for label in ALL_LABELS}
    if not txt_path.is_file():
        return result
    # Read each line and split on ':' to extract key and value
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if key in result:
            # Treat 'na', 'n/a', and empty strings as missing
            result[key] = None if val.lower() in ("na", "n/a", "") else val
    return result

# Function to copy all images from src_dir to dst_dir with a unique ID prefix
def copy_images(src_dir: Path, dst_dir: Path, uid: str):
    # Create destination directory if it does not exist
    dst_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = []
    # Enumerate and copy each image file
    for idx, img_path in enumerate(sorted(src_dir.glob("*.*")), start=1):
        ext = img_path.suffix.lower()
        target = dst_dir / f"{uid}_{idx:02d}{ext}"
        shutil.copy2(img_path, target)
        # Compute relative path to processed_dataset
        rel = os.path.relpath(target, dst_dir.parent.parent)
        relative_paths.append(rel)
    return relative_paths

# Core function to build datasets given parameters
def build_dataset(valid_json_path: str = "../../valid_reports.json", processed_dir_path: str = "../../processed_dataset", dry_run: bool = False):
    # Load valid report records
    valid_reports = json.loads(Path(valid_json_path).read_text(encoding="utf-8"))
    proc_root = Path(processed_dir_path).resolve()
    img_root = proc_root / "images"
    meta_root = proc_root / "metadata"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    # Prepare CSV writers for pftg, pfus, xray
    writers = {}
    files = {}
    for dataset_type, labels in LABEL_GROUPS.items():
        csv_path = meta_root / f"{dataset_type}_samples.csv"
        if dry_run:
            csv_file = open(os.devnull, "w", newline="", encoding="utf-8")
        else:
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = ["sample_id", "patient_id", "report_id", "img_path"] + labels
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writers[dataset_type] = writer
        files[dataset_type] = csv_file

    # Prepare CSV writer for pfs (no labels)
    pfs_csv_path = meta_root / "pfs_samples.csv"
    if dry_run:
        pfs_file = open(os.devnull, "w", newline="", encoding="utf-8")
    else:
        pfs_file = pfs_csv_path.open("w", newline="", encoding="utf-8")
    pfs_writer = csv.DictWriter(pfs_file, fieldnames=[
        "sample_id", "patient_id", "report_id", "img_path", "csv_path"
    ])
    pfs_writer.writeheader()

    # Initialize statistics for all labels
    stats = defaultdict(lambda: {"positive": 0, "negative": 0, "missing": 0})

    sample_id = 1
    # Iterate over each valid report
    for record in valid_reports:
        patient_id = record["patient_id"]
        report_id = record["report_id"]
        uid_prefix = f"{patient_id}_{report_id}"
        report_dir = Path(record["report_path"]) / "output"
        labels = parse_output_txt(report_dir / "result" / "output.txt")

        # Update statistics for each label
        for label_name, value in labels.items():
            if value is None:
                stats[label_name]["missing"] += 1
            elif str(value).lower() in (
                    "1", "yes", "true",
                    "obstruction", "intermittent",
                    "interrupted", "staccato",
                    "supervoider", "bell",
                    "acceptable", "impaired"
            ):
                stats[label_name]["positive"] += 1
            else:
                stats[label_name]["negative"] += 1

        # Process pftg, pfus, xray images
        for dataset_type in ("pftg", "pfus", "xray"):
            src_images = report_dir / dataset_type
            dst_images = img_root / dataset_type
            if src_images.exists():
                for img_rel in copy_images(src_images, dst_images, uid_prefix):
                    row = {
                        "sample_id": f"{sample_id:08d}",
                        "patient_id": patient_id,
                        "report_id": report_id,
                        "img_path": img_rel,
                        **{label: labels.get(label) for label in LABEL_GROUPS[dataset_type]}
                    }
                    writers[dataset_type].writerow(row)
                    sample_id += 1

        # Process pfs images and corresponding CSV
        pfs_images_dir = report_dir / "pfs"
        csv_src = next((pfs_images_dir).glob("*.csv"), None)
        if pfs_images_dir.exists():
            for img_rel in copy_images(pfs_images_dir, img_root / "pfs", uid_prefix):
                csv_rel = os.path.relpath(csv_src, proc_root) if csv_src else ""
                pfs_writer.writerow({
                    "sample_id": f"{sample_id:08d}",
                    "patient_id": patient_id,
                    "report_id": report_id,
                    "img_path": img_rel,
                    "csv_path": csv_rel
                })
                sample_id += 1

    # Close all open files and write label statistics (skip file writes in dry-run)
    for f in files.values():
        f.close()
    pfs_file.close()
    if not dry_run:
        (meta_root / "label_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False)
        )

    print(f"Dataset built at {proc_root}, total samples: {sample_id-1}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_json",
        default="../../valid_reports.json",
        help="Path to valid_reports.json generated by verify_reports"
    )
    parser.add_argument(
        "--processed_dir",
        default="../../processed_dataset",
        help="Root directory for output datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without writing files"
    )
    args = parser.parse_args()
    # Call the core function with parsed arguments
    build_dataset(args.valid_json, args.processed_dir, args.dry_run)

if __name__ == "__main__":
    main()
