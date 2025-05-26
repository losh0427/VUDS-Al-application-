#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 valid_reports.json 轉為四套資料集：pftg / pfus / pfs / xray
- 每張圖片 = 一筆 sample
- 影像複製到 images/<type>/
- 各類型分別產生 metadata/<type>_samples.csv
  • pftg_samples.csv   → 只含 Detrusor_instability
  • pfus_samples.csv   → 含 Flow_pattern, EMG_ES_relaxation
  • xray_samples.csv   → 其餘 7 個 X-ray label
  • pfs_samples.csv    → 圖片路徑 + csv_path（無 label）
- 仍輸出 label_stats.json（10 個 label 總計統計）

用法：
  python build_dataset.py \
    --valid_json ../../valid_reports.json \
    --processed_dir ../../processed_dataset
"""
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

LABEL_GROUPS = {
    "pftg": ["Detrusor_instability"],
    "pfus": ["Flow_pattern", "EMG_ES_relaxation"],
    "xray": [
        "Trabeculation", "Diverticulum", "Cystocele", "VUR",
        "Bladder_neck_relaxation", "External_sphincter_relaxation", "Pelvic_floor_relaxation",
    ],
    # pfs 無 label
}
ALL_LABELS = [l for labels in LABEL_GROUPS.values() for l in labels]

def parse_output_txt(txt_path: Path):
    """讀取 output.txt → dict(label: value or None)"""
    res = {k: None for k in ALL_LABELS}
    if not txt_path.is_file():
        return res
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if key in res:
            res[key] = None if val.lower() in ("na", "n/a", "") else val
    return res

def copy_images(src_dir: Path, dst_dir: Path, uid: str):
    """複製資料夾內所有影像，回傳相對路徑 list"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, img_path in enumerate(sorted(src_dir.glob("*.*")), 1):
        ext = img_path.suffix.lower()
        target = dst_dir / f"{uid}_{idx:02d}{ext}"
        shutil.copy2(img_path, target)
        # 改用 os.path.relpath 計算相對於 processed_dataset 的路徑
        rel = os.path.relpath(target, dst_dir.parent.parent)
        paths.append(rel)
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_json", default="../../valid_reports.json",
                        help="verify_reports 產生的 valid_reports.json")
    parser.add_argument("--processed_dir", default="../../processed_dataset",
                        help="輸出資料集根目錄")
    parser.add_argument("--dry-run", action="store_true",
                        help="只顯示不寫檔")
    args = parser.parse_args()

    valid_reports = json.loads(Path(args.valid_json).read_text(encoding="utf-8"))
    proc_root = Path(args.processed_dir).resolve()
    img_root = proc_root / "images"
    meta_root = proc_root / "metadata"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    # 建立各類別 CSV Writer
    writers = {}
    files = {}
    for t, labels in LABEL_GROUPS.items():
        f = (meta_root / f"{t}_samples.csv").open("w", newline="", encoding="utf-8")
        fieldnames = ["sample_id", "patient_id", "report_id", "img_path"] + labels
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        writers[t] = w
        files[t] = f

    # pfs 沒 label，另起一張表
    pfs_f = (meta_root / "pfs_samples.csv").open("w", newline="", encoding="utf-8")
    pfs_writer = csv.DictWriter(pfs_f, fieldnames=[
        "sample_id", "patient_id", "report_id", "img_path", "csv_path"
    ])
    pfs_writer.writeheader()

    # label 整體統計
    stat = defaultdict(lambda: {"positive": 0, "negative": 0, "missing": 0})

    sid = 1
    for rec in valid_reports:
        p_id = rec["patient_id"]
        r_id = rec["report_id"]
        uid_base = f"{p_id}_{r_id}"
        report_dir = Path(rec["report_path"]) / "output"
        label_dict = parse_output_txt(report_dir / "result" / "output.txt")

        # 更新統計
        for k, v in label_dict.items():
            if v is None:
                stat[k]["missing"] += 1
            elif str(v).lower() in ("1", "yes", "true",
                                     "obstruction", "intermittent",
                                     "interrupted", "staccato",
                                     "supervoider", "bell",
                                     "acceptable", "impaired"):
                stat[k]["positive"] += 1
            else:
                stat[k]["negative"] += 1

        # pftg / pfus / xray：每張圖片各成一筆
        for t in ("pftg", "pfus", "xray"):
            for img_rel in copy_images(report_dir / t, img_root / t, uid_base):
                row = {
                    "sample_id": f"{sid:08d}",
                    "patient_id": p_id,
                    "report_id": r_id,
                    "img_path": img_rel,
                    **{k: label_dict.get(k) for k in LABEL_GROUPS[t]}
                }
                writers[t].writerow(row)
                sid += 1

        # pfs：圖 + csv 一對一
        csv_src = next((report_dir / "pfs").glob("*.csv"), None)
        for img_rel in copy_images(report_dir / "pfs", img_root / "pfs", uid_base):
            csv_rel = os.path.relpath(csv_src, proc_root) if csv_src else ""
            pfs_writer.writerow({
                "sample_id": f"{sid:08d}",
                "patient_id": p_id,
                "report_id": r_id,
                "img_path": img_rel,
                "csv_path": csv_rel
            })
            sid += 1

    # 收尾：關檔 & 輸出統計
    for f in files.values():
        f.close()
    pfs_f.close()
    (meta_root / "label_stats.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False)
    )

    print(f"✅ Dataset built at {proc_root}, total samples: {sid-1}")

if __name__ == "__main__":
    main()
