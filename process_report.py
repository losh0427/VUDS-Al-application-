# -*- coding: utf-8 -*-
import os
import glob
import json
import re
import shutil
from pdf2image import convert_from_path
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Mapping of label text to output keys
LABEL_KEYS = {
    "Detrusor instability": "Detrusor_instability",
    "Flow pattern": "Flow_pattern",
    "EMG: ES relaxation": "EMG_ES_relaxation",
    "Trabeculation": "Trabeculation",
    "Diverticulum": "Diverticulum",
    "Cystocele": "Cystocele",
    "VUR": "VUR",
    "Bladder neck relaxation": "Bladder_neck_relaxation",
    "External sphincter relaxation": "External_sphincter_relaxation",
    "Pelvic floor relaxation": "Pelvic_floor_relaxation"
}

# Mapping of checkbox labels to output values
VALUE_MAP = {
    "Yes": "1", "No": "0", "Obstruction": "0", "Intermittent": "1",
    "Interrupted": "2", "Staccato": "3", "Supervoider": "4", "Bell": "5",
    "Fair": "0", "Acceptable": "1", "Impaired": "2", "Poor": "3",
    "Artifect": "4", "Delayed": "4", "Inconclusive": "5"
}

# Load checkbox offset configuration
with open("offset_config.json", "r", encoding="utf-8") as f:
    OFFSET_CFG = json.load(f)

# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def is_checked_pixel(b, g, r):
    # Determine if a pixel is part of an orange checkbox mark
    return 240 <= b <= 255 and 80 <= g <= 150 and r <= 40

def is_checked_region(img, x, y, delta=15):
    # Check if any pixel within delta radius of (x,y) is a checked pixel
    h, w = img.shape[:2]
    for dx in range(-delta, delta+1):
        for dy in range(-delta, delta+1):
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h:
                if is_checked_pixel(*img[py, px]):
                    return True
    return False

def visualize_detected_labels(img, ocr_results, annotated_dir, page_idx):
    # Draw OCR-detected label boxes and checkbox status for debugging
    debug_img = img.copy()
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key in LABEL_KEYS:
            if key in txt:
                pts = np.array(coords, np.int32)
                cv2.polylines(debug_img, [pts], True, (0,255,0), 2)
                x0, y0 = map(int, coords[0])
                for opt in OFFSET_CFG[key]:
                    px = x0 + opt['offset'][0]
                    py = y0 + opt['offset'][1]
                    checked = is_checked_region(img, px, py)
                    color = (0,0,255) if checked else (255,0,0)
                    cv2.circle(debug_img, (px, py), 5, color, -1)
    os.makedirs(annotated_dir, exist_ok=True)
    out_path = os.path.join(annotated_dir, f"annotated_page_{page_idx}.jpg")
    cv2.imwrite(out_path, debug_img)

def extract_labels_from_image(img, ocr_results):
    # Extract checkbox values from a single page image
    data = {}
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key_text, key_label in LABEL_KEYS.items():
            if key_text in txt:
                if key_label in data:
                    continue
                x0, y0 = map(int, coords[0])
                picks = []
                for opt in OFFSET_CFG[key_text]:
                    px = x0 + opt['offset'][0]
                    py = y0 + opt['offset'][1]
                    if (0 <= px < img.shape[1] and 0 <= py < img.shape[0]
                            and is_checked_region(img, px, py)):
                        picks.append(VALUE_MAP.get(opt['label']))
                if picks:
                    data[key_label] = ",".join(sorted(set(picks)))
                else:
                    data[key_label] = "NA"
                break
    return data

def process_report_folder(report_dir):
    # Process a single report folder:
    #  - Convert PDF pages to images
    #  - Run OCR and visualization
    #  - Extract all checkbox values
    #  - Save output.txt in output/result
    annotated_dir = os.path.join(report_dir, "output", "annotated")
    result_dir = os.path.join(report_dir, "output", "result")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    all_data = {v: "NA" for v in LABEL_KEYS.values()}
    flow_pattern_count = 0

    for fname in sorted(os.listdir(report_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(report_dir, fname)
        pages = convert_from_path(pdf_path, dpi=300)
        for i, page in enumerate(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            res = ocr.ocr(img, cls=True)
            visualize_detected_labels(img, res, annotated_dir, i)
            data = extract_labels_from_image(img, res)
            for k, v in data.items():
                if k == "Flow_pattern":
                    flow_pattern_count += 1
                    if flow_pattern_count == 2:
                        all_data[k] = v
                elif all_data[k] == "NA":
                    all_data[k] = v

    out_path = os.path.join(result_dir, "output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for section in [
            ["Detrusor_instability"],
            ["Flow_pattern", "EMG_ES_relaxation"],
            [
                "Trabeculation", "Diverticulum", "Cystocele", "VUR",
                "Bladder_neck_relaxation", "External_sphincter_relaxation", "Pelvic_floor_relaxation"
            ]
        ]:
            for key in section:
                f.write(f"{key}: {all_data[key]}\n")
            f.write("\n")

def batch_process_reports(root_dir="../../raw_dataset", log_path="../../report_extraction_log.json"):
    # Batch-process all patient/report folders under root_dir.
    # For each report:
    #  - Locate any PDF matching "<date_str>*.pdf" in the parent folder
    #  - Copy it into the report_dir if not already present
    #  - Call process_report_folder
    #  - Maintain a JSON log of statuses
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {}

    total_new = 0
    for patient_id in sorted(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        for report_id in sorted(os.listdir(patient_dir)):
            report_dir = os.path.join(patient_dir, report_id)
            if not os.path.isdir(report_dir):
                continue

            # Extract date string from report_id suffix
            m = re.search(r"-(\d{8})$", report_id)
            if m:
                date_str = m.group(1)
                pattern = os.path.join(patient_dir, f"{date_str}*.pdf")
                matches = glob.glob(pattern)
                if matches:
                    src_pdf = matches[0]
                    dest_pdf = os.path.join(report_dir, os.path.basename(src_pdf))
                    if not os.path.isfile(dest_pdf):
                        print(f"Copying {os.path.basename(src_pdf)} to {report_dir}")
                        shutil.copy2(src_pdf, dest_pdf)

            status = log.get(report_dir)
            if status == "done":
                print(f"Skipping (done): {report_dir}")
                continue

            pdfs = glob.glob(os.path.join(report_dir, "*.pdf"))
            if not pdfs:
                print(f"No PDF in: {report_dir}")
                log[report_dir] = "no_pdfs"
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
                continue

            print(f"Processing: {report_dir}")
            try:
                process_report_folder(report_dir)
                log[report_dir] = "done"
                total_new += 1
            except Exception as e:
                print(f"Error processing {report_dir}: {e}")
                log[report_dir] = "error"

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Batch complete. New processed: {total_new}")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../../raw_dataset",
        help="Path to the raw_dataset root directory"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="../../report_extraction_log.json",
        help="Path to the JSON log file"
    )
    args = parser.parse_args()
    batch_process_reports(root_dir=args.root_dir, log_path=args.log_path)
