# -*- coding: utf-8 -*-
import os
import glob
import json
import csv
import re
import cv2
import types
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import layoutparser as lp
from layoutparser import Rectangle

# Patch OpenCV headless issue
if not hasattr(cv2, "dnn"):
    cv2.dnn = types.SimpleNamespace()
if not hasattr(cv2.dnn, "DictValue"):
    class FakeDictValue:
        def __init__(self, *args, **kwargs): pass
    cv2.dnn.DictValue = FakeDictValue

# Sanitize path for logging by escaping non-ASCII chars
# This uses backslashreplace to convert to ASCII-safe string
def sanitize_path(path):
    # encode to ASCII bytes, replacing non-ASCII with backslash escapes
    # then decode back to ASCII string
    return path.encode('ascii', 'backslashreplace').decode('ascii')

# Setup VUDS extractor: load template config and prepare directories
def setup_vuds_extractor(template_path="template_config.json", base_input_dir="test/"):
    with open(template_path, encoding="utf-8") as f:
        templates = json.load(f)
    summary_cfg = templates.get("Pressure Flow Summary")
    if summary_cfg is None:
        raise ValueError("Missing Pressure Flow Summary in templates")

    labels = [
        "Maximum Flow", "Average Flow", "Voiding time", "Flow time", "Time to max flow",
        "Voided volume", "Flow at 2 seconds", "Acceleration", "Pressure at peak flow", "Flow at peak pressure",
        "Peak pressure", "Mean Pressure", "Opening pressure", "Closing Pressure", "BOOI",
        "BCI", "BVE", "VOID", "PVR"
    ]
    units = [
        "ml/sec", "ml/sec", "mm:ss:S", "mm:ss:S", "mm:ss:S",
        "ml", "ml/sec", "ml/sec/sec", "cmH2O", "ml/sec",
        "cmH2O", "cmH2O", "cmH2O", "cmH2O", "", "", "", "", "ml"
    ]

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    output_base = os.path.join(base_input_dir, "output")
    dirs = {
        "annotated": os.path.join(output_base, "annotated"),
        "pftg":      os.path.join(output_base, "pftg"),
        "pfus":      os.path.join(output_base, "pfus"),
        "pfs":       os.path.join(output_base, "pfs"),
        "xray":      os.path.join(output_base, "xray")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    pattern = re.compile(r".*\d+\s*(?:record|single\s*capture)$", re.IGNORECASE)
    fluoro_cfg = templates.get("Fluoroscopy in VUDS", {})

    return ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg

# Process a single image: OCR, extract regions, save outputs, annotate
def process_image(img_path, ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg):
    base = os.path.splitext(os.path.basename(img_path))[0]
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)
    np_ann = np_img.copy()

    results = ocr.ocr(np_img, cls=True)

    # Extract X-ray regions based on text pattern and fluoro_cfg offsets
    fx = fluoro_cfg.get("x", 0)
    fy = fluoro_cfg.get("offset_y", 0)
    fw = fluoro_cfg.get("width", 50)
    fh = fluoro_cfg.get("height", 10)
    xray_idx = 0
    for line in results:
        for box, (txt, _) in line:
            if pattern.match(txt.strip().lower()):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x1, y1, x2, y2 = map(int, (min(xs), min(ys), max(xs), max(ys)))
                bx0 = int(fx)
                by0 = y2 + int(fy)
                bx1 = bx0 + int(fw)
                by1 = by0 + int(fh)
                crop = np_img[by0:by1, bx0:bx1]
                out_x = os.path.join(dirs["xray"], f"{base}_xray_{xray_idx}.jpg")
                cv2.imwrite(out_x, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                xray_idx += 1
                cv2.rectangle(np_ann, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(np_ann, (bx0, by0), (bx1, by1), (255, 0, 0), 2)

    # Build text blocks for template matching
    text_blocks = []
    for line in results:
        for box, (txt, _) in line:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            text_blocks.append(lp.TextBlock(Rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), text=txt))

    summary_vals = []
    # Extract template blocks: graphs, segments, summaries
    for label, cfg in templates.items():
        for blk in text_blocks:
            text_lower = blk.text.lower()
            if label == "Pressure Flow Uroflow Segment":
                if "segment" not in text_lower:
                    continue
            else:
                if label.lower() not in text_lower:
                    continue

            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            cv2.rectangle(np_ann, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)

            sx = tx1 + cfg.get("offset_x", 0)
            sy = ty1 + cfg.get("offset_y", 0)
            ex = min(np_img.shape[1], sx + cfg["width"])
            ey = min(np_img.shape[0], sy + cfg["height"])
            crop_block = np_img[sy:ey, sx:ex]

            if label == "Pressure Flow Test Graph":
                out_b = os.path.join(dirs["pftg"], f"{base}_PFTG.jpg")
            elif label == "Pressure Flow Uroflow Segment":
                out_b = os.path.join(dirs["pfus"], f"{base}_PFUS.jpg")
            elif label == "Pressure Flow Summary":
                out_b = os.path.join(dirs["pfs"], f"{base}_PFS.jpg")
            else:
                continue

            cv2.imwrite(out_b, cv2.cvtColor(crop_block, cv2.COLOR_RGB2BGR))
            cv2.rectangle(np_ann, (sx, sy), (ex, ey), (0, 0, 255), 2)

            if label == "Pressure Flow Summary":
                for col in summary_cfg["columns"].values():
                    cx0 = sx + col["x"]
                    cy0 = sy + col["y"]
                    for i in range(col["row_count"]):
                        x0 = int(cx0)
                        y0 = int(cy0 + i * col["row_height"]) 
                        x1 = x0 + col["width"]
                        y1 = y0 + col["row_height"]
                        cell = np_img[y0:y1, x0:x1]
                        r = ocr.ocr(cell, cls=True)
                        text = "".join([itm[1][0] for itm in r[0]]) if r and r[0] else ""
                        summary_vals.append(text)
                        cv2.rectangle(np_ann, (x0, y0), (x1, y1), (0, 255, 0), 1)
            break

    # Save summary CSV if values found
    if summary_vals:
        csv_p = os.path.join(dirs["pfs"], f"{base}_PFS.csv")
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Label", "Value", "Unit", "Source"])


            for lbl, val, unit in zip(labels, summary_vals, units):
                writer.writerow([lbl, val, unit, f"{base}_PFS.jpg"])

    # Write annotated image
    out_ann = os.path.join(dirs["annotated"], f"{base}_annotated.jpg")
    cv2.imwrite(out_ann, cv2.cvtColor(np_ann, cv2.COLOR_RGB2BGR))

# Run on a single report folder
def run_on_report_folder(report_path):
    ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg = setup_vuds_extractor(base_input_dir=report_path)
    jpg_files = glob.glob(os.path.join(report_path, "*.jpg"))
    if not jpg_files:
        print(f"??  No .jpg found in {report_path}")
        return
    for jpg_file in jpg_files:
        try:
            process_image(jpg_file, ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg)
            print(f"? {os.path.basename(jpg_file)} done")
        except Exception as e:
            print(f"? error on {jpg_file}: {e}")

# Batch processing driver
def batch_process_reports(root_dir="../../raw_dataset", log_path="../../extraction_log.json"):
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {}

    total_processed = 0
    for patient_id in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for report_id in os.listdir(patient_dir):
            report_dir = os.path.join(patient_dir, report_id)
            if not os.path.isdir(report_dir):
                continue
            if log.get(report_dir) == "done":
                print(f"? Skipped already processed: {report_dir}")
                continue
            jpg_files = glob.glob(os.path.join(report_dir, "*.jpg"))
            if not jpg_files:
                print(f"??  No .jpg files found in: {report_dir}")
                log[report_dir] = "no_images"
                continue
            print(f"?? Processing: {report_dir}")
            try:
                run_on_report_folder(report_dir)
                log[report_dir] = "done"
                total_processed += 1
            except Exception as e:
                print(f"? Error processing {report_dir}: {e}")
                log[report_dir] = f"error: {str(e)}"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n? Completed batch processing. New folders processed: {total_processed}")
    print(f"?? Log saved to: {log_path}")

if __name__ == "__main__":
    batch_process_reports("../../raw_dataset", "../../extraction_log.json")
