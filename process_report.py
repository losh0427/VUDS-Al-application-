import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import json

# Configuration
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

VALUE_MAP = {
    "Yes": "1", "No": "0", "Obstruction": "0", "Intermittent": "1",
    "Interrupted": "2", "Staccato": "3", "Supervoider": "4", "Bell": "5",
    "Fair": "0", "Acceptable": "1", "Impaired": "2", "Poor": "3"
}

# 讀入 offset 配置
with open("offset_config.json", "r", encoding="utf-8") as f:
    OFFSET_CFG = json.load(f)

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def is_checked_pixel(b, g, r):
    # 橘色背景 (勾框) 或 白色 (勾選)
    if 240 <= b <= 255 and 80 <= g <= 150 and r <= 40:
        return True
    if b > 240 and g > 240 and r > 240:
        return True
    return False

def is_checked_region(img, x, y, delta=15):
    h, w = img.shape[:2]
    for dx in range(-delta, delta+1):
        for dy in range(-delta, delta+1):
            px, py = x+dx, y+dy
            if 0 <= px < w and 0 <= py < h:
                b,g,r = img[py,px]
                if is_checked_pixel(b, g, r):
                    return True
    return False

def extract_labels_from_image(img, ocr_results):
    data = {v: "NA" for v in LABEL_KEYS.values()}
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key_text, key_label in LABEL_KEYS.items():
            if key_text in txt:
                x0, y0 = int(coords[0][0]), int(coords[0][1])
                for opt in OFFSET_CFG[key_text]:
                    lbl = opt["label"]
                    dx, dy = opt["offset"]
                    px, py = x0 + dx, y0 + dy
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        if is_checked_region(img, px, py, delta=15):
                            val = VALUE_MAP.get(lbl)
                            if key_label == "Flow_pattern":
                                prev = data[key_label]
                                if prev == "NA":
                                    data[key_label] = val
                                elif val not in prev.split(","):
                                    data[key_label] = ",".join(sorted(prev.split(",") + [val]))
                            else:
                                if data[key_label] == "NA":
                                    data[key_label] = val
                break
    return data

def visualize_detected_labels(img, ocr_results, report_dir, page_idx):
    debug_img = img.copy()
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key_text in LABEL_KEYS:
            if key_text in txt:
                # 畫文字框
                pts = np.array(coords, np.int32)
                cv2.polylines(debug_img, [pts], True, (0,255,0), 2)
                x0, y0 = int(coords[0][0]), int(coords[0][1])
                for opt in OFFSET_CFG[key_text]:
                    dx, dy = opt["offset"]
                    px, py = x0 + dx, y0 + dy
                    # 偵測區域以紅點顯示
                    cv2.circle(debug_img, (px, py), 3, (0,0,255), -1)
                    cv2.putText(debug_img, opt["label"], (px+5, py-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    path = os.path.join(report_dir, f"output_debug_page_{page_idx+1}.png")
    cv2.imwrite(path, debug_img)
    print(f"🔍 debug image saved: {path}")

def process_report_folder(report_dir):
    all_data = {v: "NA" for v in LABEL_KEYS.values()}
    for fname in sorted(os.listdir(report_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        pages = convert_from_path(os.path.join(report_dir, fname), dpi=300)
        for i, page in enumerate(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            res = ocr.ocr(img, cls=True)
            visualize_detected_labels(img, res, report_dir, i)
            data = extract_labels_from_image(img, res)
            for k, v in data.items():
                if k == "Flow_pattern":
                    if all_data[k] == "NA":
                        all_data[k] = v
                    elif v != "NA":
                        merged = sorted(set(all_data[k].split(",") + v.split(",")))
                        all_data[k] = ",".join(merged)
                else:
                    if all_data[k] == "NA" and v != "NA":
                        all_data[k] = v
    out = os.path.join(report_dir, "output.txt")
    with open(out, "w", encoding="utf-8") as f:
        for section in [
            ["Detrusor_instability"],
            ["Flow_pattern","EMG_ES_relaxation"],
            ["Trabeculation","Diverticulum","Cystocele","VUR",
             "Bladder_neck_relaxation","External_sphincter_relaxation","Pelvic_floor_relaxation"]
        ]:
            for key in section:
                f.write(f"{key}: {all_data[key]}\n")
            f.write("\n")
    print(f"✅ output saved: {out}")

if __name__ == "__main__":
    process_report_folder("./report")
