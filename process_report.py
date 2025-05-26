import os
import glob
import json
import re
import shutil
from pdf2image import convert_from_path
import cv2
import numpy as np
from paddleocr import PaddleOCR

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
    "Fair": "0", "Acceptable": "1", "Impaired": "2", "Poor": "3",
    "Artifect": "4", "Delayed": "4", "Inconclusive": "5"
}

# Load offset configuration
with open("offset_config.json", "r", encoding="utf-8") as f:
    OFFSET_CFG = json.load(f)

# Initialize OCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def is_checked_pixel(b, g, r):
    """
    判斷單一像素是否為勾選標記（橘色框）
    """
    return 240 <= b <= 255 and 80 <= g <= 150 and r <= 40


def is_checked_region(img, x, y, delta=15):
    """
    在 (x,y) 周圍 delta 半徑內若有任一勾選像素即回傳 True
    """
    h, w = img.shape[:2]
    for dx in range(-delta, delta+1):
        for dy in range(-delta, delta+1):
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h and is_checked_pixel(*img[py, px]):
                return True
    return False


def visualize_detected_labels(img, ocr_results, annotated_dir, page_idx):
    """
    標註 OCR 偵測結果與 checkbox 狀態，並存至 annotated_dir
    """
    debug_img = img.copy()
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key in LABEL_KEYS:
            if key in txt:
                print(f"\n🟩 Detected Label '{key}' at {coords[0]}")
                pts = np.array(coords, np.int32)
                cv2.polylines(debug_img, [pts], True, (0,255,0), 2)
                x0, y0 = map(int, coords[0])
                for opt in OFFSET_CFG[key]:
                    px, py = x0 + opt['offset'][0], y0 + opt['offset'][1]
                    checked = is_checked_region(img, px, py)
                    mapped = VALUE_MAP.get(opt['label'], 'Unknown')
                    print(f"  → Checking '{opt['label']}' at ({px},{py}) → {'✔' if checked else '✘'} mapped '{mapped}'")
                    cv2.circle(debug_img, (px, py), 5, (0,0,255) if checked else (255,0,0), -1)
    os.makedirs(annotated_dir, exist_ok=True)
    out_path = os.path.join(annotated_dir, f"annotated_page_{page_idx}.jpg")
    cv2.imwrite(out_path, debug_img)


def extract_labels_from_image(img, ocr_results):
    """
    根據 OCR 結果與 OFFSET_CFG，回傳單張影像中所有 label 的值
    """
    data = {}  # 改為空字典，不預設 'NA'
    for line in ocr_results[0]:
        coords, (text, _) = line
        txt = text.strip()
        for key_text, key_label in LABEL_KEYS.items():
            if key_text in txt:
                # 如果這個標籤已經有值，跳過
                if key_label in data:
                    continue
                    
                x0, y0 = map(int, coords[0])
                picks = []
                for opt in OFFSET_CFG[key_text]:
                    px, py = x0 + opt['offset'][0], y0 + opt['offset'][1]
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0] and is_checked_region(img, px, py):
                        picks.append(VALUE_MAP.get(opt['label']))
                if picks:
                    unique = sorted(set(picks))
                    data[key_label] = ','.join(unique)
                else:
                    data[key_label] = 'NA'
                break
    return data


def process_report_folder(report_dir):
    """
    處理單份報告 PDF：
      - convert_from_path 轉 JPG
      - OCR 與標註
      - 抽取 checkbox 結果
      - 最終 output.txt 存至 output/result
    """
    annotated_dir = os.path.join(report_dir, 'output', 'annotated')
    result_dir    = os.path.join(report_dir, 'output', 'result')
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(result_dir,    exist_ok=True)

    # 初始化資料字典和 Flow_pattern 計數器
    all_data = {v: 'NA' for v in LABEL_KEYS.values()}
    flow_pattern_count = 0  # 追蹤 Flow_pattern 出現次數

    for fname in sorted(os.listdir(report_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(report_dir, fname)
        pages = convert_from_path(pdf_path, dpi=300)
        for i, page in enumerate(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            res = ocr.ocr(img, cls=True)
            visualize_detected_labels(img, res, annotated_dir, i)
            data = extract_labels_from_image(img, res)
            
            # 處理每個標籤
            for k, v in data.items():
                # 特殊處理 Flow_pattern
                if k == 'Flow_pattern':
                    flow_pattern_count += 1
                    # 只在第二次出現時更新值
                    if flow_pattern_count == 2:
                        all_data[k] = v
                # 其他標籤：只在第一次出現時更新
                elif all_data[k] == 'NA':
                    all_data[k] = v

    out_path = os.path.join(result_dir, 'output.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for section in [
            ['Detrusor_instability'],
            ['Flow_pattern', 'EMG_ES_relaxation'],
            ['Trabeculation', 'Diverticulum', 'Cystocele', 'VUR',
             'Bladder_neck_relaxation', 'External_sphincter_relaxation', 'Pelvic_floor_relaxation']
        ]:
            for key in section:
                f.write(f"{key}: {all_data[key]}\n")
            f.write("\n")
    print(f"✅ Results saved: {out_path}")


def batch_process_reports(root_dir='raw_dataset', log_path='report_extraction_log.json'):
    """
    批次處理 raw_dataset 內所有 patient/report，調用 process_report_folder。
    並記錄每個 report_dir 狀態於 log_path。
    """
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
    else:
        log = {}

    total_new = 0
    for patient_id in sorted(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # 新增：若 patient_dir 底下直接有 PDF，也記錄下來（但仍以子資料夾為單位處理）
        # 這裡僅呈現有需時可列印，實際處理在子資料夾複製時一併處理

        for report_id in sorted(os.listdir(patient_dir)):
            report_dir = os.path.join(patient_dir, report_id)
            if not os.path.isdir(report_dir):
                continue

            # —— 新增：從 report_id 取 YYYYMMDD，回上一層找 PDF，複製到子資料夾 —— 
            m = re.search(r'-(\d{8})$', report_id)
            if m:
                date_str = m.group(1)
                candidate_pdf = os.path.join(patient_dir, f"{date_str}報告.pdf")
                if os.path.isfile(candidate_pdf):
                    dest_pdf = os.path.join(report_dir, f"{date_str}報告.pdf")
                    if not os.path.isfile(dest_pdf):
                        print(f"📄 Found {candidate_pdf}, copying into {report_dir}")
                        shutil.copy2(candidate_pdf, dest_pdf)
            # —— 新增結束 —— 

            status = log.get(report_dir)
            if status == 'done':
                print(f"✅ Skipped (done): {report_dir}")
                continue

            pdfs = glob.glob(os.path.join(report_dir, '*.pdf'))
            if not pdfs:
                print(f"⚠️  No PDF in: {report_dir}")
                log[report_dir] = 'no_pdfs'
                continue

            print(f"🟡 Processing: {report_dir}")
            try:
                process_report_folder(report_dir)
                log[report_dir] = 'done'
                total_new += 1
            except Exception as e:
                print(f"❌ Error processing {report_dir}: {e}")
                log[report_dir] = 'error'

            # 每處理完一個就立即存 log，防止中斷
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n🏁 Batch complete. New processed: {total_new}")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 默認以命令列參數呼叫 batch_process_reports
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../../raw_dataset', help='raw_dataset 根目錄')
    parser.add_argument('--log_path', type=str, default='../../report_extraction_log.json', help='處理 log 檔')
    args = parser.parse_args()
    batch_process_reports(root_dir=args.root_dir, log_path=args.log_path)
