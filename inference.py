import argparse
import os
import glob
import re
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import layoutparser as lp
from paddleocr import PaddleOCR
import logging
from io import BytesIO
import tempfile, shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# Suppress all unnecessary logs
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('paddleocr').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode for PaddleOCR
torch.set_printoptions(profile="default")  # Suppress torch tensor printing

# Label mapping for all tasks
LABEL_MAP = {
    'pftg': {
        'Detrusor_instability': {0: 'No', 1: 'Yes'}
    },
    'pfus': {
        'Flow_pattern': {
            0: 'Obstruction',
            1: 'Intermittent',
            2: 'Interrupted',
            3: 'Staccato',
            4: 'Supervoider',
            5: 'Bell'
        },
        'EMG_ES_relaxation': {
            0: 'Fair',
            1: 'Acceptable',
            2: 'Impaired',
            3: 'Poor',
            4: 'Artifect'
        }
    },
    'xray': {
        'Trabeculation': {0: 'No', 1: 'Yes'},
        'Diverticulum': {0: 'No', 1: 'Yes'},
        'Cystocele': {0: 'No', 1: 'Yes'},
        'VUR': {0: 'No', 1: 'Yes'},
        'Bladder_neck_relaxation': {
            0: 'Fair',
            1: 'Acceptable',
            2: 'Impaired',
            3: 'Poor',
            4: 'Delayed',
            5: 'Inconclusive'
        },
        'External_sphincter_relaxation': {
            0: 'Fair',
            1: 'Acceptable',
            2: 'Impaired',
            3: 'Poor',
            4: 'Delayed',
            5: 'Inconclusive'
        },
        'Pelvic_floor_relaxation': {
            0: 'Fair',
            1: 'Acceptable',
            2: 'Impaired',
            3: 'Poor',
            4: 'Delayed',
            5: 'Inconclusive'
        }
    }
}

# Number of classes for each task
NUM_CLASSES = {
    'pftg': {'Detrusor_instability': 2},
    'pfus': {'Flow_pattern': 6, 'EMG_ES_relaxation': 5},
    'xray': {
        'Trabeculation': 2,
        'Diverticulum': 2,
        'Cystocele': 2,
        'VUR': 2,
        'Bladder_neck_relaxation': 6,
        'External_sphincter_relaxation': 6,
        'Pelvic_floor_relaxation': 6
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for VUDS-AI")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing model files")
    parser.add_argument("--img", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--output", type=str, default='output.txt',
                        help="Output label file path")
    return parser.parse_args()

def setup_image_extractor():
    """Setup OCR and patterns for image extraction"""
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    pattern = re.compile(r".*\d+\s*(?:record|single\s*capture)$", re.IGNORECASE)
    return ocr, pattern

def extract_all_regions(img_path, ocr, pattern):
    """Extract and classify all regions from an image"""
    print(f"\n🔍 Processing image: {os.path.basename(img_path)}")
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)
    print("Running OCR...")
    results = ocr.ocr(np_img, cls=True)
    regions = {
        'pftg': {'exists': False, 'regions': []},
        'pfus': {'exists': False, 'regions': []},
        'xray': {'exists': False, 'regions': []}
    }
    text_blocks = []
    for line in results:
        for box, (txt, _) in line:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            text_blocks.append(lp.TextBlock(
                lp.Rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                text=txt
            ))
    for blk in text_blocks:
        text_lower = blk.text.lower()
        if "pressure flow uroflow segment" in text_lower:
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            regions['pfus']['exists'] = True
            regions['pfus']['regions'].append(np_img[ty1:ty2, tx1:tx2])
            print(f"Found PFUS region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
        elif "pressure flow test graph" in text_lower:
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            regions['pftg']['exists'] = True
            regions['pftg']['regions'].append(np_img[ty1:ty2, tx1:tx2])
            print(f"Found PFTG region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
        elif pattern.match(text_lower.strip()):
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            pad = 50
            regions['xray']['exists'] = True
            regions['xray']['regions'].append(
                np_img[max(0, ty1-pad):min(np_img.shape[0], ty2+pad), 
                      max(0, tx1-pad):min(np_img.shape[1], tx2+pad)]
            )
            print(f"Found X-ray region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
    print("\n📊 Region Detection Summary:")
    print(f"PFUS regions found: {len(regions['pfus']['regions'])}")
    print(f"PFTG regions found: {len(regions['pftg']['regions'])}")
    print(f"X-ray regions found: {len(regions['xray']['regions'])}")
    return regions

def load_model(task, model_path):
    filename = os.path.basename(model_path)
    task_name = '_'.join(filename.split('_')[1:-1])
    if task_name not in NUM_CLASSES[task]:
        raise ValueError(f"Unknown task name '{task_name}' for task '{task}'")
    num_classes = NUM_CLASSES[task][task_name]
    model = models.resnet18(pretrained=False)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded model weights for {task_name}")
    except Exception as e:
        print(f"Warning loading weights for {task_name}: {str(e)}")
        if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
            model.fc.weight.data = state_dict['fc.weight']
            model.fc.bias.data = state_dict['fc.bias']
            print("Loaded classifier weights only")
        else:
            raise
    model.eval()
    return model, task_name

def predict_image(model, img_array):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = tfm(img_array).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).item()
    return pred

def get_default_results():
    return {key: 'NA' for key in [
        'Detrusor_instability','Flow_pattern','EMG_ES_relaxation',
        'Trabeculation','Diverticulum','Cystocele','VUR',
        'Bladder_neck_relaxation','External_sphincter_relaxation','Pelvic_floor_relaxation'
    ]}

def write_results_to_file(results, output_path):
    with open(output_path, 'w') as f:
        f.write(f"Detrusor_instability: {results['Detrusor_instability']}\n\n")
        f.write("Pressure flow study\n")
        f.write(f"Flow_pattern: {results['Flow_pattern']}\n")
        f.write(f"EMG_ES_relaxation: {results['EMG_ES_relaxation']}\n\n")
        f.write("Fluroscopy in VUDS\n")
        for lbl in ['Trabeculation','Diverticulum','Cystocele','VUR',
                    'Bladder_neck_relaxation','External_sphincter_relaxation','Pelvic_floor_relaxation']:
            f.write(f"{lbl}: {results[lbl]}\n")

def predict_all_tasks(regions, model_dir):
    results = get_default_results()
    print("\n🤖 Starting prediction for all tasks...")
    for task in ['pftg','pfus','xray']:
        if not regions[task]['exists']:
            print(f"No {task} regions, skipping")
            continue
        for mfile in os.listdir(model_dir):
            if mfile.startswith(task) and mfile.endswith('.pth'):
                model, task_name = load_model(task, os.path.join(model_dir,mfile))
                for region in regions[task]['regions']:
                    pred = predict_image(model, region)
                    results[task_name] = LABEL_MAP[task][task_name][pred]
    print("\n📊 Final Prediction Results:")
    for k,v in results.items(): print(f"{k}: {v}")
    return results

def process_images(img_paths, model_dir, ocr, pattern):
    all_results = get_default_results()
    for img_path in img_paths:
        regions = extract_all_regions(img_path, ocr, pattern)
        res = predict_all_tasks(regions, model_dir)
        for k,v in res.items():
            if v!='NA': all_results[k]=v
    print("\n📊 Combined Results:" , all_results)
    return all_results

# A4 layout constants
PAGE_W, PAGE_H = A4
MARGIN = 40
TITLE_Y = PAGE_H - 60
IMG_MAX_W = PAGE_W - 2*MARGIN
IMG_MAX_H = PAGE_H - 140

def _collect_images_by_type(dirs):
    return {typ: sorted(glob.glob(os.path.join(dirs[typ],"*.jpg")))
            for typ in ('pfus','pftg','xray')}

def make_a4_pdf(images_dict, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    for typ, imgs in images_dict.items():
        for img_p in imgs:
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(PAGE_W/2, TITLE_Y, typ.upper())
            img = ImageReader(img_p)
            iw, ih = img.getSize()
            scale = min(IMG_MAX_W/iw, IMG_MAX_H/ih)
            w, h = iw*scale, ih*scale
            x = (PAGE_W - w)/2
            y = (PAGE_H - h)/2 - 20
            c.drawImage(img, x, y, width=w, height=h)
            c.showPage()
    c.save()
    print(f"📄 A4 PDF saved → {pdf_path}")

def _predict_from_extracted(dirs, model_dir):
    results = get_default_results()
    task_dirs = {'pftg':dirs['pftg'],'pfus':dirs['pfus'],'xray':dirs['xray']}
    for task, d in task_dirs.items():
        for mfile in os.listdir(model_dir):
            if mfile.startswith(task) and mfile.endswith('.pth'):
                model, task_name = load_model(task, os.path.join(model_dir,mfile))
                for img_p in sorted(glob.glob(os.path.join(d,'*.jpg'))):
                    img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
                    pred = predict_image(model, img)
                    results[task_name] = LABEL_MAP[task][task_name][pred]
    print("\n📊 Label Prediction Summary", results)
    return results

# ---------- 1️⃣ 第一版 test_inference ----------
def test_inference(model_dir, img=None, dir_path=None, output_txt="output.txt"):
    ocr, pattern = setup_image_extractor()
    if img:
        results = process_images([img], model_dir, ocr, pattern)
    elif dir_path:
        imgs = glob.glob(os.path.join(dir_path,"*.jpg"))
        results = process_images(imgs, model_dir, ocr, pattern)
    else:
        raise ValueError("Provide --img or --dir")
    write_results_to_file(results, output_txt)
    print(f"[Test-Inference] Results → {output_txt}")
    return results

# ---------- 3️⃣ 第二版 inference_v2 ----------
def inference_v2(path, model_dir=None, pdf_name="output.pdf",
                 txt_name="output.txt", keep_output_folder=True):
    path = os.path.abspath(path)
    model_dir = os.path.abspath(model_dir or os.path.join(path,"models"))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型資料夾不存在: {model_dir}")
    jpgs = glob.glob(os.path.join(path,"*.jpg"))
    if not jpgs: raise ValueError(f"{path} 無 .jpg 圖片")
    if keep_output_folder:
        extract_base = path; tmp_root=None
    else:
        tmp_root = tempfile.mkdtemp(prefix="vuds_tmp_")
        extract_base = tmp_root
        print(f"🗄️ 暫存抽圖: {extract_base}")
    from extract_templates import setup_vuds_extractor, process_image
    ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg = setup_vuds_extractor("template_config.json", extract_base)
    for jpg in jpgs:
        process_image(jpg, ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg)
    results = _predict_from_extracted(dirs, model_dir)
    images_dict = _collect_images_by_type(dirs)
    pdf_path = os.path.join(path,pdf_name)
    if any(images_dict.values()): make_a4_pdf(images_dict,pdf_path)
    else: print("⚠️ 無裁切影像，僅 TXT")
    txt_path = os.path.join(path,txt_name)
    write_results_to_file(results,txt_path)
    print(f"📝 TXT → {txt_path}")
    if not keep_output_folder and tmp_root:
        shutil.rmtree(tmp_root,ignore_errors=True)
        print(f"🧹 刪除暫存: {tmp_root}")
    return pdf_path, txt_path


def main():
    # args = parse_args()
    # if args.img or args.dir:
    #     test_inference(args.model_dir, img=args.img, dir_path=args.dir, output_txt=args.output)
    # else:
    #     inference_v2(".", model_dir=args.model_dir)
    inference_v2("../../test_case", model_dir="../../models")

if __name__ == '__main__':
    main()
