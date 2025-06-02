# -*- coding: utf-8 -*-
# Inference pipeline for VUDS-AI – unified, runnable version
# NOTE: All comments are single-line and in English as requested.

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
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict

# Suppress unnecessary logs from libraries
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('paddleocr').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
# Remove forced CPU mode for OCR
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode for OCR
torch.set_printoptions(profile="default")  # Suppress tensor printing

# ----------------------------------------------------------------------
# Label definitions (single source of truth)
# ----------------------------------------------------------------------
LABEL_MAP = {
    'pftg': {
        'Detrusor_instability': {0: 'No', 1: 'Yes'}
    },
    'pfus': {
        'Flow_pattern': {0: 'Obstruction', 1: 'Intermittent', 2: 'Interrupted', 3: 'Staccato', 4: 'Supervoider', 5: 'Bell'},
        'EMG_ES_relaxation': {0: 'Fair', 1: 'Acceptable', 2: 'Impaired', 3: 'Poor', 4: 'Artifact'}
    },
    'xray': {
        'Trabeculation': {0: 'No', 1: 'Yes'},
        'Diverticulum': {0: 'No', 1: 'Yes'},
        'Cystocele': {0: 'No', 1: 'Yes'},
        'VUR': {0: 'No', 1: 'Yes'},
        'Bladder_neck_relaxation': {0: 'Fair', 1: 'Acceptable', 2: 'Impaired', 3: 'Poor', 4: 'Delayed', 5: 'Inconclusive'},
        'External_sphincter_relaxation': {0: 'Fair', 1: 'Acceptable', 2: 'Impaired', 3: 'Poor', 4: 'Delayed', 5: 'Inconclusive'},
        'Pelvic_floor_relaxation': {0: 'Fair', 1: 'Acceptable', 2: 'Impaired', 3: 'Poor', 4: 'Delayed', 5: 'Inconclusive'}
    }
}

# Number of classes for each label
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

# Global threshold settings
THRESHOLDS = defaultdict(lambda: 0.60)

#----------------------------------------------------------------------
# Phase 1: Model loading functions
#----------------------------------------------------------------------

def _build_model(backbone: str, num_classes: int) -> nn.Module:
    # Build a ResNet18 with adjusted final layer
    if backbone != 'resnet18':
        raise ValueError(f'Unsupported backbone: {backbone}')
    model = models.resnet18(pretrained=False)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model


def load_model_legacy(task: str, ckpt_path: str) -> Tuple[nn.Module, str]:
    # Legacy loader: filename format <task>_<label>_<backbone>.pth
    filename = os.path.basename(ckpt_path)
    parts = filename[:-4].split('_')
    if len(parts) < 3:
        raise ValueError(f'Invalid checkpoint filename: {filename}')
    if parts[0] != task:
        raise ValueError(f'Task mismatch: expected {task}, got {parts[0]}')
    label = '_'.join(parts[1:-1])
    num_classes = NUM_CLASSES[task][label]
    
    try:
        # First load the state dict to check output size
        state = torch.load(ckpt_path, map_location='cpu')
        if 'fc.weight' in state:
            expected_classes = state['fc.weight'].shape[0]
            if expected_classes != num_classes:
                raise ValueError(
                    f'Model output size mismatch for {label}: '
                    f'expected {num_classes} classes but model has {expected_classes} classes. '
                    f'Please check if you are using the correct model file.'
                )
    except Exception as e:
        raise ValueError(f'Error checking model file {filename}: {str(e)}')
    
    # If size check passes, build and load the model
    model = _build_model('resnet18', num_classes)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        raise ValueError(f'Error loading model weights: {str(e)}')
    
    model.eval()
    return model, label


def load_model_ckpt(ckpt_path: str) -> Tuple[nn.Module, Dict]:
    # Unified loader – use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and 'backbone' in ckpt:
        backbone = ckpt['backbone']
        num_classes = ckpt['num_classes']
        model = _build_model(backbone, num_classes)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        meta = {
            'backbone': backbone,
            'label': ckpt.get('label', ''),
            'model_type': ckpt.get('model_type', ''),
            'num_classes': num_classes
        }
    else:
        filename = os.path.basename(ckpt_path)
        parts = filename[:-4].split('_')
        task = parts[0]
        backbone = parts[-1]
        label = '_'.join(parts[1:-1])
        model, _ = load_model_legacy(task, ckpt_path)
        meta = {
            'backbone': backbone,
            'label': label,
            'model_type': task,
            'num_classes': NUM_CLASSES[task][label]
        }
    model.to(device)
    model.eval()
    return model, meta

#--------------------------------------------------------------------
# Phase 2: Image prediction with threshold (currently simple argmax)
#--------------------------------------------------------------------

_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model: nn.Module, img_arr: np.ndarray) -> Union[int, List[int], None]:
    """
    Predict image with threshold handling.
    
    Args:
        model: The model to use for prediction
        img_arr: Input image as numpy array
        
    Returns:
        For binary classification (num_classes == 2):
            - Returns 0 or 1
        For multi-class classification (num_classes > 2):
            - Returns [idx] if max probability >= threshold
            - Returns [] if max probability < threshold
    """
    tensor = _PREPROCESS(img_arr).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        num_classes = logits.shape[1]
        
        if num_classes == 2:
            pred = torch.argmax(logits, dim=1).item()
            return int(pred)
        else:
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            threshold = THRESHOLDS[model.__class__.__name__]
            
            if max_prob.item() >= threshold:
                return [pred_idx.item()]
            return []

#--------------------------------------------------------------------
# Phase 3: Helper utilities
#--------------------------------------------------------------------

def get_default_results() -> Dict[str, str]:
    # Initialize results to 'NA'
    keys = [
        'Detrusor_instability', 'Flow_pattern', 'EMG_ES_relaxation',
        'Trabeculation', 'Diverticulum', 'Cystocele', 'VUR',
        'Bladder_neck_relaxation', 'External_sphincter_relaxation', 'Pelvic_floor_relaxation'
    ]
    return {k: 'NA' for k in keys}


def write_results_to_file(results: Dict[str, str], output_path: str):
    # Write results in required formatted text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Detrusor_instability: {results['Detrusor_instability']}\n\n")
        f.write("Pressure flow study\n")
        f.write(f"Flow_pattern: {results['Flow_pattern']}\n")
        f.write(f"EMG_ES_relaxation: {results['EMG_ES_relaxation']}\n\n")
        f.write("Fluoroscopy in VUDS\n")
        for lbl in [
            'Trabeculation', 'Diverticulum', 'Cystocele', 'VUR',
            'Bladder_neck_relaxation', 'External_sphincter_relaxation', 'Pelvic_floor_relaxation'
        ]:
            # Check if the result is a list (multi-class prediction)
            if isinstance(results[lbl], list):
                f.write(f"{lbl}: {', '.join(results[lbl])}\n")
            else:
                f.write(f"{lbl}: {results[lbl]}\n")
    print(f"Results written to {output_path}")

#--------------------------------------------------------------------
# Phase 4: Template extraction and image collection (restored)
#--------------------------------------------------------------------

def setup_image_extractor():
    # Setup OCR engine and pattern for identifying x-ray captions
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    pattern = re.compile(r".*\d+\s*(?:record|single\s*capture)$", re.IGNORECASE)
    return ocr, pattern


def extract_all_regions(img_path: str, ocr, pattern) -> Dict[str, Dict[str, any]]:
    # Extract and classify regions (pftg, pfus, xray) from the image
    print(f"Processing image: {os.path.basename(img_path)}")
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)
    print("Running OCR...")
    results = ocr.ocr(np_img, cls=True)
    
    # Initialize regions dictionary with proper structure
    regions = {
        'pftg': {'exists': False, 'regions': []},
        'pfus': {'exists': False, 'regions': []},
        'xray': {'exists': False, 'regions': []}
    }
    
    # Process OCR results and extract text blocks
    text_blocks = []
    for line in results:
        for box, (txt, _) in line:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            text_blocks.append(
                lp.TextBlock(
                    lp.Rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), 
                    text=txt
                )
            )
    
    # Process each text block and extract regions
    for blk in text_blocks:
        text_lower = blk.text.lower()
        tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
        
        # Add padding for better region extraction
        pad = 50
        
        if "pressure flow uroflow segment" in text_lower:
            regions['pfus']['exists'] = True
            # Extract region with padding
            region = np_img[
                max(0, ty1-pad):min(np_img.shape[0], ty2+pad),
                max(0, tx1-pad):min(np_img.shape[1], tx2+pad)
            ]
            regions['pfus']['regions'].append(region)
            print(f"Found PFUS region at: {tx1},{ty1},{tx2},{ty2}")
            
        elif "pressure flow test graph" in text_lower:
            regions['pftg']['exists'] = True
            # Extract region with padding
            region = np_img[
                max(0, ty1-pad):min(np_img.shape[0], ty2+pad),
                max(0, tx1-pad):min(np_img.shape[1], tx2+pad)
            ]
            regions['pftg']['regions'].append(region)
            print(f"Found PFTG region at: {tx1},{ty1},{tx2},{ty2}")
            
        elif pattern.match(text_lower.strip()):
            regions['xray']['exists'] = True
            # Extract region with padding
            region = np_img[
                max(0, ty1-pad):min(np_img.shape[0], ty2+pad),
                max(0, tx1-pad):min(np_img.shape[1], tx2+pad)
            ]
            regions['xray']['regions'].append(region)
            print(f"Found X-ray region at: {tx1},{ty1},{tx2},{ty2}")
    
    # Print summary of found regions
    print("\nRegion Detection Summary:")
    for task in ['pfus', 'pftg', 'xray']:
        print(f"{task.upper()} regions found: {len(regions[task]['regions'])}")
    
    return regions


def _collect_images_by_type(dirs: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Collect image file paths by type from the extracted directories.
    
    Args:
        dirs: Dictionary mapping task names to their directory paths
        
    Returns:
        Dictionary mapping task types to lists of image file paths
    """
    images_dict = {}
    
    for task in ['pfus', 'pftg', 'xray']:
        task_dir = dirs.get(task)
        if not task_dir or not os.path.exists(task_dir):
            print(f"No directory found for {task}, skipping...")
            images_dict[task] = []
            continue
            
        # Get all jpg files in the directory
        jpg_files = sorted(glob.glob(os.path.join(task_dir, "*.jpg")))
        if not jpg_files:
            print(f"No jpg files found in {task_dir}")
            images_dict[task] = []
        else:
            print(f"Found {len(jpg_files)} jpg files in {task_dir}")
            images_dict[task] = jpg_files
            
    return images_dict


def make_a4_pdf(images_dict: Dict[str, List[str]], pdf_path: str):
    # Create an A4 PDF with extracted images
    PAGE_W, PAGE_H = A4
    MARGIN = 40
    IMG_MAX_W = PAGE_W - 2 * MARGIN
    IMG_MAX_H = PAGE_H - 140
    c = canvas.Canvas(pdf_path, pagesize=A4)
    for typ, imgs in images_dict.items():
        for img_p in imgs:
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(PAGE_W/2, PAGE_H - 60, typ.upper())
            img = ImageReader(img_p)
            iw, ih = img.getSize()
            scale = min(IMG_MAX_W/iw, IMG_MAX_H/ih)
            w, h = iw * scale, ih * scale
            x = (PAGE_W - w) / 2
            y = (PAGE_H - h) / 2 - 20
            c.drawImage(img, x, y, width=w, height=h)
            c.showPage()
    c.save()
    print(f"A4 PDF saved: {pdf_path}")

#----------------------------------------------------------------------
# Phase 5: Prediction routines
#----------------------------------------------------------------------

def predict_all_tasks(regions: Dict[str, Dict[str, any]], model_dir: str) -> Dict[str, str]:
    # Predict across all tasks and aggregate results
    results = get_default_results()
    print("\nStarting prediction for all tasks...")
    for task in ['pftg', 'pfus', 'xray']:
        if not regions[task]['exists']:
            print(f"No {task} regions, skipping")
            continue
        for mfile in os.listdir(model_dir):
            if mfile.startswith(task) and mfile.endswith('.pth'):
                model, task_name = load_model(task, os.path.join(model_dir, mfile))
                for region in regions[task]['regions']:
                    pred = predict_image(model, region)
                    results[task_name] = LABEL_MAP[task][task_name][pred]
    print("\nFinal Prediction Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
    return results


def process_images(img_paths: List[str], model_dir: str, ocr, pattern) -> Dict[str, str]:
    # Process list of images and collect combined results
    combined = get_default_results()
    for img_path in img_paths:
        regions = extract_all_regions(img_path, ocr, pattern)
        res = predict_all_tasks(regions, model_dir)
        for k, v in res.items():
            if v != 'NA':
                combined[k] = v
    print("\nCombined Results:", combined)
    return combined


def agg_binary_vote(pred_list: List[int]) -> Union[int, None]:
    if not pred_list:
        return None
    
    votes = {0: 0, 1: 0}
    for pred in pred_list:
        if pred in votes:
            votes[pred] += 1
    
    max_votes = max(votes.values())
    winners = [k for k, v in votes.items() if v == max_votes]
    
    if len(winners) > 1:
        return None
    
    return winners[0]

def agg_multi_union(pred_lists: List[List[int]]) -> List[int]:
    if not pred_lists:
        return []
    
    union = set()
    for preds in pred_lists:
        union.update(preds)
    
    return sorted(list(union))

def _predict_from_extracted(dirs: Dict[str, str], model_dir: str) -> Dict[str, str]:
    results = get_default_results()
    print(f"Starting prediction from extracted directories: {dirs}")
    
    xray_raw_predictions = defaultdict(list)
    
    for task in ['pftg', 'pfus', 'xray']:
        task_dir = dirs.get(task)
        if not task_dir or not os.path.exists(task_dir):
            continue
            
        jpg_files = sorted(glob.glob(os.path.join(task_dir, '*.jpg')))
        if not jpg_files:
            continue
            
        for mfile in os.listdir(model_dir):
            if mfile.startswith(task) and mfile.endswith('.pth'):
                model_path = os.path.join(model_dir, mfile)
                model, meta = load_model_ckpt(model_path)
                label_name = meta.get('label', '')
                if not label_name:
                    continue
                
                for img_p in jpg_files:
                    img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
                    pred = predict_image(model, img)
                    
                    if task == 'xray':
                        xray_raw_predictions[label_name].append(pred)
                    else:
                        if isinstance(pred, list):
                            if pred:
                                results[label_name] = LABEL_MAP[task][label_name][pred[0]]
                        else:
                            results[label_name] = LABEL_MAP[task][label_name][pred]
    
    # Process X-ray predictions
    print("\nProcessing X-ray predictions:")
    for label_name, preds in xray_raw_predictions.items():
        print(f"\n{label_name}:")
        print(f"Raw predictions: {preds}")
        
        num_classes = NUM_CLASSES['xray'][label_name]
        
        if num_classes == 2:
            final_pred = agg_binary_vote(preds)
            if final_pred is not None:
                results[label_name] = LABEL_MAP['xray'][label_name][final_pred]
                print(f"Final result: {results[label_name]}")
        else:
            final_preds = agg_multi_union(preds)
            if final_preds:
                # For multi-class, store all predicted labels
                result_labels = [LABEL_MAP['xray'][label_name][idx] for idx in final_preds]
                results[label_name] = result_labels  # Store the list of labels
                print(f"Final results: {result_labels}")
    
    return results

#--------------------------------------------------------------------
# Inference interfaces
#--------------------------------------------------------------------

def inference_v2(path: str, model_dir: str = None, pdf_name: str = "vuds.pdf", txt_name: str = "vuds.txt", keep_output_folder: bool = True) -> Tuple[str, str]:
    # Main inference function that extracts templates, predicts, and creates output files
    path = os.path.abspath(path)
    model_dir = os.path.abspath(model_dir or os.path.join(path, "models"))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    jpgs = glob.glob(os.path.join(path, "*.jpg"))
    if not jpgs:
        raise ValueError(f"No jpg images in path: {path}")
    if keep_output_folder:
        extract_base = path
    else:
        extract_base = tempfile.mkdtemp(prefix="vuds_tmp_")
        print(f"Temporary folder for extraction: {extract_base}")
    
    # Setup output directories
    output_dirs = {
        'pftg': os.path.join(extract_base, 'output', 'pftg'),
        'pfus': os.path.join(extract_base, 'output', 'pfus'),
        'xray': os.path.join(extract_base, 'output', 'xray')
    }
    
    # Create output directories if they don't exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup OCR and process images
    from extract_templates import setup_vuds_extractor, process_image
    ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg = setup_vuds_extractor("template_config.json", extract_base)
    
    # Process each image
    for jpg in jpgs:
        process_image(jpg, ocr, templates, summary_cfg, labels, units, dirs, pattern, fluoro_cfg)
    
    # Get results
    results = _predict_from_extracted(output_dirs, model_dir)
    
    # Collect images for PDF
    images_dict = _collect_images_by_type(output_dirs)
    
    # Generate PDF if there are any images
    pdf_path = os.path.join(path, pdf_name)
    if any(images_dict.values()):
        make_a4_pdf(images_dict, pdf_path)
    else:
        print("No extracted images, only TXT output")
    
    # Write results to text file
    txt_path = os.path.join(path, txt_name)
    write_results_to_file(results, txt_path)
    print(f"Text results saved to: {txt_path}")
    
    # Cleanup if needed
    if not keep_output_folder and os.path.isdir(extract_base):
        shutil.rmtree(extract_base)
        print(f"Removed temporary folder: {extract_base}")
    
    return pdf_path, txt_path

#--------------------------------------------------------------------
# Command-line entry point
#--------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for VUDS-AI")
    parser.add_argument("--model_dir", type=str, default="../../models", help="Directory containing model files")
    parser.add_argument("--dir_path", type=str, default="../../test_case", help="Path to a directory of images")
    return parser.parse_args()


def main():
    args = parse_args()
    inference_v2(path=args.dir_path, model_dir=args.model_dir, pdf_name="vuds.pdf", txt_name="vuds.txt", keep_output_folder=True)

if __name__ == '__main__':
    main()
