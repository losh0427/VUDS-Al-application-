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
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    pattern = re.compile(r".*\d+\s*(?:record|single\s*capture)$", re.IGNORECASE)
    return ocr, pattern

def extract_image_regions(img_path, ocr, pattern):
    """Extract different regions (PFUS, PFTG, X-ray) from the image"""
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)
    
    # Run OCR on full image
    results = ocr.ocr(np_img, cls=True)
    
    # Initialize regions dictionary
    regions = {
        'pftg': [],
        'pfus': [],
        'xray': []
    }
    
    # Extract text blocks
    text_blocks = []
    for line in results:
        for box, (txt, _) in line:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            text_blocks.append(lp.TextBlock(
                lp.Rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                text=txt
            ))
    
    # Process each text block to identify regions
    for blk in text_blocks:
        text_lower = blk.text.lower()
        
        # Identify PFUS
        if "pressure flow uroflow segment" in text_lower:
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            regions['pfus'].append(np_img[ty1:ty2, tx1:tx2])
            
        # Identify PFTG
        elif "pressure flow test graph" in text_lower:
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            regions['pftg'].append(np_img[ty1:ty2, tx1:tx2])
            
        # Identify X-ray regions
        elif pattern.match(text_lower.strip()):
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            # Add some padding for X-ray regions
            pad = 50
            regions['xray'].append(np_img[max(0, ty1-pad):min(np_img.shape[0], ty2+pad), 
                                         max(0, tx1-pad):min(np_img.shape[1], tx2+pad)])
    
    return regions

def load_model(task, model_path):
    """Load model for specific task"""
    # Extract full task name from model filename (e.g., 'pftg_Detrusor_instability_resnet18.pth' -> 'Detrusor_instability')
    filename = os.path.basename(model_path)
    task_name = '_'.join(filename.split('_')[1:-1])  # Get all parts between first and last underscore
    
    if task_name not in NUM_CLASSES[task]:
        raise ValueError(f"Unknown task name '{task_name}' for task '{task}'. Available tasks: {list(NUM_CLASSES[task].keys())}")
    
    num_classes = NUM_CLASSES[task][task_name]
    
    if task in ['pftg', 'pfus']:
        model = models.resnet18(pretrained=False)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    else:  # xray
        model = models.densenet121(pretrained=False)
        in_feat = model.classifier.in_features
        model.classifier = nn.Linear(in_feat, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, task_name

def predict_image(model, img_array):
    """Predict on a single image array"""
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

def process_directory(dir_path, model_dir, ocr, pattern):
    """Process all images in a directory and combine their results"""
    # Initialize results with all possible tasks
    combined_results = {
        'Detrusor_instability': 'NA',
        'Flow_pattern': 'NA',
        'EMG_ES_relaxation': 'NA',
        'Trabeculation': 'NA',
        'Diverticulum': 'NA',
        'Cystocele': 'NA',
        'VUR': 'NA',
        'Bladder_neck_relaxation': 'NA',
        'External_sphincter_relaxation': 'NA',
        'Pelvic_floor_relaxation': 'NA'
    }
    
    # Get all image files in directory
    jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
    if not jpg_files:
        raise ValueError(f"No .jpg files found in {dir_path}")
    
    # Process each image
    for img_path in jpg_files:
        try:
            # Extract regions from image
            regions = extract_image_regions(img_path, ocr, pattern)
            
            # Process each region type
            for task in ['pftg', 'pfus', 'xray']:
                if not regions[task]:  # Skip if no regions found for this task
                    continue
                    
                # Get model files for this task
                model_files = [f for f in os.listdir(model_dir) 
                             if f.startswith(task) and f.endswith('.pth')]
                
                for model_file in model_files:
                    model_path = os.path.join(model_dir, model_file)
                    model, task_name = load_model(task, model_path)
                    
                    # For each region of this type, get prediction
                    for i, region in enumerate(regions[task]):
                        pred = predict_image(model, region)
                        if task == 'xray':
                            # For xray, we might have multiple regions
                            # Take the last prediction for each task
                            combined_results[task_name] = LABEL_MAP[task][task_name][pred]
                        else:
                            combined_results[task_name] = LABEL_MAP[task][task_name][pred]
            
            print(f"✅ Processed {os.path.basename(img_path)}")
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    return combined_results

def process_single_image(img_path, model_dir, ocr, pattern):
    """Process a single image and return predictions"""
    # Extract regions from image
    regions = extract_image_regions(img_path, ocr, pattern)
    
    # Initialize results with NA for all tasks
    results = {
        'Detrusor_instability': 'NA',
        'Flow_pattern': 'NA',
        'EMG_ES_relaxation': 'NA',
        'Trabeculation': 'NA',
        'Diverticulum': 'NA',
        'Cystocele': 'NA',
        'VUR': 'NA',
        'Bladder_neck_relaxation': 'NA',
        'External_sphincter_relaxation': 'NA',
        'Pelvic_floor_relaxation': 'NA'
    }
    
    # Process each region type
    for task in ['pftg', 'pfus', 'xray']:
        if not regions[task]:  # Skip if no regions found for this task
            continue
            
        # Get model files for this task
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith(task) and f.endswith('.pth')]
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            model, task_name = load_model(task, model_path)
            
            # For each region of this type, get prediction
            for i, region in enumerate(regions[task]):
                pred = predict_image(model, region)
                if task == 'xray':
                    # For xray, we might have multiple regions
                    # Take the last prediction for each task
                    results[task_name] = LABEL_MAP[task][task_name][pred]
                else:
                    results[task_name] = LABEL_MAP[task][task_name][pred]
    
    return results

def main():
    args = parse_args()
    ocr, pattern = setup_image_extractor()
    
    if args.img:
        # Single image mode
        results = process_single_image(args.img, args.model_dir, ocr, pattern)
    elif args.dir:
        # Directory mode - process all images
        results = process_directory(args.dir, args.model_dir, ocr, pattern)
    else:
        raise ValueError("Provide either --img or --dir")
    
    # Write output.txt
    with open(args.output, 'w') as f:
        for task_name, label in results.items():
            f.write(f"{task_name}: {label}\n")
    
    print(f"Results written to {args.output}")

if __name__ == '__main__':
    main()
    
