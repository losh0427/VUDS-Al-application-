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
    
    # Run OCR on full image
    print("Running OCR...")
    results = ocr.ocr(np_img, cls=True)
    
    # Initialize regions dictionary
    regions = {
        'pftg': {'exists': False, 'regions': []},
        'pfus': {'exists': False, 'regions': []},
        'xray': {'exists': False, 'regions': []}
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
            regions['pfus']['exists'] = True
            regions['pfus']['regions'].append(np_img[ty1:ty2, tx1:tx2])
            print(f"Found PFUS region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
            
        # Identify PFTG
        elif "pressure flow test graph" in text_lower:
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            regions['pftg']['exists'] = True
            regions['pftg']['regions'].append(np_img[ty1:ty2, tx1:tx2])
            print(f"Found PFTG region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
            
        # Identify X-ray regions
        elif pattern.match(text_lower.strip()):
            tx1, ty1, tx2, ty2 = map(int, blk.coordinates)
            pad = 50
            regions['xray']['exists'] = True
            regions['xray']['regions'].append(
                np_img[max(0, ty1-pad):min(np_img.shape[0], ty2+pad), 
                      max(0, tx1-pad):min(np_img.shape[1], tx2+pad)]
            )
            print(f"Found X-ray region at coordinates: ({tx1}, {ty1}, {tx2}, {ty2})")
    
    # Print summary of found regions
    print("\n📊 Region Detection Summary:")
    print(f"PFUS regions found: {len(regions['pfus']['regions'])}")
    print(f"PFTG regions found: {len(regions['pftg']['regions'])}")
    print(f"X-ray regions found: {len(regions['xray']['regions'])}")
    
    return regions

def load_model(task, model_path):
    """Load model for specific task"""
    # Extract full task name from model filename (e.g., 'pftg_Detrusor_instability_resnet18.pth' -> 'Detrusor_instability')
    filename = os.path.basename(model_path)
    task_name = '_'.join(filename.split('_')[1:-1])  # Get all parts between first and last underscore
    
    if task_name not in NUM_CLASSES[task]:
        raise ValueError(f"Unknown task name '{task_name}' for task '{task}'. Available tasks: {list(NUM_CLASSES[task].keys())}")
    
    num_classes = NUM_CLASSES[task][task_name]
    
    # Use ResNet18 for all tasks
    model = models.resnet18(pretrained=False)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    
    try:
        # Load state dict with strict=False to handle potential architecture differences
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded model weights for {task_name}")
    except Exception as e:
        print(f"Warning: Error loading model weights for {task_name}: {str(e)}")
        print("Attempting to load with modified state dict...")
        try:
            # Try to load only the fc layer weights
            if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
                model.fc.weight.data = state_dict['fc.weight']
                model.fc.bias.data = state_dict['fc.bias']
                print("Successfully loaded only the classifier weights")
            else:
                raise ValueError("No classifier weights found in state dict")
        except Exception as e2:
            print(f"Error loading classifier weights: {str(e2)}")
            raise
    
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

def get_default_results():
    """Get default results dictionary with all tasks"""
    return {
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

def write_results_to_file(results, output_path):
    """Write results to file in the specified format"""
    with open(output_path, 'w') as f:
        # PFTG section
        f.write("Detrusor_instability: {}\n\n".format(results['Detrusor_instability']))
        
        # PFUS section
        f.write("Pressure flow study\n")
        f.write("Flow_pattern: {}\n".format(results['Flow_pattern']))
        f.write("EMG_ES_relaxation: {}\n\n".format(results['EMG_ES_relaxation']))
        
        # X-ray section
        f.write("Fluroscopy in VUDS\n")
        f.write("Trabeculation: {}\n".format(results['Trabeculation']))
        f.write("Diverticulum: {}\n".format(results['Diverticulum']))
        f.write("Cystocele: {}\n".format(results['Cystocele']))
        f.write("VUR: {}\n".format(results['VUR']))
        f.write("Bladder_neck_relaxation: {}\n".format(results['Bladder_neck_relaxation']))
        f.write("External_sphincter_relaxation: {}\n".format(results['External_sphincter_relaxation']))
        f.write("Pelvic_floor_relaxation: {}\n".format(results['Pelvic_floor_relaxation']))

def predict_all_tasks(regions, model_dir):
    """Predict all tasks based on existing regions"""
    results = get_default_results()
    print("\n🤖 Starting prediction for all tasks...")
    
    # Process each task type
    for task in ['pftg', 'pfus', 'xray']:
        if not regions[task]['exists']:
            print(f"\nNo {task.upper()} regions found, skipping predictions")
            continue
            
        print(f"\n📝 Processing {task.upper()} predictions:")
        # Get model files for this task
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith(task) and f.endswith('.pth')]
        
        for model_file in model_files:
            try:
                print(f"\nLoading model: {model_file}")
                model_path = os.path.join(model_dir, model_file)
                model, task_name = load_model(task, model_path)
                
                # For each region of this type, get prediction
                for i, region in enumerate(regions[task]['regions']):
                    print(f"Predicting region {i+1}/{len(regions[task]['regions'])} for {task_name}")
                    pred = predict_image(model, region)
                    results[task_name] = LABEL_MAP[task][task_name][pred]
                    print(f"Prediction result: {LABEL_MAP[task][task_name][pred]}")
            except Exception as e:
                print(f"❌ Error processing {model_file}: {str(e)}")
                continue
    
    print("\n📊 Final Prediction Results:")
    for task, value in results.items():
        print(f"{task}: {value}")
    
    return results

def process_images(img_paths, model_dir, ocr, pattern):
    """Process multiple images and combine results"""
    all_results = get_default_results()
    print(f"\n🔄 Processing {len(img_paths)} images...")
    
    for img_path in img_paths:
        try:
            print(f"\n{'='*50}")
            print(f"Processing image: {os.path.basename(img_path)}")
            print(f"{'='*50}")
            
            # Extract regions
            regions = extract_all_regions(img_path, ocr, pattern)
            # Get predictions
            results = predict_all_tasks(regions, model_dir)
            # Update results (take the last non-NA value)
            for key, value in results.items():
                if value != 'NA':
                    all_results[key] = value
            print(f"\n✅ Successfully processed {os.path.basename(img_path)}")
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {str(e)}")
    
    print("\n📊 Combined Results from All Images:")
    for task, value in all_results.items():
        print(f"{task}: {value}")
    
    return all_results

def main():
    args = parse_args()
    ocr, pattern = setup_image_extractor()
    
    if args.img:
        # Single image mode
        results = process_images([args.img], args.model_dir, ocr, pattern)
    elif args.dir:
        # Directory mode
        jpg_files = glob.glob(os.path.join(args.dir, "*.jpg"))
        if not jpg_files:
            raise ValueError(f"No .jpg files found in {args.dir}")
        results = process_images(jpg_files, args.model_dir, ocr, pattern)
    else:
        raise ValueError("Provide either --img or --dir")
    
    # Write output.txt in the specified format
    write_results_to_file(results, args.output)
    print(f"Results written to {args.output}")

if __name__ == '__main__':
    main()
    
