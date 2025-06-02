#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import os
from typing import Dict, Any, Optional

# API base URL
BASE_URL = "http://localhost:8000"

def call_inference(path: str, model_dir: Optional[str] = None, keep_output_folder: bool = True) -> Dict[str, Any]:
    """Call inference API"""
    url = f"{BASE_URL}/infer"
    payload = {
        "path": path,
        "model_dir": model_dir,
        "keep_output_folder": keep_output_folder
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Inference failed: {data['message']}")
    
    return data

def call_extract(root_dir: str = "../../raw_dataset", log_path: str = "../../extraction_log.json") -> Dict[str, Any]:
    """Call template extraction API"""
    url = f"{BASE_URL}/extract"
    payload = {
        "root_dir": root_dir,
        "log_path": log_path
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Extraction failed: {data['message']}")
    
    return data

def call_process(root_dir: str = "../../raw_dataset", log_path: str = "../../report_extraction_log.json") -> Dict[str, Any]:
    """Call report processing API"""
    url = f"{BASE_URL}/process"
    payload = {
        "root_dir": root_dir,
        "log_path": log_path
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Processing failed: {data['message']}")
    
    return data

def call_verify(raw_dir: str = "../../raw_dataset", out_dir: str = "../../", dry_run: bool = False) -> Dict[str, Any]:
    """Call report verification API"""
    url = f"{BASE_URL}/verify"
    payload = {
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "dry_run": dry_run
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Verification failed: {data['message']}")
    
    return data

def call_build_dataset(valid_json_path: str = "../../valid_reports.json", processed_dir_path: str = "../../processed_dataset", dry_run: bool = False) -> Dict[str, Any]:
    """Call dataset building API"""
    url = f"{BASE_URL}/build_dataset"
    payload = {
        "valid_json_path": valid_json_path,
        "processed_dir_path": processed_dir_path,
        "dry_run": dry_run
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Dataset building failed: {data['message']}")
    
    return data

def call_analyze(processed_dir_path: str = "../../processed_dataset") -> Dict[str, Any]:
    """Call label analysis API"""
    url = f"{BASE_URL}/analyze"
    payload = {
        "processed_dir_path": processed_dir_path
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Analysis failed: {data['message']}")
    
    return data

def call_train(
    model_type: str,
    data_dir: str = "../../processed_dataset",
    backbone: str = "resnet18",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 0.001,
    output_dir: str = "../../models"
) -> Dict[str, Any]:
    """Call model training API"""
    url = f"{BASE_URL}/train"
    payload = {
        "data_dir": data_dir,
        "model_type": model_type,
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "output_dir": output_dir
    }
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    
    data = resp.json()
    if not data["ok"]:
        raise Exception(f"Training failed: {data['message']}")
    
    return data

def main():
    """Main function: Execute all API calls in sequence"""
    try:
        # 1. Extract templates
        print("\n=== Starting Template Extraction ===")
        extract_result = call_extract()
        print("Template extraction successful")
        print(f"Output path: {extract_result['output_path']}")

        # 2. Process reports
        print("\n=== Starting Report Processing ===")
        process_result = call_process()
        print("Report processing successful")
        print(f"Output path: {process_result['output_path']}")

        # 3. Verify reports
        print("\n=== Starting Report Verification ===")
        verify_result = call_verify()
        print("Report verification successful")
        print(f"Valid reports: {verify_result['valid_count']}")
        print(f"Invalid reports: {verify_result['invalid_count']}")

        # 4. Build dataset
        print("\n=== Starting Dataset Building ===")
        build_result = call_build_dataset()
        print("Dataset building successful")
        print(f"Output path: {build_result['output_path']}")

        # 5. Analyze labels
        print("\n=== Starting Label Analysis ===")
        analyze_result = call_analyze()
        print("Label analysis successful")
        if analyze_result.get('data_counts'):
            print("Data counts:", analyze_result['data_counts'])
        if analyze_result.get('label_distributions'):
            print("Label distributions:", analyze_result['label_distributions'])

        # 6. Train models
        print("\n=== Starting Model Training ===")
        
        # Train PFTG model
        print("\n--- Training PFTG Model ---")
        pftg_result = call_train(
            model_type="pftg"
        )
        print("PFTG model training successful")
        print(f"Model path: {pftg_result['model_path']}")
        
        # Train PFUS model
        print("\n--- Training PFUS Model ---")
        pfus_result = call_train(
            data_dir="../../processed_dataset",
            model_type="pfus"
        )
        print("PFUS model training successful")
        print(f"Model path: {pfus_result['model_path']}")
        
        # Train XRAY model
        print("\n--- Training XRAY Model ---")
        xray_result = call_train(
            data_dir="../../processed_dataset",
            model_type="xray"
        )
        print("XRAY model training successful")
        print(f"Model path: {xray_result['model_path']}")

        # 7. Run inference
        print("\n=== Starting Inference ===")
        infer_result = call_inference(
            # path="/test/vuds_test_case",
            path="../../test_case",
            model_dir="../../models",
            keep_output_folder=False
        )
        print("Inference successful")
        print(f"PDF path: {infer_result['pdf_path']}")
        print(f"TXT path: {infer_result['txt_path']}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return

    print("\n=== All processing completed ===")

if __name__ == "__main__":
    main()
