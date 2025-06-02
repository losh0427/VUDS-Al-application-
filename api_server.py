from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import traceback
import gc
import torch

# Import the refactored inference module
from inference import inference_v2

# Initialize FastAPI application
app = FastAPI(title="VUDS-AI Inference API")

# Request schema for inference API
class InferRequest(BaseModel):
    path: str                     # Path to input folder for inference
    model_dir: Optional[str] = None  # Optional model directory override
    keep_output_folder: bool = True  # Whether to keep extraction folder

# Response schema for inference API
class InferResponse(BaseModel):
    ok: bool
    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    message: str

# Define API endpoint for inference
@app.post("/infer", response_model=InferResponse)
def run_inference(req: InferRequest):
    # Validate input path existence
    abs_path = os.path.abspath(req.path)
    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
    try:
        # Call inference function
        pdf, txt = inference_v2(
            path=abs_path,
            model_dir=req.model_dir,
            keep_output_folder=req.keep_output_folder,
        )
        # Clear GPU cache and collect garbage to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return InferResponse(
            ok=True,
            pdf_path=pdf,
            txt_path=txt,
            message="Inference completed successfully"
        )
    except Exception as e:
        tb = traceback.format_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return InferResponse(
            ok=False,
            message=f"Error: {e}\n{tb}"
        )

# Import additional modules
from extract_templates import batch_process_reports as extract_templates_function
from process_report import batch_process_reports as process_reports_function
from verify_reports import verify_reports as verify_reports_function
from analyze_labels import analyze_processed_data as analyze_labels_function
from training_process import run_training as training_process_function
from build_dataset import build_dataset as build_dataset_function

# Request schemas for template extraction
class ExtractRequest(BaseModel):
    root_dir: str = "raw_dataset"
    log_path: str = "extraction_log.json"

class ExtractResponse(BaseModel):
    ok: bool
    message: str
    output_path: Optional[str] = None

# Request schemas for report processing
class ProcessRequest(BaseModel):
    root_dir: str = "raw_dataset"
    log_path: str = "report_extraction_log.json"

class ProcessResponse(BaseModel):
    ok: bool
    message: str
    output_path: Optional[str] = None
    extracted_labels: Optional[dict] = None

# Request schemas for report verification
class VerifyRequest(BaseModel):
    raw_dir: str = "raw_dataset"
    out_dir: str = "../../"
    dry_run: bool = False

class VerifyResponse(BaseModel):
    ok: bool
    message: str
    valid_count: int
    invalid_count: int
    valid_reports: Optional[list] = None
    invalid_reports: Optional[list] = None

# Request schemas for label analysis
class AnalyzeRequest(BaseModel):
    processed_dir_path: str = "../../processed_dataset"

class AnalyzeResponse(BaseModel):
    ok: bool
    message: str
    data_counts: Optional[dict] = None
    label_distributions: Optional[dict] = None
    plot_paths: Optional[dict] = None

# Request schemas for model training
class TrainRequest(BaseModel):
    data_dir: str = "../../processed_dataset"
    model_type: str
    backbone: str = "resnet18"
    epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    output_dir: str = "../../models"

class TrainResponse(BaseModel):
    ok: bool
    message: str
    model_path: Optional[str] = None
    training_stats: Optional[dict] = None

# Request schemas for dataset building
class BuildDatasetRequest(BaseModel):
    valid_json_path: str = "../../valid_reports.json"
    processed_dir_path: str = "../../processed_dataset"
    dry_run: bool = False

class BuildDatasetResponse(BaseModel):
    ok: bool
    message: str
    total_samples: Optional[int] = None
    output_path: Optional[str] = None

# API endpoint for template extraction
@app.post("/extract", response_model=ExtractResponse)
def extract_templates(req: ExtractRequest):
    try:
        # Validate input path existence
        abs_path = os.path.abspath(req.root_dir)
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
        
        # Run template extraction
        extract_templates_function(root_dir=req.root_dir, log_path=req.log_path)
        return ExtractResponse(
            ok=True,
            message="Template extraction completed successfully",
            output_path=req.root_dir
        )
    except Exception as e:
        tb = traceback.format_exc()
        return ExtractResponse(ok=False, message=f"Error: {e}\n{tb}")

# API endpoint for report processing
@app.post("/process", response_model=ProcessResponse)
def process_reports(req: ProcessRequest):
    try:
        # Validate input path existence
        abs_path = os.path.abspath(req.root_dir)
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
        
        # Run report processing
        process_reports_function(root_dir=req.root_dir, log_path=req.log_path)
        return ProcessResponse(
            ok=True,
            message="Report processing completed successfully",
            output_path=req.root_dir
        )
    except Exception as e:
        tb = traceback.format_exc()
        return ProcessResponse(ok=False, message=f"Error: {e}\n{tb}")

# API endpoint for report verification
@app.post("/verify", response_model=VerifyResponse)
def verify_reports(req: VerifyRequest):
    try:
        # Validate input path existence
        abs_path = os.path.abspath(req.raw_dir)
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
        
        # Run report verification
        valid, invalid = verify_reports_function(raw_dir=req.raw_dir, out_dir=req.out_dir, dry_run=req.dry_run)
        return VerifyResponse(
            ok=True,
            message="Report verification completed successfully",
            valid_count=len(valid),
            invalid_count=len(invalid),
            valid_reports=valid,
            invalid_reports=invalid
        )
    except Exception as e:
        tb = traceback.format_exc()
        return VerifyResponse(
            ok=False,
            message=f"Error: {e}\n{tb}",
            valid_count=0,
            invalid_count=0
        )

# API endpoint for label analysis
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_labels(req: AnalyzeRequest):
    try:
        # Validate input path existence
        abs_path = os.path.abspath(req.processed_dir_path)
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
        
        # Run label analysis
        analyze_labels_function(processed_dir_path=req.processed_dir_path)
        return AnalyzeResponse(
            ok=True,
            message="Label analysis completed successfully"
        )
    except Exception as e:
        tb = traceback.format_exc()
        return AnalyzeResponse(ok=False, message=f"Error: {e}\n{tb}")

# API endpoint for model training
@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    try:
        # Validate input path existence
        abs_path = os.path.abspath(req.data_dir)
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Path not found: {abs_path}")
        
        # Run model training
        training_process_function(
            data_dir=req.data_dir,
            model_type=req.model_type,
            backbone=req.backbone,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            output_dir=req.output_dir
        )
        # Clear GPU cache and collect garbage after training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResponse(
            ok=True,
            message="Model training completed successfully",
            model_path=os.path.join(req.output_dir, f"{req.model_type}_{req.backbone}.pth")
        )
    except Exception as e:
        tb = traceback.format_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResponse(ok=False, message=f"Error: {e}\n{tb}")

# API endpoint for dataset building
@app.post("/build_dataset", response_model=BuildDatasetResponse)
def build_dataset(req: BuildDatasetRequest):
    try:
        # Validate input path existence
        valid_json_path = os.path.abspath(req.valid_json_path)
        if not os.path.isfile(valid_json_path):
            raise HTTPException(status_code=400, detail=f"Valid reports JSON not found: {valid_json_path}")
        
        processed_dir_path = os.path.abspath(req.processed_dir_path)
        if not os.path.isdir(processed_dir_path):
            os.makedirs(processed_dir_path)
        
        # Run dataset building
        build_dataset_function(
            valid_json_path=req.valid_json_path,
            processed_dir_path=req.processed_dir_path,
            dry_run=req.dry_run
        )
        return BuildDatasetResponse(
            ok=True,
            message="Dataset building completed successfully",
            output_path=req.processed_dir_path
        )
    except Exception as e:
        tb = traceback.format_exc()
        return BuildDatasetResponse(ok=False, message=f"Error: {e}\n{tb}")
