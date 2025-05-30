from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, traceback

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
        return InferResponse(ok=True, pdf_path=pdf, txt_path=txt, message="Inference completed successfully")
    except Exception as e:
        # Return error details in response
        tb = traceback.format_exc()
        return InferResponse(ok=False, message=f"Error: {e}\n{tb}")
