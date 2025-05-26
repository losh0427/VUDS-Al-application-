"""
啟動方式：
  uvicorn api_server:app --host 0.0.0.0 --port 8000
呼叫方式（範例）：
  curl -X POST "http://<ip>:8000/infer" \
       -H "Content-Type: application/json" \
       -d '{"path": "/workspace/case001"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, traceback

# 直接 import 你整合後的 inference 模組
from inference import inference_v2

app = FastAPI(title="VUDS-AI Inference API")

# ---------- Request / Response schema ----------
class InferRequest(BaseModel):
    path: str                     # 要推論的資料夾
    model_dir: Optional[str] = None
    keep_output_folder: bool = True

class InferResponse(BaseModel):
    ok: bool
    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    message: str

# ---------- API ----------
@app.post("/infer", response_model=InferResponse)
def run_inference(req: InferRequest):
    # 1) 檢查路徑是否存在
    abs_path = os.path.abspath(req.path)
    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=400,
                            detail=f"path not found: {abs_path}")

    try:
        # 2) 呼叫 inference_v2
        pdf, txt = inference_v2(
            path=abs_path,
            model_dir=req.model_dir,
            keep_output_folder=req.keep_output_folder,
        )
        return InferResponse(ok=True, pdf_path=pdf, txt_path=txt,
                             message="inference completed")
    except Exception as e:
        tb = traceback.format_exc()
        return InferResponse(ok=False, message=f"{e}\n{tb}")
