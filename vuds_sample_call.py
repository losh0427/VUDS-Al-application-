# sample_call.py
import requests

# API address
url = "http://localhost:8000/infer"
path = "vuds_test_case"
debug_folde_flag = False
# Parameters to send to the API
payload = {
    "path": "/test/" + path,        # The folder you want to infer
    "model_dir": "../../models",      # (Optional, default is path/models)
    "keep_output_folder": debug_folde_flag            # If you want to disable intermediate folder output
}

# POST JSON
resp = requests.post(url, json=payload)

if resp.status_code == 200:
    data = resp.json()
    if data["ok"]:
        print("Inference with success")
        print("PDF:", data["pdf_path"])
        print("TXT:", data["txt_path"])
    else:
        print("Inference failed:", data["message"])
else:
    print(f"HTTP {resp.status_code}:", resp.text)
