# sample_call.py
import requests

# API 位址
url = "http://localhost:8000/infer"

# 要傳給 API 的參數
payload = {
    "path": "../../test_case",        # 你要推論的資料夾
    "model_dir": "../../models",      # （可省略，預設 path/models）
    "keep_output_folder": True            # 如要關閉中繼資料夾輸出
}

# POST JSON
resp = requests.post(url, json=payload)

if resp.status_code == 200:
    data = resp.json()
    if data["ok"]:
        print("✅ Inference 成功")
        print("PDF:", data["pdf_path"])
        print("TXT:", data["txt_path"])
    else:
        print("❌ 失敗:", data["message"])
else:
    print(f"HTTP {resp.status_code}:", resp.text)
