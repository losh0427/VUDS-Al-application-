import os
import glob
import cv2
import torch
import layoutparser as lp
from layoutparser.models.detectron2 import Detectron2LayoutModel

input_dir = "test/"
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

model = Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    device=device,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],
)

for filepath in glob.glob(os.path.join(input_dir, "*.jpg")):
    print(f"\n🔄 Processing: {filepath}")
    image = cv2.imread(filepath)
    if image is None:
        print("⚠️ 讀圖失敗，請檢查路徑或檔案格式")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    layout = model.detect(image_rgb)
    print(f"👉 偵測到 {len(layout)} 個區塊：")
    image_with_box = image.copy()
    fname_base = os.path.splitext(os.path.basename(filepath))[0]

    for i, block in enumerate(layout):
        x1, y1, x2, y2 = map(int, block.coordinates)
        label = block.type
        print(f"  [{i}] {label} @ {block.coordinates}")

        # 畫框在整體圖片上
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_with_box, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 裁切出區塊儲存
        crop = image[y1:y2, x1:x2]
        crop_name = f"{fname_base}_block{i}_{label}.jpg"
        crop_path = os.path.join(output_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        print(f"   ⤷ 已儲存裁切圖：{crop_path}")

    # 儲存標框後的原圖
    boxed_path = os.path.join(output_dir, f"{fname_base}_boxed.jpg")
    cv2.imwrite(boxed_path, image_with_box)
    print(f"✅ 儲存標框圖像：{boxed_path}")

print("\n🏁 全部完成，請查看 output/ 資料夾")
