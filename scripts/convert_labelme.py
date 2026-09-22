import os
import json
import base64
import io
import numpy as np
import PIL.Image
import labelme

# === 路徑設定 ===
import argparse
parser = argparse.ArgumentParser(description="Convert LabelMe JSON to grayscale class-ID masks.")
parser.add_argument("--input", required=True, help="Folder containing trusted LabelMe JSON and images")
parser.add_argument("--output", required=True, help="Output folder; matching masks are overwritten")
args = parser.parse_args()
input_dir = args.input
output_dir = args.output

os.makedirs(output_dir, exist_ok=True)

# === 固定「名稱 → 類別ID」 (請依你的類別維護) ===
name_to_id = {
    'sky': 0, 'background': 0, 'background_sky': 0,
    'blade': 1,
    'tower': 2, 'pillar': 2,
    'crack': 3,
    'floor': 4, 'ground': 4,
    'fence': 5,
}

# === 類別ID → 顏色 (僅供可視化；訓練吃的是灰階ID) ===
label_colors = {
    0: (  0,   0,   0),   # sky/background
    1: (255,   0,   0),   # blade
    2: (  0,   0, 255),   # pillar/tower
    3: (  0, 255,   0),   # crack
    4: (255, 255,   0),   # floor/ground
    5: (255,   0, 255),   # fence
}

def normalize(name: str) -> str:
    return name.strip().lower()

def decode_image(imageData):
    if isinstance(imageData, str):
        imageData = base64.b64decode(imageData)
    return np.array(PIL.Image.open(io.BytesIO(imageData)))

def apply_color_map(gray):
    gray = np.array(gray)
    rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for idx, color in label_colors.items():
        rgb[gray == idx] = color
    return PIL.Image.fromarray(rgb)

def shapes_to_fixed_label(img_shape, shapes):
    """使用固定 name_to_id 產生灰階label圖；未知 label 會中止轉換"""
    # 正規化label
    fixed = []
    for s in shapes:
        s = dict(s)  # copy
        s['label'] = normalize(s['label'])
        if s['label'] not in name_to_id:
            print(f"[WARN] 未知標籤: {s['label']} -> 以 'background'(0) 處理")
            s['label'] = 'background'
        fixed.append(s)

    # 準備 LabelMe 需要的 mapping
    label_name_to_value = {'_background_': 0}
    for k, v in name_to_id.items():
        label_name_to_value[k] = v

    lbl, _ = labelme.utils.shapes_to_label(
        img_shape=img_shape, shapes=fixed, label_name_to_value=label_name_to_value
    )
    return lbl

def check_match(json_labels, mask_ids):
    expected_ids = set(name_to_id.get(normalize(x), -1) for x in json_labels)
    missing_in_mask = expected_ids - mask_ids
    extra_in_mask = mask_ids - expected_ids
    return expected_ids, missing_in_mask, extra_in_mask

# === 主流程 ===
for file in os.listdir(input_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(input_dir, file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 確保有 imageData
    if data.get("imageData") in [None, ""]:
        img_path = os.path.join(input_dir, data["imagePath"])
        with open(img_path, "rb") as img_f:
            data["imageData"] = base64.b64encode(img_f.read()).decode("utf-8")

    # 解析影像與標註
    label_file = labelme.LabelFile(filename=json_path)
    img = decode_image(label_file.imageData)

    # 產生固定mapping的灰階label
    lbl = shapes_to_fixed_label(img.shape, data["shapes"])

    # 輸出灰階
    gray_path = os.path.join(output_dir, file.replace(".json", ".png"))
    PIL.Image.fromarray(lbl.astype(np.uint8)).save(gray_path)

    # 輸出彩色可視化
#    color_img = apply_color_map(PIL.Image.fromarray(lbl.astype(np.uint8)))
#    color_path = os.path.join(output_dir, file.replace(".json", "_label_colored.png"))
#    color_img.save(color_path)

    # 取得 JSON 出現過的 labels（原始大小寫也抓，但比對時會normalize）
    json_labels = {s['label'] for s in data['shapes']}
    mask_ids = set(np.unique(lbl).tolist())

    expected_ids, miss, extra = check_match(json_labels, mask_ids)

    print(f"\n檔案 {file}:")
    print(f"  JSON 標註: {set(normalize(x) for x in json_labels)}")
    print(f"  期望 ID : {expected_ids}")
    print(f"  Mask ID  : {mask_ids}")
    if miss:
        print(f"  ⚠ Mask 沒有出現的 ID: {miss}")
    if extra:
        print(f"  ⚠ Mask 多出來的 ID : {extra}")
    if not miss and not extra:
        print("  ✅ JSON 與 mask 完全 match")

print("\n完成。若先前輸出的 mask 有誤，請刪除後用本程式重新產生。")
