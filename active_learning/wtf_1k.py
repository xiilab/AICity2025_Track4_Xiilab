import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
import glob
import re

# === Settings ===
json_paths = [
    '/DATA/jhlee/fisheye1k_merged_vnpt_nota.json',
    '/DATA/jhlee/Weighted-Boxes-Fusion/best_coco_predictions.json'
]
image_dir       = "/DATA/jhlee/dataset/Fisheye1K"
output_json     = "outputs/pseudo_wbf_output_1k_ensemble.json"
output_vis_dir  = "outputs/wbf_visualizations_1k_ensemble"
os.makedirs(output_vis_dir, exist_ok=True)

# Soft-NMS / WBF parameters
iou_thr        = 0.55
skip_box_thr   = 0.1
wbf_weights    = [1, 2]

# Debug mode
DEBUG = False

# Class map
CLASS_MAP = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

# === 1) Parse JSON files & prepare mappings ===
# For each JSON: id→filename map, filename→image_size, filename→image_id
id2fname_list = []
json_image_sizes = {}
filename_to_image_id = {}

for path in json_paths:
    with open(path) as f:
        data = json.load(f)
    # id→filename 맵
    id2fname = { img['id']: img['file_name'] for img in data['images'] }
    id2fname_list.append(id2fname)
    # 이미지 사이즈, image_id 저장
    for img in data['images']:
        fn = img['file_name']
        if fn not in json_image_sizes:
            json_image_sizes[fn] = (img.get('width', 0), img.get('height', 0))
        # 첫 등장하는 id를 저장
        filename_to_image_id.setdefault(fn, img['id'])

# === 2) Image size cache ===
image_wh_map = {}
def get_image_wh(filename):
    if filename in image_wh_map:
        return image_wh_map[filename]
    path = os.path.join(image_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    h, w = img.shape[:2]
    # Compare with JSON metadata
    jw, jh = json_image_sizes.get(filename, (0,0))
    if jw and jh and (jw != w or jh != h):
        print(f"⚠️ 크기 불일치: {filename} JSON={jw}x{jh} 실제={w}x{h}")
    image_wh_map[filename] = (w, h)
    return w, h

# === 3) Collect per-file model results ===
image_annots = defaultdict(lambda: [[] for _ in json_paths])
for idx, path in enumerate(json_paths):
    with open(path) as f:
        data = json.load(f)
    for ann in data['annotations']:
        img_id = ann['image_id']
        fn     = id2fname_list[idx].get(img_id)
        if fn is None:
            continue
        image_annots[fn][idx].append({
            "bbox":  ann["bbox"],
            "score": ann["score"],
            "label": ann["category_id"]
        })

# === 4) bbox conversion helpers ===
def coco_to_norm(box, w, h):
    x,y,ww,hh = box
    return [x/w, y/h, (x+ww)/w, (y+hh)/h]

def norm_to_coco(box, w, h):
    x1,y1,x2,y2 = box
    x = x1*w; y = y1*h
    return [x, y, (x2-x1)*w, (y2-y1)*h]

# === 5) Extract camera number ===
def extract_camera_number(filename):
    m = re.search(r'camera(\d+)', filename)
    return int(m.group(1)) if m else None

# === 6) Visualization ===
def draw_boxes(img, bboxes, labels, scores):
    for box, label, score in zip(bboxes, labels, scores):
        x,y,w,h = [int(v) for v in box]
        color = (0,255,0)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        txt = f"{CLASS_MAP.get(label,label)}:{score:.2f}"
        tw, th = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)[0]
        cv2.rectangle(img, (x,y-th-4),(x+tw+4,y), color, -1)
        cv2.putText(img, txt, (x+2,y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img

# === 7) Ensemble + save loop ===
results = []
for fn, models in tqdm(image_annots.items(), desc="Images"):
    w, h = get_image_wh(fn)
    if DEBUG:
        print(f"\n▶ Processing {fn}: sizes w={w},h={h}")
        print("  per-model annots:", [len(a) for a in models])

    # 모델별 리스트 준비
    boxes_list, scores_list, labels_list = [], [], []
    for mi, annots in enumerate(models):
        if annots:
            b_norm = [coco_to_norm(a["bbox"], w, h) for a in annots]
            s      = [a["score"] for a in annots]
            lab    = [a["label"] for a in annots]
            if DEBUG and mi==0:
                print("  Model0 첫박스:", annots[0]["bbox"], "→", b_norm[0])
            boxes_list.append(b_norm)
            scores_list.append(s)
            labels_list.append(lab)
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

    # WBF 실행
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=wbf_weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )
    if DEBUG:
        print(f"  → Ensemble: {len(boxes)} boxes")

    # Post-processing & save results
    cam = extract_camera_number(fn)
    vis_boxes, vis_labels, vis_scores = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        # Bus/Truck score 필터링 (특정 카메라 제외)
        if label in (0,4) and score<0.25 and cam not in (20,24,29):
            continue
        # append to COCO results
        coco_box = norm_to_coco(box, w, h)
        results.append({
            "image_id": filename_to_image_id.get(fn,0),
            "category_id": label,
            "bbox": [round(v,4) for v in coco_box],
            "score": round(score,4)
        })
        # for visualization
        vis_boxes.append(coco_box)
        vis_labels.append(label)
        vis_scores.append(score)

    # Save visualization
    if vis_boxes:
        img = cv2.imread(os.path.join(image_dir, fn))
        img = draw_boxes(img, vis_boxes, vis_labels, vis_scores)
        cv2.imwrite(os.path.join(output_vis_dir, fn), img)

# === 8) Write COCO JSON ===
categories = [{"id":i,"name":n} for i,n in CLASS_MAP.items()]
images = [
    {"id":filename_to_image_id[fn], "file_name":fn, "width":w, "height":h}
    for fn,(w,h) in image_wh_map.items()
]
coco_out = {"images":images, "categories":categories, "annotations":results}
with open(output_json, 'w') as f:
    json.dump(coco_out, f, indent=2)

print(f"\n✅ 완료: {len(results)} annotations saved to {output_json}")
print(f"   visualizations → {output_vis_dir}")
