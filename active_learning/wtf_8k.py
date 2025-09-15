import os
import json
import cv2
import numpy as np
from tqdm import tqdm
# from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict
import glob
from ensemble_boxes import *

# 1. Path settings
# json_paths = glob.glob("/DATA/jhlee/dataset/detr_annotation/*")
json_paths = ['/DATA/jhlee/dataset/enemble_dir/nota_lvis2_coco_fisheye8k_coco_filtered.json',
              '/DATA/jhlee/dataset/enemble_dir/nota_coco_format_fisheye8k_coco_filtered.json',]
image_dir = "/DATA/jhlee/dataset/Fisheye8K_coco/trainval/images"
output_json = "outputs/pseudo_wbf_output_8k_ensemble111_nms.json"
output_vis_dir = "outputs/wbf_visualizations_8k_ensemble111_nms"
os.makedirs(output_vis_dir, exist_ok=True)

# 2. Collect annotations per image
image_annots = defaultdict(lambda: [[] for _ in json_paths])
image_wh_map = {}
image_id_to_filename = {}  # Map image ID to file name

for idx, path in enumerate(json_paths):
    with open(path) as f:
        data = json.load(f)
        
        # Build mapping from image ID to file name
        for img_info in data['images']:
            img_id = img_info["id"]
            filename = img_info["file_name"]
            image_id_to_filename[img_id] = filename
        
        for ann in data['annotations']:
            img_id = ann["image_id"]
            box = ann["bbox"]  # [x, y, w, h]
            score = ann["score"]
            label = ann["category_id"]

            image_annots[img_id][idx].append({
                "bbox": box,
                "score": score,
                "label": label
            })

# 3. Auto extract image size
def get_image_wh(image_id):
    filename = image_id_to_filename.get(image_id, str(image_id))
    print(f"Processing image: {filename}")
    path = os.path.join(image_dir, filename)
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return img.shape[1], img.shape[0]
    raise FileNotFoundError(f"Image {filename} not found at {path}")

# 4. Normalize bbox
def coco_to_norm(box, width, height):
    x, y, w, h = box
    return [
        x / width,
        y / height,
        (x + w) / width,
        (y + h) / height
    ]

# 5. Denormalize bbox
def norm_to_coco(box, width, height):
    x1, y1, x2, y2 = box
    x = x1 * width
    y = y1 * height
    w = (x2 - x1) * width
    h = (y2 - y1) * height
    return [x, y, w, h]

# 6. Extract camera number
def extract_camera_number(filename):
    import re
    match = re.search(r'camera(\d+)', filename)
    return int(match.group(1)) if match else None

# 6. Visualization
def draw_boxes(img, bboxes, labels, scores):
    class_map = {
        0: "Bus",
        1: "Bike",
        2: "Car",
        3: "Pedestrian",
        4: "Truck"
    }

    for box, label, score in zip(bboxes, labels, scores):
        x, y, w, h = [int(coord) for coord in box]
        color = (0, 255, 0)

        # draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # draw label
        label_text = f"{class_map.get(int(label), str(label))}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size

        # background for text
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w + 4, y), color, -1)

        # text
        cv2.putText(img, label_text, (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img

# 7. Run
results = []
iou_thr = 0.55
skip_box_thr = 0.001
sigma = 0.1
weights = [1,1,1]

for img_id, all_models_annots in tqdm(image_annots.items()):
    print(all_models_annots)
    width, height = get_image_wh(img_id)
    image_wh_map[img_id] = (width, height)

    boxes_list = []
    scores_list = []
    labels_list = []

    for anns in all_models_annots:
        boxes = [coco_to_norm(a["bbox"], width, height) for a in anns]
        scores = [a["score"] for a in anns]
        labels = [a["label"] for a in anns]
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # # WBF 수행
    #     boxes, scores, labels = weighted_boxes_fusion(
    #         boxes_list, scores_list, labels_list,
    #         weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    #     )
   
    
    boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    # Save results
    filename = image_id_to_filename.get(img_id, str(img_id))
    camera_num = extract_camera_number(filename)
    
    for box, score, label in zip(boxes, scores, labels):
        # Filter out scores below 0.33 for all classes
        if score < 0.33:
            continue
            
        # Additional filter: Bus(0)/Truck(4) below 0.25 except cameras 20,24,29
        if int(label) in [0, 4] and score < 0.25 and camera_num not in [20, 24, 29]:
            continue
            
        coco_box = norm_to_coco(box, width, height)
        results.append({
            "image_id": int(img_id),
            "category_id": int(label),
            "bbox": [float(round(x, 2)) for x in coco_box],
            "score": float(round(score, 4))
        })

    # Save visualizations (with filtering)
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    
    for box, score, label in zip(boxes, scores, labels):
        # Exclude from visualization if score < 0.33
        if score < 0.33:
            continue
            
        # Exclude Bus(0)/Truck(4) < 0.25 except cameras 20,24,29
        if int(label) in [0, 4] and score < 0.25 and camera_num not in [20, 24, 29]:
            continue
        filtered_boxes.append(box)
        filtered_labels.append(label)
        filtered_scores.append(score)
    
    if filtered_boxes:  # visualize only when boxes remain after filtering
        vis_boxes = [norm_to_coco(b, width, height) for b in filtered_boxes]
        filename = image_id_to_filename.get(img_id, str(img_id))
        vis_img = cv2.imread(os.path.join(image_dir, filename))
        if vis_img is not None:
            vis_img = draw_boxes(vis_img, vis_boxes, filtered_labels, filtered_scores)
            cv2.imwrite(os.path.join(output_vis_dir, filename), vis_img)

# 8. Save COCO JSON
# Build categories
categories = [
    {"id": 0, "name": "Bus"},
    {"id": 1, "name": "Bike"},
    {"id": 2, "name": "Car"},
    {"id": 3, "name": "Pedestrian"},
    {"id": 4, "name": "Truck"}
]

# Build images
images = []
for img_id, (width, height) in image_wh_map.items():
    filename = image_id_to_filename.get(img_id, str(img_id))
    images.append({
        "id": img_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

# Build COCO structure
coco_output = {
    "images": images,
    "categories": categories,
    "annotations": results
}

with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"\n✔ WBF complete. Saved {len(results)} results")
print(f"- JSON: {output_json}")
print(f"- Visualization dir: {output_vis_dir}")