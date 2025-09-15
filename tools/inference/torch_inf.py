"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
import glob
import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import time
import json
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

class_name = {
    0: "Bus",
    1: "Bike", 
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

def get_class_threshold(class_id, other_threshold=0.4):
    """Get threshold for each class"""
    # Bus (0) and Truck (4) use fixed 0.7 threshold
    if class_id in [0, 4]:  # Bus, Truck
        return 0.7
    else:  # Bike, Car, Pedestrian
        return other_threshold

def draw(images, labels, boxes, scores, output_dir, img_name, other_threshold=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i]
        box = boxes[i]
        
        # Apply class-specific thresholds
        valid_indices = []
        for idx in range(len(lab)):
            class_id = lab[idx].item()
            class_threshold = get_class_threshold(class_id, other_threshold)
            if scr[idx] > class_threshold:
                valid_indices.append(idx)
        
        if valid_indices:
            valid_indices = torch.tensor(valid_indices)
            lab_filtered = lab[valid_indices]
            box_filtered = box[valid_indices]
            scrs_filtered = scr[valid_indices]

            for j, b in enumerate(box_filtered):
                # Draw rectangle
                draw.rectangle(list(b), outline="red", width=2)
                
                # Prepare text
                text = f"{class_name[lab_filtered[j].item()]}:{round(scrs_filtered[j].item(), 2)}"
                
                # Measure text size
                text_bbox = draw.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background (above box)
                draw.rectangle(
                    [b[0], b[1] - text_height - 4, b[0] + text_width + 4, b[1]],
                    fill="white",
                    outline="red"
                )
                
                # Draw text
                draw.text(
                    (b[0] + 2, b[1] - text_height - 2),
                    text=text,
                    fill="red",
                )

        im.save(f"{output_dir}/results_{img_name}.png")


def process_image(model, device, file_path, output_dir, coco_results=None, other_threshold=0.4, output_format='submission', image_id_counter=None, image_cnt_dict=None):
    im_pil = Image.open(file_path).convert("RGB")
    img_name = os.path.basename(file_path).split(".")[0]
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    transforms = T.Compose(
        [
            T.Resize((1920, 1920)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(im_data, orig_size)

    labels, boxes, scores = output
    
    # Draw every 100 images
    # if image_cnt_dict is not None and image_cnt_dict['count'] % 100 == 0:
    draw([im_pil], labels, boxes, scores, output_dir, img_name, other_threshold)
    # print(f"🎨 Drew visualization for image #{image_cnt_dict['count']}: {img_name}")
    
    # Increment image counter
    if image_cnt_dict is not None:
        image_cnt_dict['count'] += 1

    if coco_results is not None:
        save_predictions_coco_format(img_name, labels, boxes, scores, coco_results, output_format, image_id_counter, (w, h), other_threshold)
        # Increment counter for COCO format after processing each image
        if output_format == 'coco' and image_id_counter is not None:
            image_id_counter['count'] += 1


def process_video(model, device, file_path, output_dir):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = os.path.join(output_dir, "torch_results.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((1280, 1280)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print(f"Processing {total_frames} video frames...")
    
    # Warm up GPU
    print("🔥 Warming up GPU...")
    # for _ in range(5):
    #     dummy_frame = torch.randn(1, 3, 1280, 1280).to(device)
    #     dummy_size = torch.tensor([[orig_w, orig_h]]).to(device)
    #     with torch.no_grad():
    #         _ = model(dummy_frame, dummy_size)
    
    start_time = time.time()
    
    # Create progress bar for video processing
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(im_data, orig_size)

        labels, boxes, scores = output

        # Draw detections on the frame (every 100 frames)
        if frame_count % 100 == 0:
            draw([frame_pil], labels, boxes, scores, output_dir, f"frame_{frame_count:06d}", 0.4)

        # Convert back to OpenCV image
        frame_result = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame_result)
        
        frame_count += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    
    total_time = time.time() - start_time
    overall_fps = frame_count / total_time
    print(f"\n🎬 Video processing complete!")
    print(f"📁 Result saved: {output_video_path}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"🎯 Overall processing FPS: {overall_fps:.2f}")
    print(f"📹 Original video FPS: {video_fps:.2f}")

import json

def get_image_Id(img_name_without_extension):
    # This function expects the name without '.png'; add it back if missing
    img_name = img_name_without_extension
    if not img_name.endswith('.png'):
        img_name += '.png'
    img_name = img_name.split('.png')[0] # Kept for compatibility with original logic
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def save_predictions_coco_format(image_name, labels, boxes, scores, coco_results, output_format='submission', image_id_counter=None, image_size=None, other_threshold=0.4):
    """
    Save predictions in COCO format or submission format
    
    Args:
        image_name: Name of the image
        labels: Predicted labels
        boxes: Predicted bounding boxes
        scores: Prediction scores
        coco_results: Dict containing results for COCO format or list for submission format
        output_format: 'coco' or 'submission'
        image_id_counter: Counter for COCO format image_id (dict with 'count' key)
        image_size: Tuple of (width, height) for COCO format
        other_threshold: Threshold for filtering predictions (Bus/Truck use 0.7, others use this)
    """
    if output_format == 'coco':
        # COCO format: complete structure with images, annotations, categories
        if image_id_counter is not None:
            current_image_id = image_id_counter['count']
            
            # Add image info
            image_info = {
                "id": current_image_id,
                "file_name": f"{image_name}.png",  # Add extension back
                "width": image_size[0] if image_size else 1920,
                "height": image_size[1] if image_size else 1920
            }
            coco_results['images'].append(image_info)
            
            # Add annotations with class-specific thresholds
            for i in range(len(labels)):
                scr = scores[i]
                lab = labels[i]
                box = boxes[i]
                
                # Apply class-specific thresholds
                valid_indices = []
                for idx in range(len(lab)):
                    class_id = lab[idx].item()
                    class_threshold = get_class_threshold(class_id, other_threshold)
                    if scr[idx] > class_threshold:
                        valid_indices.append(idx)
                
                if valid_indices:
                    valid_indices = torch.tensor(valid_indices)
                    lab_filtered = lab[valid_indices]
                    box_filtered = box[valid_indices]
                    scrs_filtered = scr[valid_indices]

                    for j, b in enumerate(box_filtered):
                        x1, y1, x2, y2 = b.tolist()
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        annotation = {
                            "id": len(coco_results['annotations']),  # Unique annotation ID
                            "image_id": current_image_id,
                            "category_id": int(lab_filtered[j].item()),
                            "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                            "area": round(area, 2),
                            "iscrowd": 0,
                            "score": round(scrs_filtered[j].item(), 4)  # Add score for detections
                        }
                        coco_results['annotations'].append(annotation)
                    
    elif output_format == 'submission':
        # Submission format: simple list of predictions
        for i in range(len(labels)):
            scr = scores[i]
            lab = labels[i]
            box = boxes[i]
            
            # Apply class-specific thresholds
            valid_indices = []
            for idx in range(len(lab)):
                class_id = lab[idx].item()
                class_threshold = get_class_threshold(class_id, other_threshold)
                if scr[idx] > class_threshold:
                    valid_indices.append(idx)
            
            if valid_indices:
                valid_indices = torch.tensor(valid_indices)
                lab_filtered = lab[valid_indices]
                box_filtered = box[valid_indices]
                scrs_filtered = scr[valid_indices]

                for j, b in enumerate(box_filtered):
                    x1, y1, x2, y2 = b.tolist()
                    
                    # Submission format: special image_id conversion
                    try:
                        # Remove .png extension if present
                        image_name_clean = image_name
                        if image_name_clean.endswith('.png'):
                            image_name_clean = image_name_clean[:-4]
                        
                        submission_image_id = get_image_Id(image_name_clean)
                        
                        coco_results.append({
                            "image_id": submission_image_id,
                            "category_id": int(lab_filtered[j].item()),
                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                            "score": round(scrs_filtered[j].item(), 4)
                        })
                    except Exception as e:
                        print(f"❌ Image name processing error {image_name}: {e}. Skipped.")
                        continue


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    
    # Warm up the model
    # print("🔥 Warming up model...")
    # dummy_input = torch.randn(1, 3, 1280, 1280).to(device)
    # dummy_size = torch.tensor([[1280, 1280]]).to(device)
    # for _ in range(10):
    #     with torch.no_grad():
    #         _ = model(dummy_input, dummy_size)
    # print("✅ Model warmed up!")

    # Check if the input file is an image or a video
    file_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/annotations", exist_ok=True)
    
    # Initialize results structure based on output format
    if args.output_format == 'coco':
        coco_results = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'Bus'},
                {'id': 1, 'name': 'Bike'},
                {'id': 2, 'name': 'Car'},
                {'id': 3, 'name': 'Pedestrian'},
                {'id': 4, 'name': 'Truck'}
            ]
        }
    else:  # submission format
        coco_results = []
        
    # Initialize image_id counter for COCO format
    image_id_counter = {'count': 0} if args.output_format == 'coco' else None
    
    # Initialize image counter for drawing every 100 images
    image_cnt_dict = {'count': 0}

    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        print("📸 Processing single image...")
        process_image(model, device, file_path, output_dir, coco_results, args.threshold, args.output_format, image_id_counter, image_cnt_dict)
        print("✅ Image processing complete.")
        
    elif os.path.isdir(file_path):
        print("📁 Processing batch images...")
        image_files = glob.glob(os.path.join(file_path, "*"))
        image_files = [f for f in image_files if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        
        print(f"Found {len(image_files)} images to process...")
        
        for img_path in tqdm(image_files, desc="Processing images", unit="img"):
            process_image(model, device, img_path, output_dir, coco_results, args.threshold, args.output_format, image_id_counter, image_cnt_dict)
        
        print(f"✅ Batch processing complete!")
        print(f"🎨 Total visualizations drawn: {image_cnt_dict['count'] // 100}")
        
    elif os.path.splitext(file_path)[-1].lower() in [".mp4", ".avi", ".mov"]:
        print("🎬 Processing video...")
        process_video(model, device, file_path, output_dir)
        
    else:
        print("❌ Unsupported file format.")
        return
        
    if coco_results:
        if args.output_format == 'coco':
            output_json_path = os.path.join(output_dir + "/annotations", "coco_predictions.json")
            print(f"✅ Saved in COCO format: {output_json_path}")
            result_count = len(coco_results['annotations'])
        else:  # submission
            output_json_path = os.path.join(output_dir + "/annotations", "submission_predictions.json")
            print(f"✅ Saved in Submission format: {output_json_path}")
            result_count = len(coco_results)
            
        with open(output_json_path, "w") as f:
            json.dump(coco_results, f, indent=2)
        print(f"📊 Total {result_count} predictions saved")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D-FINE Inference with FPS Measurement")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("-i", "--input", type=str, help="Input image/video/directory path")
    parser.add_argument("-o", "--output", type=str, help="Output directory path")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--benchmark", action="store_true", help="Run FPS benchmark on dummy data")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold for Bike, Car, Pedestrian (Bus and Truck always use 0.7)")
    parser.add_argument("--output-format", type=str, default='submission', choices=['coco', 'submission'], help="Output format: 'coco' for COCO format, 'submission' for submission format")
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark mode
        cfg = YAMLConfig(args.config, resume=args.resume)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
            
        cfg.model.load_state_dict(state)
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        model = Model().to(args.device)
        # run_benchmark(model, args.device, args.warmup_runs, args.benchmark_runs)
    else:
        # Run normal inference
        if not args.input or not args.output:
            parser.error("--input and --output are required for normal inference mode")
        main(args)
