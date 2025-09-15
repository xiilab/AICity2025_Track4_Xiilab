"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Multi-GPU Inference Script
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import time
import json
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import argparse
from collections import defaultdict
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

class_name = {
    0: "Bus",
    1: "Bike", 
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

class FPSMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.preprocessing_times = []
        self.inference_times = []
        self.postprocessing_times = []
        self.total_times = []
        
    def add_timing(self, preprocess_time, inference_time, postprocess_time):
        self.preprocessing_times.append(preprocess_time)
        self.inference_times.append(inference_time)
        self.postprocessing_times.append(postprocess_time)
        self.total_times.append(preprocess_time + inference_time + postprocess_time)
    
    def get_fps_stats(self):
        if not self.total_times:
            return None
            
        # Calculate statistics
        avg_preprocess = np.mean(self.preprocessing_times) * 1000  # ms
        avg_inference = np.mean(self.inference_times) * 1000  # ms
        avg_postprocess = np.mean(self.postprocessing_times) * 1000  # ms
        avg_total = np.mean(self.total_times) * 1000  # ms
        
        fps = 1.0 / np.mean(self.total_times)
        
        return {
            'fps': fps,
            'avg_preprocess_ms': avg_preprocess,
            'avg_inference_ms': avg_inference,
            'avg_postprocess_ms': avg_postprocess,
            'avg_total_ms': avg_total,
            'min_total_ms': np.min(self.total_times) * 1000,
            'max_total_ms': np.max(self.total_times) * 1000,
            'total_frames': len(self.total_times)
        }
    
    def print_stats(self, gpu_id):
        stats = self.get_fps_stats()
        if stats is None:
            print(f"GPU {gpu_id}: No timing data available")
            return
            
        print(f"\n" + "="*60)
        print(f"📊 GPU {gpu_id} FPS MEASUREMENT RESULTS")
        print("="*60)
        print(f"🚀 Average FPS: {stats['fps']:.2f}")
        print(f"📸 Total Frames Processed: {stats['total_frames']}")
        print(f"⏱️  Average Processing Time: {stats['avg_total_ms']:.2f} ms")
        print(f"⚡ Min Processing Time: {stats['min_total_ms']:.2f} ms")
        print(f"🐌 Max Processing Time: {stats['max_total_ms']:.2f} ms")
        print("\n📋 Breakdown by Stage:")
        print(f"   🔄 Preprocessing:  {stats['avg_preprocess_ms']:.2f} ms ({stats['avg_preprocess_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"   🧠 Inference:      {stats['avg_inference_ms']:.2f} ms ({stats['avg_inference_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"   📦 Postprocessing: {stats['avg_postprocess_ms']:.2f} ms ({stats['avg_postprocess_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"GPU {gpu_id} " + "="*55)

def draw(images, labels, boxes, scores, output_dir, img_names, thrh=0.7):
    """Draw detection results on batch of images"""
    for i, im in enumerate(images):
        if i >= len(img_names):
            break
            
        draw_obj = ImageDraw.Draw(im)
        img_name = img_names[i]

        if i < len(scores):
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scr[scr > thrh]

            for j, b in enumerate(box):
                # Draw rectangle
                draw_obj.rectangle(list(b), outline="red", width=2)
                
                # Prepare text
                text = f"{class_name[lab[j].item()]}:{round(scrs[j].item(), 2)}"
                
                # Measure text size
                text_bbox = draw_obj.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background (above box)
                draw_obj.rectangle(
                    [b[0], b[1] - text_height - 4, b[0] + text_width + 4, b[1]],
                    fill="white",
                    outline="red"
                )
                
                # Draw text
                draw_obj.text(
                    (b[0] + 2, b[1] - text_height - 2),
                    text=text,
                    fill="red",
                )

        im.save(f"{output_dir}/results_{img_name}.png")

def get_image_Id(img_name_without_extension):
    """Parse image name to get unique image ID"""
    try:
        img_name = img_name_without_extension
        if img_name.endswith('.png'):
            img_name = img_name[:-4]
        
        sceneList = ['M', 'A', 'E', 'N']
        
        # Handle different naming conventions
        if 'camera' in img_name:
            parts = img_name.split('_')
            
            # Find camera index - look for part containing 'camera'
            camera_part = None
            scene_part = None
            frame_part = None
            
            for i, part in enumerate(parts):
                if 'camera' in part:
                    # Extract camera number from 'camera3' -> 3
                    camera_part = part.replace('camera', '')
                    # Look for scene in subsequent parts
                    if i + 1 < len(parts) and parts[i + 1] in sceneList:
                        scene_part = parts[i + 1]
                    # Look for frame number in subsequent parts
                    if i + 2 < len(parts) and parts[i + 2].isdigit():
                        frame_part = parts[i + 2]
                    break
            
            if camera_part and scene_part and frame_part:
                cameraIndx = int(camera_part)
                sceneIndx = sceneList.index(scene_part)
                frameIndx = int(frame_part)
                imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
                return imageId
        
        # Fallback: use hash of the filename if parsing fails
        return abs(hash(img_name)) % 1000000
        
    except Exception as e:
        # Final fallback: use hash of the filename
        return abs(hash(img_name_without_extension)) % 1000000

def save_predictions_coco_format(image_name, labels, boxes, scores, coco_results, images_info, thrh=0.7):
    """Save predictions in COCO format with robust error handling"""
    try:
        for i in range(len(labels)):
            if i >= len(scores):
                break
                
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scr[scr > thrh]

            for j, b in enumerate(box):
                try:
                    x1, y1, x2, y2 = b.tolist()
                    
                    # Remove .png extension if present
                    image_name_clean = image_name
                    if image_name_clean.endswith('.png'):
                        image_name_clean = image_name_clean[:-4]
                    
                    submission_image_id = get_image_Id(image_name_clean)
                    
                    # Calculate area
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    annotation = {
                        "id": len(coco_results) + 1,  # Unique annotation ID
                        "category_id": int(lab[j].item()),
                        "image_id": submission_image_id,
                        "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                        "segmentation": [],
                        "area": round(area, 2),
                        "iscrowd": 0,
                        "score": round(scrs[j].item(), 4)
                    }
                    
                    coco_results.append(annotation)
                    
                except Exception as e:
                    # Skip individual detection if there's an error
                    continue
                    
    except Exception as e:
        print(f"❌ Error saving predictions {image_name}: {e}. Skipped.")
        return

def process_image_batch(model, device, image_paths, output_dir, gpu_id, result_queue, progress_queue, batch_size=8):
    """Process images in batches on a specific GPU"""
    print(f"🔥 GPU {gpu_id}: Starting batch processing {len(image_paths)} images with batch size {batch_size}")
    
    # Initialize FPS meter and results for this GPU
    fps_meter = FPSMeter()
    coco_results = []
    images_info = []
    
    # Warm up the model
    print(f"🔥 GPU {gpu_id}: Warming up model...")
    dummy_input = torch.randn(batch_size, 3, 1920, 1920).to(device)
    dummy_size = torch.tensor([[1920, 1920] for _ in range(batch_size)]).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input, dummy_size)
    print(f"✅ GPU {gpu_id}: Model warmed up!")
    
    # Process images in batches
    processed_count = 0
    transforms = T.Compose([
        T.Resize((1920, 1920)),
        T.ToTensor(),
    ])
    
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        current_batch_size = len(batch_paths)
        
        try:
            # Preprocessing
            preprocess_start = time.time()
            batch_images = []
            batch_orig_sizes = []
            batch_pil_images = []
            batch_img_names = []
            
            for file_path in batch_paths:
                try:
                    im_pil = Image.open(file_path).convert("RGB")
                    img_name = os.path.basename(file_path).split(".")[0]
                    w, h = im_pil.size
                    
                    # Transform image
                    im_data = transforms(im_pil)
                    
                    batch_images.append(im_data)
                    batch_orig_sizes.append([w, h])
                    batch_pil_images.append(im_pil)
                    batch_img_names.append(img_name)
                    
                    # Store image info for COCO format
                    image_id = get_image_Id(img_name)
                    image_info = {
                        "id": image_id,
                        "file_name": os.path.basename(file_path),
                        "width": w,
                        "height": h
                    }
                    images_info.append(image_info)
                    
                except Exception as e:
                    print(f"❌ GPU {gpu_id}: Error loading {file_path}: {e}")
                    continue
            
            if not batch_images:
                print(f"❌ GPU {gpu_id}: No valid images in batch {batch_start//batch_size + 1}")
                continue
            
            # Stack batch tensors
            batch_tensor = torch.stack(batch_images).to(device)
            batch_sizes_tensor = torch.tensor(batch_orig_sizes).to(device)
            
            preprocess_time = time.time() - preprocess_start

            # Inference
            inference_start = time.time()
            with torch.no_grad():
                outputs = model(batch_tensor, batch_sizes_tensor)
            
            # GPU synchronization for accurate timing
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start

            # Postprocessing
            postprocess_start = time.time()
            labels, boxes, scores = outputs
            
            # Draw results for each image in batch
            draw(batch_pil_images, labels, boxes, scores, output_dir, batch_img_names)
            
            # Save predictions for batch
            for i, img_name in enumerate(batch_img_names):
                if i < len(labels):
                    save_predictions_coco_format(img_name, [labels[i]], [boxes[i]], [scores[i]], coco_results, images_info)
            
            postprocess_time = time.time() - postprocess_start

            # Record timing
            fps_meter.add_timing(preprocess_time, inference_time, postprocess_time)
            
            processed_count += len(batch_img_names)
            
            # Send progress update - handle queue full gracefully
            try:
                progress_queue.put((gpu_id, processed_count, len(image_paths)), timeout=0.1)
            except:
                pass  # Skip if queue is full
            
            # Log batch progress
            if (batch_start // batch_size + 1) % 10 == 0:
                batch_fps = len(batch_img_names) / (preprocess_time + inference_time + postprocess_time)
                print(f"🚀 GPU {gpu_id}: Batch {batch_start//batch_size + 1}, Size: {len(batch_img_names)}, FPS: {batch_fps:.2f}")
                
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Error processing batch {batch_start//batch_size + 1}: {e}")
            continue
    
    # Print GPU-specific statistics
    fps_meter.print_stats(gpu_id)
    
    # Send results back to main process
    result_queue.put({
        'gpu_id': gpu_id,
        'coco_results': coco_results,
        'images_info': images_info,
        'fps_stats': fps_meter.get_fps_stats(),
        'processed_count': processed_count
    })
    
    print(f"✅ GPU {gpu_id}: Completed processing {processed_count} images in batches")

def gpu_worker(gpu_id, image_paths, config_path, resume_path, output_dir, result_queue, progress_queue, batch_size):
    """Worker function that runs on each GPU"""
    try:
        # Set the specific GPU
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # Load model
        cfg = YAMLConfig(config_path, resume=resume_path)
        
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(resume_path, map_location="cpu")
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

        model = Model().to(device)
        model.eval()  # Set to evaluation mode
        
        # Process the assigned images in batches
        process_image_batch(model, device, image_paths, output_dir, gpu_id, result_queue, progress_queue, batch_size)
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Worker error: {e}")
        print(f"❌ GPU {gpu_id}: Traceback: {traceback.format_exc()}")
        result_queue.put({
            'gpu_id': gpu_id,
            'error': str(e),
            'processed_count': 0
        })

def split_data(data_list, num_splits):
    """Split data into roughly equal chunks"""
    chunk_size = len(data_list) // num_splits
    remainder = len(data_list) % num_splits
    
    chunks = []
    start = 0
    
    for i in range(num_splits):
        # Add one extra item to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data_list[start:end])
        start = end
    
    return chunks

def progress_monitor(progress_queue, total_images, num_gpus, stop_event):
    """Monitor progress from all GPUs"""
    gpu_progress = {i: 0 for i in range(num_gpus)}
    last_update = time.time()
    
    while not stop_event.is_set():
        try:
            gpu_id, current, total_for_gpu = progress_queue.get(timeout=2)
            gpu_progress[gpu_id] = current
            
            # Update progress every 2 seconds to avoid spam
            if time.time() - last_update > 2:
                total_processed = sum(gpu_progress.values())
                overall_progress = (total_processed / total_images) * 100
                
                print(f"📊 Overall Progress: {total_processed}/{total_images} ({overall_progress:.1f}%) - "
                      f"GPU Progress: {dict(gpu_progress)}")
                last_update = time.time()
                      
                if total_processed >= total_images:
                    break
                    
        except Exception as e:
            # Check if we should stop
            if stop_event.is_set():
                break
            continue
    
    print("📊 Progress monitor stopped")

def get_optimal_batch_size(gpu_memory_gb):
    """Get optimal batch size based on GPU memory"""
    if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
        return 16
    elif gpu_memory_gb >= 16:  # RTX 4080, V100, etc.
        return 12
    elif gpu_memory_gb >= 12:  # RTX 3080 Ti, RTX 4070 Ti, etc.
        return 8
    elif gpu_memory_gb >= 8:   # RTX 3070, RTX 4060 Ti, etc.
        return 6
    elif gpu_memory_gb >= 6:   # RTX 3060, etc.
        return 4
    else:
        return 2

def main(args):
    """Main function for multi-GPU batch inference"""
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus) if args.num_gpus > 0 else available_gpus
    
    print(f"🎮 Available GPUs: {available_gpus}")
    print(f"🚀 Using {num_gpus} GPUs for batch inference")
    
    # Auto-detect optimal batch size if not specified
    if args.batch_size == 0:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        args.batch_size = get_optimal_batch_size(gpu_memory)
        print(f"🧠 Auto-detected optimal batch size: {args.batch_size} (GPU memory: {gpu_memory:.1f} GB)")
    else:
        print(f"📦 Using specified batch size: {args.batch_size}")
    
    # Setup directories
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/annotations", exist_ok=True)
    
    # Get list of images to process
    if os.path.isdir(args.input):
        print("📁 Processing batch images from directory...")
        image_files = glob.glob(os.path.join(args.input, "*"))
        image_files = [f for f in image_files if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    elif os.path.splitext(args.input)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        print("📸 Processing single image...")
        image_files = [args.input]
    else:
        print("❌ Unsupported input format. Please provide an image file or directory.")
        return
    
    if not image_files:
        print("❌ No valid image files found!")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Split images across GPUs
    image_chunks = split_data(image_files, num_gpus)
    
    print(f"📊 Data distribution:")
    for i, chunk in enumerate(image_chunks):
        estimated_batches = (len(chunk) + args.batch_size - 1) // args.batch_size
        print(f"   GPU {i}: {len(chunk)} images (~{estimated_batches} batches)")
    
    # Create queues and manager for communication
    manager = Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()
    stop_event = manager.Event()
    
    # Start progress monitor
    progress_process = Process(target=progress_monitor, args=(progress_queue, len(image_files), num_gpus, stop_event))
    progress_process.start()
    
    # Start GPU workers
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        if len(image_chunks[gpu_id]) > 0:  # Only start process if there are images to process
            p = Process(
                target=gpu_worker,
                args=(gpu_id, image_chunks[gpu_id], args.config, args.resume, output_dir, result_queue, progress_queue, args.batch_size)
            )
            p.start()
            processes.append(p)
    
    # Collect results
    all_coco_results = []
    all_images_info = []
    total_processed = 0
    gpu_stats = {}
    
    # Wait for all processes to complete and collect results
    for p in processes:
        p.join()
    
    # Get results from queue
    while not result_queue.empty():
        try:
            result = result_queue.get(timeout=1)
            if 'error' in result:
                print(f"❌ GPU {result['gpu_id']} encountered error: {result['error']}")
            else:
                all_coco_results.extend(result['coco_results'])
                all_images_info.extend(result.get('images_info', []))
                total_processed += result['processed_count']
                gpu_stats[result['gpu_id']] = result['fps_stats']
        except:
            break
    
    # Stop progress monitor
    stop_event.set()
    progress_process.join(timeout=5)
    if progress_process.is_alive():
        progress_process.terminate()
        progress_process.join()
    
    total_time = time.time() - start_time
    
    # Save combined results in full COCO format
    if all_coco_results:
        # Create complete COCO format
        coco_output = {
            "info": {
                "description": "D-FINE Multi-GPU Batch Inference Results",
                "version": "1.0",
                "year": 2024,
                "contributor": "D-FINE",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": [
                {"id": 0, "name": "Bus"},
                {"id": 1, "name": "Bike"},
                {"id": 2, "name": "Car"},
                {"id": 3, "name": "Pedestrian"},
                {"id": 4, "name": "Truck"}
            ],
            "images": [],
            "annotations": all_coco_results
        }
        
        # Use collected images info and remove duplicates
        unique_images = {}
        for img_info in all_images_info:
            img_id = img_info["id"]
            if img_id not in unique_images:
                unique_images[img_id] = img_info
        
        coco_output["images"] = list(unique_images.values())
        
        # Save complete COCO format
        submission_json_path = os.path.join(output_dir + "/annotations", "coco_predictions.json")
        with open(submission_json_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        print(f"✅ COCO format results saved: {submission_json_path}")
        print(f"📊 Total annotations: {len(all_coco_results)}")
        print(f"📊 Total images: {len(coco_output['images'])}")
        
        # Also save simple format for submission if needed
        # simple_json_path = os.path.join(output_dir + "/annotations", "submission_predictions.json")
        # with open(simple_json_path, "w") as f:
        #     json.dump(all_coco_results, f, indent=2)
        # print(f"✅ Simple format also saved: {simple_json_path}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("🎯 MULTI-GPU BATCH INFERENCE COMPLETE")
    print("="*80)
    print(f"🚀 Total GPUs used: {num_gpus}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📸 Total images processed: {total_processed}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    if total_processed > 0:
        print(f"🎯 Overall processing FPS: {total_processed / total_time:.2f}")
    
    # Calculate combined statistics
    if gpu_stats:
        all_fps = [stats['fps'] for stats in gpu_stats.values() if stats]
        all_inference_times = [stats['avg_inference_ms'] for stats in gpu_stats.values() if stats]
        
        if all_fps:
            print(f"📊 Average GPU FPS: {np.mean(all_fps):.2f}")
            print(f"📊 Average Inference Time: {np.mean(all_inference_times):.2f} ms")
            print(f"🚀 Batch processing speedup: ~{args.batch_size}x per GPU")
    
    # Save combined statistics
    combined_stats = {
        'total_images': total_processed,
        'total_time_seconds': total_time,
        'overall_fps': total_processed / total_time if total_processed > 0 else 0,
        'num_gpus_used': num_gpus,
        'gpu_individual_stats': gpu_stats
    }
    
    stats_file = os.path.join(output_dir, "multi_gpu_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(combined_stats, f, indent=2)
    print(f"📈 Combined statistics saved to: {stats_file}")
    print("="*80)

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="D-FINE Multi-GPU Batch Inference")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image/directory path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use (0 = use all available)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU (0 = auto-detect optimal size)")
    
    args = parser.parse_args()
    main(args) 