"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Multi-GPU Batch Inference Script - OPTIMIZED VERSION

Key optimizations:
1. Fixed image count mismatch issue
2. Eliminated file name processing inconsistencies  
3. Improved error handling and validation
4. Added comprehensive statistics and validation
"""

import os
import sys
import glob
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
import traceback
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

class_name = {
    0: "Bus",
    1: "Bike", 
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, target_size=(1920, 1920)):
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transform
                transform = T.Compose([
                    T.Resize(self.target_size),
                    T.ToTensor(),
                ])
                image_tensor = transform(image)
            
            return {
                'image': image_tensor,
                'orig_size': torch.tensor([orig_w, orig_h]),
                'image_path': image_path,
                'pil_image': image,
                'success': True
            }
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            # Return dummy data for failed images
            dummy_tensor = torch.zeros(3, self.target_size[0], self.target_size[1])
            dummy_image = Image.new('RGB', (224, 224), color='black')
            return {
                'image': dummy_tensor,
                'orig_size': torch.tensor([224, 224]),
                'image_path': image_path,
                'pil_image': dummy_image,
                'success': False
            }

class FPSMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.preprocessing_times = []
        self.inference_times = []
        self.postprocessing_times = []
        self.total_times = []
        self.batch_sizes = []
        
    def add_timing(self, preprocess_time, inference_time, postprocess_time, batch_size=1):
        self.preprocessing_times.append(preprocess_time)
        self.inference_times.append(inference_time)
        self.postprocessing_times.append(postprocess_time)
        self.total_times.append(preprocess_time + inference_time + postprocess_time)
        self.batch_sizes.append(batch_size)
    
    def get_fps_stats(self):
        if not self.total_times:
            return None
            
        # Calculate statistics
        avg_preprocess = np.mean(self.preprocessing_times) * 1000  # ms
        avg_inference = np.mean(self.inference_times) * 1000  # ms
        avg_postprocess = np.mean(self.postprocessing_times) * 1000  # ms
        avg_total = np.mean(self.total_times) * 1000  # ms
        
        total_images = sum(self.batch_sizes)
        total_time = sum(self.total_times)
        fps = total_images / total_time if total_time > 0 else 0
        
        avg_batch_size = np.mean(self.batch_sizes)
        
        return {
            'fps': fps,
            'avg_preprocess_ms': avg_preprocess,
            'avg_inference_ms': avg_inference,
            'avg_postprocess_ms': avg_postprocess,
            'avg_total_ms': avg_total,
            'min_total_ms': np.min(self.total_times) * 1000,
            'max_total_ms': np.max(self.total_times) * 1000,
            'total_batches': len(self.total_times),
            'total_images': total_images,
            'avg_batch_size': avg_batch_size
        }
    
    def print_stats(self, gpu_id):
        stats = self.get_fps_stats()
        if stats is None:
            print(f"GPU {gpu_id}: No timing data available")
            return
            
        print(f"\n" + "="*60)
        print(f"📊 GPU {gpu_id} BATCH PROCESSING RESULTS")
        print("="*60)
        print(f"🚀 Average FPS: {stats['fps']:.2f}")
        print(f"📸 Total Images Processed: {stats['total_images']}")
        print(f"📦 Total Batches Processed: {stats['total_batches']}")
        print(f"📏 Average Batch Size: {stats['avg_batch_size']:.1f}")
        print(f"⏱️  Average Processing Time per Batch: {stats['avg_total_ms']:.2f} ms")
        print(f"⚡ Min Batch Processing Time: {stats['min_total_ms']:.2f} ms")
        print(f"🐌 Max Batch Processing Time: {stats['max_total_ms']:.2f} ms")
        print("\n📋 Breakdown by Stage:")
        print(f"   🔄 Preprocessing:  {stats['avg_preprocess_ms']:.2f} ms ({stats['avg_preprocess_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"   🧠 Inference:      {stats['avg_inference_ms']:.2f} ms ({stats['avg_inference_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"   📦 Postprocessing: {stats['avg_postprocess_ms']:.2f} ms ({stats['avg_postprocess_ms']/stats['avg_total_ms']*100:.1f}%)")
        print(f"GPU {gpu_id} " + "="*55)

def draw(images, labels, boxes, scores, output_dir, file_names, thrh=0.6):
    """Draw detection results on images"""
    for i, (im, file_name) in enumerate(zip(images, file_names)):
        if not hasattr(im, 'size'):  # Skip invalid images
            continue
            
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            # Draw rectangle
            draw.rectangle(list(b), outline="red", width=2)
            
            # Prepare text
            text = f"{class_name[lab[j].item()]}:{round(scrs[j].item(), 2)}"
            
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

        # Save using file name without extension
        base_name = os.path.splitext(file_name)[0]
        im.save(f"{output_dir}/results_{base_name}.png")

def save_predictions_coco_format_batch(file_names, labels, boxes, scores, coco_results, image_info_map, thrh=0.6):
    """Save predictions for a batch of images - OPTIMIZED"""
    for i, file_name in enumerate(file_names):
        if len(labels) <= i:  # Skip if no predictions for this image
            continue
            
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        # Directly match by file name
        if file_name in image_info_map:
            image_id = image_info_map[file_name]['id']
            
            for j, b in enumerate(box):
                x1, y1, x2, y2 = b.tolist()
                
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(lab[j].item()),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(scrs[j].item(), 4)
                })
        else:
            print(f"❌ Image info not found: {file_name}")

def process_batch_images(model, device, image_paths, output_dir, gpu_id, result_queue, progress_queue, image_info_map, batch_size=8, target_size=(1920, 1920)):
    """Process images in batches on a specific GPU - OPTIMIZED"""
    print(f"🔥 GPU {gpu_id}: Starting batch processing {len(image_paths)} images with batch size {batch_size}")
    
    # Initialize FPS meter and results for this GPU
    fps_meter = FPSMeter()
    coco_results = []
    processed_files = []  # Track successfully processed files
    
    # Create dataset and dataloader with custom collate function
    def custom_collate_fn(batch):
        """Custom collate function to handle PIL images"""
        # Separate successful and failed images
        successful_items = [item for item in batch if item['success']]
        failed_items = [item for item in batch if not item['success']]
        
        if not successful_items:
            return None  # No successful images in this batch
        
        images = torch.stack([item['image'] for item in successful_items])
        orig_sizes = torch.stack([item['orig_size'] for item in successful_items])
        image_paths = [item['image_path'] for item in successful_items]
        pil_images = [item['pil_image'] for item in successful_items]
        
        result = {
            'image': images,
            'orig_size': orig_sizes,
            'image_path': image_paths,
            'pil_image': pil_images,
            'failed_count': len(failed_items)
        }
        
        return result
    
    dataset = ImageDataset(image_paths, target_size=target_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    # Warm up the model
    print(f"🔥 GPU {gpu_id}: Warming up model...")
    dummy_input = torch.randn(batch_size, 3, target_size[0], target_size[1]).to(device)
    dummy_sizes = torch.tensor([[target_size[0], target_size[1]] for _ in range(batch_size)]).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input, dummy_sizes)
    print(f"✅ GPU {gpu_id}: Model warmed up!")
    
    processed_images = 0
    failed_images = 0
    
    # Process batches
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_data is None:  # Skip if no successful images in batch
            continue
            
        try:
            # Preprocessing
            preprocess_start = time.time()
            
            batch_images = batch_data['image'].to(device)
            batch_orig_sizes = batch_data['orig_size'].to(device)
            batch_paths = batch_data['image_path']
            batch_pil_images = batch_data['pil_image']
            failed_images += batch_data['failed_count']
            
            current_batch_size = len(batch_paths)
            preprocess_time = time.time() - preprocess_start

            # Inference
            inference_start = time.time()
            with torch.no_grad():
                outputs = model(batch_images, batch_orig_sizes)
            
            # GPU synchronization for accurate timing
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            inference_time = time.time() - inference_start

            # Postprocessing
            postprocess_start = time.time()
            labels, boxes, scores = outputs
            
            # Extract file names (with extension)
            file_names = [os.path.basename(path) for path in batch_paths]
            processed_files.extend(file_names)
            
            # Draw results
            draw(batch_pil_images, labels, boxes, scores, output_dir, file_names)
            
            # Save predictions
            save_predictions_coco_format_batch(file_names, labels, boxes, scores, coco_results, image_info_map)
            
            postprocess_time = time.time() - postprocess_start

            # Record timing
            fps_meter.add_timing(preprocess_time, inference_time, postprocess_time, current_batch_size)
            
            processed_images += current_batch_size
            
            # Send progress update
            try:
                progress_queue.put((gpu_id, processed_images, len(image_paths)), timeout=0.1)
            except:
                pass
                
            # Log batch progress
            if batch_idx % 10 == 0:
                batch_fps = current_batch_size / (preprocess_time + inference_time + postprocess_time)
                print(f"🚀 GPU {gpu_id}: Batch {batch_idx+1}, Size: {current_batch_size}, FPS: {batch_fps:.2f}")
                
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Error processing batch {batch_idx}: {e}")
            continue
    
    # Print GPU-specific statistics
    fps_meter.print_stats(gpu_id)
    
    if failed_images > 0:
        print(f"⚠️  GPU {gpu_id}: {failed_images} images failed to load")
    
    # Send results back to main process
    result_queue.put({
        'gpu_id': gpu_id,
        'coco_results': coco_results,
        'fps_stats': fps_meter.get_fps_stats(),
        'processed_count': processed_images,
        'failed_count': failed_images,
        'processed_files': processed_files
    })
    
    print(f"✅ GPU {gpu_id}: Completed processing {processed_images} images ({failed_images} failed)")

def gpu_worker(gpu_id, image_paths, config_path, resume_path, output_dir, result_queue, progress_queue, global_image_info_map, batch_size, target_size):
    """Worker function that runs on each GPU - OPTIMIZED"""
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
        model.eval()
        
        # Extract only the relevant image info for this GPU's images
        relevant_image_info = {}
        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            if file_name in global_image_info_map:
                relevant_image_info[file_name] = global_image_info_map[file_name]
        
        print(f"🔥 GPU {gpu_id}: Using {len(relevant_image_info)} image mappings")
        
        # Process the assigned images in batches
        process_batch_images(model, device, image_paths, output_dir, gpu_id, result_queue, progress_queue, relevant_image_info, batch_size, target_size)
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Worker error: {e}")
        print(f"❌ GPU {gpu_id}: Traceback: {traceback.format_exc()}")
        result_queue.put({
            'gpu_id': gpu_id,
            'error': str(e),
            'processed_count': 0,
            'failed_count': 0,
            'processed_files': []
        })

def split_data(data_list, num_splits):
    """Split data into roughly equal chunks"""
    chunk_size = len(data_list) // num_splits
    remainder = len(data_list) % num_splits
    
    chunks = []
    start = 0
    
    for i in range(num_splits):
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
            
            if time.time() - last_update > 3:
                total_processed = sum(gpu_progress.values())
                overall_progress = (total_processed / total_images) * 100
                
                print(f"📊 Overall Progress: {total_processed}/{total_images} ({overall_progress:.1f}%) - "
                      f"GPU Progress: {dict(gpu_progress)}")
                last_update = time.time()
                      
                if total_processed >= total_images:
                    break
                    
        except Exception as e:
            if stop_event.is_set():
                break
            continue
    
    print("📊 Progress monitor stopped")

def get_optimal_batch_size(gpu_memory_gb):
    """Get optimal batch size based on GPU memory"""
    if gpu_memory_gb >= 24:
        return 16
    elif gpu_memory_gb >= 16:
        return 12
    elif gpu_memory_gb >= 12:
        return 8
    elif gpu_memory_gb >= 8:
        return 6
    elif gpu_memory_gb >= 6:
        return 4
    else:
        return 2

def main(args):
    """Main function for multi-GPU batch inference - OPTIMIZED"""
    
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
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
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
    
    # Create GLOBAL image info map FIRST - this ensures consistent mapping
    print("📋 Creating global image info map...")
    global_image_info_map = {}
    coco_images = []
    
    for idx, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️  Could not read dimensions for {file_name}: {e}")
            width, height = 1920, 1920  # Default
        
        image_id = idx + 1
        global_image_info_map[file_name] = {
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height
        }
        
        coco_images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })
    
    print(f"✅ Created global image info map with {len(global_image_info_map)} entries")
    
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
    
    # Convert global_image_info_map to manager dict for multiprocessing
    shared_image_info_map = manager.dict(global_image_info_map)
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    print(f"🎯 Target image size: {target_size}")
    
    # Start progress monitor
    progress_process = Process(target=progress_monitor, args=(progress_queue, len(image_files), num_gpus, stop_event))
    progress_process.start()
    
    # Start GPU workers
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        if len(image_chunks[gpu_id]) > 0:
            p = Process(
                target=gpu_worker,
                args=(gpu_id, image_chunks[gpu_id], args.config, args.resume, output_dir, result_queue, progress_queue, shared_image_info_map, args.batch_size, target_size)
            )
            p.start()
            processes.append(p)
    
    # Collect results
    all_coco_results = []
    total_processed = 0
    total_failed = 0
    gpu_stats = {}
    all_processed_files = []
    
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
                total_processed += result['processed_count']
                total_failed += result.get('failed_count', 0)
                gpu_stats[result['gpu_id']] = result['fps_stats']
                all_processed_files.extend(result.get('processed_files', []))
        except:
            break
    
    # Stop progress monitor
    stop_event.set()
    progress_process.join(timeout=5)
    if progress_process.is_alive():
        progress_process.terminate()
        progress_process.join()
    
    total_time = time.time() - start_time
    
    # Create complete COCO format annotations
    if all_coco_results:
        coco_format = {
            "info": {
                "description": "D-FINE Multi-GPU Batch Inference Results - OPTIMIZED",
                "version": "1.0",
                "year": 2024,
                "contributor": "D-FINE",
                "date_created": "2024"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "categories": [
                {"id": 0, "name": "Bus", "supercategory": "vehicle"},
                {"id": 1, "name": "Bike", "supercategory": "vehicle"},
                {"id": 2, "name": "Car", "supercategory": "vehicle"},
                {"id": 3, "name": "Pedestrian", "supercategory": "person"},
                {"id": 4, "name": "Truck", "supercategory": "vehicle"}
            ],
            "images": coco_images,  # Use pre-created image list
            "annotations": []
        }
        
        # Add annotations with proper IDs
        for ann_id, result in enumerate(all_coco_results, 1):
            bbox = result["bbox"]
            area = bbox[2] * bbox[3]
            
            annotation = {
                "id": ann_id,
                "image_id": result["image_id"],
                "category_id": result["category_id"],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
                "score": result["score"]
            }
            coco_format["annotations"].append(annotation)
        
        # Save complete COCO format
        coco_json_path = os.path.join(output_dir + "/annotations", "pseudo_train_v3_optimized.json")
        with open(coco_json_path, "w") as f:
            json.dump(coco_format, f, indent=2)
        print(f"✅ COCO format results saved: {coco_json_path}")
        print(f"📊 Total images in COCO: {len(coco_format['images'])}")
        print(f"📊 Total annotations: {len(coco_format['annotations'])}")
        
        # Also save simple submission format
        submission_json_path = os.path.join(output_dir + "/annotations", "submission_predictions_optimized.json")
        with open(submission_json_path, "w") as f:
            json.dump(all_coco_results, f, indent=2)
        print(f"✅ Submission format saved: {submission_json_path}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("🎯 MULTI-GPU BATCH INFERENCE COMPLETE - OPTIMIZED")
    print("="*80)
    print(f"🚀 Total GPUs used: {num_gpus}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📸 Total images processed: {total_processed}")
    print(f"❌ Total images failed: {total_failed}")
    print(f"📊 Success rate: {(total_processed/(total_processed+total_failed)*100):.1f}%")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    if total_processed > 0:
        print(f"🎯 Overall processing FPS: {total_processed / total_time:.2f}")
    
    # Validate image count consistency
    expected_images = len(image_files)
    coco_images_count = len(coco_images) if coco_images else 0
    processed_unique = len(set(all_processed_files))
    
    print(f"\n🔍 IMAGE COUNT VALIDATION:")
    print(f"   Expected total images: {expected_images}")
    print(f"   Images in COCO format: {coco_images_count}")
    print(f"   Unique processed files: {processed_unique}")
    print(f"   Processing success rate: {(processed_unique/expected_images*100):.1f}%")
    
    if coco_images_count != expected_images:
        print(f"⚠️  WARNING: Image count mismatch detected!")
    else:
        print(f"✅ Image count validation passed!")
    
    # Calculate combined statistics
    if gpu_stats:
        all_fps = [stats['fps'] for stats in gpu_stats.values() if stats]
        all_inference_times = [stats['avg_inference_ms'] for stats in gpu_stats.values() if stats]
        total_batches = sum([stats['total_batches'] for stats in gpu_stats.values() if stats])
        avg_batch_sizes = [stats['avg_batch_size'] for stats in gpu_stats.values() if stats]
        
        if all_fps:
            print(f"📊 Average GPU FPS: {np.mean(all_fps):.2f}")
            print(f"📊 Average Inference Time: {np.mean(all_inference_times):.2f} ms")
            print(f"📊 Total Batches Processed: {total_batches}")
            print(f"📊 Average Batch Size: {np.mean(avg_batch_sizes):.1f}")
    
    # Save combined statistics
    combined_stats = {
        'total_images': total_processed,
        'total_failed': total_failed,
        'expected_images': expected_images,
        'coco_images_count': coco_images_count,
        'total_time_seconds': total_time,
        'overall_fps': total_processed / total_time if total_processed > 0 else 0,
        'batch_size': args.batch_size,
        'target_size': target_size,
        'num_gpus_used': num_gpus,
        'gpu_individual_stats': gpu_stats,
        'processed_files_count': len(all_processed_files),
        'unique_processed_files': processed_unique
    }
    
    stats_file = os.path.join(output_dir, "multi_gpu_batch_statistics_optimized.json")
    with open(stats_file, "w") as f:
        json.dump(combined_stats, f, indent=2)
    print(f"📈 Combined statistics saved to: {stats_file}")
    print("="*80)

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="D-FINE Multi-GPU Batch Inference - OPTIMIZED")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image/directory path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use (0 = use all available)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size per GPU (0 = auto-detect optimal size)")
    parser.add_argument("--target-size", type=str, default="1920,1920", help="Target image size as 'width,height'")
    
    args = parser.parse_args()
    main(args) 