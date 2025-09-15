## AICity Track 4 Fisheye D-FINE: Fine-grained Distribution Refinement for DETR-based Object Detection

### 1. Setup

- Docker (inference for Jetson)
```bash
docker pull xiilab/aicity_iccv_2025_track4_jetson:latest
docker run -it --ipc=host --gpus all \
  -v $(pwd):/workspace \
  xiilab/aicity_iccv_2025_track4_jetson:latest
```

- Server (train/test on server)
```bash
pip install -r requirements.txt
```

- Training Docker (server)
  - Environment: 4 × NVIDIA A40 GPUs
  - Docker Image: `xiilab/aicity_iccv_2025_track4_train:latest` ([Docker Hub](https://hub.docker.com/r/xiilab/aicity_iccv_2025_track4_train))
  - Container Launcher: `run_docker_train.sh`
  - Training Script: `run_pipeline.sh` (multi-GPU via PyTorch `torchrun`)
  - Model Export: `convert_pytorch_to_onnx.py` (export to ONNX)


### 2. Dataset structure and location

- Data root: `dataset`
- Annotation JSON must be in COCO format

Example directory
```text
dataset
├─ train/                 # image folder (example)
├─ val/                   # image folder (example)
└─ annotations/
   ├─ instances_train.json
   └─ instances_val.json
```

Set paths in `configs/dataset/*.yml`
```yaml
train_dataloader:
  dataset:
    img_folder: dataset/train/
    ann_file: dataset/annotations/instances_train.json
val_dataloader:
  dataset:
    img_folder: dataset/val/
    ann_file: dataset/annotations/instances_val.json
```

### 3. Training

Basic run
```bash
python train.py -c configs/dfine/dfine_hgnetv2_l_coco.yml -t pretrained.pth
```

Common options
- `--use-amp`, `--device cuda:0`
- `--output-dir`, `--summary-dir`
- `--best-metric {ap,f1}`

Example
```bash
python train.py \
  -c configs/dfine/dfine_hgnetv2_l_coco.yml \
  --use-amp --device cuda:0 --best-metric f1 \
  --output-dir output/run1 \
  --summary-dir output/run1/summary
```

Notes
- Stage1 augmentation module: `data_aug_gen.py`
- Stage1/Stage2 pretrained checkpoints: Hugging Face [Xiilab-model/AICity_track4](https://huggingface.co/Xiilab-model/AICity_track4)

Stage1 augmentation example
```bash
python data_aug_gen.py \
  -i path/to/images \
  -j path/to/annotations.json \
  -o output/augmented/images \
  -a output/augmented/annotations.json \
  -n 3 \
  --validation-dir output/augmentation_validation
```

Pseudo-label generation (WBF ensemble)
- Script: `active_learning/wtf_1k.py`
- Purpose: Ensemble multiple COCO prediction JSONs with WBF to produce a pseudo-label COCO JSON and visualizations
- Usage:
  1) Edit top variables in the script: `json_paths` (list of input prediction JSONs), `image_dir`, `output_json`, `output_vis_dir`
  2) Optional params: `iou_thr`, `skip_box_thr`, `wbf_weights`, `DEBUG`
  3) Run:
```bash
python active_learning/wtf_1k.py
```

Pseudo-label generation (batch inference, Stage1 checkpoint)
- Script: `active_learning/dfine_pseudo_inference.sh`
- Purpose: Run batch inference on unlabeled images using a trained Stage1 checkpoint to generate pseudo predictions
- Run (recommended):
```bash
bash active_learning/dfine_pseudo_inference.sh
```
- Direct equivalent command:
```bash
python tools/inference/torch_inf_multi_gpu_batch_optimized.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  -r output/dfine_hgnetv2_l_custom/best_stg1.pth \
  -i path/to/unlabeled/images \
  -o output/pseudo_predictions \
  --batch-size 16 \
  --num-gpus 4
```

### 4. Evaluation

```bash
python train.py -c configs/dfine/dfine_hgnetv2_l_coco.yml --test-only --resume /path/to/checkpoint.pth
```
- Prints COCO mAP and additional metrics (F1/Precision/Recall/IoU)

### 5. Inference and submission

Script: `tools/inference/torch_inf.py`

Single image
```bash
python tools/inference/torch_inf.py \
  -c configs/dfine/dfine_hgnetv2_l_coco.yml \
  -r output/dfine_hgnetv2_l_custom/best_stg1.pth \
  -i /path/to/image.png -o /path/to/out --device cuda:0 \
  --output-format submission --threshold 0.4
```

Folder/Video
```bash
python tools/inference/torch_inf.py -c <cfg> -r <ckpt> -i /path/to/dir_or_video -o /path/to/out --device cuda:0
```

Output
- Visualization: `<out>/`
- Annotation JSON: `<out>/annotations/`
- Submission format: `--output-format submission`

See `train.py` and `configs/` for details and checkpoint saving rules.

- Multi-GPU batch inference
```bash
python tools/inference/torch_inf_multi_gpu_batch.py -h
```
Example
```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/inference/torch_inf_multi_gpu_batch.py \
  -c <cfg> -r <ckpt> -i /path/to/dir -o /path/to/out
```


