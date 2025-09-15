## AICity Track 4 Fisheye D-FINE: Fine-grained Distribution Refinement for DETR-based Object Detection

### 1. 환경 셋팅

- Docker (inference for jetson)
```bash
docker pull xiilab/aicity_iccv_2025_track4_jetson:latest
docker run -it --ipc=host --gpus all \
  -v $(pwd):/workspace \
  xiilab/aicity_iccv_2025_track4_jetson:latest
```

- Server (train/test with server):
```bash
pip install -r requirements.txt
```


### 2. 데이터셋 구조 및 위치

- 데이터 루트: `dataset`
- 어노테이션 JSON: COCO format 필수

예시 디렉토리
```text
dataset
├─ train/                 # 이미지 폴더(예시)
├─ val/                   # 이미지 폴더(예시)
└─ annotations/
   ├─ instances_train.json
   └─ instances_val.json
```

`configs/dataset/*.yml`에서 경로 설정 예시
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

### 3. 학습

기본 실행
```bash
python train.py -c configs/dfine/dfine_hgnetv2_l_coco.yml -t pretrained.pth
```

자주 쓰는 옵션
- `--use-amp`, `--device cuda:0`
- `--output-dir`, `--summary-dir`
- `--best-metric {ap,f1}`

예시
```bash
python train.py \
  -c configs/dfine/dfine_hgnetv2_l_coco.yml \
  --use-amp --device cuda:0 --best-metric f1 \
  --output-dir output/run1 \
  --summary-dir output/run1/summary
```

참고
- Stage1 증강 모듈: `data_aug_gen.py` (학습 초기 단계 데이터 증강 로직)
- Stage1/Stage2 사전학습 체크포인트: Hugging Face [Xiilab-model/AICity_track4](https://huggingface.co/Xiilab-model/AICity_track4)

Stage1 증강 실행 예시
```bash
python data_aug_gen.py \
  -i path/to/images \
  -j path/to/annotations.json \
  -o output/augmented/images \
  -a output/augmented/annotations.json \
  -n 3 \
  --validation-dir output/augmentation_validation
```

의사라벨 생성(WBF 앙상블)
- 스크립트: `active_learning/wtf_1k.py`
- 역할: 여러 COCO 예측 JSON을 WBF로 앙상블하여 의사라벨 COCO JSON과 시각화를 생성
- 사용법:
  1) 스크립트 상단 변수 수정: `json_paths`(입력 예측 JSON 리스트), `image_dir`, `output_json`, `output_vis_dir`
  2) 선택 파라미터: `iou_thr`, `skip_box_thr`, `wbf_weights`, `DEBUG`
  3) 실행:
```bash
python active_learning/wtf_1k.py
```
- 필요 패키지 설치:
```bash
pip install ensemble-boxes
```

의사라벨 생성(배치 추론, Stage1 체크포인트)
- 스크립트: `active_learning/dfine_pseudo_inference.sh`
- 역할: Stage1에서 학습 완료된 체크포인트로 미라벨 이미지 폴더에 대해 배치 추론을 수행하여 의사라벨 예측을 생성
- 실행 (권장):
```bash
bash active_learning/dfine_pseudo_inference.sh
```
- 직접 실행(동등 명령):
```bash
python tools/inference/torch_inf_multi_gpu_batch_optimized.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  -r output/dfine_hgnetv2_l_custom/best_stg1.pth \
  -i path/to/unlabeled/images \
  -o output/pseudo_predictions \
  --batch-size 16 \
  --num-gpus 4
```

### 4. 평가

```bash
python train.py -c configs/dfine/dfine_hgnetv2_l_coco.yml --test-only --resume /path/to/checkpoint.pth
```
- COCO mAP 및 추가 지표(F1/Precision/Recall/IoU)를 출력합니다.

### 5. 추론 및 제출

스크립트: `tools/inference/torch_inf.py`

싱글 이미지
```bash
python tools/inference/torch_inf.py \
  -c configs/dfine/dfine_hgnetv2_l_coco.yml \
  -r output/dfine_hgnetv2_l_custom/best_stg1.pth \
  -i /path/to/image.png -o /path/to/out --device cuda:0 \
  --output-format submission --threshold 0.4
```

폴더/비디오
```bash
python tools/inference/torch_inf.py -c <cfg> -r <ckpt> -i /path/to/dir_or_video -o /path/to/out --device cuda:0
```

출력
- 시각화: `<out>/`
- 어노테이션 JSON: `<out>/annotations/`
- 제출용 포맷: `--output-format submission`

참고: 세부 옵션과 체크포인트 저장 규칙은 `train.py`, `configs/`를 참고하세요.

- 멀티 GPU 배치 추론
```bash
python tools/inference/torch_inf_multi_gpu_batch.py -h
```
예시
```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/inference/torch_inf_multi_gpu_batch.py \
  -c <cfg> -r <ckpt> -i /path/to/dir -o /path/to/out
```