## D-FINE: Fine-grained Distribution Refinement for DETR-based Object Detection

간단한 학습/검증/추론 사용법과 구성 방법을 정리했습니다. 본 코드는 RT-DETR 기반으로 학습/평가 루프와 구성 시스템을 확장했습니다.

### 주요 특징
- YAML 기반 구성(`configs/`)으로 모델/데이터/옵티마이저/스케줄러 일괄 관리
- 2-Stage 학습 로직(멀티스케일 스케줄 + EMA 재시작 옵션)
- 평가: COCO mAP + 추가 지표(F1/Precision/Recall/IoU)
- AMP 지원, EMA, TensorBoard/W&B 로깅, 분산 학습(DDP)

### 리포지토리 구조(요약)
- `train.py`: 엔트리포인트. YAML 로드 → 솔버 실행
- `src/core/`: 레지스트리/팩토리(`@register`, `create`), 구성 로더(`YAMLConfig`)
- `src/solver/`: 학습/검증 루프(`det_engine.py`), 솔버(`det_solver.py`)
- `src/zoo/dfine/`: 모델 정의(`dfine.py`), 디코더/포스트프로세서 등
- `src/data/`: 데이터셋, 변환, 커스텀 `DataLoader`/collate_fn
- `configs/`: 데이터셋/런타임/모델/하이퍼파라미터 YAML 모음
- `tools/inference/torch_inf.py`: 이미지/폴더/비디오 추론 유틸

### 요구사항(권장)
- TBD
```

### 데이터셋 경로 설정
- COCO 예시: `configs/dataset/coco_detection.yml`에서 경로 수정
```yaml
train_dataloader:
  dataset:
    img_folder: /data/COCO2017/train2017/
    ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
  dataset:
    img_folder: /data/COCO2017/val2017/
    ann_file: /data/COCO2017/annotations/instances_val2017.json
```

### 학습(Training)
기본 실행:
```bash
python /DATA/jhlee/D-FINE/train.py -c /DATA/jhlee/D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml
```

자주 쓰는 옵션:
- `--use-amp`: mixed precision
- `--output-dir`, `--summary-dir`: 결과/로그 경로
- `--device cuda:0` 또는 환경 기본값 사용
- `--best-metric {ap,f1}`: 베스트 모델 저장 기준(F1 또는 COCO AP50:95)

예시:
```bash
python /DATA/jhlee/D-FINE/train.py \
  -c /DATA/jhlee/D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml \
  --use-amp --device cuda:0 --best-metric f1 \
  --output-dir /DATA/jhlee/D-FINE/output/run1 \
  --summary-dir /DATA/jhlee/D-FINE/output/run1/summary
```

Resume/Tuning:
```bash
python train.py -c <cfg> --resume /path/to/checkpoint.pth
python train.py -c <cfg> --tuning /path/to/checkpoint.pth
```

Stage 전환(요약):
- `train_dataloader.collate_fn.stop_epoch` 이전을 Stage1로 간주, 이후에 Stage2 진입
- 기준 메트릭 개선 시 `best_stg1.pth` 또는 `best_f1_stg1.pth` 저장(Stage2는 `best_stg2.pth`/`best_f1_stg2.pth`)

### 검증/평가(Validation/Eval)
- 즉시 평가 모드:
```bash
python /DATA/jhlee/D-FINE/train.py -c <cfg> --test-only --resume /path/to/checkpoint.pth
```
- 지표: COCO mAP(`CocoEvaluator`) + 추가 지표(F1/Precision/Recall/IoU 등 `Validator`)

### 추론(Inference)
스크립트: `tools/inference/torch_inf.py`

싱글 이미지:
```bash
python /DATA/jhlee/D-FINE/tools/inference/torch_inf.py \
  -c /DATA/jhlee/D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml \
  -r /DATA/jhlee/D-FINE/output/dfine_hgnetv2_l_custom/best_stg1.pth \
  -i /path/to/image.png -o /path/to/out --device cuda:0 \
  --output-format submission --threshold 0.4
```

폴더 배치:
```bash
python /DATA/jhlee/D-FINE/tools/inference/torch_inf.py -c <cfg> -r <ckpt> -i /path/to/dir -o /path/to/out --device cuda:0
```

비디오:
```bash
python /DATA/jhlee/D-FINE/tools/inference/torch_inf.py -c <cfg> -r <ckpt> -i /path/to/video.mp4 -o /path/to/out --device cuda:0
```

출력 형식:
- `--output-format submission` 또는 `coco` 선택
- 시각화 이미지는 `out/`에 저장, 어노테이션 JSON은 `out/annotations/`에 저장

클래스별 임계값:
- Bus/Truck은 0.7 고정, 그 외 클래스(Bike/Car/Pedestrian)는 `--threshold` 사용(기본 0.4)

### 출력/체크포인트
- `output/<exp>/last.pth`, `checkpointXXXX.pth`, `best_stg{1,2}.pth` 또는 `best_f1_stg{1,2}.pth`
- `eval/`에 COCO 평가 상태 저장, `summary/`에 TensorBoard 로그

### 구성 커스터마이징 팁
- 베스트 메트릭: `configs/runtime.yml`의 `best_metric` 또는 CLI `--best-metric {ap,f1}`
- 멀티스케일/스테이지 분기: 각 모델 YAML의 `train_dataloader.collate_fn.{base_size_repeat,stop_epoch}`
- 옵티마이저 파라미터 그룹: 정규식으로 백본/인코더/디코더 분리(`optimizer.params`)

### 고성능/안정화 팁
- AMP(`--use-amp`) + GradScaler, Gradient Clipping(`clip_max_norm`), EMA 사용 고려
- 데이터로드 워커(`num_workers`)와 배치 크기는 GPU 메모리에 맞게 조정

### 라이선스/감사
- 본 프로젝트 일부는 RT-DETR 코드를 기반으로 수정/확장되었습니다.

### Jetson AGX Orin Docker 추론(INT8 TensorRT)

- **Platform**: Jetson AGX Orin
- **JetPack**: 6.1 (30W mode)
- **Docker Image**: `xiilab/aicity_iccv_2025_track4_jetson:latest` (Docker Hub)
- **Entry Point**: `total_run.sh`

#### 워크플로우
1. `xiilab.onnx` → `xiilab_int8.engine`로 INT8 TensorRT 캘리브레이션 변환
2. CUDA Graph + GPU 전처리를 사용하는 C++ TensorRT 엔진(`dfine_inference.cpp`)으로 실시간 추론 실행

컨테이너 내부 동작:
- 엔진 생성: `onnx_to_tensorrt_memory_efficient.py`
- 추론 실행: C++ 엔진(`dfine_inference.cpp`), CUDA Graph 및 GPU-side preprocessing 사용

#### 사전 준비
- ONNX 파일명은 반드시 `xiilab.onnx` 여야 합니다. (PyTorch ckpt 미지원, ONNX만 지원)
- 캘리브레이션 데이터는 `/data/FishEye1K_eval/`(또는 동등 경로)에 위치해야 합니다.

데이터 디렉토리 구조 예시:
```text
/data/FishEye1K_eval
├─ annotations/
└─ *.png
```

#### 실행 방법
1) 호스트에서 데이터 경로 환경변수 설정:
```bash
export DATA_DIR="/home/agx2/data/aicitychallenge_2025/team_3"
```

2) 컨테이너 실행 및 `/data`로 마운트:
```bash
docker run -it --ipc=host --runtime=nvidia -v $DATA_DIR:/data xiilab/aicity_iccv_2025_track4_jetson:latest
```

3) 컨테이너 진입 후 엔트리포인트 `total_run.sh`가 아래를 자동 수행합니다.
- INT8 엔진 생성(`xiilab.onnx` → `xiilab_int8.engine`)
- C++ 엔진으로 배치/실시간 추론 수행

#### 출력
- 추론 결과 JSON: `/data/predictions.json`

#### 참고/주의
- ONNX 파일은 `xiilab.onnx`로 준비해야 합니다.
- 데이터 구조가 맞지 않으면 추론 오류가 발생할 수 있습니다. 위 디렉토리 규칙을 확인하세요.

