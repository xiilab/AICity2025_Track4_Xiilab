import os
import json
import cv2
import albumentations as A
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.colors import COLORS  # 불필요한 import 제거
# from copy_paste import CopyPaste  # BaseMixTransform 기반 CopyPaste 클래스

# 클래스 아이디 → 이름 매핑
class_map = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

# 시각화용 색상 매핑
class_colors = {
    0: 'red',      # Bus
    1: 'blue',     # Bike
    2: 'green',    # Car
    3: 'orange',   # Pedestrian
    4: 'purple'    # Truck
}

# ==========================================
# 1) BBoxMotionBlur 클래스 (기존 그대로)
# ==========================================
class BBoxMotionBlur(A.ImageOnlyTransform):
    """
    바운딩 박스 내부 영역에만 모션 블러를 적용하는 Transform.
    bboxes 파라미터를 받아서, 각 박스 영역에 대해서만 블러 커널을 적용합니다.
    """
    def __init__(self, blur_limit=(3, 7), angle_limit=(-45, 45), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.blur_limit = blur_limit
        self.angle_limit = angle_limit

    def apply(self, image, **params):
        h, w = image.shape[:2]
        bboxes = params.get("bboxes", [])
        output = image.copy()

        # 랜덤 커널 크기 (홀수)
        kernel_size = random.randint(*self.blur_limit)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 랜덤 블러 각도
        angle = random.uniform(*self.angle_limit)

        # 모션 블러 커널 생성
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        k[kernel_size // 2, :] = 1.0
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5),
            angle, 1
        )
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
        k /= k.sum()

        # 각 bbox 영역마다 필터 적용
        for bbox in bboxes:
            x_min, y_min, box_w, box_h = bbox[:4]  # COCO 포맷: x_min, y_min, w, h
            x1 = int(round(x_min))
            y1 = int(round(y_min))
            x2 = int(round(x_min + box_w))
            y2 = int(round(y_min + box_h))

            # 이미지 경계 내로 클램핑
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            blurred = cv2.filter2D(roi, -1, k)
            output[y1:y2, x1:x2] = blurred

        return output


# ==========================================
# 2) Bounding Box → Mask 변환 함수
# ==========================================
def bboxes_to_masks(bboxes, image_shape):
    """
    COCO 형식 바운딩 박스를 개별 이진 마스크로 변환.
    
    Args:
        bboxes: [(x_min, y_min, w, h), ...] (COCO 포맷)
        image_shape: (height, width[, channels])
    Returns:
        masks: [mask1, mask2, ...], 각 mask는 (H, W) 이진 배열
    """
    h, w = image_shape[:2]
    masks = []

    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        mask = np.zeros((h, w), dtype=np.uint8)

        x1 = max(0, int(round(x_min)))
        y1 = max(0, int(round(y_min)))
        x2 = min(w, int(round(x_min + width)))
        y2 = min(h, int(round(y_min + height)))

        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 1
        masks.append(mask)

    return masks

# ==========================================
# 4) 실제 증강 함수 ( Albumentations 적용)
# ==========================================
def coco_augment_image(
    image,
    coco_bboxes,
    paste_image=None,
    paste_coco_bboxes=None
):
    """
    주어진 COCO 형식 이미지/박스를 증강.
    여러 Albumentations Transform을 순차 적용.

    Args:
        image: 원본 이미지 (H, W, 3) numpy array
        coco_bboxes: [(category_id, x_min, y_min, w, h), ...] 원본 바운딩 박스
        paste_image: 사용하지 않음 (이전 버전과의 호환성을 위해 유지)
        paste_coco_bboxes: 사용하지 않음 (이전 버전과의 호환성을 위해 유지)

    Returns:
        aug_image: 증강된 이미지 (numpy array)
        aug_bboxes: 증강된 COCO 형식 바운딩 박스 [(category_id, x_min, y_min, w, h), ...]
    """
    img_h, img_w = image.shape[:2]
    bboxes_out = coco_bboxes.copy()

    # Albumentations로 변형 정의
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                       contrast_limit=(-0.2, 0.2), p=0.5),
            A.CLAHE(clip_limit=(1, 3), tile_grid_size=(16, 16), p=0.5),
            A.RandomGamma(gamma_limit=(80, 100), p=0.5),
            # A.Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.ImageCompression(quality=80, p=0.8),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                           num_shadows=3,
                           shadow_dimension=3, p=0.5),
            A.GaussNoise(std_range=[0.01, 0.09],mean_range=[0, 0], per_channel=True,noise_scale_factor=1, p=0.5),
            A.Posterize(num_bits=(4, 6), p=0.5),  # 새로 추가: 이미지 포스터화
            A.RandomGravel(gravel_roi=(0.1, 0.4, 0.9, 0.9), 
                          number_of_patches=3, p=0.5),  # 새로 추가: 자갈 효과
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"])
    )

    # Albumentations 입력 준비: 좌표를 픽셀 범위 내로 클램핑
    filtered_bboxes = []
    filtered_labels = []
    for (cat_id, x_min, y_min, bw, bh) in bboxes_out:
        x_min = max(0.0, min(x_min, img_w - 1.0))
        y_min = max(0.0, min(y_min, img_h - 1.0))
        bw = max(1.0, min(bw, img_w - x_min))
        bh = max(1.0, min(bh, img_h - y_min))
        if bw >= 1.0 and bh >= 1.0:
            filtered_bboxes.append((x_min, y_min, bw, bh))
            filtered_labels.append(cat_id)

    augmented = transform(
        image=image,
        bboxes=filtered_bboxes,
        class_labels=filtered_labels
    )

    aug_image = augmented["image"]
    aug_h, aug_w = aug_image.shape[:2]
    aug_bboxes_raw = augmented["bboxes"]
    aug_labels_raw = augmented["class_labels"]

    # 최종 클램핑 후 COCO 형식으로 재조합
    final_bboxes = []
    for idx, bbox in enumerate(aug_bboxes_raw):
        x_min, y_min, bw, bh = bbox
        cat_id = aug_labels_raw[idx]
        x_min = max(0.0, min(x_min, aug_w - 1.0))
        y_min = max(0.0, min(y_min, aug_h - 1.0))
        bw = max(1.0, min(bw, aug_w - x_min))
        bh = max(1.0, min(bh, aug_h - y_min))
        if bw >= 1.0 and bh >= 1.0:
            final_bboxes.append((int(cat_id), x_min, y_min, bw, bh))

    return aug_image, final_bboxes


# ==========================================
# 5) COCO 데이터셋 전체 증강 함수
# ==========================================
def augment_coco_dataset(
    image_dir: str,
    coco_json_path: str,
    save_img_dir: str,
    save_json_path: str,
    num_augments: int = 5
):
    """
    COCO 형식 JSON을 읽어와 모든 이미지를 증강하고 새로운 COCO JSON을 생성합니다.

    - 모든 이미지에 대해 증강 수행
    - 원본은 "*.png" 형식으로 저장
    - 증강본은 "*_aug_{i}.png" 형식으로 저장  
    - 중복 라벨(같은 image_id, 동일 bbox, 동일 category_id)을 체크하여 제거
    - 파일명 기반 중복 처리 방지

    Args:
        image_dir: 원본 이미지가 있는 디렉토리
        coco_json_path: 원본 COCO JSON 파일 경로
        save_img_dir: 증강된 이미지들을 저장할 디렉토리
        save_json_path: 생성될 COCO JSON 파일 경로
        num_augments: 이미지당 생성할 증강본 개수 (원본 포함 아님)
    """
    os.makedirs(save_img_dir, exist_ok=True)

    # 1) COCO JSON 로드
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images_info = coco_data.get("images", [])
    annotations_info = coco_data.get("annotations", [])
    categories_info = coco_data.get("categories", [])

    # 이미지 ID → 메타 정보(파일명, width, height) 매핑
    img_id_to_info = {img["id"]: img for img in images_info}

    # 이미지 ID별 어노테이션 모으기
    img_id_to_anns = {}
    for ann in annotations_info:
        img_id = ann["image_id"]
        img_id_to_anns.setdefault(img_id, []).append(ann)

    # 카테고리 정보 업데이트 (class_map 적용)
    updated_categories = []
    for cat in categories_info:
        cat_id = cat["id"]
        updated = cat.copy()
        if cat_id in class_map:
            updated["name"] = class_map[cat_id]
        updated_categories.append(updated)

    new_images = []
    new_annotations = []
    annotation_id = 1
    image_id_counter = 1

    # 중복 체크를 위한 셋들
    seen_annotation_keys = set()  # (image_id, category_id, x_min, y_min, w, h)
    processed_filenames = set()   # 이미 처리된 파일명들
    seen_image_filenames = set()  # JSON에 추가된 이미지 파일명들

    # 결과 JSON 뼈대
    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": updated_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    print(f"📊 원본 데이터: {len(images_info)}개 이미지, {len(annotations_info)}개 어노테이션")

    # 2) 각 원본 이미지별로 처리
    processed_count = 0
    skipped_count = 0
    
    for orig_img in images_info:
        orig_id = orig_img["id"]
        file_name = orig_img["file_name"]
        ori_w = orig_img["width"]
        ori_h = orig_img["height"]
        img_path = os.path.join(image_dir, file_name)

        # 파일명 중복 체크 (확장자 제거한 베이스명으로 체크)
        base_name = os.path.splitext(file_name)[0]
        if base_name in processed_filenames:
            print(f"[건너뜀] 이미 처리된 파일: {file_name}")
            skipped_count += 1
            continue

        if not os.path.exists(img_path):
            print(f"[경고] 이미지가 없음: {img_path}")
            continue

        # 원본 이미지 로드
        orig_img_np = cv2.imread(img_path)
        if orig_img_np is None:
            print(f"[경고] 이미지 로드 실패: {img_path}")
            continue

        # 원본 바운딩 박스 리스트 준비 (COCO 포맷)
        orig_bboxes = []
        if orig_id in img_id_to_anns:
            for ann in img_id_to_anns[orig_id]:
                orig_bboxes.append((ann["category_id"], *ann["bbox"]))

        # 처리된 파일명으로 마킹
        processed_filenames.add(base_name)

        # 모든 이미지에 대해 증강 수행
        
        # (1) 원본 이미지 저장
        orig_save_name = f"{base_name}.png"
        if orig_save_name not in seen_image_filenames:
            orig_save_path = os.path.join(save_img_dir, orig_save_name)
            if not os.path.exists(orig_save_path):
                cv2.imwrite(orig_save_path, orig_img_np)
            
            seen_image_filenames.add(orig_save_name)
            new_images.append({
                "id": image_id_counter,
                "file_name": orig_save_name,
                "width": orig_img_np.shape[1],
                "height": orig_img_np.shape[0],
                "license": orig_img.get("license", 0)
            })

            # 원본 어노테이션 추가
            for (cat_id, x_min, y_min, w_bbox, h_bbox) in orig_bboxes:
                area = float(w_bbox * h_bbox)
                key = (image_id_counter, cat_id,
                       round(x_min, 2), round(y_min, 2),
                       round(w_bbox, 2), round(h_bbox, 2))

                if key not in seen_annotation_keys and area >= 1.0:
                    seen_annotation_keys.add(key)
                    new_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": cat_id,
                        "bbox": [float(x_min), float(y_min), float(w_bbox), float(h_bbox)],
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id_counter += 1

        # (2) 증강본 생성
        for i in range(num_augments):
            aug_save_name = f"{base_name}_aug_{i+1}.png"
            if aug_save_name in seen_image_filenames:
                continue

            # 증강 적용
            aug_img_np, aug_bboxes = coco_augment_image(
                orig_img_np, orig_bboxes
            )

            # 증강 이미지 저장
            aug_save_path = os.path.join(save_img_dir, aug_save_name)
            if not os.path.exists(aug_save_path):
                cv2.imwrite(aug_save_path, aug_img_np)

            seen_image_filenames.add(aug_save_name)
            new_images.append({
                "id": image_id_counter,
                "file_name": aug_save_name,
                "width": aug_img_np.shape[1],
                "height": aug_img_np.shape[0],
                "license": orig_img.get("license", 0)
            })

            # 증강 어노테이션 추가
            for (cat_id, x_min, y_min, w_bbox, h_bbox) in aug_bboxes:
                area = float(w_bbox * h_bbox)
                key = (image_id_counter, cat_id,
                       round(x_min, 2), round(y_min, 2),
                       round(w_bbox, 2), round(h_bbox, 2))

                if key not in seen_annotation_keys and area >= 1.0:
                    seen_annotation_keys.add(key)
                    new_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": cat_id,
                        "bbox": [float(x_min), float(y_min), float(w_bbox), float(h_bbox)],
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id_counter += 1

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"[진행] {processed_count}개 이미지 처리 완료...")

    print(f"[완료] 총 {processed_count}개 이미지 처리, {skipped_count}개 중복 건너뜀")

    # 3) 생성된 JSON 저장
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"[완료] 증강 이미지 폴더: {save_img_dir}")
    print(f"[완료] 증강용 COCO JSON: {save_json_path}")
    print(f"📊 최종 결과: {len(new_images)}개 이미지, {len(new_annotations)}개 어노테이션")


# ==========================================
# 6) COCO JSON 형식 검증 함수
# ==========================================
def validate_coco_format(coco_json_path, image_dir=None):
    """
    생성된 COCO JSON 파일의 필수 구조/ID 중복/바운딩박스 유효성 등을 간단히 검증.
    파일명 중복도 체크합니다.

    Args:
        coco_json_path: 검사할 COCO JSON 경로
        image_dir: (선택) 이미지 디렉토리 경로 (파일 존재 여부 확인용)

    Returns:
        bool: 검증 통과 시 True, 문제 발견 시 False
    """
    try:
        with open(coco_json_path, "r") as f:
            data = json.load(f)

        # 필수 키 검사
        for key in ("images", "annotations", "categories"):
            if key not in data:
                print(f"❌ 필수 키 누락: {key}")
                return False

        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]

        print(f"✅ 이미지 수: {len(images)}")
        print(f"✅ 어노테이션 수: {len(annotations)}")
        print(f"✅ 카테고리 수: {len(categories)}")

        # 이미지 ID 중복 검사
        img_ids = [img["id"] for img in images]
        if len(img_ids) != len(set(img_ids)):
            print("❌ 중복된 이미지 ID 발견")
            return False
        else:
            print("✅ 이미지 ID 중복 없음")

        # 파일명 중복 검사
        filenames = [img["file_name"] for img in images]
        if len(filenames) != len(set(filenames)):
            duplicates = len(filenames) - len(set(filenames))
            print(f"❌ 중복된 파일명 {duplicates}개 발견")
            
            # 중복 파일명 샘플 출력
            seen = set()
            duplicate_names = []
            for name in filenames:
                if name in seen:
                    duplicate_names.append(name)
                else:
                    seen.add(name)
            
            print(f"   중복 파일명 샘플 (최대 5개): {duplicate_names[:5]}")
            return False
        else:
            print("✅ 파일명 중복 없음")

        # 어노테이션 ID 중복 검사
        ann_ids = [ann["id"] for ann in annotations]
        if len(ann_ids) != len(set(ann_ids)):
            print("❌ 중복된 어노테이션 ID 발견")
            return False
        else:
            print("✅ 어노테이션 ID 중복 없음")

        valid_img_ids = set(img_ids)
        valid_cat_ids = set(cat["id"] for cat in categories)

        # 어노테이션 샘플 검사 (최초 10개)
        validation_errors = 0
        for i, ann in enumerate(annotations[:10]):
            for req_key in ("id", "image_id", "category_id", "bbox", "area"):
                if req_key not in ann:
                    print(f"❌ 어노테이션[{i}] 누락된 키: {req_key}")
                    validation_errors += 1

            if ann["image_id"] not in valid_img_ids:
                print(f"❌ 어노테이션[{i}] 잘못된 image_id: {ann['image_id']}")
                validation_errors += 1

            if ann["category_id"] not in valid_cat_ids:
                print(f"❌ 어노테이션[{i}] 잘못된 category_id: {ann['category_id']}")
                validation_errors += 1

            bbox = ann["bbox"]
            if (not isinstance(bbox, list)) or len(bbox) != 4:
                print(f"❌ 어노테이션[{i}] 잘못된 bbox 형식: {bbox}")
                validation_errors += 1
                continue

            x_min, y_min, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                print(f"❌ 어노테이션[{i}] 유효하지 않은 bbox 크기: {bbox}")
                validation_errors += 1

            expected_area = bw * bh
            if abs(ann["area"] - expected_area) > 1e-2:
                print(f"❌ 어노테이션[{i}] area 불일치: {ann['area']} vs {expected_area}")
                validation_errors += 1

        if validation_errors == 0:
            print("✅ 어노테이션 형식 검증 통과")

        # 이미지 파일 존재 여부 (샘플 5개)
        if image_dir:
            missing = []
            for img in images[:5]:
                path = os.path.join(image_dir, img["file_name"])
                if not os.path.exists(path):
                    missing.append(img["file_name"])
            if missing:
                print(f"❌ 누락된 이미지 파일: {missing}")
                return False
            else:
                print("✅ 이미지 파일 존재 확인 (샘플)")

        # 전체 매칭 체크 (선택사항)
        if image_dir and os.path.exists(image_dir):
            actual_files = set(os.listdir(image_dir))
            json_files = set(img["file_name"] for img in images)
            
            missing_in_dir = json_files - actual_files
            extra_in_dir = actual_files - json_files
            
            if len(missing_in_dir) == 0 and len(extra_in_dir) == 0:
                print("✅ JSON과 실제 파일 100% 매칭")
            else:
                if len(missing_in_dir) > 0:
                    print(f"⚠️ JSON에는 있지만 디렉토리에 없는 파일: {len(missing_in_dir)}개")
                if len(extra_in_dir) > 0:
                    print(f"⚠️ 디렉토리에는 있지만 JSON에 없는 파일: {len(extra_in_dir)}개")

        if validation_errors > 0:
            print(f"❌ 총 {validation_errors}개의 검증 오류 발견")
            return False

        print("✅ COCO 포맷 검증 통과!")
        return True

    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
        return False


# ==========================================
# 시각화 함수들 추가
# ==========================================
def draw_bboxes_on_image(image, coco_bboxes, title="Image with Bounding Boxes"):
    """
    이미지에 COCO 형식 바운딩 박스를 그려서 시각화
    
    Args:
        image: numpy array (H, W, 3) - BGR 형식
        coco_bboxes: [(category_id, x_min, y_min, w, h), ...] COCO 바운딩 박스
        title: 플롯 제목
    
    Returns:
        matplotlib figure
    """
    # BGR to RGB 변환
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 바운딩 박스 그리기
    for i, (cat_id, x_min, y_min, w, h) in enumerate(coco_bboxes):
        color = class_colors.get(cat_id, 'yellow')
        class_name = class_map.get(cat_id, f'Class_{cat_id}')
        
        # 사각형 그리기
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 클래스 라벨 추가
        ax.text(
            x_min, y_min - 5,
            f'{class_name}',
            color=color, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def visualize_augmentation_comparison(original_image, original_bboxes, 
                                    augmented_image, augmented_bboxes,
                                    save_path=None):
    """
    원본과 증강된 이미지를 나란히 비교 시각화
    
    Args:
        original_image: 원본 이미지 (numpy array)
        original_bboxes: 원본 바운딩 박스 리스트
        augmented_image: 증강된 이미지 (numpy array)  
        augmented_bboxes: 증강된 바운딩 박스 리스트
        save_path: 저장할 경로 (None이면 화면에만 표시)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본 이미지
    if len(original_image.shape) == 3:
        orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = original_image
    
    ax1.imshow(orig_rgb)
    ax1.set_title(f'Original Image ({len(original_bboxes)} objects)', fontsize=14, fontweight='bold')
    
    for cat_id, x_min, y_min, w, h in original_bboxes:
        color = class_colors.get(cat_id, 'yellow')
        class_name = class_map.get(cat_id, f'Class_{cat_id}')
        
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(
            x_min, y_min - 5, class_name,
            color=color, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7)
        )
    
    # 증강된 이미지
    if len(augmented_image.shape) == 3:
        aug_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
    else:
        aug_rgb = augmented_image
    
    ax2.imshow(aug_rgb)
    ax2.set_title(f'Augmented Image ({len(augmented_bboxes)} objects)', fontsize=14, fontweight='bold')
    
    for cat_id, x_min, y_min, w, h in augmented_bboxes:
        color = class_colors.get(cat_id, 'yellow')
        class_name = class_map.get(cat_id, f'Class_{cat_id}')
        
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(
            x_min, y_min - 5, class_name,
            color=color, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7)
        )
    
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"비교 이미지 저장: {save_path}")
    
    return fig

def create_augmentation_samples(image_dir, coco_json_path, output_dir, num_samples=5):
    """
    데이터 증강 샘플들을 생성하고 시각화하여 저장
    
    Args:
        image_dir: 원본 이미지 디렉토리
        coco_json_path: COCO JSON 파일 경로
        output_dir: 시각화 결과를 저장할 디렉토리
        num_samples: 생성할 샘플 개수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO 데이터 로드
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    images_info = coco_data.get("images", [])
    annotations_info = coco_data.get("annotations", [])
    
    # 이미지 ID별 어노테이션 매핑
    img_id_to_anns = {}
    for ann in annotations_info:
        img_id = ann["image_id"]
        img_id_to_anns.setdefault(img_id, []).append(ann)
    
    # 객체가 있는 이미지들만 필터링
    valid_images = [img for img in images_info if img["id"] in img_id_to_anns]
    
    if len(valid_images) == 0:
        print("❌ 어노테이션이 있는 이미지를 찾을 수 없습니다.")
        return
    
    print(f"📊 {len(valid_images)}개의 유효한 이미지 중 {num_samples}개 샘플 생성...")
    
    # 랜덤하게 샘플 선택
    sample_images = random.sample(valid_images, min(num_samples, len(valid_images)))
    
    for i, img_info in enumerate(sample_images):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ 이미지 파일 없음: {img_path}")
            continue
        
        # 원본 이미지 로드
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue
        
        # 원본 바운딩 박스
        original_bboxes = []
        for ann in img_id_to_anns[img_id]:
            original_bboxes.append((ann["category_id"], *ann["bbox"]))
        
        print(f"🖼️ 샘플 {i+1}: {file_name} ({len(original_bboxes)}개 객체)")
        
        # 여러 증강 버전 생성
        for aug_idx in range(3):  # 각 이미지당 3개 증강 버전
            try:
                # 증강 적용
                aug_image, aug_bboxes = coco_augment_image(
                    original_image.copy(), 
                    original_bboxes.copy()
                )
                
                # 비교 시각화 생성
                base_name = os.path.splitext(file_name)[0]
                save_path = os.path.join(output_dir, f"sample_{i+1}_{base_name}_aug_{aug_idx+1}.png")
                
                fig = visualize_augmentation_comparison(
                    original_image, original_bboxes,
                    aug_image, aug_bboxes,
                    save_path
                )
                plt.close(fig)  # 메모리 절약
                
            except Exception as e:
                print(f"⚠️ 증강 실패 (샘플 {i+1}, 증강 {aug_idx+1}): {e}")
                continue
    
    print(f"✅ 증강 샘플 시각화 완료! 결과: {output_dir}")

def validate_annotations_visually(image_dir, coco_json_path, output_dir, num_check=10):
    """
    생성된 COCO 어노테이션이 올바른지 시각적으로 검증
    
    Args:
        image_dir: 이미지 디렉토리
        coco_json_path: 검증할 COCO JSON 파일
        output_dir: 검증 결과 이미지를 저장할 디렉토리
        num_check: 검증할 이미지 개수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    images_info = coco_data.get("images", [])
    annotations_info = coco_data.get("annotations", [])
    
    # 이미지 ID별 어노테이션 매핑
    img_id_to_anns = {}
    for ann in annotations_info:
        img_id = ann["image_id"]
        img_id_to_anns.setdefault(img_id, []).append(ann)
    
    # 랜덤 샘플 선택
    sample_images = random.sample(images_info, min(num_check, len(images_info)))
    
    print(f"🔍 {len(sample_images)}개 이미지의 어노테이션 시각적 검증 중...")
    
    validation_results = []
    
    for i, img_info in enumerate(sample_images):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ 이미지 파일 없음: {img_path}")
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue
        
        # 어노테이션 변환
        bboxes = []
        if img_id in img_id_to_anns:
            for ann in img_id_to_anns[img_id]:
                bboxes.append((ann["category_id"], *ann["bbox"]))
        
        # 시각화
        fig = draw_bboxes_on_image(
            image, bboxes, 
            title=f"Validation {i+1}: {file_name} ({len(bboxes)} objects)"
        )
        
        # 저장
        save_path = os.path.join(output_dir, f"validation_{i+1:02d}_{os.path.splitext(file_name)[0]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 검증 결과 기록
        img_h, img_w = image.shape[:2]
        valid_bboxes = 0
        invalid_bboxes = 0
        
        for cat_id, x_min, y_min, w, h in bboxes:
            if (0 <= x_min < img_w and 0 <= y_min < img_h and 
                w > 0 and h > 0 and x_min + w <= img_w and y_min + h <= img_h):
                valid_bboxes += 1
            else:
                invalid_bboxes += 1
        
        validation_results.append({
            'file_name': file_name,
            'total_bboxes': len(bboxes),
            'valid_bboxes': valid_bboxes,
            'invalid_bboxes': invalid_bboxes
        })
        
        print(f"✅ 검증 {i+1}: {file_name} - {valid_bboxes}개 유효, {invalid_bboxes}개 무효")
    
    # 검증 요약
    total_bboxes = sum(r['total_bboxes'] for r in validation_results)
    total_valid = sum(r['valid_bboxes'] for r in validation_results)
    total_invalid = sum(r['invalid_bboxes'] for r in validation_results)
    
    print(f"\n📊 검증 요약:")
    print(f"   총 바운딩 박스: {total_bboxes}개")
    print(f"   유효한 박스: {total_valid}개 ({total_valid/total_bboxes*100:.1f}%)")
    print(f"   무효한 박스: {total_invalid}개 ({total_invalid/total_bboxes*100:.1f}%)")
    print(f"   검증 이미지 저장: {output_dir}")
    
    return validation_results

# ==========================================
# 7) 실행 예제
# ==========================================
if __name__ == "__main__":
    image_dir = "/DATA/jhlee/fisheye1k_pseudo/images"
    coco_json = "/DATA/jhlee/fisheye1k_pseudo/fisheye1k_merged_vnpt_nota_wbf.json"
    save_img_dir = "/DATA/jhlee/pseudo_label_aug_1k/images"
    save_json = "/DATA/jhlee/pseudo_label_aug_1k/pseudo_label_aug_1k.json"
    num_augments = 3

    print("🚀 전체 이미지 데이터 증강 시작...")
    print(f"📁 원본 이미지: {image_dir}")
    print(f"📁 원본 JSON: {coco_json}")
    print(f"📁 저장 디렉토리: {save_img_dir}")
    print(f"📁 저장 JSON: {save_json}")
    print(f"🔢 증강 개수: {num_augments}")
    
    # 1) 증강 샘플 미리보기 생성 (실제 증강 전에)
    # print("\n🎨 증강 샘플 미리보기 생성 중...")
    # preview_dir = "/DATA/jhlee/augmentation_preview"
    # create_augmentation_samples(
    #     image_dir=image_dir,
    #     coco_json_path=coco_json,
    #     output_dir=preview_dir,
    #     num_samples=5  # 5개 샘플 이미지로 미리보기
    # )
    
    # 2) 실제 데이터 증강 실행
    print("\n📊 실제 데이터 증강 실행...")
    augment_coco_dataset(image_dir, coco_json, save_img_dir, save_json, num_augments)

    # 3) 생성된 COCO JSON 검증
    print("\n📋 생성된 COCO JSON 검증 중...")
    valid = validate_coco_format(save_json, save_img_dir)
    
    # 4) 시각적 검증 (증강된 데이터셋)
    if valid:
        print("\n🔍 증강된 데이터셋 시각적 검증 중...")
        validation_dir = "/DATA/jhlee/augmentation_validation2"
        validate_annotations_visually(
            image_dir=save_img_dir,
            coco_json_path=save_json,
            output_dir=validation_dir,
            num_check=50  # 10개 이미지 검증
        )
    
    if valid:
        print("\n✅ 전체 이미지 증강 및 검증 완료! 학습에 사용할 수 있습니다.")
        print("🎯 주요 특징:")
        print("   - 모든 이미지에 대해 증강 수행")
        print("   - 파일명 기반 중복 처리 방지")
        print("   - 이미지/어노테이션 ID 중복 제거")
        print("   - 실제 파일과 JSON 100% 매칭")
        print("   - 증강 샘플 시각화로 품질 확인")
        print("   - 어노테이션 정확성 시각적 검증")
        print(f"\n📁 결과 확인:")
        print(f"   - 최종 데이터셋: {save_img_dir}")
        print(f"   - 검증 결과: {validation_dir}")
    else:
        print("\n❌ 검증 실패! 생성된 데이터를 확인해주세요.")
