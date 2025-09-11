"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional
import random
import math

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import numpy as np
import cv2

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomResizedCrop = register()(T.RandomResizedCrop)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # Check for empty bounding boxes before applying RandomIoUCrop
        sample = inputs[0] if len(inputs) == 1 else inputs
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
            if isinstance(target, dict) and 'boxes' in target:
                boxes = target['boxes']
                # Skip IoU crop if no boxes or empty boxes
                if boxes.numel() == 0 or boxes.shape[0] == 0:
                    return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            # spatial_size is (H, W), we need (W, H) for boxes (x, y, x, y)  
            h, w = spatial_size
            # Ensure normalization values are positive to prevent division issues
            norm_values = torch.tensor([w, h, w, h], dtype=inpt.dtype, device=inpt.device)
            norm_values = torch.clamp(norm_values, min=1.0)  # Prevent division by zero/small values
            
            # Check for invalid values before normalization
            if torch.isnan(inpt).any() or torch.isinf(inpt).any():
                print("⚠️ Critical: NaN/Inf detected in boxes before normalization")
                inpt = torch.nan_to_num(inpt, nan=0.0, posinf=float(w), neginf=0.0)
            
            inpt = inpt / norm_values[None]
            
            # Clamp normalized boxes to valid range [0, 1] and ensure valid dimensions
            inpt = torch.clamp(inpt, min=0.0, max=1.0)
            
            # Additional check for cxcywh format - ensure width and height are positive
            if self.fmt.lower() == 'cxcywh':
                # For cxcywh: [cx, cy, w, h]
                min_size = 1e-2  # 더 큰 최소값으로 안정성 향상
                
                # Check for invalid values
                if torch.any(inpt[:, 2:] <= 0):
                    print("⚠️ Critical: Non-positive width/height detected in cxcywh format")
                    print(f"   Invalid values: {inpt[inpt[:, 2:] <= 0]}")
                
                inpt[:, 2:] = torch.clamp(inpt[:, 2:], min=min_size)  # w, h
                
                # Additional safety: ensure centers are in valid range considering width/height
                max_cx = 1.0 - inpt[:, 2] / 2  # cx can't be too close to right edge
                max_cy = 1.0 - inpt[:, 3] / 2  # cy can't be too close to bottom edge
                min_cx = inpt[:, 2] / 2        # cx can't be too close to left edge
                min_cy = inpt[:, 3] / 2        # cy can't be too close to top edge
                
                inpt[:, 0] = torch.clamp(inpt[:, 0], min=min_cx, max=max_cx)  # cx
                inpt[:, 1] = torch.clamp(inpt[:, 1], min=min_cy, max=max_cy)  # cy
                
                # Final safety check
                if torch.any(inpt[:, 2:] <= 0):
                    print("⚠️ Critical: Still have non-positive width/height after clamping!")
                    inpt[:, 2:] = torch.clamp(inpt[:, 2:], min=min_size)
                
                # Check for NaN/Inf after all operations
                if torch.isnan(inpt).any() or torch.isinf(inpt).any():
                    print("⚠️ Critical: NaN/Inf detected after cxcywh processing")
                    inpt = torch.nan_to_num(inpt, nan=0.5, posinf=1.0, neginf=0.0)

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt


# ==========================================
# Custom Augmentation Transforms (from data_aug.py)
# ==========================================

@register()
class RandomBrightnessContrast(T.Transform):
    """Random brightness and contrast adjustment"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5):
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        brightness_factor = 1.0 + random.uniform(*self.brightness_limit)
        contrast_factor = 1.0 + random.uniform(*self.contrast_limit)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                inpt = F.adjust_brightness(inpt, brightness_factor)
                inpt = F.adjust_contrast(inpt, contrast_factor)
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class CLAHE(T.Transform):
    """Contrast Limited Adaptive Histogram Equalization"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, clip_limit=(1, 3), tile_grid_size=(16, 16), p=1.0):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        clip_limit = random.uniform(*self.clip_limit)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                # Convert to numpy for CLAHE processing
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = inpt.numpy().transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.tile_grid_size)
                if len(img_np.shape) == 3:
                    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    img_np = clahe.apply(img_np)
                
                # Convert back
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(img_np)
                else:
                    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class RandomGamma(T.Transform):
    """Random gamma correction"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, gamma_limit=(80, 120), p=0.3):
        super().__init__()
        self.gamma_limit = gamma_limit
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        gamma = random.uniform(*self.gamma_limit) / 100.0
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                inpt = F.adjust_gamma(inpt, gamma)
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class BBoxMotionBlur(T.Transform):
    """Motion blur applied only to bounding box regions"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, blur_limit=(5, 7), angle_limit=(-30, 30), p=0.5):
        super().__init__()
        self.blur_limit = blur_limit
        self.angle_limit = angle_limit
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Find image and bboxes
        image_input = None
        bbox_input = None
        outputs = []
        
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
            elif isinstance(inpt, BoundingBoxes):
                bbox_input = inpt
            outputs.append(inpt)
        
        if image_input is None or bbox_input is None:
            return outputs if len(outputs) > 1 else outputs[0]
        
        # Convert image to numpy
        if isinstance(image_input, PIL.Image.Image):
            img_np = np.array(image_input)
        else:
            img_np = image_input.numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        
        # Generate motion blur kernel
        kernel_size = random.randint(*self.blur_limit)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        angle = random.uniform(*self.angle_limit)
        
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        k[kernel_size // 2, :] = 1.0
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5),
            angle, 1
        )
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
        k /= k.sum()
        
        # Apply blur to bbox regions
        output_img = img_np.copy()
        spatial_size = getattr(bbox_input, _boxes_keys[1])
        
        # Convert normalized boxes to pixel coordinates
        boxes = bbox_input.clone()
        if bbox_input.format.value.lower() == 'cxcywh':
            # Convert from center format to xyxy
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        if boxes.max() <= 1.0:
            boxes = boxes * torch.tensor([w, h, w, h])
        
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            roi = output_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            blurred = cv2.filter2D(roi, -1, k)
            output_img[y1:y2, x1:x2] = blurred
        
        # Convert back to original format
        for i, inpt in enumerate(inputs):
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    outputs[i] = PIL.Image.fromarray(output_img)
                else:
                    img_tensor = torch.from_numpy(output_img.transpose(2, 0, 1)).float() / 255.0
                    outputs[i] = Image(img_tensor)
        
        return outputs if len(outputs) > 1 else outputs[0]


# RandomRotate is now implemented as SimpleRandomRotate (defined later in this file)


@register()
class GaussianNoise(T.Transform):
    """Add Gaussian noise to image"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, var_limit=(10.0, 50.0), p=0.4):
        super().__init__()
        self.var_limit = var_limit
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Use variance directly instead of square root
        variance = random.uniform(*self.var_limit)
        std = variance  # Use variance as standard deviation for more visible noise
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt).astype(np.float32)
                else:
                    img_np = inpt.numpy().transpose(1, 2, 0) * 255.0
                
                # Generate noise with the specified standard deviation
                noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
                noisy_img = img_np + noise
                
                # Clip to valid range
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(noisy_img)
                else:
                    img_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class JPEGCompression(T.Transform):
    """JPEG compression simulation"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, quality_lower=85, quality_upper=100, p=1.0):
        super().__init__()
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        quality = random.randint(self.quality_lower, self.quality_upper)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    # Simulate JPEG compression
                    import io
                    buffer = io.BytesIO()
                    inpt.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    inpt = PIL.Image.open(buffer)
                else:
                    # Convert to PIL, compress, convert back
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    pil_img = PIL.Image.fromarray(img_np)
                    
                    import io
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    compressed_pil = PIL.Image.open(buffer)
                    
                    compressed_np = np.array(compressed_pil)
                    img_tensor = torch.from_numpy(compressed_np.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class RandomShadow(T.Transform):
    """Add random shadows to image"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                 num_shadows_upper=2, shadow_dimension=5, p=0.3):
        super().__init__()
        self.shadow_roi = shadow_roi
        self.num_shadows_lower = num_shadows_lower
        self.num_shadows_upper = num_shadows_upper
        self.shadow_dimension = shadow_dimension
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        num_shadows = random.randint(self.num_shadows_lower, self.num_shadows_upper)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                
                for _ in range(num_shadows):
                    # Create random shadow polygon
                    x1 = int(w * self.shadow_roi[0])
                    y1 = int(h * self.shadow_roi[1])
                    x2 = int(w * self.shadow_roi[2])
                    y2 = int(h * self.shadow_roi[3])
                    
                    # Random shadow vertices
                    vertices = []
                    for _ in range(self.shadow_dimension):
                        x = random.randint(x1, x2)
                        y = random.randint(y1, y2)
                        vertices.append([x, y])
                    
                    vertices = np.array(vertices, dtype=np.int32)
                    
                    # Create shadow mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [vertices], 255)
                    
                    # Apply shadow (darken)
                    shadow_intensity = random.uniform(0.3, 0.7)
                    img_np = img_np.astype(np.float32)
                    img_np[mask > 0] *= shadow_intensity
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(img_np)
                else:
                    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class CoarseDropout(T.Transform):
    """CoarseDropout of rectangular regions in the image (similar to Albumentations CoarseDropout)"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, max_holes=8, max_height=8, max_width=8, min_holes=None, 
                 min_height=None, min_width=None, fill_value=0, p=0.5):
        super().__init__()
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Number of holes to create
        num_holes = random.randint(self.min_holes, self.max_holes)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                
                # Create holes
                for _ in range(num_holes):
                    # Random hole dimensions
                    hole_height = random.randint(self.min_height, self.max_height)
                    hole_width = random.randint(self.min_width, self.max_width)
                    
                    # Ensure hole dimensions don't exceed image dimensions
                    hole_height = min(hole_height, h)
                    hole_width = min(hole_width, w)
                    
                    # Random position for the hole
                    y1 = random.randint(0, max(0, h - hole_height))
                    x1 = random.randint(0, max(0, w - hole_width))
                    y2 = y1 + hole_height
                    x2 = x1 + hole_width
                    
                    # Fill the hole
                    if isinstance(self.fill_value, (int, float)):
                        img_np[y1:y2, x1:x2] = self.fill_value
                    elif isinstance(self.fill_value, (list, tuple)):
                        # Per-channel fill values
                        if len(img_np.shape) == 3:
                            for c in range(min(len(self.fill_value), img_np.shape[2])):
                                img_np[y1:y2, x1:x2, c] = self.fill_value[c]
                        else:
                            img_np[y1:y2, x1:x2] = self.fill_value[0]
                
                # Convert back to original format
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(img_np)
                else:
                    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class Transpose(T.Transform):
    """Transpose the input by swapping its rows and columns (similar to Albumentations Transpose)"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    # For PIL Image, transpose using PIL's transpose method
                    inpt = inpt.transpose(PIL.Image.Transpose.TRANSPOSE)
                else:
                    # For torch Image, transpose the tensor
                    # Image tensor shape is (C, H, W), we want to swap H and W
                    img_tensor = inpt.permute(0, 2, 1)  # (C, H, W) -> (C, W, H)
                    inpt = Image(img_tensor)
            elif isinstance(inpt, BoundingBoxes):
                # For bounding boxes, we need to swap coordinates and adjust for new dimensions
                spatial_size = getattr(inpt, _boxes_keys[1])  # (H, W)
                new_spatial_size = (spatial_size[1], spatial_size[0])  # (W, H)
                
                boxes = inpt.clone()
                
                # Convert to xyxy format if needed
                if inpt.format.value.lower() == 'cxcywh':
                    boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
                
                # Swap x and y coordinates and adjust for new dimensions
                # Original: [x_min, y_min, x_max, y_max]
                # After transpose: [y_min, x_min, y_max, x_max] but adjusted for new dimensions
                if boxes.max() <= 1.0:  # Normalized coordinates
                    # For normalized coordinates, just swap x and y
                    new_boxes = torch.stack([
                        boxes[:, 1],  # y_min -> x_min
                        boxes[:, 0],  # x_min -> y_min  
                        boxes[:, 3],  # y_max -> x_max
                        boxes[:, 2]   # x_max -> y_max
                    ], dim=1)
                else:  # Absolute coordinates
                    # For absolute coordinates, swap and adjust for dimension change
                    h, w = spatial_size
                    new_boxes = torch.stack([
                        boxes[:, 1],  # y_min -> x_min
                        boxes[:, 0],  # x_min -> y_min
                        boxes[:, 3],  # y_max -> x_max  
                        boxes[:, 2]   # x_max -> y_max
                    ], dim=1)
                
                # Convert back to original format if needed
                if inpt.format.value.lower() == 'cxcywh':
                    new_boxes = torchvision.ops.box_convert(new_boxes, 'xyxy', 'cxcywh')
                
                # Create new BoundingBoxes with swapped spatial size
                inpt = convert_to_tv_tensor(
                    new_boxes,
                    key="boxes",
                    box_format=inpt.format.value,
                    spatial_size=new_spatial_size
                )
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class ResizeWithPadding(T.Transform):
    """Resize with padding to maintain aspect ratio"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (width, height)
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, *inputs: Any) -> Any:
        # Find image input to get original size
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get original size
        if isinstance(image_input, PIL.Image.Image):
            orig_w, orig_h = image_input.size
        else:
            _, orig_h, orig_w = image_input.shape
        
        target_w, target_h = self.size
        
        # Calculate scale to fit within target size while maintaining aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        
        # Calculate new size after scaling
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Calculate padding needed
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        # Padding: (left, top, right, bottom)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                # First resize to maintain aspect ratio
                inpt = F.resize(inpt, (new_h, new_w))
                # Then pad to target size
                inpt = F.pad(inpt, padding=padding, fill=self.fill, padding_mode=self.padding_mode)
            elif isinstance(inpt, BoundingBoxes):
                # Update bounding boxes for scaling and padding
                boxes = inpt.clone()
                
                # Scale boxes
                boxes = boxes * scale
                
                # Adjust for padding offset
                boxes[:, 0] += pad_left  # x coordinates
                boxes[:, 1] += pad_top   # y coordinates
                if inpt.format.value.lower() == 'xyxy':
                    boxes[:, 2] += pad_left  # x2 coordinates
                    boxes[:, 3] += pad_top   # y2 coordinates
                
                # Update spatial size
                spatial_size = (target_w, target_h)
                inpt = convert_to_tv_tensor(
                    boxes, key="boxes", 
                    box_format=inpt.format.value, 
                    spatial_size=spatial_size
                )
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FixedRandomHorizontalFlip:
    """Horizontally flip image and bounding boxes."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        flipped_image = F.hflip(image)
        boxes = target['boxes']
        coords = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        fmt = boxes.format.value.lower() if hasattr(boxes, 'format') else 'xyxy'
        xyxy = coords if fmt == 'xyxy' else torchvision.ops.box_convert(coords, fmt, 'xyxy')
        w = image.size[0] if isinstance(image, PIL.Image.Image) else image.shape[-1]
        x1 = w - xyxy[:, 2]
        x2 = w - xyxy[:, 0]
        flipped = torch.stack([x1, xyxy[:,1], x2, xyxy[:,3]], dim=1)
        new_coords = flipped if fmt=='xyxy' else torchvision.ops.box_convert(flipped, 'xyxy', fmt)
        new_boxes = convert_to_tv_tensor(new_coords,
                                         key='boxes',
                                         box_format=boxes.format.value,
                                         spatial_size=getattr(boxes, _boxes_keys[1]))
        new_target = target.copy()
        new_target['boxes'] = new_boxes
        return flipped_image, new_target

@register()
class FixedRandomVerticalFlip:
    """Vertically flip image and bounding boxes."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        flipped_image = F.vflip(image)
        boxes = target['boxes']
        coords = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        fmt = boxes.format.value.lower() if hasattr(boxes, 'format') else 'xyxy'
        xyxy = coords if fmt == 'xyxy' else torchvision.ops.box_convert(coords, fmt, 'xyxy')
        h = image.size[1] if isinstance(image, PIL.Image.Image) else image.shape[1]
        y1 = h - xyxy[:,3]
        y2 = h - xyxy[:,1]
        flipped = torch.stack([xyxy[:,0], y1, xyxy[:,2], y2], dim=1)
        new_coords = flipped if fmt=='xyxy' else torchvision.ops.box_convert(flipped, 'xyxy', fmt)
        new_boxes = convert_to_tv_tensor(new_coords,
                                         key='boxes',
                                         box_format=boxes.format.value,
                                         spatial_size=getattr(boxes, _boxes_keys[1]))
        new_target = target.copy()
        new_target['boxes'] = new_boxes
        return flipped_image, new_target

@register()
class SimpleRandomRotate(T.Transform):
    """
    Simple and robust random rotation transform based on PIL/torchvision.
    
    이미지와 바운딩박스를 함께 랜덤 회전시키는 Transform.
    사용자 제공 코드 기반으로 간소화된 구현.
    
    Args:
        limit (float): 최대 회전 각도 (degrees). ±limit 범위에서 랜덤 선택
        p (float): 적용 확률
        fill (int or tuple): 회전 후 생기는 빈 영역 채울 색값
        expand (bool): True면 회전 후 이미지가 잘리지 않도록 캔버스 확장
    """
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, limit=30, p=0.5, fill=0, expand=False):
        super().__init__()
        self.limit = limit
        self.p = p
        self.fill = fill
        self.expand = expand

    @staticmethod
    def _rotate_box(boxes, angle, img_size, expand):
        """
        boxes: Tensor[N,4] (xmin, ymin, xmax, ymax) 
        angle: float, degrees
        img_size: (width, height) before rotation
        expand: bool
        """
        w, h = img_size
        cx, cy = w / 2.0, h / 2.0
        theta = math.radians(angle)

        # 새로운 캔버스 크기 계산
        if expand:
            new_w = abs(w * math.cos(theta)) + abs(h * math.sin(theta))
            new_h = abs(w * math.sin(theta)) + abs(h * math.cos(theta))
        else:
            new_w, new_h = w, h
        nx_cx, nx_cy = new_w / 2.0, new_h / 2.0

        # 각 bbox의 네 모서리 (N,4,2)
        x1, y1, x2, y2 = boxes.unbind(-1)
        corners = torch.stack([
            torch.stack([x1, y1], dim=1),  # top-left
            torch.stack([x2, y1], dim=1),  # top-right
            torch.stack([x2, y2], dim=1),  # bottom-right
            torch.stack([x1, y2], dim=1),  # bottom-left
        ], dim=1)  # (N,4,2)

        # 회전 함수
        def rotate_pts(pts):
            # pts: (N,4,2)
            x_rel = pts[...,0] - cx
            y_rel = pts[...,1] - cy
            x_rot = x_rel * math.cos(theta) - y_rel * math.sin(theta) + nx_cx
            y_rot = x_rel * math.sin(theta) + y_rel * math.cos(theta) + nx_cy
            return torch.stack([x_rot, y_rot], dim=-1)

        # 모든 코너 회전
        rot_corners = rotate_pts(corners)
        x_coords = rot_corners[...,0]
        y_coords = rot_corners[...,1]
        
        # 회전된 코너들로부터 axis-aligned bbox 생성
        new_x1 = x_coords.min(dim=1)[0]
        new_y1 = y_coords.min(dim=1)[0]
        new_x2 = x_coords.max(dim=1)[0]
        new_y2 = y_coords.max(dim=1)[0]
        
        # 이미지 경계 내로 클램핑
        new_x1 = torch.clamp(new_x1, 0, new_w)
        new_y1 = torch.clamp(new_y1, 0, new_h)
        new_x2 = torch.clamp(new_x2, 0, new_w)
        new_y2 = torch.clamp(new_y2, 0, new_h)
        
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    def __call__(self, *inputs: Any) -> Any:
        # 디버깅용 로그 (비활성화)
        # print(f"RandomRotate called with {len(inputs)} inputs, p={self.p}")
        
        rand_val = torch.rand(1).item()
        if rand_val >= self.p:
            # print(f"Skipping rotation: rand={rand_val:.3f} >= p={self.p}")
            return inputs if len(inputs) > 1 else inputs[0]
            
        # 랜덤 각도 선택
        angle = random.uniform(-self.limit, self.limit)
        # print(f"Applying rotation: angle={angle:.1f}°")
        
        # 이미지와 bbox 찾기
        image_input = None
        bbox_input = None
        
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
            elif isinstance(inpt, BoundingBoxes):
                bbox_input = inpt
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # 원본 이미지 크기
        if isinstance(image_input, PIL.Image.Image):
            w, h = image_input.size
        else:
            # Handle both CHW and HWC formats
            if len(image_input.shape) == 3:
                if image_input.shape[0] <= 4:  # CHW format (channels first)
                    _, h, w = image_input.shape
                else:  # HWC format (channels last)
                    h, w, _ = image_input.shape
            else:
                return inputs if len(inputs) > 1 else inputs[0]
        
        orig_size = (w, h)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                # PIL Image 회전 (간단하고 신뢰성 있음)
                import torchvision.transforms.functional as TF
                inpt = TF.rotate(inpt, angle, fill=self.fill, expand=self.expand)
            elif isinstance(inpt, BoundingBoxes) and bbox_input is not None:
                # bbox 회전
                if inpt.numel() == 0:
                    # 빈 bbox인 경우 그대로 유지
                    pass
                else:
                    # 포맷 변환
                    original_format = inpt.format.value
                    if original_format.lower() == 'cxcywh':
                        xyxy_boxes = torchvision.ops.box_convert(inpt, 'cxcywh', 'xyxy')
                    elif original_format.lower() not in ['xyxy', 'xywh']:
                        xyxy_boxes = torchvision.ops.box_convert(inpt, original_format.lower(), 'xyxy')
                    else:
                        xyxy_boxes = inpt
                    
                    # bbox 회전
                    new_boxes_tensor = self._rotate_box(xyxy_boxes, angle, orig_size, self.expand)
                    
                    # 유효한 bbox 필터링 (최소 1픽셀 크기)
                    valid_mask = (new_boxes_tensor[:, 2] - new_boxes_tensor[:, 0] >= 1) & \
                                 (new_boxes_tensor[:, 3] - new_boxes_tensor[:, 1] >= 1)
                    
                    if valid_mask.any():
                        # 유효한 bbox만 유지
                        new_boxes_tensor = new_boxes_tensor[valid_mask]
                        
                        # 포맷 변환
                        if original_format.lower() == 'cxcywh':
                            new_coords = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
                        elif original_format.lower() not in ['xyxy']:
                            new_coords = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', original_format.lower())
                        else:
                            new_coords = new_boxes_tensor
                        
                        # 새로운 이미지 크기 (expand된 경우)
                        if self.expand:
                            # 회전된 이미지에서 크기 가져오기
                            rotated_image = None
                            for inp in outputs:
                                if isinstance(inp, (PIL.Image.Image, Image)):
                                    rotated_image = inp
                                    break
                            
                            if rotated_image is not None:
                                if isinstance(rotated_image, PIL.Image.Image):
                                    new_w, new_h = rotated_image.size
                                else:
                                    if len(rotated_image.shape) == 3:
                                        if rotated_image.shape[0] <= 4:  # CHW format
                                            _, new_h, new_w = rotated_image.shape
                                        else:  # HWC format
                                            new_h, new_w, _ = rotated_image.shape
                            else:
                                new_w, new_h = w, h
                        else:
                            new_w, new_h = w, h
                        
                        # BoundingBoxes 객체 생성
                        spatial_size = (new_h, new_w)
                        inpt = convert_to_tv_tensor(
                            new_coords,
                            key="boxes",
                            box_format=original_format,
                            spatial_size=spatial_size
                        )
                    else:
                        # 모든 bbox가 필터링된 경우 빈 bbox 생성
                        empty_boxes = torch.empty((0, 4), dtype=inpt.dtype, device=inpt.device)
                        spatial_size = getattr(inpt, _boxes_keys[1])
                        inpt = convert_to_tv_tensor(
                            empty_boxes,
                            key="boxes",
                            box_format=inpt.format.value,
                            spatial_size=spatial_size
                        )
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


# Albumentations 기반 RandomRotate
class AlbumentationsRandomRotate:
    """
    Albumentations 기반 RandomRotate - 매우 검증된 구현
    """
    def __init__(self, limit=30, p=0.5, fill=0):
        import albumentations as A
        self.limit = limit
        self.p = p
        self.fill = fill
        
        # Albumentations transform 생성
        self.transform = A.Compose([
            A.Rotate(limit=limit, p=1.0, border_mode=0, fill=fill)  # 올바른 파라미터는 fill
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
        
        # PIL Image를 numpy array로 변환
        if isinstance(image, PIL.Image.Image):
            img_np = np.array(image)
        else:
            # Tensor인 경우 numpy로 변환
            if len(image.shape) == 3 and image.shape[0] <= 4:  # CHW format
                img_np = image.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = (image.numpy() * 255).astype(np.uint8) if image.max() <= 1.0 else image.numpy().astype(np.uint8)
        
        h, w = img_np.shape[:2]
        
        # D-FINE BoundingBoxes를 albumentations format으로 변환
        boxes = target['boxes']
        coords = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        fmt = boxes.format.value.lower() if hasattr(boxes, 'format') else 'xyxy'
        
        # xyxy format으로 변환
        if fmt == 'xyxy':
            xyxy = coords
        else:
            xyxy = torchvision.ops.box_convert(coords, fmt, 'xyxy')
        
        if xyxy.numel() == 0:
            return image, target
        
        # Pixel coordinates를 normalized coordinates로 변환 (albumentations format)
        if xyxy.max() > 1.0:  # pixel coordinates
            norm_boxes = xyxy / torch.tensor([w, h, w, h], device=xyxy.device, dtype=xyxy.dtype)
        else:  # already normalized
            norm_boxes = xyxy
        
        # albumentations format: [x_min, y_min, x_max, y_max] (normalized)
        albu_boxes = norm_boxes.cpu().numpy().tolist()
        
        # Labels 추출 (있으면)
        if 'labels' in target and len(target['labels']) == len(albu_boxes):
            class_labels = target['labels'].cpu().numpy().tolist() if hasattr(target['labels'], 'cpu') else target['labels'].tolist()
        else:
            class_labels = [1] * len(albu_boxes)  # 기본값
        
        try:
            # Albumentations 적용
            result = self.transform(image=img_np, bboxes=albu_boxes, class_labels=class_labels)
            
            rotated_img_np = result['image']
            rotated_boxes = result['bboxes']
            rotated_labels = result['class_labels']
            
        except Exception as e:
            print(f"Albumentations transform failed: {e}")
            return image, target
        
        # 결과를 다시 PIL Image로 변환
        if isinstance(image, PIL.Image.Image):
            rotated_image = PIL.Image.fromarray(rotated_img_np)
        else:
            # Tensor로 변환
            rotated_image = torch.from_numpy(rotated_img_np.transpose(2, 0, 1)).float() / 255.0
        
        # 빈 결과 처리
        if not rotated_boxes:
            # 모든 bbox가 필터링된 경우
            empty_boxes = torch.empty((0, 4), dtype=coords.dtype, device=coords.device)
            empty_labels = torch.empty((0,), dtype=target['labels'].dtype, device=target['labels'].device) if 'labels' in target else None
            
            new_target = target.copy()
            new_target['boxes'] = convert_to_tv_tensor(
                empty_boxes, key='boxes', box_format=boxes.format.value, spatial_size=(h, w)
            )
            if empty_labels is not None:
                new_target['labels'] = empty_labels
            
            return rotated_image, new_target
        
        # Normalized coordinates를 다시 원본 format으로 변환
        rotated_boxes_tensor = torch.tensor(rotated_boxes, dtype=coords.dtype, device=coords.device)
        
        # Pixel coordinates로 변환 (필요시)
        if xyxy.max() > 1.0:  # 원본이 pixel coordinates였으면
            rotated_boxes_tensor = rotated_boxes_tensor * torch.tensor([w, h, w, h], device=coords.device, dtype=coords.dtype)
        
        # 원본 format으로 변환
        if fmt != 'xyxy':
            rotated_boxes_tensor = torchvision.ops.box_convert(rotated_boxes_tensor, 'xyxy', fmt)
        
        # BoundingBoxes 객체 생성
        new_boxes = convert_to_tv_tensor(
            rotated_boxes_tensor, key='boxes', box_format=boxes.format.value, spatial_size=(h, w)
        )
        
        # Target 업데이트
        new_target = target.copy()
        new_target['boxes'] = new_boxes
        
        # Labels 업데이트 (필터링된 경우)
        if 'labels' in target:
            if len(rotated_labels) != len(target['labels']):
                # 일부 bbox가 필터링된 경우
                new_labels = torch.tensor(rotated_labels, dtype=target['labels'].dtype, device=target['labels'].device)
                new_target['labels'] = new_labels
                
                # 다른 관련 필드들도 같은 길이로 맞춤
                for key in ['area', 'iscrowd', 'image_id']:
                    if key in target and hasattr(target[key], '__len__') and len(target[key]) == len(target['labels']):
                        # 임시로 동일한 길이로 맞춤 (실제로는 더 정교한 처리 필요)
                        if len(rotated_labels) > 0:
                            new_target[key] = target[key][:len(rotated_labels)]
        
        return rotated_image, new_target


# Pipeline용 SimpleRandomRotate (image, target 방식)
class SimpleRandomRotateForPipeline:
    """
    Pipeline 전용 SimpleRandomRotate - (image, target) 방식으로 호출
    이전 FixedRandomRotate와 동일한 방식으로 작동하되, 정확한 4-코너 회전 사용
    """
    def __init__(self, limit=30, p=0.5, fill=0, expand=False):
        self.limit = limit
        self.p = p
        self.fill = fill
        self.expand = expand

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target
            
        angle = random.uniform(-self.limit, self.limit)
        rotated_image = F.rotate(image, angle, fill=self.fill)
        boxes = target['boxes']
        coords = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        fmt = boxes.format.value.lower() if hasattr(boxes, 'format') else 'xyxy'
        xyxy = coords if fmt=='xyxy' else torchvision.ops.box_convert(coords, fmt, 'xyxy')
        
        # 이미지 크기 확인
        if hasattr(image, 'shape'):  # Tensor
            _, h, w = image.shape
        else:  # PIL Image
            w, h = image.size
        
        # 이 시점에서는 항상 픽셀 좌표 (정규화는 ConvertBoxes에서 마지막에 수행)
        
        cx, cy = w/2, h/2
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        # 올바른 회전 로직: 각 bbox의 네 모서리를 모두 회전
        new_boxes = []
        valid_indices = []  # 유효한 bbox의 인덱스 추적
        
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            
            # bbox의 네 모서리 좌표
            corners = torch.tensor([
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ], device=box.device, dtype=box.dtype)
            
            # 이미지 중심으로 평행이동
            corners_centered = corners - torch.tensor([cx, cy], device=box.device, dtype=box.dtype)
            
            # 회전 행렬 적용
            rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                                    device=box.device, dtype=box.dtype)
            rotated_corners = corners_centered @ rot_matrix.T
            
            # 다시 이미지 좌표계로 변환
            rotated_corners = rotated_corners + torch.tensor([cx, cy], device=box.device, dtype=box.dtype)
            
            # 회전된 모서리들로부터 axis-aligned bbox 계산
            x_coords = rotated_corners[:, 0]
            y_coords = rotated_corners[:, 1]
            
            new_x1 = x_coords.min()
            new_y1 = y_coords.min()
            new_x2 = x_coords.max()
            new_y2 = y_coords.max()
            
            # 이미지 경계 내로 클램핑
            new_x1 = torch.clamp(new_x1, 0, w)
            new_y1 = torch.clamp(new_y1, 0, h)
            new_x2 = torch.clamp(new_x2, 0, w)
            new_y2 = torch.clamp(new_y2, 0, h)
            
            # 유효한 bbox인지 확인 (최소 1픽셀 크기)
            if new_x2 - new_x1 >= 1 and new_y2 - new_y1 >= 1:
                new_boxes.append(torch.tensor([new_x1, new_y1, new_x2, new_y2], 
                                            device=box.device, dtype=box.dtype))
                valid_indices.append(i)  # 유효한 인덱스 기록
        
        if not new_boxes:
            # 모든 bbox가 필터링된 경우 원본 반환
            return rotated_image, target
            
        out = torch.stack(new_boxes)
        new_coords = out if fmt=='xyxy' else torchvision.ops.box_convert(out, 'xyxy', fmt)
        new_boxes = convert_to_tv_tensor(new_coords,
                                         key='boxes',
                                         box_format=boxes.format.value,
                                         spatial_size=(h,w))
        new_target = target.copy()
        new_target['boxes'] = new_boxes
        
        # 유효한 bbox에 해당하는 다른 필드들도 함께 필터링
        if len(valid_indices) < len(xyxy):
            # valid_indices를 tensor로 변환 (안전한 indexing을 위해)
            valid_indices_tensor = torch.tensor(valid_indices, device=xyxy.device)
            # bbox와 같은 길이를 가져야 하는 필드들 필터링
            for key in ['labels', 'area', 'iscrowd', 'image_id']:
                if key in target and hasattr(target[key], '__len__') and len(target[key]) == len(xyxy):
                    new_target[key] = target[key][valid_indices_tensor]
        
        return rotated_image, new_target

# 기존 호환성을 위한 alias들
@register()
class FixedRandomRotate(SimpleRandomRotate):
    """Alias for SimpleRandomRotate for backward compatibility"""
    pass

# RandomRotate 등록 (SimpleRandomRotate를 사용)
@register()
class RandomRotate(SimpleRandomRotate):
    """RandomRotate implementation using SimpleRandomRotate"""
    pass

@register()
class GeometricAugmentationPipeline:
    """Apply horizontal flip, vertical flip, then rotation."""
    def __init__(self, h_flip_p=0.5, v_flip_p=0.1, rotate_p=0.7, rotate_limit=30, fill=0):
        self.hflip = FixedRandomHorizontalFlip(p=h_flip_p)
        self.vflip = FixedRandomVerticalFlip(p=v_flip_p)
        self.rotate = AlbumentationsRandomRotate(limit=rotate_limit, p=rotate_p, fill=fill)

    def __call__(self, sample):
        # Expect sample to be (image, target, ...)
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
            extra = sample[2:] if len(sample) > 2 else ()
            image, target = self.hflip(image, target)
            image, target = self.vflip(image, target)
            image, target = self.rotate(image, target)
            return (image, target) + extra
        # If single image or unsupported format, return as is
        return sample

@register()
class RandomErase(T.Transform):
    """Random Erasing augmentation
    
    Randomly selects a rectangle region in an image and erases its pixels
    with random values or zero values.
    """
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img = np.array(inpt)
                else:
                    img = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w, c = img.shape
                area = h * w
                
                for _ in range(100):  # Try up to 100 times
                    target_area = random.uniform(*self.scale) * area
                    aspect_ratio = random.uniform(*self.ratio)
                    
                    h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
                    w_erase = int(round(math.sqrt(target_area / aspect_ratio)))
                    
                    if w_erase < w and h_erase < h:
                        x1 = random.randint(0, w - w_erase)
                        y1 = random.randint(0, h - h_erase)
                        
                        if isinstance(self.value, (int, float)):
                            img[y1:y1+h_erase, x1:x1+w_erase] = self.value
                        elif self.value == 'random':
                            img[y1:y1+h_erase, x1:x1+w_erase] = np.random.randint(0, 256, (h_erase, w_erase, c))
                        
                        break
                
                # Convert back to original format
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(img)
                else:
                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class CopyPaste(T.Transform):
    """Copy-Paste augmentation for object detection
    
    Copies objects from one image and pastes them onto another image.
    """
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes, Mask)
    
    def __init__(self, p=0.5, max_paste_objects=3):
        super().__init__()
        self.p = p
        self.max_paste_objects = max_paste_objects
        self._paste_image = None
        self._paste_boxes = None
        self._paste_masks = None
    
    def set_paste_data(self, paste_image, paste_boxes, paste_masks=None):
        """Set the source image, boxes, and optionally masks for copy-paste"""
        self._paste_image = paste_image
        self._paste_boxes = paste_boxes
        self._paste_masks = paste_masks
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p or self._paste_image is None or self._paste_boxes is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        
        # Select random objects to paste
        num_objects = min(len(self._paste_boxes), self.max_paste_objects)
        if num_objects == 0:
            return inputs if len(inputs) > 1 else inputs[0]
        
        paste_indices = random.sample(range(len(self._paste_boxes)), 
                                    random.randint(1, num_objects))
        
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                # Perform copy-paste operation
                if isinstance(inpt, PIL.Image.Image):
                    target_img = np.array(inpt)
                    source_img = np.array(self._paste_image)
                else:
                    target_img = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    source_img = (self._paste_image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h_target, w_target = target_img.shape[:2]
                h_source, w_source = source_img.shape[:2]
                
                # Resize source image if needed
                if (h_source, w_source) != (h_target, w_target):
                    source_img = cv2.resize(source_img, (w_target, h_target))
                
                # Paste selected objects
                for idx in paste_indices:
                    # Get bounding box (assuming normalized xyxy format)
                    box = self._paste_boxes[idx]
                    
                    # Convert to pixel coordinates
                    x1 = int(box[0] * w_target)
                    y1 = int(box[1] * h_target)
                    x2 = int(box[2] * w_target)
                    y2 = int(box[3] * h_target)
                    
                    # Ensure valid box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_target, x2), min(h_target, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Simple copy-paste (could be enhanced with masks)
                        target_img[y1:y2, x1:x2] = source_img[y1:y2, x1:x2]
                
                # Convert back to original format
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(target_img)
                else:
                    img_tensor = torch.from_numpy(target_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
                    
            elif isinstance(inpt, BoundingBoxes):
                # Add pasted object boxes to existing boxes
                paste_boxes = self._paste_boxes[paste_indices]
                
                # Convert to same format as input boxes
                if len(paste_boxes) > 0:
                    # Assuming paste_boxes are in xyxy format, convert if needed
                    if inpt.format.value.lower() != 'xyxy':
                        paste_boxes_converted = torchvision.ops.box_convert(
                            paste_boxes, 'xyxy', inpt.format.value.lower()
                        )
                    else:
                        paste_boxes_converted = paste_boxes
                    
                    # Combine with existing boxes
                    combined_boxes = torch.cat([inpt, paste_boxes_converted], dim=0)
                    spatial_size = getattr(inpt, _boxes_keys[1])
                    
                    inpt = convert_to_tv_tensor(
                        combined_boxes,
                        key="boxes",
                        box_format=inpt.format.value,
                        spatial_size=spatial_size
                    )
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]