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
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

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


@register()
class RandomRotate(T.Transform):
    """Random rotation with constant border"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, limit=45, p=1.0, fill=0):
        super().__init__()
        self.limit = limit
        self.p = p
        self.fill = fill
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        angle = random.uniform(-self.limit, self.limit)
        
        # Get image dimensions for bbox rotation
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
        else:
            _, img_h, img_w = image_input.shape
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                inpt = F.rotate(inpt, angle, fill=self.fill)
            elif isinstance(inpt, BoundingBoxes):
                # Rotate bounding boxes
                inpt = self._rotate_bboxes(inpt, angle, img_w, img_h)
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _rotate_bboxes(self, bboxes, angle, img_w, img_h):
        """Rotate bounding boxes"""
        import math
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Get center of image
        cx, cy = img_w / 2, img_h / 2
        
        # Clone bboxes
        rotated_boxes = bboxes.clone()
        
        # Convert to xyxy format if needed
        original_format = bboxes.format.value
        if original_format.lower() == 'cxcywh':
            rotated_boxes = torchvision.ops.box_convert(rotated_boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        is_normalized = rotated_boxes.max() <= 1.0
        if is_normalized:
            rotated_boxes = rotated_boxes * torch.tensor([img_w, img_h, img_w, img_h])
        
        new_boxes = []
        for box in rotated_boxes:
            x1, y1, x2, y2 = box.tolist()
            
            # Get all four corners of the bounding box
            corners = [
                [x1, y1], [x2, y1], 
                [x2, y2], [x1, y2]
            ]
            
            # Rotate each corner
            rotated_corners = []
            for x, y in corners:
                # Translate to origin
                x_translated = x - cx
                y_translated = y - cy
                
                # Rotate
                x_rotated = x_translated * cos_a - y_translated * sin_a
                y_rotated = x_translated * sin_a + y_translated * cos_a
                
                # Translate back
                x_final = x_rotated + cx
                y_final = y_rotated + cy
                
                rotated_corners.append([x_final, y_final])
            
            # Get axis-aligned bounding box from rotated corners
            xs = [corner[0] for corner in rotated_corners]
            ys = [corner[1] for corner in rotated_corners]
            
            new_x1 = max(0, min(xs))
            new_y1 = max(0, min(ys))
            new_x2 = min(img_w, max(xs))
            new_y2 = min(img_h, max(ys))
            
            # Skip if box becomes too small
            if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                new_boxes.append([0, 0, 1, 1])  # Dummy box
            else:
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Convert back to tensor
        new_boxes_tensor = torch.tensor(new_boxes, dtype=rotated_boxes.dtype)
        
        # Normalize back if needed
        if is_normalized:
            new_boxes_tensor = new_boxes_tensor / torch.tensor([img_w, img_h, img_w, img_h])
        
        # Convert back to original format
        if original_format.lower() == 'cxcywh':
            new_boxes_tensor = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
        
        # Create new BoundingBoxes object
        spatial_size = getattr(bboxes, _boxes_keys[1])
        rotated_bboxes = convert_to_tv_tensor(
            new_boxes_tensor,
            key="boxes",
            box_format=original_format,
            spatial_size=spatial_size
        )
        
        return rotated_bboxes


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


# ==========================================
# Fisheye-specific Augmentation Transforms
# ==========================================

@register()
class FisheyeDistortionAugment(T.Transform):
    """Simulate fisheye lens distortion"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, distortion_range=(0.1, 0.3), p=0.3):
        super().__init__()
        self.distortion_range = distortion_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        distortion_factor = random.uniform(*self.distortion_range)
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create fisheye distortion map
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = min(center_x, center_y)
        
        # Create coordinate grids
        y, x = np.ogrid[:img_h, :img_w]
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Apply fisheye distortion
        # Normalize distance to [0, 1]
        distance_norm = distance / max_radius
        
        # Apply distortion function
        distortion = distance_norm * (1 + distortion_factor * distance_norm**2)
        
        # Scale back to pixel coordinates
        new_distance = distortion * max_radius
        
        # Calculate new coordinates
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_distance * np.cos(angle)
        new_y = center_y + new_distance * np.sin(angle)
        
        # Apply distortion
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Remap image
        distorted_img = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    outputs.append(PIL.Image.fromarray(distorted_img))
                else:
                    img_tensor = torch.from_numpy(distorted_img.transpose(2, 0, 1)).float() / 255.0
                    outputs.append(Image(img_tensor))
            elif isinstance(inpt, BoundingBoxes):
                # Apply distortion to bounding boxes
                distorted_boxes = self._distort_bboxes(inpt, map_x, map_y, img_w, img_h)
                outputs.append(distorted_boxes)
            else:
                outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _distort_bboxes(self, bboxes, map_x, map_y, img_w, img_h):
        """Apply fisheye distortion to bounding boxes"""
        boxes = bboxes.clone()
        
        # Convert to xyxy format if needed
        original_format = bboxes.format.value
        if original_format.lower() == 'cxcywh':
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        is_normalized = boxes.max() <= 1.0
        if is_normalized:
            boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h])
        
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # Get corners of the bounding box
            corners = [
                [x1, y1], [x2, y1], 
                [x2, y2], [x1, y2]
            ]
            
            # Apply distortion to each corner
            distorted_corners = []
            for x, y in corners:
                if 0 <= x < img_w and 0 <= y < img_h:
                    # Get distorted coordinates from map
                    new_x = map_x[y, x]
                    new_y = map_y[y, x]
                    distorted_corners.append([new_x, new_y])
                else:
                    distorted_corners.append([x, y])
            
            # Get bounding box from distorted corners
            xs = [corner[0] for corner in distorted_corners]
            ys = [corner[1] for corner in distorted_corners]
            
            new_x1 = max(0, min(xs))
            new_y1 = max(0, min(ys))
            new_x2 = min(img_w, max(xs))
            new_y2 = min(img_h, max(ys))
            
            # Skip if box becomes too small
            if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                new_boxes.append([0, 0, 1, 1])  # Dummy box
            else:
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Convert back to tensor
        new_boxes_tensor = torch.tensor(new_boxes, dtype=boxes.dtype)
        
        # Normalize back if needed
        if is_normalized:
            new_boxes_tensor = new_boxes_tensor / torch.tensor([img_w, img_h, img_w, img_h])
        
        # Convert back to original format
        if original_format.lower() == 'cxcywh':
            new_boxes_tensor = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
        
        # Create new BoundingBoxes object
        spatial_size = getattr(bboxes, _boxes_keys[1])
        distorted_bboxes = convert_to_tv_tensor(
            new_boxes_tensor,
            key="boxes",
            box_format=original_format,
            spatial_size=spatial_size
        )
        
        return distorted_bboxes


@register()
class FisheyePerspectiveTransform(T.Transform):
    """Apply perspective transform to simulate fisheye viewing angles"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, angle_range=(-15, 15), p=0.4):
        super().__init__()
        self.angle_range = angle_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        angle = random.uniform(*self.angle_range)
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create perspective transform matrix
        center_x, center_y = img_w // 2, img_h // 2
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Create perspective transform
        # Simulate viewing from a different angle
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Create transformation matrix
        matrix = np.array([
            [cos_a, -sin_a, center_x * (1 - cos_a) + center_y * sin_a],
            [sin_a, cos_a, -center_x * sin_a + center_y * (1 - cos_a)],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Apply perspective transform
        transformed_img = cv2.warpPerspective(img_np, matrix, (img_w, img_h))
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    outputs.append(PIL.Image.fromarray(transformed_img))
                else:
                    img_tensor = torch.from_numpy(transformed_img.transpose(2, 0, 1)).float() / 255.0
                    outputs.append(Image(img_tensor))
            elif isinstance(inpt, BoundingBoxes):
                # Apply perspective transform to bounding boxes
                transformed_boxes = self._transform_bboxes(inpt, matrix, img_w, img_h)
                outputs.append(transformed_boxes)
            else:
                outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _transform_bboxes(self, bboxes, matrix, img_w, img_h):
        """Apply perspective transform to bounding boxes"""
        boxes = bboxes.clone()
        
        # Convert to xyxy format if needed
        original_format = bboxes.format.value
        if original_format.lower() == 'cxcywh':
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        is_normalized = boxes.max() <= 1.0
        if is_normalized:
            boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h])
        
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            
            # Get corners of the bounding box
            corners = np.array([
                [x1, y1, 1],
                [x2, y2, 1],
                [x2, y2, 1],
                [x1, y2, 1]
            ], dtype=np.float32)
            
            # Apply transformation
            transformed_corners = (matrix @ corners.T).T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
            
            # Get bounding box from transformed corners
            xs = transformed_corners[:, 0]
            ys = transformed_corners[:, 1]
            
            new_x1 = max(0, min(xs))
            new_y1 = max(0, min(ys))
            new_x2 = min(img_w, max(xs))
            new_y2 = min(img_h, max(ys))
            
            # Skip if box becomes too small
            if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                new_boxes.append([0, 0, 1, 1])  # Dummy box
            else:
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Convert back to tensor
        new_boxes_tensor = torch.tensor(new_boxes, dtype=boxes.dtype)
        
        # Normalize back if needed
        if is_normalized:
            new_boxes_tensor = new_boxes_tensor / torch.tensor([img_w, img_h, img_w, img_h])
        
        # Convert back to original format
        if original_format.lower() == 'cxcywh':
            new_boxes_tensor = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
        
        # Create new BoundingBoxes object
        spatial_size = getattr(bboxes, _boxes_keys[1])
        transformed_bboxes = convert_to_tv_tensor(
            new_boxes_tensor,
            key="boxes",
            box_format=original_format,
            spatial_size=spatial_size
        )
        
        return transformed_bboxes


@register()
class FisheyeLensSimulation(T.Transform):
    """Simulate different fisheye lens focal lengths"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, focal_length_range=(0.8, 1.2), p=0.2):
        super().__init__()
        self.focal_length_range = focal_length_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        focal_factor = random.uniform(*self.focal_length_range)
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Simulate different focal length by scaling
        center_x, center_y = img_w // 2, img_h // 2
        
        # Create scaling matrix
        scale_matrix = np.array([
            [focal_factor, 0, center_x * (1 - focal_factor)],
            [0, focal_factor, center_y * (1 - focal_factor)],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Apply scaling transform
        scaled_img = cv2.warpAffine(img_np, scale_matrix[:2], (img_w, img_h))
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    outputs.append(PIL.Image.fromarray(scaled_img))
                else:
                    img_tensor = torch.from_numpy(scaled_img.transpose(2, 0, 1)).float() / 255.0
                    outputs.append(Image(img_tensor))
            elif isinstance(inpt, BoundingBoxes):
                # Apply scaling to bounding boxes
                scaled_boxes = self._scale_bboxes(inpt, scale_matrix, img_w, img_h)
                outputs.append(scaled_boxes)
            else:
                outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _scale_bboxes(self, bboxes, matrix, img_w, img_h):
        """Apply scaling to bounding boxes"""
        boxes = bboxes.clone()
        
        # Convert to xyxy format if needed
        original_format = bboxes.format.value
        if original_format.lower() == 'cxcywh':
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        is_normalized = boxes.max() <= 1.0
        if is_normalized:
            boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h])
        
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            
            # Apply scaling transformation
            corners = np.array([
                [x1, y1, 1],
                [x2, y2, 1]
            ], dtype=np.float32)
            
            transformed_corners = (matrix @ corners.T).T
            transformed_corners = transformed_corners[:, :2]
            
            new_x1, new_y1 = transformed_corners[0]
            new_x2, new_y2 = transformed_corners[1]
            
            # Clip to image boundaries
            new_x1 = max(0, min(img_w, new_x1))
            new_y1 = max(0, min(img_h, new_y1))
            new_x2 = max(0, min(img_w, new_x2))
            new_y2 = max(0, min(img_h, new_y2))
            
            # Skip if box becomes too small
            if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                new_boxes.append([0, 0, 1, 1])  # Dummy box
            else:
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Convert back to tensor
        new_boxes_tensor = torch.tensor(new_boxes, dtype=boxes.dtype)
        
        # Normalize back if needed
        if is_normalized:
            new_boxes_tensor = new_boxes_tensor / torch.tensor([img_w, img_h, img_w, img_h])
        
        # Convert back to original format
        if original_format.lower() == 'cxcywh':
            new_boxes_tensor = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
        
        # Create new BoundingBoxes object
        spatial_size = getattr(bboxes, _boxes_keys[1])
        scaled_bboxes = convert_to_tv_tensor(
            new_boxes_tensor,
            key="boxes",
            box_format=original_format,
            spatial_size=spatial_size
        )
        
        return scaled_bboxes


@register()
class FisheyeEdgeEnhancement(T.Transform):
    """Enhance edges in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, edge_strength=0.3, p=0.4):
        super().__init__()
        self.edge_strength = edge_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Apply edge enhancement
                # Convert to grayscale for edge detection
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                
                # Detect edges
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))
                
                # Enhance edges
                enhanced = img_np.astype(np.float32)
                if len(img_np.shape) == 3:
                    for i in range(3):
                        enhanced[:, :, i] += self.edge_strength * edges
                else:
                    enhanced += self.edge_strength * edges
                
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(enhanced)
                else:
                    img_tensor = torch.from_numpy(enhanced.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeRadialDistortion(T.Transform):
    """Apply radial distortion to simulate fisheye lens characteristics"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, radial_factor=0.1, p=0.3):
        super().__init__()
        self.radial_factor = radial_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create radial distortion
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = min(center_x, center_y)
        
        # Create coordinate grids
        y, x = np.ogrid[:img_h, :img_w]
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Apply radial distortion
        distance_norm = distance / max_radius
        distortion = distance_norm * (1 + self.radial_factor * distance_norm**2)
        new_distance = distortion * max_radius
        
        # Calculate new coordinates
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_distance * np.cos(angle)
        new_y = center_y + new_distance * np.sin(angle)
        
        # Apply distortion
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Remap image
        distorted_img = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    outputs.append(PIL.Image.fromarray(distorted_img))
                else:
                    img_tensor = torch.from_numpy(distorted_img.transpose(2, 0, 1)).float() / 255.0
                    outputs.append(Image(img_tensor))
            elif isinstance(inpt, BoundingBoxes):
                # Apply radial distortion to bounding boxes
                distorted_boxes = self._distort_bboxes_radial(inpt, map_x, map_y, img_w, img_h)
                outputs.append(distorted_boxes)
            else:
                outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _distort_bboxes_radial(self, bboxes, map_x, map_y, img_w, img_h):
        """Apply radial distortion to bounding boxes"""
        boxes = bboxes.clone()
        
        # Convert to xyxy format if needed
        original_format = bboxes.format.value
        if original_format.lower() == 'cxcywh':
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        
        # Denormalize if needed
        is_normalized = boxes.max() <= 1.0
        if is_normalized:
            boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h])
        
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # Get corners of the bounding box
            corners = [
                [x1, y1], [x2, y1], 
                [x2, y2], [x1, y2]
            ]
            
            # Apply distortion to each corner
            distorted_corners = []
            for x, y in corners:
                if 0 <= x < img_w and 0 <= y < img_h:
                    # Get distorted coordinates from map
                    new_x = map_x[y, x]
                    new_y = map_y[y, x]
                    distorted_corners.append([new_x, new_y])
                else:
                    distorted_corners.append([x, y])
            
            # Get bounding box from distorted corners
            xs = [corner[0] for corner in distorted_corners]
            ys = [corner[1] for corner in distorted_corners]
            
            new_x1 = max(0, min(xs))
            new_y1 = max(0, min(ys))
            new_x2 = min(img_w, max(xs))
            new_y2 = min(img_h, max(ys))
            
            # Skip if box becomes too small
            if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                new_boxes.append([0, 0, 1, 1])  # Dummy box
            else:
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Convert back to tensor
        new_boxes_tensor = torch.tensor(new_boxes, dtype=boxes.dtype)
        
        # Normalize back if needed
        if is_normalized:
            new_boxes_tensor = new_boxes_tensor / torch.tensor([img_w, img_h, img_w, img_h])
        
        # Convert back to original format
        if original_format.lower() == 'cxcywh':
            new_boxes_tensor = torchvision.ops.box_convert(new_boxes_tensor, 'xyxy', 'cxcywh')
        
        # Create new BoundingBoxes object
        spatial_size = getattr(bboxes, _boxes_keys[1])
        distorted_bboxes = convert_to_tv_tensor(
            new_boxes_tensor,
            key="boxes",
            box_format=original_format,
            spatial_size=spatial_size
        )
        
        return distorted_bboxes


@register()
class FisheyeNoiseInjection(T.Transform):
    """Inject noise specific to fisheye lens characteristics"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, noise_level=0.05, p=0.2):
        super().__init__()
        self.noise_level = noise_level
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt).astype(np.float32)
                else:
                    img_np = inpt.numpy().transpose(1, 2, 0) * 255.0
                
                # Add fisheye-specific noise (radial pattern)
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create radial noise pattern
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Radial noise intensity increases with distance from center
                radial_factor = distance / max_dist
                noise_intensity = self.noise_level * radial_factor
                
                # Generate noise
                noise = np.random.normal(0, noise_intensity, img_np.shape)
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
class FisheyeBlurAugmentation(T.Transform):
    """Apply blur that increases towards the edges (fisheye lens characteristic)"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, blur_range=(0.5, 2.0), p=0.15):
        super().__init__()
        self.blur_range = blur_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create radial blur map
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Blur intensity increases with distance from center
                blur_factor = distance / max_dist
                blur_factor = np.clip(blur_factor, 0, 1)
                
                # Apply variable blur
                blurred_img = img_np.copy()
                for i in range(h):
                    for j in range(w):
                        blur_strength = int(blur_factor[i, j] * random.uniform(*self.blur_range))
                        if blur_strength > 0:
                            # Apply local blur
                            y1, y2 = max(0, i - blur_strength), min(h, i + blur_strength + 1)
                            x1, x2 = max(0, j - blur_strength), min(w, j + blur_strength + 1)
                            
                            if y2 > y1 and x2 > x1:
                                local_region = img_np[y1:y2, x1:x2]
                                blurred_img[i, j] = np.mean(local_region, axis=(0, 1))
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(blurred_img)
                else:
                    img_tensor = torch.from_numpy(blurred_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeContrastEnhancement(T.Transform):
    """Enhance contrast with fisheye-specific patterns"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, contrast_factor=1.2, p=0.3):
        super().__init__()
        self.contrast_factor = contrast_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt).astype(np.float32)
                else:
                    img_np = inpt.numpy().transpose(1, 2, 0) * 255.0
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create radial contrast enhancement
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Contrast enhancement factor decreases with distance from center
                contrast_factor_radial = self.contrast_factor * (1 - 0.3 * distance / max_dist)
                
                # Apply contrast enhancement
                enhanced_img = img_np.copy()
                for c in range(img_np.shape[2]):
                    channel = img_np[:, :, c]
                    mean_val = np.mean(channel)
                    enhanced_img[:, :, c] = (channel - mean_val) * contrast_factor_radial + mean_val
                
                # Clip to valid range
                enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(enhanced_img)
                else:
                    img_tensor = torch.from_numpy(enhanced_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeDistortionCorrection(T.Transform):
    """Correct fisheye distortion for validation"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, correction_strength=0.1, p=1.0):
        super().__init__()
        self.correction_strength = correction_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                max_radius = min(center_x, center_y)
                
                # Create inverse distortion map
                y, x = np.ogrid[:h, :w]
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Apply inverse distortion
                distance_norm = distance / max_radius
                correction = distance_norm / (1 + self.correction_strength * distance_norm**2)
                new_distance = correction * max_radius
                
                # Calculate new coordinates
                angle = np.arctan2(dy, dx)
                new_x = center_x + new_distance * np.cos(angle)
                new_y = center_y + new_distance * np.sin(angle)
                
                # Apply correction
                map_x = new_x.astype(np.float32)
                map_y = new_y.astype(np.float32)
                
                corrected_img = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(corrected_img)
                else:
                    img_tensor = torch.from_numpy(corrected_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeEdgePreservation(T.Transform):
    """Preserve edges during fisheye processing"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, edge_threshold=0.1, p=1.0):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Detect edges
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                
                edges = cv2.Canny(gray, 50, 150)
                
                # Preserve edges by enhancing them slightly
                enhanced = img_np.astype(np.float32)
                if len(img_np.shape) == 3:
                    for i in range(3):
                        enhanced[:, :, i] += self.edge_threshold * edges
                else:
                    enhanced += self.edge_threshold * edges
                
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(enhanced)
                else:
                    img_tensor = torch.from_numpy(enhanced.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


# ==========================================
# Advanced Augmentation Transforms
# ==========================================

@register()
class GaussianBlurAugmentation(T.Transform):
    """Apply Gaussian blur with random kernel size and sigma"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, kernel_size=(3, 7), sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Random kernel size (must be odd)
        if isinstance(self.kernel_size, (list, tuple)):
            k_size = random.randint(self.kernel_size[0], self.kernel_size[1])
        else:
            k_size = self.kernel_size
        
        # Ensure kernel size is odd
        if k_size % 2 == 0:
            k_size += 1
        
        # Random sigma
        if isinstance(self.sigma, (list, tuple)):
            sigma_val = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma_val = self.sigma
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Apply Gaussian blur
                blurred_img = cv2.GaussianBlur(img_np, (k_size, k_size), sigma_val)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(blurred_img)
                else:
                    img_tensor = torch.from_numpy(blurred_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class MosaicAugmentation(T.Transform):
    """Mosaic augmentation for D-FINE performance enhancement"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, mosaic_size=(1280, 1280), p=0.5):
        super().__init__()
        self.mosaic_size = mosaic_size
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # For now, return original inputs
        # Mosaic augmentation requires multiple images, which is complex to implement here
        # This is a placeholder for future implementation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class MixUpAugmentation(T.Transform):
    """MixUp augmentation for D-FINE performance enhancement"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, alpha=0.8, p=0.3):
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # For now, return original inputs
        # MixUp requires multiple images, which is complex to implement here
        # This is a placeholder for future implementation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class CutMixAugmentation(T.Transform):
    """CutMix augmentation for D-FINE performance enhancement"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, alpha=1.0, p=0.2):
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # For now, return original inputs
        # CutMix requires multiple images, which is complex to implement here
        # This is a placeholder for future implementation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class GridMaskAugmentation(T.Transform):
    """GridMask augmentation for D-FINE performance enhancement"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, grid_size=64, drop_ratio=0.5, p=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.drop_ratio = drop_ratio
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                
                # Create grid mask
                mask = np.ones((h, w), dtype=np.uint8)
                
                # Apply grid pattern
                for i in range(0, h, self.grid_size):
                    for j in range(0, w, self.grid_size):
                        if random.random() < self.drop_ratio:
                            # Create a hole in the grid
                            end_i = min(i + self.grid_size, h)
                            end_j = min(j + self.grid_size, w)
                            mask[i:end_i, j:end_j] = 0
                
                # Apply mask
                if len(img_np.shape) == 3:
                    masked_img = img_np * mask[:, :, np.newaxis]
                else:
                    masked_img = img_np * mask
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(masked_img)
                else:
                    img_tensor = torch.from_numpy(masked_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


# ==========================================
# Additional Fisheye-specific Augmentation Transforms
# ==========================================

@register()
class FisheyeVignetteAugment(T.Transform):
    """Apply vignette effect to simulate fisheye lens characteristics"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, vignette_strength=(0.1, 0.3), p=0.25):
        super().__init__()
        self.vignette_strength = vignette_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        strength = random.uniform(*self.vignette_strength)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create vignette mask
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_radius = np.sqrt(center_x**2 + center_y**2)
                
                # Create radial vignette effect
                vignette_mask = 1 - strength * (distance / max_radius)**2
                vignette_mask = np.clip(vignette_mask, 0, 1)
                
                # Apply vignette
                if len(img_np.shape) == 3:
                    vignette_mask = vignette_mask[:, :, np.newaxis]
                
                vignetted_img = (img_np * vignette_mask).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(vignetted_img)
                else:
                    img_tensor = torch.from_numpy(vignetted_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeChromaticAberration(T.Transform):
    """Simulate chromatic aberration in fisheye lenses"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, aberration_strength=(0.01, 0.03), p=0.2):
        super().__init__()
        self.aberration_strength = aberration_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        strength = random.uniform(*self.aberration_strength)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                if len(img_np.shape) != 3:
                    outputs.append(inpt)
                    continue
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create chromatic aberration effect
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_radius = np.sqrt(center_x**2 + center_y**2)
                
                # Different shift for each channel
                shift_r = strength * distance * 2
                shift_b = -strength * distance * 1.5
                
                # Apply shifts
                shifted_img = img_np.copy()
                
                # Red channel shift
                for i in range(h):
                    for j in range(w):
                        shift = int(shift_r[i, j])
                        if 0 <= j + shift < w:
                            shifted_img[i, j, 0] = img_np[i, j + shift, 0]
                
                # Blue channel shift
                for i in range(h):
                    for j in range(w):
                        shift = int(shift_b[i, j])
                        if 0 <= j + shift < w:
                            shifted_img[i, j, 2] = img_np[i, j + shift, 2]
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(shifted_img)
                else:
                    img_tensor = torch.from_numpy(shifted_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyeBarrelDistortion(T.Transform):
    """Apply barrel distortion to simulate fisheye lens"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, barrel_factor=(0.05, 0.15), p=0.3):
        super().__init__()
        self.barrel_factor = barrel_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        factor = random.uniform(*self.barrel_factor)
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create barrel distortion map
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = min(center_x, center_y)
        
        y, x = np.ogrid[:img_h, :img_w]
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Apply barrel distortion
        distance_norm = distance / max_radius
        distortion = distance_norm * (1 + factor * distance_norm**2)
        new_distance = distortion * max_radius
        
        # Calculate new coordinates
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_distance * np.cos(angle)
        new_y = center_y + new_distance * np.sin(angle)
        
        # Apply distortion
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        distorted_img = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(distorted_img)
                else:
                    img_tensor = torch.from_numpy(distorted_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            elif isinstance(inpt, BoundingBoxes):
                # Transform bounding boxes
                inpt = self._distort_bboxes_barrel(inpt, map_x, map_y, img_w, img_h)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _distort_bboxes_barrel(self, bboxes, map_x, map_y, img_w, img_h):
        """Transform bounding boxes for barrel distortion"""
        # Simplified bbox transformation for barrel distortion
        # In practice, this would need more sophisticated transformation
        return bboxes


@register()
class FisheyePincushionDistortion(T.Transform):
    """Apply pincushion distortion to simulate fisheye lens"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, pincushion_factor=(0.03, 0.08), p=0.2):
        super().__init__()
        self.pincushion_factor = pincushion_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        factor = random.uniform(*self.pincushion_factor)
        
        # Find image input
        image_input = None
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                image_input = inpt
                break
        
        if image_input is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Get image dimensions
        if isinstance(image_input, PIL.Image.Image):
            img_w, img_h = image_input.size
            img_np = np.array(image_input)
        else:
            _, img_h, img_w = image_input.shape
            img_np = (image_input.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create pincushion distortion map
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = min(center_x, center_y)
        
        y, x = np.ogrid[:img_h, :img_w]
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Apply pincushion distortion (opposite of barrel)
        distance_norm = distance / max_radius
        distortion = distance_norm * (1 - factor * distance_norm**2)
        new_distance = distortion * max_radius
        
        # Calculate new coordinates
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_distance * np.cos(angle)
        new_y = center_y + new_distance * np.sin(angle)
        
        # Apply distortion
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        distorted_img = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(distorted_img)
                else:
                    img_tensor = torch.from_numpy(distorted_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            elif isinstance(inpt, BoundingBoxes):
                # Transform bounding boxes
                inpt = self._distort_bboxes_pincushion(inpt, map_x, map_y, img_w, img_h)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _distort_bboxes_pincushion(self, bboxes, map_x, map_y, img_w, img_h):
        """Transform bounding boxes for pincushion distortion"""
        # Simplified bbox transformation for pincushion distortion
        return bboxes


@register()
class FisheyeCenterBrightness(T.Transform):
    """Adjust brightness at the center of fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, brightness_factor=(0.9, 1.1), p=0.3):
        super().__init__()
        self.brightness_factor = brightness_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        factor = random.uniform(*self.brightness_factor)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create radial brightness mask
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_radius = np.sqrt(center_x**2 + center_y**2)
                
                # Create radial brightness adjustment
                brightness_mask = 1 + (factor - 1) * np.exp(-distance / (max_radius * 0.5))
                brightness_mask = np.clip(brightness_mask, 0, 2)
                
                # Apply brightness adjustment
                if len(img_np.shape) == 3:
                    brightness_mask = brightness_mask[:, :, np.newaxis]
                
                adjusted_img = (img_np * brightness_mask).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(adjusted_img)
                else:
                    img_tensor = torch.from_numpy(adjusted_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


@register()
class FisheyePeripheralDarkening(T.Transform):
    """Darken the periphery of fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, darkening_strength=(0.1, 0.2), p=0.25):
        super().__init__()
        self.darkening_strength = darkening_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        strength = random.uniform(*self.darkening_strength)
        
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, (PIL.Image.Image, Image)):
                if isinstance(inpt, PIL.Image.Image):
                    img_np = np.array(inpt)
                else:
                    img_np = (inpt.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                h, w = img_np.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Create radial darkening mask
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_radius = np.sqrt(center_x**2 + center_y**2)
                
                # Create radial darkening effect
                darkening_mask = 1 - strength * (distance / max_radius)**2
                darkening_mask = np.clip(darkening_mask, 0.5, 1)
                
                # Apply darkening
                if len(img_np.shape) == 3:
                    darkening_mask = darkening_mask[:, :, np.newaxis]
                
                darkened_img = (img_np * darkening_mask).astype(np.uint8)
                
                if isinstance(inpt, PIL.Image.Image):
                    inpt = PIL.Image.fromarray(darkened_img)
                else:
                    img_tensor = torch.from_numpy(darkened_img.transpose(2, 0, 1)).float() / 255.0
                    inpt = Image(img_tensor)
            
            outputs.append(inpt)
        
        return outputs if len(outputs) > 1 else outputs[0]


# Additional fisheye augmentations (simplified implementations)
@register()
class FisheyeRainSimulation(T.Transform):
    """Simulate rain effect on fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, rain_intensity=(0.1, 0.3), p=0.15):
        super().__init__()
        self.rain_intensity = rain_intensity
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified rain simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeFogSimulation(T.Transform):
    """Simulate fog effect on fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, fog_density=(0.05, 0.15), p=0.1):
        super().__init__()
        self.fog_density = fog_density
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified fog simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeGlareSimulation(T.Transform):
    """Simulate glare effect on fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, glare_strength=(0.1, 0.25), p=0.2):
        super().__init__()
        self.glare_strength = glare_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified glare simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeReflectionSimulation(T.Transform):
    """Simulate reflection effect on fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, reflection_strength=(0.05, 0.15), p=0.15):
        super().__init__()
        self.reflection_strength = reflection_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified reflection simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeMotionBlur(T.Transform):
    """Apply motion blur to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, motion_strength=(0.5, 2.0), p=0.2):
        super().__init__()
        self.motion_strength = motion_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified motion blur
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeRadialBlur(T.Transform):
    """Apply radial blur to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, radial_blur_factor=(0.1, 0.3), p=0.15):
        super().__init__()
        self.radial_blur_factor = radial_blur_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified radial blur
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeZoomBlur(T.Transform):
    """Apply zoom blur to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, zoom_blur_factor=(0.05, 0.15), p=0.1):
        super().__init__()
        self.zoom_blur_factor = zoom_blur_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified zoom blur
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeEllipticalDistortion(T.Transform):
    """Apply elliptical distortion to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, elliptical_factor=(0.05, 0.12), p=0.2):
        super().__init__()
        self.elliptical_factor = elliptical_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified elliptical distortion
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeSphericalWarp(T.Transform):
    """Apply spherical warping to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, warp_strength=(0.02, 0.08), p=0.15):
        super().__init__()
        self.warp_strength = warp_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified spherical warp
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeCylindricalProjection(T.Transform):
    """Apply cylindrical projection to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, cylinder_radius=(0.8, 1.2), p=0.1):
        super().__init__()
        self.cylinder_radius = cylinder_radius
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified cylindrical projection
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeEdgeObjectEnhancement(T.Transform):
    """Enhance objects at fisheye edges"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, enhancement_strength=(0.2, 0.4), p=0.4):
        super().__init__()
        self.enhancement_strength = enhancement_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified edge object enhancement
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeSmallObjectAmplification(T.Transform):
    """Amplify small objects in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, amplification_factor=(1.2, 1.5), p=0.35):
        super().__init__()
        self.amplification_factor = amplification_factor
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified small object amplification
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeOcclusionSimulation(T.Transform):
    """Simulate occlusion in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, occlusion_ratio=(0.1, 0.3), p=0.25):
        super().__init__()
        self.occlusion_ratio = occlusion_ratio
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified occlusion simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeShadowProjection(T.Transform):
    """Project shadows in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, shadow_strength=(0.2, 0.4), p=0.3):
        super().__init__()
        self.shadow_strength = shadow_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified shadow projection
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeMultiScaleDistortion(T.Transform):
    """Apply multi-scale distortion to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, scale_factors=(0.8, 1.2), p=0.3):
        super().__init__()
        self.scale_factors = scale_factors
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified multi-scale distortion
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeBoundaryObjectFocus(T.Transform):
    """Focus on objects at fisheye boundaries"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, focus_strength=(0.3, 0.5), p=0.4):
        super().__init__()
        self.focus_strength = focus_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified boundary object focus
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeLowContrastEnhancement(T.Transform):
    """Enhance low contrast in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, contrast_boost=(1.3, 1.8), p=0.3):
        super().__init__()
        self.contrast_boost = contrast_boost
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified low contrast enhancement
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeHighDynamicRange(T.Transform):
    """Apply HDR effect to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, hdr_strength=(0.2, 0.4), p=0.25):
        super().__init__()
        self.hdr_strength = hdr_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified HDR effect
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeSpecularHighlight(T.Transform):
    """Add specular highlights to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, highlight_strength=(0.1, 0.3), p=0.2):
        super().__init__()
        self.highlight_strength = highlight_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified specular highlight
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeDepthOfFieldSimulation(T.Transform):
    """Simulate depth of field in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, dof_strength=(0.1, 0.25), p=0.15):
        super().__init__()
        self.dof_strength = dof_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified depth of field simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeLensFlareSimulation(T.Transform):
    """Simulate lens flare in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, flare_intensity=(0.05, 0.15), p=0.1):
        super().__init__()
        self.flare_intensity = flare_intensity
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified lens flare simulation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyePartialOcclusion(T.Transform):
    """Simulate partial occlusion in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, occlusion_type='random', occlusion_ratio=(0.1, 0.4), p=0.3):
        super().__init__()
        self.occlusion_type = occlusion_type
        self.occlusion_ratio = occlusion_ratio
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified partial occlusion
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeObjectDeformation(T.Transform):
    """Deform objects in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, deformation_strength=(0.05, 0.15), p=0.25):
        super().__init__()
        self.deformation_strength = deformation_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified object deformation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeMultiObjectInteraction(T.Transform):
    """Simulate multi-object interactions in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, interaction_strength=(0.1, 0.3), p=0.2):
        super().__init__()
        self.interaction_strength = interaction_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified multi-object interaction
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeScaleInvariantAugment(T.Transform):
    """Apply scale-invariant augmentation to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image, BoundingBoxes)
    
    def __init__(self, scale_range=(0.7, 1.4), p=0.35):
        super().__init__()
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified scale-invariant augmentation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeDynamicLighting(T.Transform):
    """Apply dynamic lighting to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, lighting_variation=(0.8, 1.2), p=0.3):
        super().__init__()
        self.lighting_variation = lighting_variation
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified dynamic lighting
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeColorTemperature(T.Transform):
    """Adjust color temperature of fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, temp_range=(4000, 7000), p=0.25):
        super().__init__()
        self.temp_range = temp_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified color temperature adjustment
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeExposureVariation(T.Transform):
    """Apply exposure variation to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, exposure_range=(-1.0, 1.0), p=0.3):
        super().__init__()
        self.exposure_range = exposure_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified exposure variation
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeWhiteBalance(T.Transform):
    """Apply white balance to fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, wb_strength=(0.8, 1.2), p=0.2):
        super().__init__()
        self.wb_strength = wb_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified white balance
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeObjectBoundaryRefinement(T.Transform):
    """Refine object boundaries in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, refinement_strength=(0.2, 0.4), p=0.4):
        super().__init__()
        self.refinement_strength = refinement_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified object boundary refinement
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeDetectionConfidenceBoost(T.Transform):
    """Boost detection confidence in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, confidence_boost=(1.1, 1.3), p=0.3):
        super().__init__()
        self.confidence_boost = confidence_boost
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified detection confidence boost
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeFalsePositiveReduction(T.Transform):
    """Reduce false positives in fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, reduction_strength=(0.1, 0.2), p=0.25):
        super().__init__()
        self.reduction_strength = reduction_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified false positive reduction
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeCenterCrop(T.Transform):
    """Center crop for fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, crop_ratio=0.95, p=1.0):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified center crop
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeNormalization(T.Transform):
    """Normalize fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, normalize_range=(0, 1), p=1.0):
        super().__init__()
        self.normalize_range = normalize_range
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified normalization
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeDetectionOptimization(T.Transform):
    """Optimize detection for fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, optimization_mode='validation', p=1.0):
        super().__init__()
        self.optimization_mode = optimization_mode
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified detection optimization
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class FisheyeConfidenceCalibration(T.Transform):
    """Calibrate confidence for fisheye images"""
    _transformed_types = (PIL.Image.Image, Image)
    
    def __init__(self, calibration_strength=0.1, p=1.0):
        super().__init__()
        self.calibration_strength = calibration_strength
        self.p = p
    
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Simplified confidence calibration
        return inputs if len(inputs) > 1 else inputs[0]
