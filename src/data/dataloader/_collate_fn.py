"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
import numpy as np
import PIL.Image

from ...core import register

__all__ = [
    "BatchImageCollateFunction",
]


@register()
class BatchImageCollateFunction(nn.Module):
    def __init__(
        self,
        base_size=640,
        base_size_repeat=1,
        stop_epoch=float("inf"),
    ):
        super().__init__()
        self.base_size = base_size
        self.base_size_repeat = base_size_repeat
        self.stop_epoch = stop_epoch
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, batch):
        if self.epoch <= self.stop_epoch:
            # multi-scale training
            if isinstance(self.base_size, int):
                base_size = random.choice(
                    [self.base_size - i * 32 for i in range(self.base_size_repeat)]
                    + [self.base_size + i * 32 for i in range(self.base_size_repeat)]
                )
            else:
                base_size = random.choice(self.base_size)
        else:
            if isinstance(self.base_size, int):
                base_size = self.base_size
            else:
                base_size = self.base_size[0]

        images, targets = list(zip(*batch))
        images = self._image_forward(images, base_size)
        targets = self._targets_forward(targets)

        return {"images": images, "targets": targets}

    def _targets_forward(self, targets):
        return list(targets)

    def _image_forward(self, images, base_size):
        h = w = base_size

        def resized(image):
            if isinstance(image, torch.Tensor):
                return F.interpolate(image[None], size=[w, h], mode="bilinear", align_corners=False)[0]
            else:
                return TF.resize(image, size=[w, h])

        if isinstance(images[0], torch.Tensor):
            return torch.stack([resized(im) for im in images])
        else:
            return [resized(im) for im in images]


@register()
class MixUpCutMixCollateFunction(nn.Module):
    """Collate function with MixUp and CutMix augmentation support
    
    Applies MixUp and CutMix at the batch level for object detection.
    """
    
    def __init__(self, 
                 base_size=640, 
                 base_size_repeat=1,
                 stop_epoch=float('inf'),
                 use_mixup=False,
                 mixup_alpha=0.2,
                 mixup_prob=0.5,
                 use_cutmix=False,
                 cutmix_alpha=1.0,
                 cutmix_prob=0.5):
        super().__init__()
        self.base_size = base_size
        self.base_size_repeat = base_size_repeat
        self.stop_epoch = stop_epoch
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def forward(self, batch):
        """
        Apply MixUp/CutMix augmentation to batch
        """
        # Standard collate first
        images, targets = list(zip(*batch))
        
        # Resize images first
        if self.epoch <= self.stop_epoch:
            # Multi-scale training
            if isinstance(self.base_size, int):
                base_size = random.choice(
                    [self.base_size - i * 32 for i in range(self.base_size_repeat)]
                    + [self.base_size + i * 32 for i in range(self.base_size_repeat)]
                )
            else:
                base_size = random.choice(self.base_size)
        else:
            if isinstance(self.base_size, int):
                base_size = self.base_size
            else:
                base_size = self.base_size[0]
        
        # Resize images
        resized_images = self._image_forward(images, base_size)
        batch_targets = list(targets)
        
        # Apply MixUp or CutMix (but not both in same batch)
        if self.epoch < self.stop_epoch and len(resized_images) > 1:
            # Randomly choose between MixUp and CutMix
            use_mixup = self.use_mixup and random.random() < self.mixup_prob
            use_cutmix = self.use_cutmix and random.random() < self.cutmix_prob and not use_mixup
            
            if use_mixup:
                resized_images, batch_targets = self._apply_mixup(resized_images, batch_targets)
            elif use_cutmix:
                resized_images, batch_targets = self._apply_cutmix(resized_images, batch_targets)
        
        return {
            'images': resized_images,
            'targets': batch_targets
        }
    
    def _image_forward(self, images, base_size):
        """Resize images to target size"""
        h = w = base_size

        def resized(image):
            if isinstance(image, torch.Tensor):
                return F.interpolate(image[None], size=[w, h], mode="bilinear", align_corners=False)[0]
            else:
                return TF.resize(image, size=[w, h])

        if isinstance(images[0], torch.Tensor):
            return torch.stack([resized(im) for im in images])
        else:
            return [resized(im) for im in images]
    
    def _apply_mixup(self, images, targets):
        """Apply MixUp augmentation to batch"""
        if isinstance(images, list):
            batch_images = torch.stack(images)
        else:
            batch_images = images
            
        batch_size = batch_images.size(0)
        
        # Generate mixing coefficients
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        # Random permutation for mixing
        indices = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        # Mix targets (concatenate bounding boxes)
        mixed_targets = []
        for i in range(batch_size):
            target_a = dict(targets[i])
            target_b = dict(targets[indices[i]])
            
            # Concatenate boxes and labels
            if 'boxes' in target_a and 'boxes' in target_b:
                mixed_boxes = torch.cat([target_a['boxes'], target_b['boxes']], dim=0)
                target_a['boxes'] = mixed_boxes
                
            if 'labels' in target_a and 'labels' in target_b:
                mixed_labels = torch.cat([target_a['labels'], target_b['labels']], dim=0)
                target_a['labels'] = mixed_labels
            
            # Add mixing information for loss computation
            target_a['mixup_lam'] = lam
            target_a['mixup_index'] = indices[i].item()
            
            mixed_targets.append(target_a)
        
        return mixed_images, mixed_targets
    
    def _apply_cutmix(self, images, targets):
        """Apply CutMix augmentation to batch"""
        if isinstance(images, list):
            batch_images = torch.stack(images)
        else:
            batch_images = images
            
        batch_size = batch_images.size(0)
        
        # Generate mixing coefficient
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 0.5
        
        # Random permutation for mixing
        indices = torch.randperm(batch_size)
        
        # Get image dimensions
        _, _, h, w = batch_images.shape
        
        # Generate cut region
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random position for cut
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_images = batch_images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = batch_images[indices, :, y1:y2, x1:x2]
        
        # Mix targets (concatenate boxes for simplicity)
        mixed_targets = []
        for i in range(batch_size):
            target_a = dict(targets[i])
            target_b = dict(targets[indices[i]])
            
            # For simplicity, concatenate all boxes (could be improved with proper filtering)
            if 'boxes' in target_a and 'boxes' in target_b:
                mixed_boxes = torch.cat([target_a['boxes'], target_b['boxes']], dim=0)
                target_a['boxes'] = mixed_boxes
                
            if 'labels' in target_a and 'labels' in target_b:
                mixed_labels = torch.cat([target_a['labels'], target_b['labels']], dim=0)
                target_a['labels'] = mixed_labels
            
            # Add CutMix information
            target_a['cutmix_lam'] = lam
            target_a['cutmix_index'] = indices[i].item()
            target_a['cutmix_region'] = (x1, y1, x2, y2)
            
            mixed_targets.append(target_a)
        
        return mixed_images, mixed_targets 