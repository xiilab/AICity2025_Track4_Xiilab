"""
Robust Augmentation for D-FINE
Domain generalization via real image augmentations
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Dict, List, Tuple, Union
import numpy as np


class RobustAugmentationWrapper:
    """
    Apply image-level augmentations to create multiple views
    """
    
    def __init__(
        self,
        aug_types: List[str] = ['brightness', 'contrast', 'hue', 'rotation', 'noise'],
        num_views: int = 3,
        aug_strength: float = 0.3,
        apply_prob: float = 0.8
    ):
        self.aug_types = aug_types
        self.num_views = num_views
        self.aug_strength = aug_strength
        self.apply_prob = apply_prob
        
        # Augmentation transforms
        self.transforms = {
            'brightness': self._brightness_aug,
            'contrast': self._contrast_aug,
            'hue': self._hue_aug,
            'rotation': self._rotation_aug,
            'noise': self._noise_aug,
            'blur': self._blur_aug,
            'sharpness': self._sharpness_aug
        }
    
    def _brightness_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Brightness adjustment"""
        factor = 1.0 + (random.random() - 0.5) * self.aug_strength * 2
        factor = torch.clamp(torch.tensor(factor), 0.5, 1.5)
        return torch.clamp(images * factor, 0.0, 1.0)
    
    def _contrast_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Contrast adjustment"""
        factor = 1.0 + (random.random() - 0.5) * self.aug_strength * 2
        factor = torch.clamp(torch.tensor(factor), 0.5, 1.5)
        
        # Contrast adjustment: (image - mean) * factor + mean
        mean_val = images.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((images - mean_val) * factor + mean_val, 0.0, 1.0)
    
    def _hue_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Hue adjustment (approximation)"""
        # Simple RGB channel rotation approximation
        hue_shift = (random.random() - 0.5) * self.aug_strength * 0.2
        
        # Simple color shift (channel roll)
        if random.random() < 0.5:
            shifted = torch.roll(images, shifts=1, dims=1)  # RGB -> GBR
        else:
            shifted = torch.roll(images, shifts=-1, dims=1)  # RGB -> BRG
        
        # Blend original and shifted
        alpha = abs(hue_shift)
        return (1 - alpha) * images + alpha * shifted
    
    def _rotation_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Small-angle rotation (placeholder)"""
        angle = (random.random() - 0.5) * self.aug_strength * 30  # ±15도
        
        # Placeholder: return input; implement proper rotation if needed
        return images  # 실제 구현시 proper rotation 필요
    
    def _noise_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        noise_level = self.aug_strength * 0.1
        noise = torch.randn_like(images) * noise_level
        return torch.clamp(images + noise, 0.0, 1.0)
    
    def _blur_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Blur effect (mean filter approximation)"""
        # 3x3 평균 필터 근사
        kernel_size = 3
        padding = kernel_size // 2
        
        # Simple blur (consider Gaussian blur for better quality)
        blurred = F.avg_pool2d(images, kernel_size, stride=1, padding=padding)
        
        # Original과 blurred의 blend
        alpha = self.aug_strength * 0.5
        return (1 - alpha) * images + alpha * blurred
    
    def _sharpness_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Sharpness adjustment (unsharp mask)"""
        # 간단한 unsharp mask
        blurred = F.avg_pool2d(images, 3, stride=1, padding=1)
        enhanced = images + self.aug_strength * (images - blurred)
        return torch.clamp(enhanced, 0.0, 1.0)
    
    def apply_augmentation(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply augmentations to images and create multiple views
        
        Args:
            images: Tensor of shape [B, C, H, W]
            
        Returns:
            List of augmented images
        """
        views = [images]  # Original view
        
        for i in range(self.num_views - 1):
            if random.random() < self.apply_prob:
                # Randomly choose an augmentation
                aug_type = random.choice(self.aug_types)
                augmented = self.transforms[aug_type](images.clone())
                views.append(augmented)
            else:
                # No augmentation (copy original)
                views.append(images.clone())
        
        return views


class FeatureAugmentationWrapper:
    """
    Apply feature-level augmentations (more efficient)
    """
    
    def __init__(
        self,
        aug_types: List[str] = ['dropout', 'noise', 'scale', 'channel_shuffle'],
        num_views: int = 3,
        aug_strength: float = 0.1
    ):
        self.aug_types = aug_types
        self.num_views = num_views
        self.aug_strength = aug_strength
    
    def apply_feature_augmentation(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply augmentations to features
        
        Args:
            features: Tensor of shape [B, N, D] (e.g., pred_logits)
            
        Returns:
            List of augmented features
        """
        views = [features]  # Original view
        
        for i in range(self.num_views - 1):
            aug_type = random.choice(self.aug_types)
            
            if aug_type == 'dropout':
                # Feature dropout
                dropout_rate = self.aug_strength
                mask = torch.rand_like(features) > dropout_rate
                augmented = features * mask.float()
                
            elif aug_type == 'noise':
                # Gaussian noise
                noise = torch.randn_like(features) * self.aug_strength
                augmented = features + noise
                
            elif aug_type == 'scale':
                # Random scaling
                scale = 1.0 + (random.random() - 0.5) * self.aug_strength * 2
                augmented = features * scale
                
            elif aug_type == 'channel_shuffle':
                # Channel dimension shuffle
                if len(features.shape) == 3:  # [B, N, D]
                    perm = torch.randperm(features.size(-1))
                    augmented = features[..., perm]
                else:
                    augmented = features
                    
            else:
                augmented = features
                
            views.append(augmented)
        
        return views


# Integration with RobustLossWrapper
def get_augmented_views(
    images: torch.Tensor = None,
    features: torch.Tensor = None,
    aug_config: Dict = None
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Apply augmentations to images or features (or both)
    
    Args:
        images: input images (optional)
        features: input features (optional)
        aug_config: augmentation configuration
        
    Returns:
        Augmented views
    """
    if aug_config is None:
        aug_config = {
            'use_image_aug': False,
            'use_feature_aug': True,
            'num_views': 3,
            'aug_strength': 0.1
        }
    
    # When using both image-level and feature-level augmentations
    if (aug_config.get('use_image_aug', False) and images is not None and 
        aug_config.get('use_feature_aug', False) and features is not None):
        
        # Combine image and feature augmentations
        img_aug = RobustAugmentationWrapper(
            aug_types=aug_config.get('image_aug_types', ['brightness', 'contrast', 'hue', 'noise']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        feat_aug = FeatureAugmentationWrapper(
            aug_types=aug_config.get('feature_aug_types', ['dropout', 'noise', 'scale']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        
        # Augment both images and features
        img_views = img_aug.apply_augmentation(images)
        feat_views = feat_aug.apply_feature_augmentation(features)
        
        # Return feature views (commonly used in loss)
        return feat_views
        
    elif aug_config.get('use_image_aug', False) and images is not None:
        # Image-level augmentation only
        img_aug = RobustAugmentationWrapper(
            aug_types=aug_config.get('image_aug_types', ['brightness', 'contrast', 'hue', 'noise']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        return img_aug.apply_augmentation(images)
    
    elif features is not None:
        # Feature-level augmentation only (default)
        feat_aug = FeatureAugmentationWrapper(
            aug_types=aug_config.get('feature_aug_types', ['dropout', 'noise', 'scale']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        return feat_aug.apply_feature_augmentation(features)
    
    else:
        raise ValueError("Either images or features must be provided")