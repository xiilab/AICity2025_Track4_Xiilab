"""
Robust Augmentation for D-FINE
실제 이미지 augmentation을 통한 domain generalization
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
    실제 이미지 augmentation을 적용하여 multiple view 생성
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
        """밝기 조정"""
        factor = 1.0 + (random.random() - 0.5) * self.aug_strength * 2
        factor = torch.clamp(torch.tensor(factor), 0.5, 1.5)
        return torch.clamp(images * factor, 0.0, 1.0)
    
    def _contrast_aug(self, images: torch.Tensor) -> torch.Tensor:
        """대비 조정"""
        factor = 1.0 + (random.random() - 0.5) * self.aug_strength * 2
        factor = torch.clamp(torch.tensor(factor), 0.5, 1.5)
        
        # Contrast adjustment: (image - mean) * factor + mean
        mean_val = images.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((images - mean_val) * factor + mean_val, 0.0, 1.0)
    
    def _hue_aug(self, images: torch.Tensor) -> torch.Tensor:
        """색조 조정"""
        # RGB to HSV 변환 후 H 채널 조정 (간단한 근사)
        hue_shift = (random.random() - 0.5) * self.aug_strength * 0.2
        
        # 간단한 색상 shift (RGB 채널 순환)
        if random.random() < 0.5:
            shifted = torch.roll(images, shifts=1, dims=1)  # RGB -> GBR
        else:
            shifted = torch.roll(images, shifts=-1, dims=1)  # RGB -> BRG
        
        # Original과 shifted의 blend
        alpha = abs(hue_shift)
        return (1 - alpha) * images + alpha * shifted
    
    def _rotation_aug(self, images: torch.Tensor) -> torch.Tensor:
        """회전 (작은 각도)"""
        angle = (random.random() - 0.5) * self.aug_strength * 30  # ±15도
        
        # 간단한 rotation approximation (실제로는 interpolation 필요)
        # 여기서는 작은 transformation으로 근사
        return images  # 실제 구현시 proper rotation 필요
    
    def _noise_aug(self, images: torch.Tensor) -> torch.Tensor:
        """가우시안 노이즈 추가"""
        noise_level = self.aug_strength * 0.1
        noise = torch.randn_like(images) * noise_level
        return torch.clamp(images + noise, 0.0, 1.0)
    
    def _blur_aug(self, images: torch.Tensor) -> torch.Tensor:
        """블러 효과 (간단한 평균 필터)"""
        # 3x3 평균 필터 근사
        kernel_size = 3
        padding = kernel_size // 2
        
        # 간단한 블러 (실제로는 Gaussian blur 사용 권장)
        blurred = F.avg_pool2d(images, kernel_size, stride=1, padding=padding)
        
        # Original과 blurred의 blend
        alpha = self.aug_strength * 0.5
        return (1 - alpha) * images + alpha * blurred
    
    def _sharpness_aug(self, images: torch.Tensor) -> torch.Tensor:
        """선명도 조정"""
        # 간단한 unsharp mask
        blurred = F.avg_pool2d(images, 3, stride=1, padding=1)
        enhanced = images + self.aug_strength * (images - blurred)
        return torch.clamp(enhanced, 0.0, 1.0)
    
    def apply_augmentation(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        이미지에 augmentation 적용하여 multiple view 생성
        
        Args:
            images: [B, C, H, W] 형태의 이미지 텐서
            
        Returns:
            List of augmented images
        """
        views = [images]  # Original view
        
        for i in range(self.num_views - 1):
            if random.random() < self.apply_prob:
                # 랜덤하게 augmentation 선택
                aug_type = random.choice(self.aug_types)
                augmented = self.transforms[aug_type](images.clone())
                views.append(augmented)
            else:
                # No augmentation (copy original)
                views.append(images.clone())
        
        return views


class FeatureAugmentationWrapper:
    """
    Feature level에서 augmentation 적용 (더 효율적)
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
        Feature에 augmentation 적용
        
        Args:
            features: [B, N, D] 형태의 feature 텐서 (예: pred_logits)
            
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
    이미지 또는 feature에 augmentation 적용 (혹은 둘 다)
    
    Args:
        images: 입력 이미지 (optional)
        features: 입력 feature (optional)  
        aug_config: augmentation 설정
        
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
    
    # Image level과 Feature level 둘 다 사용하는 경우
    if (aug_config.get('use_image_aug', False) and images is not None and 
        aug_config.get('use_feature_aug', False) and features is not None):
        
        # 이미지와 피쳐 augmentation을 조합
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
        
        # 이미지와 피쳐를 동시에 augment
        img_views = img_aug.apply_augmentation(images)
        feat_views = feat_aug.apply_feature_augmentation(features)
        
        # 피쳐 결과를 반환 (loss에서 주로 사용)
        return feat_views
        
    elif aug_config.get('use_image_aug', False) and images is not None:
        # 이미지 level augmentation만
        img_aug = RobustAugmentationWrapper(
            aug_types=aug_config.get('image_aug_types', ['brightness', 'contrast', 'hue', 'noise']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        return img_aug.apply_augmentation(images)
    
    elif features is not None:
        # Feature level augmentation만 (기본값)
        feat_aug = FeatureAugmentationWrapper(
            aug_types=aug_config.get('feature_aug_types', ['dropout', 'noise', 'scale']),
            num_views=aug_config['num_views'],
            aug_strength=aug_config['aug_strength']
        )
        return feat_aug.apply_feature_augmentation(features)
    
    else:
        raise ValueError("Either images or features must be provided") 