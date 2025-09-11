"""
Enhanced Multi-Scale Feature Fusion (EMSFF) for D-FINE
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math

from .utils import get_activation
from ...core import register


class EnhancedMultiScaleFeatureFusion(nn.Module):
    """
    Enhanced Multi-Scale Feature Fusion (EMSFF) module
    
    This module improves feature fusion for lightweight models by incorporating:
    1. Attention-weighted feature fusion
    2. Cross-scale feature interaction
    3. Adaptive channel weighting
    4. Efficient feature pyramid construction
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_levels: int = 3,
        attention_type: str = "channel",  # "channel", "spatial", "both"
        fusion_method: str = "adaptive",  # "adaptive", "concat", "add"
        activation: str = "relu",
        use_deformable: bool = False,
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.attention_type = attention_type
        self.fusion_method = fusion_method
        
        # Lateral convolutions for channel alignment
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])
        
        # Feature pyramid convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                get_activation(activation)
            ) for _ in range(num_levels)
        ])
        
        # Attention modules
        if attention_type in ["channel", "both"]:
            self.channel_attention = nn.ModuleList([
                ChannelAttention(out_channels) for _ in range(num_levels)
            ])
        
        if attention_type in ["spatial", "both"]:
            self.spatial_attention = nn.ModuleList([
                SpatialAttention() for _ in range(num_levels)
            ])
        
        # Cross-scale interaction modules
        self.cross_scale_fusion = CrossScaleFusion(
            out_channels, num_levels, fusion_method
        )
        
        # Adaptive feature weighting
        self.adaptive_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                get_activation(activation),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])
        
        # Feature enhancement modules
        self.feature_enhancers = nn.ModuleList([
            FeatureEnhancer(out_channels, activation) for _ in range(num_levels)
        ])
        
        # Optional deformable convolutions
        if use_deformable:
            self.deformable_convs = nn.ModuleList([
                DeformableConv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(num_levels)
            ])
        else:
            self.deformable_convs = None
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of enhanced multi-scale feature fusion
        
        Args:
            features: List of feature maps from different scales
            
        Returns:
            List of enhanced feature maps
        """
        
        # Step 1: Lateral connections for channel alignment
        laterals = []
        for i, (feat, conv) in enumerate(zip(features, self.lateral_convs)):
            lateral = conv(feat)
            laterals.append(lateral)
        
        # Step 2: Top-down pathway with attention
        enhanced_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Top level - no upsampling needed
                current_feat = laterals[i]
            else:
                # Upsample and add
                upsampled = F.interpolate(
                    enhanced_features[0], 
                    size=laterals[i].shape[-2:], 
                    mode='nearest'
                )
                current_feat = laterals[i] + upsampled
            
            # Apply attention mechanisms
            if self.attention_type in ["channel", "both"]:
                current_feat = self.channel_attention[i](current_feat)
            
            if self.attention_type in ["spatial", "both"]:
                current_feat = self.spatial_attention[i](current_feat)
            
            # Apply adaptive weighting
            weight = self.adaptive_weights[i](current_feat)
            current_feat = current_feat * weight
            
            enhanced_features.insert(0, current_feat)
        
        # Step 3: Cross-scale feature interaction
        enhanced_features = self.cross_scale_fusion(enhanced_features)
        
        # Step 4: Final feature enhancement
        final_features = []
        for i, feat in enumerate(enhanced_features):
            # Apply feature enhancer
            enhanced_feat = self.feature_enhancers[i](feat)
            
            # Apply deformable convolution if available
            if self.deformable_convs is not None:
                enhanced_feat = self.deformable_convs[i](enhanced_feat)
            
            # Apply final FPN convolution
            final_feat = self.fpn_convs[i](enhanced_feat)
            final_features.append(final_feat)
        
        return final_features


class ChannelAttention(nn.Module):
    """Channel attention module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


class CrossScaleFusion(nn.Module):
    """Cross-scale feature fusion module"""
    
    def __init__(
        self,
        channels: int,
        num_levels: int,
        fusion_method: str = "adaptive"
    ):
        super().__init__()
        
        self.channels = channels
        self.num_levels = num_levels
        self.fusion_method = fusion_method
        
        if fusion_method == "adaptive":
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(
                torch.ones(num_levels, num_levels) / num_levels
            )
            
            # Cross-scale convolutions
            self.cross_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(channels, channels, 1) if i != j else nn.Identity()
                    for j in range(num_levels)
                ]) for i in range(num_levels)
            ])
        
        elif fusion_method == "concat":
            self.fusion_conv = nn.Conv2d(
                channels * num_levels, channels, 1
            )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform cross-scale feature fusion
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            List of fused feature maps
        """
        
        if self.fusion_method == "adaptive":
            fused_features = []
            
            for i, target_feat in enumerate(features):
                target_size = target_feat.shape[-2:]
                fused_feat = torch.zeros_like(target_feat)
                
                for j, source_feat in enumerate(features):
                    # Resize source feature to target size
                    if source_feat.shape[-2:] != target_size:
                        resized_feat = F.interpolate(
                            source_feat, size=target_size, mode='bilinear', align_corners=False
                        )
                    else:
                        resized_feat = source_feat
                    
                    # Apply cross-scale convolution
                    processed_feat = self.cross_convs[i][j](resized_feat)
                    
                    # Weight and accumulate
                    weight = self.fusion_weights[i, j]
                    fused_feat += weight * processed_feat
                
                fused_features.append(fused_feat)
            
            return fused_features
        
        elif self.fusion_method == "concat":
            # Resize all features to the largest size
            target_size = features[0].shape[-2:]
            resized_features = []
            
            for feat in features:
                if feat.shape[-2:] != target_size:
                    resized_feat = F.interpolate(
                        feat, size=target_size, mode='bilinear', align_corners=False
                    )
                else:
                    resized_feat = feat
                resized_features.append(resized_feat)
            
            # Concatenate and fuse
            concat_feat = torch.cat(resized_features, dim=1)
            fused_feat = self.fusion_conv(concat_feat)
            
            # Resize back to original sizes
            fused_features = []
            for original_feat in features:
                if original_feat.shape[-2:] != target_size:
                    resized_back = F.interpolate(
                        fused_feat, size=original_feat.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                else:
                    resized_back = fused_feat
                fused_features.append(resized_back)
            
            return fused_features
        
        else:  # "add"
            return features


class FeatureEnhancer(nn.Module):
    """Feature enhancement module with residual connections"""
    
    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.se_module = SqueezeExcitation(channels)
        
        self.act2 = get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply squeeze-and-excitation
        out = self.se_module(out)
        
        # Residual connection
        out += identity
        out = self.act2(out)
        
        return out


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.global_pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        
        return x * scale


class DeformableConv2d(nn.Module):
    """Simplified deformable convolution implementation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        
        # Regular convolution
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Initialize offset conv to zero
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict offsets
        offset = self.offset_conv(x)
        
        # For simplicity, we'll just use regular convolution
        # In a full implementation, you would apply the offsets
        # to perform deformable convolution
        return self.regular_conv(x)


@register()
class AdaptiveFeaturePyramid(nn.Module):
    """
    Adaptive Feature Pyramid Network with dynamic layer selection
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_extra_levels: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.num_backbone_levels = len(in_channels_list)
        self.num_extra_levels = num_extra_levels
        self.total_levels = self.num_backbone_levels + num_extra_levels
        
        # Enhanced multi-scale feature fusion
        self.emsff = EnhancedMultiScaleFeatureFusion(
            in_channels_list, out_channels, len(in_channels_list),
            attention_type="both", fusion_method="adaptive"
        )
        
        # Extra levels for small object detection
        self.extra_convs = nn.ModuleList()
        for i in range(num_extra_levels):
            if i == 0:
                # First extra level from last backbone level
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )
            else:
                # Subsequent extra levels
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )
        
        # Level importance predictor
        self.level_importance = nn.Sequential(
            nn.Linear(out_channels, self.total_levels),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self, 
        backbone_features: List[torch.Tensor],
        use_all_levels: bool = True,
        return_weights: bool = False
    ) -> List[torch.Tensor]:
        """
        Forward pass of adaptive feature pyramid
        
        Args:
            backbone_features: Features from backbone
            use_all_levels: Whether to use all pyramid levels
            return_weights: Whether to return level importance weights
            
        Returns:
            List of pyramid features, or tuple of (features, weights) if return_weights=True
        """
        
        # Enhanced multi-scale feature fusion
        enhanced_features = self.emsff(backbone_features)
        
        # Generate extra levels
        pyramid_features = enhanced_features.copy()
        current_feat = enhanced_features[-1]
        
        for extra_conv in self.extra_convs:
            current_feat = extra_conv(current_feat)
            pyramid_features.append(current_feat)
        
        # Predict level importance
        importance_input = torch.stack([
            F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            for feat in pyramid_features
        ], dim=1)  # [B, num_levels, C]
        
        level_weights = self.level_importance(
            importance_input.mean(dim=1)  # [B, C]
        )  # [B, num_levels]
        
        # Apply importance weighting if not using all levels
        if not use_all_levels:
            # Select top-k important levels
            k = min(4, len(pyramid_features))  # Use at most 4 levels
            _, top_indices = torch.topk(level_weights.mean(dim=0), k)
            
            selected_features = [pyramid_features[i] for i in sorted(top_indices)]
            selected_weights = level_weights[:, sorted(top_indices)]
            
            if return_weights:
                return selected_features, selected_weights
            else:
                return selected_features
        
        if return_weights:
            return pyramid_features, level_weights
        else:
            return pyramid_features 