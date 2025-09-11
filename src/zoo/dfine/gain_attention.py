"""
GAIN: Guided Attention Inference Network Module for D-FINE
Based on "Tell Me Where to Look: Guided Attention Inference Network" (CVPR 2018)
Author: Kunpeng Li et al.

This module integrates guided attention mechanism into D-FINE decoder for better
feature localization and attention refinement.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Tuple, List


class GuidedAttentionModule(nn.Module):
    """
    Core GAIN module that generates attention maps and applies guided supervision.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        attention_dropout: float = 0.1,
        self_guided: bool = True,
        attention_supervision: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.self_guided = self_guided
        self.attention_supervision = attention_supervision
        
        # Attention generation layers
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Self-guided attention refinement
        if self_guided:
            self.self_guide_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid()
            )
            
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(attention_dropout)
        )
        
        # Attention refinement loss components
        if attention_supervision:
            self.attention_loss_fn = nn.MSELoss()
            
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def generate_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate attention map from input features
        Args:
            features: Input feature tensor [B, C, H, W]
        Returns:
            attention_map: Attention map [B, 1, H, W]
        """
        attention_map = self.attention_conv(features)
        return attention_map
    
    def apply_self_guidance(self, features: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        """
        Apply self-guided attention refinement
        Args:
            features: Input feature tensor [B, C, H, W]
            attention_map: Current attention map [B, 1, H, W]
        Returns:
            refined_attention: Refined attention map [B, 1, H, W]
        """
        if not self.self_guided:
            return attention_map
            
        # Generate self-guidance signal
        self_guide = self.self_guide_conv(features)
        
        # Combine original attention with self-guidance
        # Use element-wise multiplication to enhance regions of interest
        refined_attention = attention_map * (1 + self_guide)
        refined_attention = torch.clamp(refined_attention, 0, 1)
        
        return refined_attention
    
    def compute_attention_loss(
        self, 
        attention_map: torch.Tensor, 
        target_attention: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention supervision loss
        Args:
            attention_map: Generated attention map [B, 1, H, W]
            target_attention: Target attention map [B, 1, H, W] (if available)
            gt_masks: Ground truth masks [B, H, W] (for self-supervision)
        Returns:
            loss: Attention supervision loss
        """
        if not self.attention_supervision:
            return torch.tensor(0.0, device=attention_map.device)
        
        if target_attention is not None:
            # Direct supervision with provided attention maps
            return self.attention_loss_fn(attention_map, target_attention)
        
        elif gt_masks is not None:
            # Self-supervision using ground truth masks
            # Resize gt_masks to match attention map size
            gt_masks = gt_masks.unsqueeze(1).float()
            if gt_masks.shape[-2:] != attention_map.shape[-2:]:
                gt_masks = F.interpolate(gt_masks, size=attention_map.shape[-2:], mode='nearest')
            
            # Encourage attention to focus on object regions
            return self.attention_loss_fn(attention_map, gt_masks)
        
        else:
            # Self-guided loss: encourage attention diversity and completeness
            b, _, h, w = attention_map.shape
            
            # Diversity loss: prevent attention collapse
            attention_flat = attention_map.view(b, -1)
            attention_mean = attention_flat.mean(dim=1, keepdim=True)
            diversity_loss = F.mse_loss(attention_flat, attention_mean.expand_as(attention_flat))
            
            # Completeness loss: encourage wider coverage
            attention_sum = attention_map.sum(dim=(2, 3))
            target_coverage = torch.ones_like(attention_sum) * (h * w * 0.3)  # Target 30% coverage
            completeness_loss = F.mse_loss(attention_sum, target_coverage)
            
            return diversity_loss + 0.1 * completeness_loss
    
    def forward(
        self, 
        features: torch.Tensor, 
        target_attention: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of GAIN module
        Args:
            features: Input features [B, C, H, W]
            target_attention: Target attention for supervision [B, 1, H, W]
            gt_masks: Ground truth masks for self-supervision [B, H, W]
        Returns:
            enhanced_features: Attention-enhanced features [B, hidden_dim, H, W]
            attention_map: Generated attention map [B, 1, H, W]
            attention_loss: Attention supervision loss
        """
        # Generate initial attention map
        attention_map = self.generate_attention_map(features)
        
        # Apply self-guided refinement
        attention_map = self.apply_self_guidance(features, attention_map)
        
        # Enhance features with attention
        enhanced_features = self.feature_enhance(features)
        enhanced_features = enhanced_features * attention_map
        
        # Compute attention loss
        attention_loss = self.compute_attention_loss(attention_map, target_attention, gt_masks)
        
        return enhanced_features, attention_map, attention_loss


class GAINEnhancedMSDeformableAttention(nn.Module):
    """
    Enhanced MSDeformableAttention with GAIN mechanism
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        embed_dim: int = 256,
        attention_weight: float = 0.3,
        self_guided: bool = True,
        attention_supervision: bool = False,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.embed_dim = embed_dim
        self.attention_weight = attention_weight
        
        # GAIN attention module for query enhancement
        self.gain_module = GuidedAttentionModule(
            in_channels=embed_dim,
            hidden_dim=embed_dim,
            self_guided=self_guided,
            attention_supervision=attention_supervision,
        )
        
        # Additional projection for feature fusion
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
        gain_features: Optional[torch.Tensor] = None,
        target_attention: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Enhanced forward pass with GAIN attention
        Args:
            query: Query tensor [bs, query_length, C]
            reference_points: Reference points for deformable attention
            value: Value tensor [bs, value_length, C]
            value_spatial_shapes: Spatial shapes for multi-scale features
            gain_features: Features for GAIN processing [B, C, H, W]
            target_attention: Target attention maps for supervision
            gt_masks: Ground truth masks for self-supervision
        Returns:
            output: Enhanced output features
            attention_map: GAIN attention map (if gain_features provided)
            attention_loss: GAIN attention loss (if supervision enabled)
        """
        # Original deformable attention
        original_output = self.original_attention(query, reference_points, value, value_spatial_shapes)
        
        attention_map = None
        attention_loss = None
        
        # Apply GAIN enhancement if spatial features are provided
        if gain_features is not None:
            enhanced_features, attention_map, attention_loss = self.gain_module(
                gain_features, target_attention, gt_masks
            )
            
            # Convert enhanced spatial features to sequence format
            B, C, H, W = enhanced_features.shape
            enhanced_seq = enhanced_features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # Ensure sequence length matches
            if enhanced_seq.shape[1] != original_output.shape[1]:
                # Interpolate or pad to match dimensions
                if enhanced_seq.shape[1] > original_output.shape[1]:
                    enhanced_seq = enhanced_seq[:, :original_output.shape[1], :]
                else:
                    # Pad with zeros
                    pad_length = original_output.shape[1] - enhanced_seq.shape[1]
                    padding = torch.zeros(B, pad_length, C, device=enhanced_seq.device)
                    enhanced_seq = torch.cat([enhanced_seq, padding], dim=1)
            
            # Fuse original and enhanced features
            fused_features = torch.cat([original_output, enhanced_seq], dim=-1)
            fused_output = self.fusion_proj(fused_features)
            
            # Weighted combination
            output = (1 - self.attention_weight) * original_output + self.attention_weight * fused_output
        else:
            output = original_output
        
        return output, attention_map, attention_loss


class GAINEnhancedTransformerDecoderLayer(nn.Module):
    """
    Enhanced TransformerDecoderLayer with GAIN attention mechanism
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int = 256,
        attention_weight: float = 0.3,
        self_guided: bool = True,
        attention_supervision: bool = False,
    ):
        super().__init__()
        
        # Store original layer for delegation
        self.original_layer = original_layer
        
        # Create GAIN-enhanced cross attention
        if hasattr(original_layer, 'cross_attn'):
            self.enhanced_cross_attn = GAINEnhancedMSDeformableAttention(
                original_attention=original_layer.cross_attn,
                embed_dim=d_model,
                attention_weight=attention_weight,
                self_guided=self_guided,
                attention_supervision=attention_supervision,
            )
            self.use_gain = True
        else:
            self.use_gain = False
    
    def __getattr__(self, name):
        """Delegate attribute access to original layer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)
    
    def forward(
        self, 
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[int],
        attn_mask: Optional[torch.Tensor] = None,
        query_pos_embed: Optional[torch.Tensor] = None,
        gain_features: Optional[torch.Tensor] = None,
        target_attention: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced forward pass with GAIN attention
        Returns only tensor to maintain compatibility with D-FINE
        """
        if self.use_gain:
            # Use GAIN-enhanced cross attention - following D-FINE structure
            # Self attention
            q = k = self.original_layer.with_pos_embed(target, query_pos_embed)
            target2, _ = self.original_layer.self_attn(q, k, value=target, attn_mask=attn_mask)
            target = target + self.original_layer.dropout1(target2)
            target = self.original_layer.norm1(target)
            
            # Enhanced cross attention with GAIN
            target2, attention_map, attention_loss = self.enhanced_cross_attn(
                query=self.original_layer.with_pos_embed(target, query_pos_embed),
                reference_points=reference_points,
                value=value,
                value_spatial_shapes=spatial_shapes,
                gain_features=gain_features,
                target_attention=target_attention,
                gt_masks=gt_masks,
            )
            
            # Store attention outputs for later collection
            if hasattr(self, 'layer_idx'):
                # Store in class attributes for collection
                if not hasattr(self, '_attention_maps'):
                    self._attention_maps = {}
                if not hasattr(self, '_attention_losses'):
                    self._attention_losses = []
                
                if attention_map is not None:
                    self._attention_maps[f'layer_{self.layer_idx}'] = attention_map
                if attention_loss is not None:
                    self._attention_losses.append(attention_loss)
            
            # Use gateway instead of norm2 (D-FINE specific)
            target = self.original_layer.gateway(target, self.original_layer.dropout2(target2))
            
            # FFN (unchanged)
            target2 = self.original_layer.forward_ffn(target)
            target = target + self.original_layer.dropout4(target2)
            target = self.original_layer.norm3(target.clamp(min=-65504, max=65504))
            
            return target
        else:
            # Fallback to original layer
            return self.original_layer(target, reference_points, value, spatial_shapes, 
                                     attn_mask, query_pos_embed)


def enhance_dfine_decoder_with_gain(
    original_decoder: nn.Module,
    enhanced_layers: Optional[List[int]] = None,
    attention_weight: float = 0.3,
    self_guided: bool = True,
    attention_supervision: bool = False,
) -> nn.Module:
    """
    Enhance D-FINE decoder with GAIN attention mechanism
    
    Args:
        original_decoder: Original D-FINE TransformerDecoder
        enhanced_layers: List of layer indices to enhance (default: [2, 4])
        attention_weight: Weight for combining original and enhanced features
        self_guided: Enable self-guided attention refinement
        attention_supervision: Enable attention supervision loss
    
    Returns:
        Enhanced decoder with GAIN attention
    """
    if enhanced_layers is None:
        enhanced_layers = [2, 4]  # Enhance middle layers by default
    
    # Ensure enhanced_layers contains integers (handle string conversion from config)
    enhanced_layers = [int(idx) for idx in enhanced_layers]
    
    # Create a copy of the original decoder
    enhanced_decoder = copy.deepcopy(original_decoder)
    
    # Replace specific layers with GAIN-enhanced versions
    for layer_idx in enhanced_layers:
        if layer_idx < len(enhanced_decoder.layers):
            original_layer = enhanced_decoder.layers[layer_idx]
            enhanced_layer = GAINEnhancedTransformerDecoderLayer(
                original_layer=original_layer,
                d_model=getattr(original_layer, 'd_model', 256),
                attention_weight=attention_weight,
                self_guided=self_guided,
                attention_supervision=attention_supervision,
            )
            enhanced_layer.layer_idx = layer_idx  # Store layer index for loss collection
            enhanced_decoder.layers[layer_idx] = enhanced_layer
    
    # Store enhanced layer indices for reference
    enhanced_decoder.gain_enhanced_layers = enhanced_layers
    
    # Add method to collect attention losses from layers
    def collect_attention_losses(self):
        losses = []
        for layer in self.layers:
            if hasattr(layer, '_attention_losses'):
                losses.extend(layer._attention_losses)
                layer._attention_losses.clear()  # Clear after collection
        return losses
    
    # Add method to collect attention maps from layers
    def collect_attention_maps(self):
        maps = {}
        for layer in self.layers:
            if hasattr(layer, '_attention_maps'):
                maps.update(layer._attention_maps)
                layer._attention_maps.clear()  # Clear after collection
        return maps
    
    # Bind methods to decoder
    enhanced_decoder.collect_attention_losses = collect_attention_losses.__get__(enhanced_decoder)
    enhanced_decoder.collect_attention_maps = collect_attention_maps.__get__(enhanced_decoder)
    
    return enhanced_decoder


# Usage example and integration utilities
class GAINLoss(nn.Module):
    """
    GAIN attention supervision loss
    """
    
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
    
    def forward(self, attention_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute total GAIN attention loss
        Args:
            attention_losses: List of attention losses from different layers
        Returns:
            total_loss: Weighted sum of attention losses
        """
        if not attention_losses:
            return torch.tensor(0.0, requires_grad=True)
        
        valid_losses = [loss for loss in attention_losses if loss is not None]
        if not valid_losses:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = sum(valid_losses) / len(valid_losses)
        return self.loss_weight * total_loss 