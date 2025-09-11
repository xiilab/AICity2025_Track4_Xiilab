"""
Enhanced D-FINE Decoder with GAIN Attention Integration
This file demonstrates how to integrate GAIN attention mechanism into D-FINE decoder
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

from .dfine_decoder import DFINETransformer
from .gain_attention import (
    enhance_dfine_decoder_with_gain,
    GAINEnhancedTransformerDecoderLayer,
    GAINLoss,
    GuidedAttentionModule
)
from ...core import register


@register()
class DFINETransformerWithGAIN(DFINETransformer):
    """
    Enhanced D-FINE Transformer with GAIN attention mechanism
    """
    
    def __init__(
        self,
        # Original D-FINE parameters
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        # GAIN-specific parameters
        gain_enabled=True,
        gain_layers=None,
        gain_attention_weight=0.3,
        gain_self_guided=True,
        gain_attention_supervision=False,
        gain_loss_weight=1.0,
    ):
        # Initialize the base D-FINE transformer first
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
        )
        
        # GAIN-specific parameters
        self.gain_enabled = gain_enabled
        self.gain_layers = gain_layers or [2, 4]  # Enhance layers 2 and 4 by default
        # Ensure gain_layers contains integers (handle string conversion from config)
        if self.gain_layers:
            self.gain_layers = [int(idx) for idx in self.gain_layers]
        self.gain_attention_weight = gain_attention_weight
        self.gain_attention_supervision = gain_attention_supervision
        
        if self.gain_enabled:
            # Enhance the decoder with GAIN attention
            self.decoder = enhance_dfine_decoder_with_gain(
                original_decoder=self.decoder,
                enhanced_layers=self.gain_layers,
                attention_weight=gain_attention_weight,
                self_guided=gain_self_guided,
                attention_supervision=gain_attention_supervision,
            )
            
            # Add GAIN loss computation
            self.gain_loss_fn = GAINLoss(loss_weight=gain_loss_weight)
            
            # Storage for attention maps and losses
            self.attention_maps = {}
            self.attention_losses = []
    
    def forward(self, feats, targets=None, **kwargs):
        """
        Enhanced forward pass with GAIN attention
        """
        # Clear previous attention data
        if hasattr(self, 'attention_maps'):
            self.attention_maps.clear()
        if hasattr(self, 'attention_losses'):
            self.attention_losses.clear()
        
        # Call parent forward method
        outputs = super().forward(feats, targets, **kwargs)
        
        # Add GAIN loss to outputs if available
        if hasattr(self, 'attention_losses') and self.attention_losses:
            gain_loss = self.gain_loss_fn(self.attention_losses)
            outputs['gain_loss'] = gain_loss
        
        return outputs
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return attention maps for visualization"""
        return getattr(self, 'attention_maps', {})
    
    def get_gain_loss(self) -> torch.Tensor:
        """Return GAIN attention loss"""
        if hasattr(self, 'attention_losses') and self.attention_losses:
            return self.gain_loss_fn(self.attention_losses)
        return torch.tensor(0.0)
    
    def collect_attention_loss(self, layer_idx: int, attention_loss: torch.Tensor):
        """Collect attention loss from GAIN-enhanced layers"""
        if not hasattr(self, 'attention_losses'):
            self.attention_losses = []
        
        if attention_loss is not None:
            self.attention_losses.append(attention_loss)
    
    def collect_attention_map(self, layer_idx: int, attention_map: torch.Tensor):
        """Collect attention map from GAIN-enhanced layers"""
        if not hasattr(self, 'attention_maps'):
            self.attention_maps = {}
        
        if attention_map is not None:
            self.attention_maps[f'layer_{layer_idx}'] = attention_map


# Example configuration for creating GAIN-enhanced D-FINE
def create_gain_enhanced_dfine_config():
    """
    Create a configuration for GAIN-enhanced D-FINE
    """
    return {
        # Base D-FINE configuration
        'num_classes': 80,
        'hidden_dim': 256,
        'num_queries': 300,
        'feat_channels': [512, 1024, 2048],
        'feat_strides': [8, 16, 32],
        'num_levels': 3,
        'num_points': 4,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.0,
        'activation': 'relu',
        'num_denoising': 100,
        'label_noise_ratio': 0.5,
        'box_noise_scale': 1.0,
        'learn_query_content': False,
        'eval_spatial_size': None,
        'eval_idx': -1,
        'eps': 1e-2,
        'aux_loss': True,
        'cross_attn_method': 'default',
        'query_select_method': 'default',
        'reg_max': 32,
        'reg_scale': 4.0,
        'layer_scale': 1,
        
        # GAIN-specific configuration
        'gain_enabled': True,
        'gain_layers': [2, 4],  # Apply GAIN to layers 2 and 4
        'gain_attention_weight': 0.3,  # Weight for combining original and enhanced features
        'gain_self_guided': True,  # Enable self-guided attention refinement
        'gain_attention_supervision': False,  # Enable if you have attention supervision data
        'gain_loss_weight': 0.1,  # Weight for GAIN loss in total loss
    }


# Loss function that incorporates GAIN attention loss
class GAINAwareLoss(nn.Module):
    """
    Loss function that incorporates GAIN attention loss
    """
    
    def __init__(
        self,
        original_loss_fn: nn.Module,
        gain_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.original_loss_fn = original_loss_fn
        self.gain_loss_weight = gain_loss_weight
    
    def forward(self, outputs, targets):
        """
        Compute total loss including GAIN attention loss
        """
        # Original loss computation
        loss_dict = self.original_loss_fn(outputs, targets)
        
        # Add GAIN loss if available
        if 'gain_loss' in outputs:
            gain_loss = outputs['gain_loss']
            loss_dict['gain_loss'] = gain_loss
            
            # Add to total loss
            if 'loss' in loss_dict:
                loss_dict['loss'] += self.gain_loss_weight * gain_loss
            else:
                # Find total loss (usually the sum of all individual losses)
                total_loss = sum([v for k, v in loss_dict.items() if k != 'gain_loss' and isinstance(v, torch.Tensor)])
                loss_dict['loss'] = total_loss + self.gain_loss_weight * gain_loss
        
        return loss_dict


# Utility functions for visualization and analysis
def visualize_attention_maps(
    model: DFINETransformerWithGAIN,
    images: torch.Tensor,
    save_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Visualize attention maps generated by GAIN
    
    Args:
        model: GAIN-enhanced D-FINE model
        images: Input images [B, C, H, W]
        save_path: Path to save visualization (optional)
    
    Returns:
        Dictionary of attention maps
    """
    model.eval()
    with torch.no_grad():
        # Forward pass to generate attention maps
        _ = model(images)
        attention_maps = model.get_attention_maps()
    
    # Process and optionally save attention maps
    processed_maps = {}
    for layer_name, attn_map in attention_maps.items():
        if attn_map is not None:
            # Normalize attention map to [0, 1]
            attn_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            processed_maps[layer_name] = attn_normalized
            
            if save_path:
                # Save individual attention maps
                import torchvision.transforms.functional as TF
                for i in range(attn_normalized.shape[0]):
                    attn_img = attn_normalized[i, 0]  # [H, W]
                    TF.to_pil_image(attn_img).save(f"{save_path}_{layer_name}_batch_{i}.png")
    
    return processed_maps


# Integration test function
def test_gain_integration():
    """
    Test GAIN integration with D-FINE
    """
    print("?? Testing GAIN integration with D-FINE...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create GAIN-enhanced D-FINE
        model = DFINETransformerWithGAIN(
            num_classes=80,
            hidden_dim=256,
            num_queries=300,
            gain_enabled=True,
            gain_layers=[2, 4],
            gain_attention_weight=0.3,
        ).to(device)
        
        # Test input
        batch_size = 2
        feats = [
            torch.randn(batch_size, 512, 64, 64).to(device),    # Level 0
            torch.randn(batch_size, 1024, 32, 32).to(device),   # Level 1
            torch.randn(batch_size, 2048, 16, 16).to(device),   # Level 2
        ]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(feats)
        
        print(f"? Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Check for GAIN loss
        if 'gain_loss' in outputs:
            print(f"  GAIN loss: {outputs['gain_loss'].item():.6f}")
        else:
            print("  ??  No GAIN loss found in outputs")
        
        # Check attention maps
        attention_maps = model.get_attention_maps()
        print(f"  Attention maps: {len(attention_maps)} collected")
        
        return True
        
    except Exception as e:
        print(f"? GAIN integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_gain_integration() 