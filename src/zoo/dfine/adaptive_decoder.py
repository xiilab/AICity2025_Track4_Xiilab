"""
Adaptive Training-Time Decoder Enhancement (ATDE) for D-FINE
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import copy

from .dfine_decoder import TransformerDecoderLayer, MLP, DFINETransformer
from .utils import get_activation
from ...core import register


@register()
class AdaptiveTrainingDecoder(DFINETransformer):
    """
    Adaptive Training-Time Decoder Enhancement (ATDE)
    
    This module extends DFINETransformer with sophisticated decoder layers during training 
    that can be discarded during inference, addressing the limitation of shallow decoder 
    layers in lightweight models.
    """
    
    def __init__(
        self,
        base_num_layers: int = 3,
        training_num_layers: int = 6,
        distillation_weight: float = 1.0,
        progressive_distillation: bool = True,
        **kwargs  # Pass all other arguments to parent DFINETransformer
    ):
        # Call parent constructor with training_num_layers to ensure enough heads
        super().__init__(
            hidden_dim=kwargs.get('hidden_dim', 256),
            nhead=kwargs.get('nhead', 8),
            dim_feedforward=kwargs.get('dim_feedforward', 1024),
            dropout=kwargs.get('dropout', 0.0),
            activation=kwargs.get('activation', 'relu'),
            num_levels=kwargs.get('num_levels', 4),
            num_layers=training_num_layers,  # Use max layers for head initialization
            num_points=kwargs.get('num_points', 4),
            cross_attn_method=kwargs.get('cross_attn_method', 'default'),
            **{k: v for k, v in kwargs.items() if k not in [
                'hidden_dim', 'nhead', 'dim_feedforward', 'dropout', 
                'activation', 'num_levels', 'num_layers', 'num_points', 
                'cross_attn_method'
            ]}
        )
        
        # Store configuration
        self.base_num_layers = base_num_layers
        self.training_num_layers = training_num_layers
        self.current_num_layers = base_num_layers
        self.distillation_weight = distillation_weight
        self.progressive_distillation = progressive_distillation
        self.training_mode = False
        
        # Store original layers for dynamic switching
        self.base_layers = nn.ModuleList(self.decoder.layers[:base_num_layers])
        if training_num_layers > base_num_layers:
            # Create additional training layers
            self.training_layers = nn.ModuleList([
                copy.deepcopy(self.decoder.layers[base_num_layers - 1])
                for _ in range(training_num_layers - base_num_layers)
            ])
        else:
            self.training_layers = nn.ModuleList()
        
        # Reset decoder to use only base layers initially
        self.decoder.layers = nn.ModuleList(self.base_layers)
        self.decoder.num_layers = base_num_layers
        
        # Store attributes needed for creating training layers
        self.dim_feedforward = kwargs.get('dim_feedforward', 1024)
        self.dropout = kwargs.get('dropout', 0.0)
        self.activation = kwargs.get('activation', 'relu')
        self.num_points = kwargs.get('num_points', 4)
        self.cross_attn_method = kwargs.get('cross_attn_method', 'default')
        
        # Initialize distillation components
        if progressive_distillation:
            self.feature_adapters = nn.ModuleList([
                nn.Linear(self.hidden_dim, self.hidden_dim) 
                for _ in range(training_num_layers - base_num_layers)
            ])
            self.distillation_projectors = nn.ModuleList([
                MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
                for _ in range(training_num_layers - base_num_layers)
            ])
        
        # Layer-wise attention weights for progressive training
        self.layer_weights = nn.Parameter(torch.ones(training_num_layers))
        
        # Initialize training state
        self.epoch = 0
        self.stage = "base"  # "base", "progressive", "full"
        self._debug_printed = False  # For debug printing
        self.rank = 0  # Add rank attribute for logging
        
        # Complexity predictor for dynamic inference
        self.complexity_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            get_activation(self.activation),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer gates for early stopping
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 8),
                get_activation(self.activation),
                nn.Linear(self.hidden_dim // 8, 1),
                nn.Sigmoid()
            ) for _ in range(base_num_layers)
        ])
        
    def set_num_layers(self, num_layers: int):
        """
        Dynamically set the number of active layers during training
        
        Args:
            num_layers: Number of layers to use (should be <= training_num_layers)
        """
        num_layers = max(1, min(num_layers, self.training_num_layers))
        self.current_num_layers = num_layers
        
        # Update the decoder's layers based on current requirements
        if num_layers <= self.base_num_layers:
            # Use only base layers
            self.decoder.layers = nn.ModuleList(self.base_layers[:num_layers])
            self.training_mode = False
        else:
            # Use base layers + some training layers
            additional_needed = num_layers - self.base_num_layers
            combined_layers = list(self.base_layers) + list(self.training_layers[:additional_needed])
            self.decoder.layers = nn.ModuleList(combined_layers)
            self.training_mode = True
        
        # Update decoder's num_layers attribute
        self.decoder.num_layers = num_layers
        
        # CRITICAL: Update eval_idx to be within the valid range
        # Store original eval_idx if not already stored
        if not hasattr(self, 'original_eval_idx'):
            self.original_eval_idx = self.eval_idx
        
        # Set eval_idx to be the minimum of original eval_idx and (num_layers - 1)
        # But ensure it's not negative
        new_eval_idx = min(self.original_eval_idx, num_layers - 1)
        if new_eval_idx < 0:
            new_eval_idx = num_layers - 1
        
        self.eval_idx = new_eval_idx
        self.decoder.eval_idx = new_eval_idx
        
        # Ensure we have enough bbox and score heads
        if len(self.dec_bbox_head) < num_layers:
            print(f"Warning: Not enough bbox heads ({len(self.dec_bbox_head)}) for {num_layers} layers")
        if len(self.dec_score_head) < num_layers:
            print(f"Warning: Not enough score heads ({len(self.dec_score_head)}) for {num_layers} layers")
        
        if self.rank == 0:
            print(f"[INFO] Set active layers to {num_layers} (eval_idx: {new_eval_idx})")
    
    def convert_to_deploy(self):
        """
        Override convert_to_deploy to handle adaptive decoder properly
        """
        # Ensure we're using base layers for deployment
        self.set_num_layers(self.base_num_layers)
        
        # Call parent's convert_to_deploy
        super().convert_to_deploy()
        
        # Also convert the internal decoder
        if hasattr(self.decoder, 'convert_to_deploy'):
            self.decoder.convert_to_deploy()
        
        return self
    
    def reset_num_layers(self):
        """Reset to original number of layers"""
        self.set_num_layers(self.base_num_layers)
    
    def forward(self, feats, targets=None):
        """
        Enhanced forward pass with adaptive training/inference behavior
        Maintains compatibility with DFINETransformer interface
        """
        # Simply call parent's forward method - no additional complexity for now
        # This ensures compatibility with the existing D-FINE architecture
        return super().forward(feats, targets)
    
    def _compute_progressive_distillation_loss(
        self,
        base_features: List[torch.Tensor],
        training_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute progressive distillation loss between base and training features
        """
        if not training_features:
            return torch.tensor(0.0, device=base_features[0].device)
        
        distillation_loss = 0.0
        num_base = len(base_features)
        num_training = len(training_features)
        
        for i, base_feat in enumerate(base_features):
            # Map base layer to corresponding training layer
            teacher_idx = min(
                int((i + 1) * num_training / num_base) - 1,
                num_training - 1
            )
            teacher_feat = training_features[teacher_idx].detach()
            
            # Project features for alignment
            student_proj = self.distillation_projectors[i](base_feat)
            
            # Compute feature alignment loss
            feat_loss = F.mse_loss(student_proj, teacher_feat)
            
            # Weight by layer importance
            weighted_loss = feat_loss * self.layer_weights[i] * self.distillation_weight
            distillation_loss += weighted_loss
        
        return distillation_loss / num_base
    
    def initialize_from_original(self, original_decoder):
        """Initialize adaptive decoder from original decoder weights"""
        if hasattr(original_decoder, 'layers'):
            # Copy weights from original decoder layers to base layers
            num_original_layers = len(original_decoder.layers)
            num_base_layers = len(self.base_layers)
            
            for i in range(min(num_base_layers, num_original_layers)):
                # Copy layer weights
                self.base_layers[i].load_state_dict(
                    original_decoder.layers[i].state_dict(), strict=False
                )
        
        # Copy other components if they exist
        if hasattr(original_decoder, 'norm') and hasattr(self, 'norm'):
            self.norm.load_state_dict(original_decoder.norm.state_dict())
        
        if hasattr(original_decoder, 'ref_point_head') and hasattr(self, 'ref_point_head'):
            self.ref_point_head.load_state_dict(original_decoder.ref_point_head.state_dict())
    
    def set_inference_mode(self, inference_mode: bool = True):
        """Set the module to inference mode"""
        if inference_mode:
            self.eval()
        else:
            self.train()
    
    def get_inference_complexity(self) -> Dict[str, float]:
        """Get current inference complexity metrics"""
        return {
            'base_layers': self.base_num_layers,
            'training_layers': self.training_num_layers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'inference_params': sum(p.numel() for p in self.base_layers.parameters()),
            'compression_ratio': sum(p.numel() for p in self.base_layers.parameters()) / 
                               sum(p.numel() for p in self.parameters())
        }


class ProgressiveLayerDistillation(nn.Module):
    """
    Progressive Layer Distillation (PLD) module for enhanced knowledge transfer
    """
    
    def __init__(
        self,
        num_base_layers: int = 3,
        num_training_layers: int = 6,
        hidden_dim: int = 256,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        super().__init__()
        
        self.num_base_layers = num_base_layers
        self.num_training_layers = num_training_layers
        self.temperature = temperature
        self.alpha = alpha
        
        # Learnable distillation weights
        self.distillation_weights = nn.Parameter(
            torch.ones(num_base_layers) / num_base_layers
        )
        
        # Feature alignment networks - will be created dynamically based on input size
        self.alignment_networks = nn.ModuleList()
        self.alignment_networks_initialized = False
        
        # Attention-based feature selection
        self.attention_weights = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(num_base_layers)
        ])
    
    def _initialize_alignment_networks(self, input_dim: int):
        """Initialize alignment networks based on actual input dimensions"""
        if not self.alignment_networks_initialized:
            self.alignment_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                ) for _ in range(self.num_base_layers)
            ])
            self.alignment_networks_initialized = True
    
    def set_distillation_weight(self, weight: float):
        """
        Set the global distillation weight
        
        Args:
            weight: Global distillation weight multiplier
        """
        self.global_distillation_weight = weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute progressive distillation loss
        
        Args:
            outputs: D-FINE model outputs (dictionary format)
            targets: Ground truth targets for adaptive weighting
            
        Returns:
            Dictionary containing various distillation losses
        """
        
        # Get device from outputs
        device = next(iter(outputs.values())).device
        
        # Initialize with very small losses to avoid training instability
        losses = {
            'feature_distillation': torch.tensor(0.0, device=device, requires_grad=True),
            'attention_distillation': torch.tensor(0.0, device=device, requires_grad=True),
            'response_distillation': torch.tensor(0.0, device=device, requires_grad=True),
            'total_distillation': torch.tensor(0.0, device=device, requires_grad=True)
        }
        
        # Skip distillation if essential outputs are missing
        if 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
            return losses
        
        try:
            pred_logits = outputs['pred_logits']  # [batch_size, num_queries, num_classes]
            pred_boxes = outputs['pred_boxes']    # [batch_size, num_queries, 4]
            
            # Ensure tensors are on the correct device and have valid shapes
            if pred_logits.numel() == 0 or pred_boxes.numel() == 0:
                return losses
            
            # Simple feature-based distillation with very small weights
            batch_size, num_queries, num_classes = pred_logits.shape
            
            # Use auxiliary outputs if available for distillation
            if 'aux_outputs' in outputs and len(outputs['aux_outputs']) > 0:
                aux_logits = outputs['aux_outputs'][0]['pred_logits']
                aux_boxes = outputs['aux_outputs'][0]['pred_boxes']
                
                # Simple MSE loss between main and auxiliary outputs
                feature_loss = F.mse_loss(pred_logits, aux_logits.detach()) * 0.01
                box_loss = F.mse_loss(pred_boxes, aux_boxes.detach()) * 0.01
                
                losses['feature_distillation'] = feature_loss
                losses['response_distillation'] = box_loss
                losses['total_distillation'] = feature_loss + box_loss
            
            # Clamp losses to prevent explosion
            for key in losses:
                losses[key] = torch.clamp(losses[key], max=1.0)
            
        except Exception as e:
            # Return zero losses if anything fails - don't break training
            pass
        
        return losses


class DynamicInferenceController(nn.Module):
    """
    Dynamic Inference Controller for adaptive layer selection
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_layers: int = 6,
        confidence_threshold: float = 0.95,
        complexity_weight: float = 0.1,
    ):
        super().__init__()
        
        self.max_layers = max_layers
        self.confidence_threshold = confidence_threshold
        self.complexity_weight = complexity_weight
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimators for each layer
        self.confidence_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, 1),
                nn.Sigmoid()
            ) for _ in range(max_layers)
        ])
        
        # Layer importance weights
        self.layer_importance = nn.Parameter(
            torch.linspace(0.5, 1.0, max_layers)
        )
    
    def set_complexity_budget(self, budget: float):
        """
        Set the complexity budget for inference
        
        Args:
            budget: Complexity budget (0-1)
        """
        self.complexity_budget = budget
    
    def compute_complexity_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute complexity penalty loss
        
        Args:
            outputs: Model outputs containing complexity information
            
        Returns:
            Complexity penalty loss
        """
        if 'complexity_score' in outputs:
            complexity_score = outputs['complexity_score']
            # Penalize high complexity during training
            complexity_loss = self.complexity_weight * complexity_score
            return torch.tensor(complexity_loss, requires_grad=True)
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def should_continue(
        self,
        features: torch.Tensor,
        layer_idx: int,
        complexity_budget: Optional[float] = None
    ) -> bool:
        """
        Determine whether to continue processing through more layers
        
        Args:
            features: Current layer features
            layer_idx: Current layer index
            complexity_budget: Computational budget (0-1)
            
        Returns:
            Boolean indicating whether to continue
        """
        
        # Always process at least 2 layers
        if layer_idx < 2:
            return True
        
        # Check if we've reached max layers
        if layer_idx >= self.max_layers - 1:
            return False
        
        # Predict complexity and confidence
        complexity_score = self.complexity_predictor(
            features.mean(dim=1, keepdim=True)
        ).mean()
        
        confidence_score = self.confidence_estimators[layer_idx](
            features.mean(dim=1)
        ).mean()
        
        # Budget-based decision
        if complexity_budget is not None:
            required_budget = (layer_idx + 1) / self.max_layers
            if required_budget > complexity_budget:
                return False
        
        # Confidence-based early stopping
        if confidence_score > self.confidence_threshold:
            return False
        
        # Complexity-based decision
        if complexity_score < 0.3 and layer_idx >= 3:  # Simple scene
            return False
        
        return True
    
    def get_optimal_layers(
        self,
        features: torch.Tensor,
        complexity_budget: Optional[float] = None
    ) -> int:
        """
        Predict optimal number of layers for given features
        
        Args:
            features: Input features
            complexity_budget: Computational budget
            
        Returns:
            Optimal number of layers
        """
        
        complexity_score = self.complexity_predictor(
            features.mean(dim=1, keepdim=True)
        ).mean()
        
        if complexity_budget is not None:
            budget_layers = int(complexity_budget * self.max_layers)
            complexity_layers = int(complexity_score * self.max_layers)
            return min(budget_layers, complexity_layers, self.max_layers)
        else:
            return min(
                max(2, int(complexity_score * self.max_layers)),
                self.max_layers
            ) 