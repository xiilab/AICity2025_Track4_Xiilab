"""
Contrastive Enhanced D-FINE Criterion
Integrates contrastive learning loss with existing D-FINE losses
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ...core import register
from .dfine_criterion import DFINECriterion
from .contrastive_learning import ContrastiveLoss


@register()
class ContrastiveDFINECriterion(DFINECriterion):
    """
    Enhanced D-FINE Criterion with Contrastive Learning
    """
    
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        # Contrastive learning parameters
        contrastive_weight=1.0,
        contrastive_temperature=0.07,
        contrastive_pos_weight=2.0,
        contrastive_neg_weight=1.0,
        contrastive_margin=0.5,
        use_hard_negatives=True,
        hard_negative_ratio=0.3,
        contrastive_layers=[3, 4, 5],
        enable_contrastive=True
    ):
        """
        Args:
            contrastive_weight: Weight for contrastive loss
            contrastive_temperature: Temperature for contrastive learning
            contrastive_pos_weight: Weight for positive samples in contrastive loss
            contrastive_neg_weight: Weight for negative samples in contrastive loss
            contrastive_margin: Margin for contrastive loss
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_ratio: Ratio of hard negatives to mine
            contrastive_layers: Which decoder layers to apply contrastive learning
            enable_contrastive: Whether to enable contrastive learning
        """
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices
        )
        
        self.enable_contrastive = enable_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_layers = contrastive_layers
        
        if self.enable_contrastive:
            # Initialize contrastive loss module
            self.contrastive_loss = ContrastiveLoss(
                temperature=contrastive_temperature,
                pos_weight=contrastive_pos_weight,
                neg_weight=contrastive_neg_weight,
                margin=contrastive_margin,
                use_hard_negatives=use_hard_negatives,
                hard_negative_ratio=hard_negative_ratio
            )
            
            # Add contrastive losses to weight dict
            contrastive_losses = [
                'contrastive_total_loss',
                'contrastive_classification_loss', 
                'contrastive_contrastive_loss',
                'contrastive_hard_negative_loss',
                'contrastive_consistency_loss'
            ]
            
            for loss_name in contrastive_losses:
                if loss_name not in self.weight_dict:
                    if 'total' in loss_name:
                        self.weight_dict[loss_name] = contrastive_weight
                    else:
                        self.weight_dict[loss_name] = contrastive_weight * 0.1
    
    def forward(self, outputs, targets, **kwargs):
        """
        Compute losses including contrastive learning
        
        Args:
            outputs: Model outputs including contrastive outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Compute base D-FINE losses
        base_losses = super().forward(outputs, targets, **kwargs)
        
        # Add contrastive losses if enabled and available
        if self.enable_contrastive and 'contrastive_outputs' in outputs:
            contrastive_losses = self.compute_contrastive_losses(outputs, targets)
            base_losses.update(contrastive_losses)
        
        return base_losses
    
    def compute_contrastive_losses(
        self, 
        outputs: Dict, 
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive learning losses
        
        Args:
            outputs: Model outputs containing contrastive_outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of contrastive losses
        """
        if 'contrastive_outputs' not in outputs:
            return {}
        
        contrastive_outputs = outputs['contrastive_outputs']
        
        # Get matched indices from base matcher
        if hasattr(self, '_last_matched_indices'):
            matched_indices = self._last_matched_indices
        else:
            # Compute matching if not available
            with torch.no_grad():
                matched_indices = self.matcher(outputs, targets)
                self._last_matched_indices = matched_indices
        
        total_losses = {}
        
        # Compute contrastive loss for each layer
        for layer_key, layer_outputs in contrastive_outputs.items():
            if int(layer_key) in self.contrastive_layers:
                features = layer_outputs['enhanced_features']
                pos_neg_scores = layer_outputs['pos_neg_scores']
                
                # Compute contrastive loss for this layer
                layer_losses = self.contrastive_loss(
                    features, pos_neg_scores, targets, matched_indices
                )
                
                # Add layer prefix to loss names
                for loss_name, loss_value in layer_losses.items():
                    prefixed_name = f"layer_{layer_key}_{loss_name}"
                    total_losses[prefixed_name] = loss_value
        
        # Compute average losses across layers
        if total_losses:
            avg_losses = {}
            loss_types = set('_'.join(name.split('_')[2:]) for name in total_losses.keys())
            
            for loss_type in loss_types:
                layer_losses = [v for k, v in total_losses.items() if k.endswith(loss_type)]
                if layer_losses:
                    avg_losses[f"avg_{loss_type}"] = torch.stack(layer_losses).mean()
            
            total_losses.update(avg_losses)
        
        return total_losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Enhanced get_loss method that handles contrastive losses
        """
        # Handle contrastive losses
        if loss.startswith('contrastive_') or loss.startswith('layer_') or loss.startswith('avg_'):
            # These losses are computed in compute_contrastive_losses
            return {}
        
        # Handle base D-FINE losses
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)


class ContrastiveAwareMatcher(nn.Module):
    """
    Enhanced matcher that considers contrastive learning outputs
    """
    
    def __init__(
        self,
        base_matcher,
        use_contrastive_scores=True,
        contrastive_weight=0.1
    ):
        super().__init__()
        self.base_matcher = base_matcher
        self.use_contrastive_scores = use_contrastive_scores
        self.contrastive_weight = contrastive_weight
    
    def forward(self, outputs, targets):
        """
        Enhanced matching that considers contrastive scores
        """
        # Get base matching
        base_indices = self.base_matcher(outputs, targets)
        
        if not self.use_contrastive_scores or 'contrastive_outputs' not in outputs:
            return base_indices
        
        # Enhance matching with contrastive scores
        enhanced_indices = []
        contrastive_outputs = outputs['contrastive_outputs']
        
        for batch_idx, (base_query_idx, base_target_idx) in enumerate(base_indices):
            # Get contrastive scores for this batch
            batch_contrastive_scores = []
            
            for layer_key, layer_outputs in contrastive_outputs.items():
                if 'pos_neg_probs' in layer_outputs:
                    pos_probs = layer_outputs['pos_neg_probs'][batch_idx, :, 1]  # Positive probabilities
                    batch_contrastive_scores.append(pos_probs)
            
            if batch_contrastive_scores:
                # Average contrastive scores across layers
                avg_contrastive_scores = torch.stack(batch_contrastive_scores).mean(dim=0)
                
                # Enhance base matching with contrastive scores
                enhanced_query_idx, enhanced_target_idx = self._enhance_matching(
                    base_query_idx, base_target_idx, avg_contrastive_scores
                )
                
                enhanced_indices.append((enhanced_query_idx, enhanced_target_idx))
            else:
                enhanced_indices.append((base_query_idx, base_target_idx))
        
        return enhanced_indices
    
    def _enhance_matching(
        self, 
        query_idx: torch.Tensor, 
        target_idx: torch.Tensor, 
        contrastive_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhance matching using contrastive scores
        """
        if len(query_idx) == 0:
            return query_idx, target_idx
        
        # Get contrastive scores for matched queries
        matched_scores = contrastive_scores[query_idx]
        
        # Filter out low-confidence matches
        confidence_threshold = 0.3
        high_confidence_mask = matched_scores > confidence_threshold
        
        if high_confidence_mask.sum() > 0:
            # Keep only high-confidence matches
            enhanced_query_idx = query_idx[high_confidence_mask]
            enhanced_target_idx = target_idx[high_confidence_mask]
        else:
            # If no high-confidence matches, keep original
            enhanced_query_idx = query_idx
            enhanced_target_idx = target_idx
        
        return enhanced_query_idx, enhanced_target_idx


@register()
class ContrastiveEnhancedTrainer(nn.Module):
    """
    Training wrapper that handles contrastive learning integration
    """
    
    def __init__(
        self,
        model,
        criterion,
        enable_contrastive_warmup=True,
        contrastive_warmup_epochs=5,
        contrastive_start_weight=0.1,
        contrastive_end_weight=1.0
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.enable_contrastive_warmup = enable_contrastive_warmup
        self.contrastive_warmup_epochs = contrastive_warmup_epochs
        self.contrastive_start_weight = contrastive_start_weight
        self.contrastive_end_weight = contrastive_end_weight
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup scheduling"""
        self.current_epoch = epoch
        
        if self.enable_contrastive_warmup and hasattr(self.criterion, 'contrastive_weight'):
            # Compute warmup weight
            if epoch < self.contrastive_warmup_epochs:
                warmup_ratio = epoch / self.contrastive_warmup_epochs
                current_weight = (
                    self.contrastive_start_weight + 
                    (self.contrastive_end_weight - self.contrastive_start_weight) * warmup_ratio
                )
            else:
                current_weight = self.contrastive_end_weight
            
            # Update criterion weight
            self.criterion.contrastive_weight = current_weight
            
            # Update weight dict
            for loss_name in self.criterion.weight_dict:
                if 'contrastive' in loss_name:
                    if 'total' in loss_name:
                        self.criterion.weight_dict[loss_name] = current_weight
                    else:
                        self.criterion.weight_dict[loss_name] = current_weight * 0.1
    
    def forward(self, samples, targets):
        """Forward pass with contrastive learning"""
        # Forward through model
        outputs = self.model(samples)
        
        # Compute losses
        loss_dict = self.criterion(outputs, targets)
        
        # Add epoch info for logging
        loss_dict['contrastive_epoch'] = torch.tensor(self.current_epoch, dtype=torch.float32)
        if hasattr(self.criterion, 'contrastive_weight'):
            loss_dict['contrastive_weight'] = torch.tensor(
                self.criterion.contrastive_weight, dtype=torch.float32
            )
        
        return outputs, loss_dict 