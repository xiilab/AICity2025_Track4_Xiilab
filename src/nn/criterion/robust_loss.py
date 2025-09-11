import torch
import torch.nn.functional as F
from typing import Dict

from ...core import register
from .oadg_losses import (
    ContrastiveLossPlus, JSDLoss, supcontrast,
    smooth_l1_dg_loss, l1_dg_loss, jsd_loss, jsdv1_3_loss
)
from ...data.augmentation.robust_augmentation import get_augmented_views


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@register()
class RobustLossWrapper:
    """
    Enhanced Robust Loss Wrapper for D-FINE
    
    Supports:
    - Calibration Loss: For well-calibrated predictions
    - BBox Consistency Loss: For consistent box predictions
    - OA-DG Losses: JSD and Contrastive for domain generalization
    
    NOTE: Real domain robustness now comes from enhanced image-level 
    augmentation in dataloader_robust_enhanced.yml rather than 
    feature-level augmentation.
    """
    def __init__(
        self,
        base_criterion=None,
        criterion=None,
        # OA-DG parameters...
        use_oadg_losses=True,
        oadg_weight=0.1,
        oadg_loss_types=('jsd', 'contrastive'),
        oadg_start_epoch=0,
        oadg_aug_config=None,
        # Calibration & BBox Consistency
        use_calibration_loss=True,
        calibration_weight=0.05,
        use_bbox_consistency_loss=True,
        bbox_consistency_weight=0.1,
        # Sampling ratio
        sample_ratio=0.5
    ):
        self.base_criterion = base_criterion or criterion
        self.use_oadg_losses = use_oadg_losses
        self.oadg_weight = oadg_weight
        self.oadg_loss_types = list(oadg_loss_types)
        self.oadg_start_epoch = oadg_start_epoch
        self.oadg_aug_config = oadg_aug_config or {
            'use_image_aug': False,
            'use_feature_aug': True,
            'num_views': 3,
            'aug_strength': 0.05,
            'feature_aug_types': ['dropout', 'noise']
        }
        self.use_calibration_loss = use_calibration_loss
        self.calibration_weight = calibration_weight
        self.use_bbox_consistency_loss = use_bbox_consistency_loss
        self.bbox_consistency_weight = bbox_consistency_weight
        self.sample_ratio = sample_ratio
        self._warned_image_aug = False
    # def __init__(
    #     self,
    #     base_criterion=None,
    #     criterion=None,  # Backward compatibility
    #     # OA-DG parameters
    #     use_oadg_losses=True,
    #     oadg_weight=0.1,
    #     oadg_loss_types=('jsd', 'contrastive'),
    #     oadg_start_epoch=0,
    #     oadg_aug_config=None,  # OA-DG augmentation config from cfg
    #     # Calibration & BBox Consistency
    #     use_calibration_loss=True,
    #     calibration_weight=0.05,
    #     use_bbox_consistency_loss=True,
    #     bbox_consistency_weight=0.1,
    #     # Legacy compatibility
    #     fdr_loss_weight=None,
    #     oadg_loss_weight=None,
    #     # Sampling ratio for efficiency
    #     sample_ratio=0.5
    # ):
    #     # Handle backward compatibility
    #     self.base_criterion = base_criterion or criterion
        
    #     # OA-DG settings
    #     self.use_oadg_losses = use_oadg_losses
    #     self.oadg_weight = oadg_loss_weight or oadg_weight
    #     self.oadg_loss_types = list(oadg_loss_types)
    #     self.oadg_start_epoch = oadg_start_epoch
    #     self.oadg_aug_config = oadg_aug_config or {
    #         'use_image_aug': False,
    #         'use_feature_aug': True,
    #         'num_views': 3,
    #         'aug_strength': 0.05,
    #         'feature_aug_types': ['dropout', 'noise']
    #     }
        
    #     # Calibration & BBox Consistency settings
    #     self.use_calibration_loss = use_calibration_loss
    #     self.calibration_weight = fdr_loss_weight or calibration_weight
    #     self.use_bbox_consistency_loss = use_bbox_consistency_loss
    #     self.bbox_consistency_weight = bbox_consistency_weight
        
    #     # Sampling
    #     self.sample_ratio = sample_ratio
        
    #     # Warning flag to show message only once
    #     self._warned_image_aug = False
        
        print(f"🔧 RobustLossWrapper initialized:")
        print(f"   - Calibration: {self.use_calibration_loss} (weight: {self.calibration_weight})")
        print(f"   - BBox Consistency: {self.use_bbox_consistency_loss} (weight: {self.bbox_consistency_weight})")
        print(f"   - OA-DG: {self.use_oadg_losses} (weight: {self.oadg_weight}, types: {self.oadg_loss_types})")
        print(f"   - Enhanced image augmentation active in dataloader")

    @classmethod
    def phase1(cls, base_criterion, sample_ratio=0.5,
               calibration_weight=0.05, bbox_consistency_weight=0.1):
        """Phase 1: only calibration and bbox consistency losses."""
        return cls(
            base_criterion,
            use_oadg_losses=False,
            use_calibration_loss=True,
            calibration_weight=calibration_weight,
            use_bbox_consistency_loss=True,
            bbox_consistency_weight=bbox_consistency_weight,
            sample_ratio=sample_ratio
        )

    @classmethod
    def phase2_jsd(cls, base_criterion, sample_ratio=0.5, oadg_weight=0.1):
        """Phase 2: add Jensen-Shannon Divergence (JSD) only."""
        return cls(
            base_criterion,
            use_oadg_losses=True,
            oadg_weight=oadg_weight,
            oadg_loss_types=('jsd',),
            use_calibration_loss=False,
            use_bbox_consistency_loss=False,
            sample_ratio=sample_ratio
        )

    @classmethod
    def phase2_contrastive(cls, base_criterion, sample_ratio=0.5, oadg_weight=0.1):
        """Phase 2: add Contrastive loss only."""
        return cls(
            base_criterion,
            use_oadg_losses=True,
            oadg_weight=oadg_weight,
            oadg_loss_types=('contrastive',),
            use_calibration_loss=False,
            use_bbox_consistency_loss=False,
            sample_ratio=sample_ratio
        )

    def __call__(self, outputs, targets, **kwargs):
        """
        Compute robust losses with proper image-level augmentation support
        """
        # Get epoch for conditional losses
        epoch = kwargs.get('epoch', 0)
        
        # Base detection losses
        if self.base_criterion is not None:
            base_losses = self.base_criterion(outputs, targets, **kwargs)
        else:
            raise ValueError("No base criterion provided to RobustLossWrapper")
            
        robust_losses: Dict[str, torch.Tensor] = {}

        # OA-DG Losses (conditional on epoch)
        if self.use_oadg_losses and epoch >= self.oadg_start_epoch:
            for loss_type in self.oadg_loss_types:
                oadg_loss = self.compute_oadg_losses(outputs, targets, loss_type)
                robust_losses[f'loss_oadg_{loss_type}'] = oadg_loss * self.oadg_weight
                
        # Calibration Loss
        if self.use_calibration_loss:
            cal_loss = self.compute_calibration_loss(outputs, targets)
            robust_losses['loss_calibration'] = cal_loss * self.calibration_weight
            
        # BBox Consistency Loss
        if self.use_bbox_consistency_loss:
            bbox_loss = self.compute_bbox_consistency_loss(outputs, targets)
            robust_losses['loss_bbox_consistency'] = bbox_loss * self.bbox_consistency_weight

        return {**base_losses, **robust_losses}

    def compute_oadg_losses(self, outputs, targets, loss_type='jsd'):
        """
        Compute Out-of-distribution Aware Domain Generalization losses
        
        NOTE: This now uses minimal feature-level augmentation since the real
        domain generalization comes from strong image-level augmentation in 
        the enhanced dataloader (dataloader_robust_enhanced.yml).
        """
        if 'pred_logits' not in outputs:
            return torch.tensor(0.0, device=list(outputs.values())[0].device)
        
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        try:
            if loss_type == 'jsd':
                # Use config-based augmentation (can be image-level or feature-level)
                if self.oadg_aug_config.get('use_image_aug', False):
                    # Image-level augmentation requested but images not available at loss stage
                    # Fall back to feature-level with minimal perturbation for consistency
                    if not self._warned_image_aug:
                        print("⚠️ Image-level augmentation requested but not available at loss computation stage")
                        print("   Using minimal feature-level augmentation for OA-DG consistency")
                        self._warned_image_aug = True
                    aug_config = {
                        'use_image_aug': False,
                        'use_feature_aug': True,
                        'num_views': self.oadg_aug_config.get('num_views', 3),
                        'aug_strength': 0.02,  # Very minimal for consistency only
                        'feature_aug_types': ['dropout']  # Only dropout for minimal perturbation
                    }
                else:
                    # Use config-provided feature augmentation settings
                    aug_config = self.oadg_aug_config
                
                augmented_views = get_augmented_views(features=pred_logits, aug_config=aug_config)
                
                # Direct JSD computation
                pred_orig, pred_aug1, pred_aug2 = augmented_views[0], augmented_views[1], augmented_views[2]
                
                # Softmax to get probability distributions
                eps = 1e-8
                p_orig = F.softmax(pred_orig, dim=-1) + eps
                p_aug1 = F.softmax(pred_aug1, dim=-1) + eps  
                p_aug2 = F.softmax(pred_aug2, dim=-1) + eps
                
                # Mixture distribution
                p_mixture = (p_orig + p_aug1 + p_aug2) / 3.0
                
                # KL divergences
                kl1 = F.kl_div(torch.log(p_mixture), p_orig, reduction='batchmean')
                kl2 = F.kl_div(torch.log(p_mixture), p_aug1, reduction='batchmean')
                kl3 = F.kl_div(torch.log(p_mixture), p_aug2, reduction='batchmean')
                
                # Jensen-Shannon Divergence
                jsd = (kl1 + kl2 + kl3) / 3.0
                return jsd
                
            elif loss_type == 'contrastive':
                # Simplified contrastive loss
                features = pred_logits.view(-1, pred_logits.size(-1))
                
                # Create positive and negative pairs
                pos_sim = F.cosine_similarity(features, features, dim=1).mean()
                neg_sim = F.cosine_similarity(features, torch.roll(features, 1, 0), dim=1).mean()
                
                # Contrastive loss: maximize positive similarity, minimize negative
                contrastive_loss = F.relu(0.5 - pos_sim + neg_sim)
                return contrastive_loss
                
            else:
                return torch.tensor(0.0, device=device)
                
        except Exception as e:
            print(f"⚠️ Warning: OADG loss computation failed: {e}")
            return torch.tensor(0.0, device=device)

    def compute_calibration_loss(self, outputs, targets):
        """
        Calibration Loss adapted for Fisheye without explicit FOV info
        """
        if 'pred_logits' not in outputs:
            return torch.tensor(0.0, device=outputs.get('pred_boxes', torch.tensor([])).device)

        logits = outputs['pred_logits']  # [B, N, C]
        device = logits.device
        B, N, C = logits.shape

        # 1) radial distance from center (0.5, 0.5)
        centers = outputs['pred_boxes'][..., :2]
        dx = centers[..., 0] - 0.5
        dy = centers[..., 1] - 0.5
        radial = torch.sqrt(dx**2 + dy**2 + 1e-8) / (2**0.5 * 0.5)  # Added 1e-8 for numerical stability
        radial = radial.clamp(0.0, 1.0)

        # 2) variable temperature
        t_center, t_edge = 1.5, 3.0
        temperature = t_center + (t_edge - t_center) * radial.unsqueeze(-1)
        scaled = logits / temperature

        # 3) softmax & confidence penalty
        probs = F.softmax(scaled, dim=-1)
        maxp = probs.max(dim=-1)[0]
        conf_penalty = F.relu(maxp - 0.85).mean()

        # 4) entropy regularization
        base_ent = torch.log(torch.tensor(C, device=device))
        alpha_c, alpha_e = 0.5, 0.7
        target_ent = base_ent * (alpha_c + (alpha_e - alpha_c) * radial)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        ent_loss = F.mse_loss(entropy, target_ent).mean()

        return conf_penalty + 0.1 * ent_loss

    def compute_bbox_consistency_loss(self, outputs, targets):
        """
        Consistency Loss adapted via approximate spherical unprojection
        """
        if 'pred_boxes' not in outputs:
            return torch.tensor(0.0, device=outputs.get('pred_logits', torch.tensor([])).device)

        boxes = outputs['pred_boxes']  # [B, N, 4]
        device = boxes.device
        B, N, _ = boxes.shape

        centers = boxes[..., :2]
        dx = centers[..., 0] - 0.5
        dy = centers[..., 1] - 0.5
        r = torch.sqrt(dx**2 + dy**2).clamp(0,1)

        theta = r * (torch.pi / 2)
        phi = torch.atan2(dy, dx)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        dirs = torch.stack([x, y, z], dim=-1)

        dirs_exp = dirs.view(B, N, 1, 3)
        dots = (dirs_exp * dirs_exp.transpose(1,2)).sum(-1).clamp(-1,1)
        sph_dist = torch.acos(dots)

        nearby = (sph_dist < 0.2) & (sph_dist > 0.02)
        if nearby.sum() == 0:
            return torch.tensor(0.0, device=device)

        sizes = boxes[..., 2:]
        size_dist = torch.cdist(sizes, sizes, p=2)
        loss = (size_dist * nearby.float()).sum() / (nearby.sum().float() + 1e-8)
        return loss


    # def compute_calibration_loss(self, outputs, targets):
    #     """
    #     Calibration Loss for well-calibrated predictions
        
    #     This complements the image-level augmentation by ensuring the model
    #     outputs are properly calibrated for robust performance.
    #     """
    #     if 'pred_logits' not in outputs:
    #         return torch.tensor(0.0, device=outputs.get('pred_boxes', torch.tensor([])).device)
            
    #     logits = outputs['pred_logits']
    #     device = logits.device
        
    #     # Temperature scaling for calibration
    #     temperature = 2.0
    #     scaled_logits = logits / temperature
    #     probs = F.softmax(scaled_logits, dim=-1)
        
    #     # Confidence calibration: penalize overconfident predictions
    #     max_probs = torch.max(probs, dim=-1)[0]
    #     confidence_penalty = F.relu(max_probs - 0.85).mean()
        
    #     # Entropy regularization for better calibration
    #     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    #     target_entropy = torch.log(torch.tensor(logits.size(-1), device=device)) * 0.6
    #     entropy_loss = F.mse_loss(entropy.mean(), target_entropy)
        
    #     calibration_loss = confidence_penalty + entropy_loss * 0.1
    #     return calibration_loss

    # def compute_bbox_consistency_loss(self, outputs, targets):
    #     """
    #     BBox Consistency Loss for stable box predictions
        
    #     This encourages consistent bounding box predictions across
    #     similar spatial regions, complementing image-level augmentation.
    #     """
    #     if 'pred_boxes' not in outputs:
    #         return torch.tensor(0.0, device=outputs.get('pred_logits', torch.tensor([])).device)
        
    #     pred_boxes = outputs['pred_boxes']
    #     device = pred_boxes.device
    #     batch_size, num_queries = pred_boxes.shape[:2]
        
    #     try:
    #         # Spatial consistency: boxes at similar locations should have similar sizes
    #         centers = pred_boxes[..., :2]  # cx, cy
    #         sizes = pred_boxes[..., 2:]    # w, h
            
    #         # Compute pairwise center distances
    #         center_dist = torch.cdist(centers.view(batch_size, num_queries, 2), 
    #                                  centers.view(batch_size, num_queries, 2))
            
    #         # For nearby centers, encourage similar sizes
    #         nearby_mask = (center_dist < 0.15) & (center_dist > 0.01)  # Avoid self-comparison
            
    #         if nearby_mask.sum() > 0:
    #             # Size consistency for nearby boxes
    #             size_dist = torch.cdist(sizes.view(batch_size, num_queries, 2),
    #                                   sizes.view(batch_size, num_queries, 2))
    #             consistency_loss = (size_dist * nearby_mask.float()).sum() / (nearby_mask.sum() + 1e-8)
    #         else:
    #             consistency_loss = torch.tensor(0.0, device=device)
                
    #         return consistency_loss
            
    #     except Exception as e:
    #         print(f"⚠️ Warning: BBox consistency loss computation failed: {e}")
    #         return torch.tensor(0.0, device=device)


# ============================================================================
# ENHANCED ROBUST TRAINING DOCUMENTATION
# ============================================================================

"""
🎯 ENHANCED ROBUST TRAINING METHODOLOGY

This implementation combines the best of both approaches:

1. ✅ PRIMARY ROBUSTNESS: Enhanced Image-Level Augmentation
   Location: dataloader_robust_enhanced.yml
   - RandomPhotometricDistort: Real lighting/color variations
   - GaussianNoise: Real sensor noise simulation
   - JPEGCompression: Real compression artifacts
   - GeometricAugmentationPipeline: Real perspective changes
   - CoarseDropout: Real occlusion simulation
   - MixUp/CutMix: Real composite scenarios
   
2. ✅ SECONDARY ROBUSTNESS: Prediction-Level Losses
   - Calibration Loss: Well-calibrated confidence scores
   - BBox Consistency Loss: Stable spatial predictions
   - OA-DG Losses: Minimal feature-level consistency (JSD/Contrastive)

3. ✅ TRAINING PHASES:
   - Phase 1: Calibration + BBox Consistency (stabilization)
   - Phase 2: Add OA-DG losses (domain generalization)

BENEFITS:
- True visual robustness from image augmentation
- Prediction stability from calibration losses
- Domain generalization from OA-DG losses
- Efficient training with proper loss balancing

USAGE:
- Enhanced augmentation: ENABLED in dataloader
- Feature augmentation: MINIMAL (only for consistency)
- All robust losses: ACTIVE and properly balanced

This provides comprehensive robust training for object detection.
"""
 