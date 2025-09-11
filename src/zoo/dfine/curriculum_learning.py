"""
Curriculum Learning for D-FINE Lightweight Models
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    total_epochs: int = 80
    warmup_epochs: int = 10
    easy_stage_ratio: float = 0.3
    medium_stage_ratio: float = 0.4
    hard_stage_ratio: float = 0.3
    min_decoder_layers: int = 3
    max_decoder_layers: int = 6
    min_distillation_weight: float = 0.1
    max_distillation_weight: float = 1.0
    difficulty_metrics: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_metrics is None:
            self.difficulty_metrics = ['object_size', 'crowd_density', 'occlusion_level']


class CurriculumTrainingScheduler:
    """
    Curriculum Learning Scheduler for D-FINE
    
    This scheduler implements a progressive training strategy that:
    1. Starts with easy samples and simple architecture
    2. Gradually increases sample difficulty and model complexity
    3. Adapts distillation weights based on training progress
    4. Provides dynamic sample filtering and weighting
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_epoch = 0
        self.current_stage = "warmup"
        
        # Stage boundaries
        self.easy_end = int(config.total_epochs * config.easy_stage_ratio)
        self.medium_end = self.easy_end + int(config.total_epochs * config.medium_stage_ratio)
        
        # Difficulty assessor
        self.difficulty_assessor = DifficultyAssessor()
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'sample_difficulties': [],
            'model_complexities': [],
            'distillation_weights': []
        }
    
    def update_epoch(self, epoch: int):
        """Update current epoch and stage"""
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            self.current_stage = "warmup"
        elif epoch < self.easy_end:
            self.current_stage = "easy"
        elif epoch < self.medium_end:
            self.current_stage = "medium"
        else:
            self.current_stage = "hard"
    
    def get_training_config(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Get training configuration for current epoch
        
        Args:
            epoch: Epoch number (uses current_epoch if None)
            
        Returns:
            Dictionary containing training configuration
        """
        if epoch is not None:
            self.update_epoch(epoch)
        
        progress = self.current_epoch / self.config.total_epochs
        
        # Progressive model complexity
        if self.current_stage == "warmup":
            num_decoder_layers = self.config.min_decoder_layers
            distillation_weight = 0.0
            sample_difficulty = "easy"
            complexity_budget = 0.3
        elif self.current_stage == "easy":
            # Very gradual increase in complexity
            stage_progress = (self.current_epoch - self.config.warmup_epochs) / (self.easy_end - self.config.warmup_epochs)
            # Only add 1 layer maximum during easy stage
            num_decoder_layers = min(
                self.config.min_decoder_layers + int(stage_progress * 1),
                self.config.max_decoder_layers
            )
            distillation_weight = self.config.min_distillation_weight * stage_progress * 0.5  # Reduce weight
            sample_difficulty = "easy"
            complexity_budget = 0.3 + 0.2 * stage_progress
        elif self.current_stage == "medium":
            # Gradual increase to max layers
            stage_progress = (self.current_epoch - self.easy_end) / (self.medium_end - self.easy_end)
            # Gradually increase to max layers
            num_decoder_layers = min(
                self.config.min_decoder_layers + 1 + int(stage_progress * (self.config.max_decoder_layers - self.config.min_decoder_layers - 1)),
                self.config.max_decoder_layers
            )
            distillation_weight = self.config.min_distillation_weight + 0.3 * stage_progress * (
                self.config.max_distillation_weight - self.config.min_distillation_weight)
            sample_difficulty = "mixed"
            complexity_budget = 0.5 + 0.3 * stage_progress
        else:  # hard stage
            # Full complexity
            num_decoder_layers = self.config.max_decoder_layers
            distillation_weight = self.config.max_distillation_weight * 0.5  # Reduce final weight
            sample_difficulty = "hard"
            complexity_budget = 1.0
        
        # Adaptive learning rate scaling
        lr_scale = self._get_lr_scale(progress)
        
        # Sample weighting strategy
        sample_weights = self._get_sample_weights()
        
        config = {
            'num_decoder_layers': num_decoder_layers,
            'distillation_weight': distillation_weight,
            'sample_difficulty': sample_difficulty,
            'complexity_budget': complexity_budget,
            'lr_scale': lr_scale,
            'sample_weights': sample_weights,
            'stage': self.current_stage,
            'progress': progress,
            'epoch': self.current_epoch
        }
        
        # Update statistics
        self.training_stats['model_complexities'].append(num_decoder_layers)
        self.training_stats['distillation_weights'].append(distillation_weight)
        
        return config
    
    def _get_lr_scale(self, progress: float) -> float:
        """Get learning rate scale based on training progress"""
        if self.current_stage == "warmup":
            return 0.1  # Lower LR during warmup
        elif self.current_stage == "easy":
            return 1.0  # Normal LR
        elif self.current_stage == "medium":
            return 0.8  # Slightly reduced LR
        else:
            return 0.5  # Lower LR for fine-tuning
    
    def _get_sample_weights(self) -> Dict[str, float]:
        """Get sample weighting strategy"""
        if self.current_stage == "warmup":
            return {'easy': 1.0, 'medium': 0.0, 'hard': 0.0}
        elif self.current_stage == "easy":
            return {'easy': 0.8, 'medium': 0.2, 'hard': 0.0}
        elif self.current_stage == "medium":
            return {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        else:
            return {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
    
    def filter_samples(
        self, 
        samples: List[Dict], 
        targets: List[Dict],
        difficulty_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict], List[float]]:
        """
        Filter samples based on current curriculum stage
        
        Args:
            samples: List of input samples
            targets: List of target annotations
            difficulty_threshold: Threshold for sample difficulty
            
        Returns:
            Tuple of (filtered_samples, filtered_targets, sample_weights)
        """
        if self.current_stage == "warmup":
            # Use only easiest samples
            return self._filter_by_difficulty(samples, targets, max_difficulty=0.3)
        elif self.current_stage == "easy":
            # Use easy to medium samples
            return self._filter_by_difficulty(samples, targets, max_difficulty=0.6)
        elif self.current_stage == "medium":
            # Use all but hardest samples
            return self._filter_by_difficulty(samples, targets, max_difficulty=0.9)
        else:
            # Use all samples
            difficulties = [self.difficulty_assessor.assess_sample(s, t) for s, t in zip(samples, targets)]
            weights = [1.0] * len(samples)
            return samples, targets, weights
    
    def _filter_by_difficulty(
        self, 
        samples: List[Dict], 
        targets: List[Dict], 
        max_difficulty: float
    ) -> Tuple[List[Dict], List[Dict], List[float]]:
        """Filter samples by maximum difficulty"""
        filtered_samples = []
        filtered_targets = []
        sample_weights = []
        
        for sample, target in zip(samples, targets):
            difficulty = self.difficulty_assessor.assess_sample(sample, target)
            
            if difficulty <= max_difficulty:
                filtered_samples.append(sample)
                filtered_targets.append(target)
                # Weight easier samples higher in early stages
                weight = 1.0 + (max_difficulty - difficulty) * 0.5
                sample_weights.append(weight)
        
        return filtered_samples, filtered_targets, sample_weights
    
    def get_loss_weights(self, outputs: Dict, targets: List[Dict]) -> Dict[str, float]:
        """
        Get adaptive loss weights based on current stage
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss weights
        """
        base_weights = {
            'classification': 1.0,
            'bbox_regression': 1.0,
            'distillation': self.get_training_config()['distillation_weight']
        }
        
        # Adaptive weighting based on stage
        if self.current_stage == "warmup":
            # Focus on classification in warmup
            base_weights['classification'] = 2.0
            base_weights['bbox_regression'] = 0.5
        elif self.current_stage == "easy":
            # Balanced weights
            pass
        elif self.current_stage == "medium":
            # Increase bbox regression weight
            base_weights['bbox_regression'] = 1.5
        else:
            # Focus on distillation in final stage
            base_weights['distillation'] = 2.0
        
        return base_weights
    
    def update_statistics(self, epoch_loss: float, sample_difficulties: List[float]):
        """Update training statistics"""
        self.training_stats['epoch_losses'].append(epoch_loss)
        self.training_stats['sample_difficulties'].extend(sample_difficulties)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'avg_epoch_loss': np.mean(self.training_stats['epoch_losses'][-10:]) if self.training_stats['epoch_losses'] else 0,
            'avg_sample_difficulty': np.mean(self.training_stats['sample_difficulties'][-1000:]) if self.training_stats['sample_difficulties'] else 0,
            'current_model_complexity': self.training_stats['model_complexities'][-1] if self.training_stats['model_complexities'] else 0,
            'current_distillation_weight': self.training_stats['distillation_weights'][-1] if self.training_stats['distillation_weights'] else 0
        }


class DifficultyAssessor:
    """
    Assess the difficulty of training samples
    """
    
    def __init__(self):
        self.difficulty_cache = {}
    
    def assess_sample(self, sample: Dict, target: Dict) -> float:
        """
        Assess the difficulty of a single sample
        
        Args:
            sample: Input sample (image data)
            target: Target annotations
            
        Returns:
            Difficulty score between 0 (easy) and 1 (hard)
        """
        # Use sample ID as cache key if available
        sample_id = sample.get('image_id', id(sample))
        if sample_id in self.difficulty_cache:
            return self.difficulty_cache[sample_id]
        
        difficulty_scores = []
        
        # Object size difficulty
        object_size_difficulty = self._assess_object_size(target)
        difficulty_scores.append(object_size_difficulty)
        
        # Crowd density difficulty
        crowd_density_difficulty = self._assess_crowd_density(target)
        difficulty_scores.append(crowd_density_difficulty)
        
        # Occlusion difficulty
        occlusion_difficulty = self._assess_occlusion(target)
        difficulty_scores.append(occlusion_difficulty)
        
        # Scale variation difficulty
        scale_variation_difficulty = self._assess_scale_variation(target)
        difficulty_scores.append(scale_variation_difficulty)
        
        # Overall difficulty (weighted average)
        weights = [0.3, 0.25, 0.25, 0.2]  # Weights for each difficulty metric
        overall_difficulty = sum(w * s for w, s in zip(weights, difficulty_scores))
        
        # Cache the result
        self.difficulty_cache[sample_id] = overall_difficulty
        
        return overall_difficulty
    
    def _assess_object_size(self, target: Dict) -> float:
        """Assess difficulty based on object sizes"""
        if 'boxes' not in target or len(target['boxes']) == 0:
            return 0.0
        
        boxes = target['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # Calculate object areas (assuming boxes are in [x1, y1, x2, y2] format)
        areas = []
        for box in boxes:
            if len(box) >= 4:
                width = abs(box[2] - box[0])
                height = abs(box[3] - box[1])
                area = width * height
                areas.append(area)
        
        if not areas:
            return 0.0
        
        # Smaller objects are more difficult
        avg_area = np.mean(areas)
        min_area = np.min(areas)
        
        # Normalize difficulty (smaller objects = higher difficulty)
        size_difficulty = 1.0 - min(1.0, avg_area / 10000)  # Assuming image size ~640x640
        small_object_penalty = 1.0 - min(1.0, min_area / 1000)
        
        return (size_difficulty + small_object_penalty) / 2
    
    def _assess_crowd_density(self, target: Dict) -> float:
        """Assess difficulty based on object density"""
        if 'boxes' not in target:
            return 0.0
        
        num_objects = len(target['boxes'])
        
        # More objects = higher difficulty
        if num_objects <= 2:
            return 0.1
        elif num_objects <= 5:
            return 0.3
        elif num_objects <= 10:
            return 0.6
        else:
            return min(1.0, 0.6 + (num_objects - 10) * 0.04)
    
    def _assess_occlusion(self, target: Dict) -> float:
        """Assess difficulty based on object occlusion"""
        if 'boxes' not in target or len(target['boxes']) < 2:
            return 0.0
        
        boxes = target['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # Calculate IoU between all pairs of boxes
        overlaps = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self._calculate_iou(boxes[i], boxes[j])
                overlaps.append(iou)
        
        if not overlaps:
            return 0.0
        
        # Higher overlap = higher difficulty
        max_overlap = max(overlaps)
        avg_overlap = np.mean(overlaps)
        
        return (max_overlap + avg_overlap) / 2
    
    def _assess_scale_variation(self, target: Dict) -> float:
        """Assess difficulty based on scale variation"""
        if 'boxes' not in target or len(target['boxes']) < 2:
            return 0.0
        
        boxes = target['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # Calculate areas
        areas = []
        for box in boxes:
            if len(box) >= 4:
                width = abs(box[2] - box[0])
                height = abs(box[3] - box[1])
                area = width * height
                areas.append(area)
        
        if len(areas) < 2:
            return 0.0
        
        # Higher scale variation = higher difficulty
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        
        if area_mean == 0:
            return 0.0
        
        coefficient_of_variation = area_std / area_mean
        return min(1.0, coefficient_of_variation)
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


class AdaptiveSampleWeighting:
    """
    Adaptive sample weighting based on training progress and sample difficulty
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
        self.loss_history = []
    
    def compute_sample_weights(
        self,
        losses: torch.Tensor,
        difficulties: List[float],
        stage: str
    ) -> torch.Tensor:
        """
        Compute adaptive sample weights
        
        Args:
            losses: Per-sample losses
            difficulties: Per-sample difficulty scores
            stage: Current curriculum stage
            
        Returns:
            Sample weights tensor
        """
        device = losses.device
        difficulties = torch.tensor(difficulties, device=device)
        
        # Base weights from focal loss style
        pt = torch.exp(-losses)
        focal_weights = self.alpha * (1 - pt) ** self.gamma
        
        # Difficulty-based weights
        if stage == "warmup":
            # Prefer easier samples
            difficulty_weights = 1.0 - difficulties
        elif stage == "easy":
            # Slight preference for easier samples
            difficulty_weights = 1.0 - 0.5 * difficulties
        elif stage == "medium":
            # Balanced weighting
            difficulty_weights = torch.ones_like(difficulties)
        else:
            # Prefer harder samples
            difficulty_weights = 0.5 + 0.5 * difficulties
        
        # Combine weights
        combined_weights = focal_weights * difficulty_weights
        
        # Normalize weights
        combined_weights = combined_weights / combined_weights.mean()
        
        return combined_weights
    
    def update_loss_history(self, epoch_loss: float):
        """Update loss history for adaptive weighting"""
        self.loss_history.append(epoch_loss)
        
        # Keep only recent history
        if len(self.loss_history) > 50:
            self.loss_history = self.loss_history[-50:] 