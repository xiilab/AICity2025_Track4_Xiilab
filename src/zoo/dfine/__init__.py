"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .dfine import DFINE
from .dfine_criterion import DFINECriterion, DFINEGAINCriterion
from .dfine_decoder import DFINETransformer
from .dfine_decoder_with_gain import DFINETransformerWithGAIN
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .postprocessor import DFINEPostProcessor
from .distortion_aware_neck import HybridDistortionAwareNeck
from .gain_attention import GuidedAttentionModule, GAINLoss

# Enhanced modules
from .adaptive_decoder import AdaptiveTrainingDecoder, ProgressiveLayerDistillation, DynamicInferenceController
from .enhanced_feature_fusion import AdaptiveFeaturePyramid, EnhancedMultiScaleFeatureFusion
from .curriculum_learning import CurriculumTrainingScheduler, CurriculumConfig, DifficultyAssessor, AdaptiveSampleWeighting
from .distortion_aware_neck import HybridDistortionAwareNeck
# Contrastive learning modules
from .contrastive_learning import (
    PositiveNegativeAttention, 
    ContrastiveLoss, 
    ContrastiveEnhancedDecoder, 
    ContrastiveDFINETransformer
)
from .contrastive_criterion import (
    ContrastiveDFINECriterion, 
    ContrastiveAwareMatcher, 
    ContrastiveEnhancedTrainer
)
