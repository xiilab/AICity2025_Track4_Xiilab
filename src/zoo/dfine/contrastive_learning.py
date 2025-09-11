"""
Contrastive Learning Module for D-FINE
Adds positive/negative sample discrimination and contrastive loss for improved detection performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from ...core import register


class PositiveNegativeAttention(nn.Module):
    """
    Attention layer that discriminates between positive and negative samples
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for positive/negative discrimination
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Positive/Negative classification head
        self.pos_neg_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # [negative_score, positive_score]
        )
        
        # Feature enhancement for positive samples
        self.positive_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Feature suppression for negative samples
        self.negative_suppressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Suppression weights
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        query_features: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: [batch_size, num_queries, hidden_dim]
            target_labels: [batch_size, num_queries] - ground truth labels for supervision
            return_attention: whether to return attention weights
            
        Returns:
            Dictionary containing enhanced features and classification scores
        """
        batch_size, num_queries, _ = query_features.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query_features)  # [B, N, D]
        K = self.k_proj(query_features)  # [B, N, D]
        V = self.v_proj(query_features)  # [B, N, D]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, V)
        attended_features = attended_features.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.hidden_dim
        )
        
        # Positive/Negative classification
        pos_neg_scores = self.pos_neg_classifier(attended_features)  # [B, N, 2]
        pos_neg_probs = F.softmax(pos_neg_scores, dim=-1)
        
        # Get positive and negative probabilities
        neg_probs = pos_neg_probs[:, :, 0]  # [B, N]
        pos_probs = pos_neg_probs[:, :, 1]  # [B, N]
        
        # Create masks for positive and negative samples
        pos_mask = (pos_probs > self.pos_threshold).float()
        neg_mask = (neg_probs > self.neg_threshold).float()
        
        # Enhance positive samples
        positive_enhanced = self.positive_enhancer(attended_features)
        positive_enhanced = positive_enhanced * pos_mask.unsqueeze(-1)
        
        # Suppress negative samples
        negative_suppression = self.negative_suppressor(attended_features)
        negative_suppressed = attended_features * negative_suppression
        negative_suppressed = negative_suppressed * neg_mask.unsqueeze(-1)
        
        # Combine enhanced and suppressed features
        enhanced_features = (
            query_features +  # Residual connection
            positive_enhanced +
            negative_suppressed
        )
        
        enhanced_features = self.layer_norm(enhanced_features)
        
        outputs = {
            'enhanced_features': enhanced_features,
            'pos_neg_scores': pos_neg_scores,
            'pos_neg_probs': pos_neg_probs,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'positive_enhanced': positive_enhanced,
            'negative_suppressed': negative_suppressed
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for positive/negative sample discrimination
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        margin: float = 0.5,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.3
    ):
        super().__init__()
        
        self.temperature = temperature
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.margin = margin
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        features: torch.Tensor,
        pos_neg_scores: torch.Tensor,
        targets: List[Dict],
        matched_indices: Optional[List[Tuple]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, num_queries, hidden_dim]
            pos_neg_scores: [batch_size, num_queries, 2] - [neg_score, pos_score]
            targets: List of target dictionaries
            matched_indices: List of (query_indices, target_indices) tuples
            
        Returns:
            Dictionary of loss components
        """
        batch_size, num_queries, hidden_dim = features.shape
        device = features.device
        
        # Create positive/negative labels
        pos_neg_labels = torch.zeros(batch_size, num_queries, dtype=torch.long, device=device)
        
        if matched_indices is not None:
            for batch_idx, (query_indices, target_indices) in enumerate(matched_indices):
                if len(query_indices) > 0:
                    pos_neg_labels[batch_idx, query_indices] = 1  # Positive samples
        
        # Classification loss for positive/negative discrimination
        pos_neg_scores_flat = pos_neg_scores.view(-1, 2)
        pos_neg_labels_flat = pos_neg_labels.view(-1)
        
        classification_loss = self.cross_entropy(pos_neg_scores_flat, pos_neg_labels_flat)
        classification_loss = classification_loss.view(batch_size, num_queries)
        
        # Weight positive and negative samples differently
        pos_mask = (pos_neg_labels == 1).float()
        neg_mask = (pos_neg_labels == 0).float()
        
        weighted_classification_loss = (
            classification_loss * pos_mask * self.pos_weight +
            classification_loss * neg_mask * self.neg_weight
        )
        
        # Contrastive loss between positive and negative features
        contrastive_loss = self._compute_contrastive_loss(
            features, pos_neg_labels, pos_mask, neg_mask
        )
        
        # Hard negative mining
        if self.use_hard_negatives:
            hard_negative_loss = self._compute_hard_negative_loss(
                features, pos_neg_scores, pos_neg_labels
            )
        else:
            hard_negative_loss = torch.tensor(0.0, device=device)
        
        # Feature consistency loss
        consistency_loss = self._compute_consistency_loss(features, pos_neg_labels)
        
        # Total loss
        total_loss = (
            weighted_classification_loss.mean() +
            contrastive_loss +
            hard_negative_loss +
            0.1 * consistency_loss
        )
        
        return {
            'contrastive_total_loss': total_loss,
            'contrastive_classification_loss': weighted_classification_loss.mean(),
            'contrastive_contrastive_loss': contrastive_loss,
            'contrastive_hard_negative_loss': hard_negative_loss,
            'contrastive_consistency_loss': consistency_loss,
            'pos_neg_labels': pos_neg_labels
        }
    
    def _compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between positive and negative samples"""
        batch_size, num_queries, hidden_dim = features.shape
        
        # Normalize features
        features_norm = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features_norm, features_norm.transpose(-2, -1))
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create positive and negative pair masks
        pos_pairs = torch.matmul(pos_mask.unsqueeze(-1), pos_mask.unsqueeze(-2))
        neg_pairs = torch.matmul(neg_mask.unsqueeze(-1), neg_mask.unsqueeze(-2))
        
        # Remove self-similarity
        eye_mask = torch.eye(num_queries, device=features.device).unsqueeze(0).expand(batch_size, -1, -1)
        pos_pairs = pos_pairs * (1 - eye_mask)
        neg_pairs = neg_pairs * (1 - eye_mask)
        
        # Positive loss: maximize similarity between positive samples
        pos_loss = -similarity_matrix * pos_pairs
        pos_loss = pos_loss.sum() / (pos_pairs.sum() + 1e-8)
        
        # Negative loss: minimize similarity between negative samples
        neg_loss = F.relu(similarity_matrix - self.margin) * neg_pairs
        neg_loss = neg_loss.sum() / (neg_pairs.sum() + 1e-8)
        
        return pos_loss + neg_loss
    
    def _compute_hard_negative_loss(
        self,
        features: torch.Tensor,
        pos_neg_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for hard negative samples"""
        batch_size, num_queries = labels.shape
        
        # Get negative samples
        neg_mask = (labels == 0)
        
        if neg_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Get confidence scores for negative samples
        neg_confidence = F.softmax(pos_neg_scores, dim=-1)[:, :, 0]  # Negative confidence
        neg_confidence = neg_confidence * neg_mask.float()
        
        # Select hard negatives (low confidence negatives)
        num_hard_negatives = max(1, int(neg_mask.sum().item() * self.hard_negative_ratio))
        
        # Get indices of hard negatives
        neg_confidence_flat = neg_confidence.view(-1)
        neg_mask_flat = neg_mask.view(-1)
        
        if neg_mask_flat.sum() > 0:
            neg_indices = torch.where(neg_mask_flat)[0]
            neg_scores = neg_confidence_flat[neg_indices]
            
            if len(neg_scores) >= num_hard_negatives:
                _, hard_neg_indices = torch.topk(neg_scores, num_hard_negatives, largest=False)
                hard_neg_global_indices = neg_indices[hard_neg_indices]
                
                # Create hard negative mask
                hard_neg_mask = torch.zeros_like(neg_mask_flat)
                hard_neg_mask[hard_neg_global_indices] = 1
                hard_neg_mask = hard_neg_mask.view(batch_size, num_queries)
                
                # Compute loss for hard negatives
                hard_neg_loss = self.cross_entropy(
                    pos_neg_scores.view(-1, 2),
                    labels.view(-1)
                )
                hard_neg_loss = hard_neg_loss.view(batch_size, num_queries)
                hard_neg_loss = (hard_neg_loss * hard_neg_mask.float()).sum() / (hard_neg_mask.sum() + 1e-8)
                
                return hard_neg_loss
        
        return torch.tensor(0.0, device=features.device)
    
    def _compute_consistency_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature consistency loss within same class"""
        batch_size, num_queries, hidden_dim = features.shape
        
        # Normalize features
        features_norm = F.normalize(features, dim=-1)
        
        # Compute consistency loss for positive samples
        pos_mask = (labels == 1).float()
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Get positive features
        pos_features = features_norm * pos_mask.unsqueeze(-1)
        
        # Compute mean positive feature per batch
        pos_mean = pos_features.sum(dim=1) / (pos_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # Consistency loss: positive features should be similar to mean
        consistency_loss = 0
        for batch_idx in range(batch_size):
            if pos_mask[batch_idx].sum() > 0:
                pos_indices = torch.where(pos_mask[batch_idx])[0]
                batch_pos_features = features_norm[batch_idx, pos_indices]
                batch_mean = pos_mean[batch_idx].unsqueeze(0)
                
                consistency_loss += F.mse_loss(batch_pos_features, batch_mean.expand_as(batch_pos_features))
        
        return consistency_loss / batch_size


class ContrastiveEnhancedDecoder(nn.Module):
    """
    Enhanced decoder with contrastive learning capabilities
    """
    
    def __init__(
        self,
        base_decoder: nn.Module,
        hidden_dim: int = 256,
        num_heads: int = 8,
        contrastive_layers: List[int] = [3, 4, 5],  # Which decoder layers to apply contrastive learning
        temperature: float = 0.07,
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.3
    ):
        super().__init__()
        
        self.base_decoder = base_decoder
        self.hidden_dim = hidden_dim
        self.contrastive_layers = contrastive_layers
        
        # Add contrastive attention modules
        self.contrastive_attentions = nn.ModuleDict()
        for layer_idx in contrastive_layers:
            self.contrastive_attentions[str(layer_idx)] = PositiveNegativeAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                temperature=temperature,
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold
            )
        
        # Contrastive loss module
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        
        # Feature fusion for contrastive enhanced features
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in contrastive_layers
        ])
    
    def forward(self, *args, **kwargs):
        """Forward pass with contrastive learning integration"""
        # Get base decoder outputs
        base_outputs = self.base_decoder(*args, **kwargs)
        
        # Extract intermediate features if available
        if hasattr(self.base_decoder, 'intermediate_features'):
            intermediate_features = self.base_decoder.intermediate_features
        else:
            # If not available, use final features for all layers
            intermediate_features = {
                str(i): base_outputs.get('pred_logits', base_outputs.get('logits'))
                for i in self.contrastive_layers
            }
        
        # Apply contrastive learning to specified layers
        contrastive_outputs = {}
        enhanced_features = {}
        
        for i, layer_idx in enumerate(self.contrastive_layers):
            layer_key = str(layer_idx)
            if layer_key in intermediate_features:
                features = intermediate_features[layer_key]
                
                # Apply contrastive attention
                contrastive_result = self.contrastive_attentions[layer_key](features)
                
                # Fuse original and enhanced features
                original_features = features
                enhanced_feat = contrastive_result['enhanced_features']
                
                fused_features = torch.cat([original_features, enhanced_feat], dim=-1)
                fused_features = self.feature_fusion[i](fused_features)
                
                enhanced_features[layer_key] = fused_features
                contrastive_outputs[layer_key] = contrastive_result
        
        # Update base outputs with enhanced features
        enhanced_outputs = base_outputs.copy() if isinstance(base_outputs, dict) else {}
        enhanced_outputs.update({
            'contrastive_outputs': contrastive_outputs,
            'enhanced_features': enhanced_features
        })
        
        return enhanced_outputs
    
    def compute_contrastive_loss(
        self,
        contrastive_outputs: Dict,
        targets: List[Dict],
        matched_indices: Optional[List[Tuple]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss for all layers"""
        total_losses = {}
        
        for layer_key, layer_outputs in contrastive_outputs.items():
            features = layer_outputs['enhanced_features']
            pos_neg_scores = layer_outputs['pos_neg_scores']
            
            layer_losses = self.contrastive_loss(
                features, pos_neg_scores, targets, matched_indices
            )
            
            # Add layer prefix to loss names
            for loss_name, loss_value in layer_losses.items():
                prefixed_name = f"{layer_key}_{loss_name}"
                total_losses[prefixed_name] = loss_value
        
        # Compute average losses across layers
        if total_losses:
            avg_losses = {}
            loss_types = set(name.split('_', 1)[1] for name in total_losses.keys())
            
            for loss_type in loss_types:
                layer_losses = [v for k, v in total_losses.items() if k.endswith(loss_type)]
                if layer_losses:
                    avg_losses[f"avg_{loss_type}"] = torch.stack(layer_losses).mean()
            
            total_losses.update(avg_losses)
        
        return total_losses


@register()
class ContrastiveDFINETransformer(nn.Module):
    """
    D-FINE Transformer with integrated contrastive learning
    """
    
    def __init__(
        self,
        base_transformer_config: Dict,
        contrastive_config: Optional[Dict] = None
    ):
        super().__init__()
        
        # Import and create base D-FINE transformer
        from .dfine_decoder import DFINETransformer
        self.base_transformer = DFINETransformer(**base_transformer_config)
        
        # Contrastive learning configuration
        if contrastive_config is None:
            contrastive_config = {
                'hidden_dim': base_transformer_config.get('hidden_dim', 256),
                'num_heads': 8,
                'contrastive_layers': [3, 4, 5],
                'temperature': 0.07,
                'pos_threshold': 0.5,
                'neg_threshold': 0.3
            }
        
        # Wrap decoder with contrastive enhancement
        self.contrastive_decoder = ContrastiveEnhancedDecoder(
            base_decoder=self.base_transformer.decoder,
            **contrastive_config
        )
        
        # Replace base decoder
        self.base_transformer.decoder = self.contrastive_decoder
    
    def forward(self, *args, **kwargs):
        """Forward pass with contrastive learning"""
        return self.base_transformer(*args, **kwargs)
    
    def compute_contrastive_loss(self, outputs, targets, matched_indices=None):
        """Compute contrastive loss"""
        if 'contrastive_outputs' in outputs:
            return self.contrastive_decoder.compute_contrastive_loss(
                outputs['contrastive_outputs'], targets, matched_indices
            )
        return {} 