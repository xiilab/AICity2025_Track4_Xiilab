"""
OA-DG Loss Functions for D-FINE
Adapted from OA-DG repository for domain generalization robust training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...core import register


def weighted_loss2(loss_func):
    """Create a weighted version of a given loss function.
    
    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the loss function. The decorated function will
    have the signature like `loss_func(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs)`.
    """

    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


# =============================================================================
# Smooth L1 Loss Functions
# =============================================================================

@weighted_loss2
def smooth_l1_loss(pred, target, num_views=3, beta=1.0):
    """Smooth L1 loss for multiple views.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        num_views (int): Number of views/augmentations.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig = torch.chunk(pred, num_views)[0]
    target = torch.chunk(target, num_views)[0]

    assert beta > 0
    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    diff = torch.abs(pred_orig - target)
    loss_orig = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return loss_orig


@weighted_loss2
def l1_loss(pred, target, num_views=3):
    """L1 loss for multiple views.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        num_views (int): Number of views/augmentations.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig = torch.chunk(pred, num_views)[0]
    target = torch.chunk(target, num_views)[0]

    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    loss_orig = torch.abs(pred_orig - target)

    return loss_orig


def smooth_l1_dg_loss(pred, target, beta=1.0, weight=None, reduction='mean', avg_factor=None):
    """Smooth L1 Domain Generalization loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    assert beta > 0
    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == pred_aug1.size()
    diff = (torch.abs(pred_orig - pred_aug1) + torch.abs(pred_orig - pred_aug2)) / 2
    additional_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    additional_loss = additional_loss.mean()

    return additional_loss


def l1_dg_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    """L1 Domain Generalization loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == pred_aug1.size()
    additional_loss = (torch.abs(pred_orig - pred_aug1) + torch.abs(pred_aug1 - pred_aug2) + torch.abs(pred_aug2 - pred_orig)) / 3
    additional_loss = additional_loss.mean()

    return additional_loss


# =============================================================================
# Cross Entropy Loss Functions  
# =============================================================================

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  num_views=3,
                  avg='1.0'):
    """Calculate the CrossEntropy loss for multiple views.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        num_views (int): Number of views/augmentations.
        avg (str): Averaging method.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    pred_orig = torch.chunk(pred, num_views)[0]
    label = torch.chunk(label, num_views)[0]
    avg_factor = avg_factor / num_views if avg=='1.1' else avg_factor

    loss = F.cross_entropy(
        pred_orig,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = torch.chunk(weight, num_views)[0]
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         num_views=3,
                         avg='1.0'):
    """Calculate the binary CrossEntropy loss for multiple views.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        num_views (int): Number of views/augmentations.
        avg (str): Averaging method.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    
    # weighted element-wise losses
    if weight is not None:
        weight = torch.chunk(weight, num_views)[0]
        weight = weight.float()

    pred_orig = torch.chunk(pred, num_views)[0]
    label = torch.chunk(label, num_views)[0]
    avg_factor = avg_factor / num_views if avg=='1.1' else avg_factor

    # Clamp inputs for numerical stability
    pred_orig_safe = torch.clamp(pred_orig, min=-50.0, max=50.0)
    pred_orig_safe = torch.nan_to_num(pred_orig_safe, nan=0.0, posinf=50.0, neginf=-50.0)
    label_safe = torch.clamp(label.float(), min=0.0, max=1.0)
    label_safe = torch.nan_to_num(label_safe, nan=0.0, posinf=1.0, neginf=0.0)

    loss = F.binary_cross_entropy_with_logits(
        pred_orig_safe, label_safe, pos_weight=class_weight, reduction='none')

    # Replace NaNs/Infs
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


# =============================================================================
# Jensen-Shannon Divergence Loss Functions
# =============================================================================

def jsd_loss(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None):
    """Calculate the JSD loss for domain generalization.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    try:
        pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

        # Clamp inputs for numerical stability
        pred_orig = torch.clamp(pred_orig, min=-10.0, max=10.0)
        pred_aug1 = torch.clamp(pred_aug1, min=-10.0, max=10.0)
        pred_aug2 = torch.clamp(pred_aug2, min=-10.0, max=10.0)
        
        # Replace NaNs/Infs
        pred_orig = torch.nan_to_num(pred_orig, nan=0.0, posinf=10.0, neginf=-10.0)
        pred_aug1 = torch.nan_to_num(pred_aug1, nan=0.0, posinf=10.0, neginf=-10.0)
        pred_aug2 = torch.nan_to_num(pred_aug2, nan=0.0, posinf=10.0, neginf=-10.0)

        # Safe softmax
        p_clean = F.softmax(pred_orig, dim=1)
        p_aug1 = F.softmax(pred_aug1, dim=1)  
        p_aug2 = F.softmax(pred_aug2, dim=1)
        
        # Stabilize probabilities (avoid extremely small values)
        eps = 1e-8
        p_clean = torch.clamp(p_clean, min=eps, max=1-eps)
        p_aug1 = torch.clamp(p_aug1, min=eps, max=1-eps)
        p_aug2 = torch.clamp(p_aug2, min=eps, max=1-eps)
        
        # Replace NaNs/Infs
        p_clean = torch.nan_to_num(p_clean, nan=eps, posinf=1-eps, neginf=eps)
        p_aug1 = torch.nan_to_num(p_aug1, nan=eps, posinf=1-eps, neginf=eps)
        p_aug2 = torch.nan_to_num(p_aug2, nan=eps, posinf=1-eps, neginf=eps)

        # Mixture distribution (without log)
        p_mixture = (p_clean + p_aug1 + p_aug2) / 3.0
        p_mixture = torch.clamp(p_mixture, min=eps, max=1-eps)
        p_mixture = torch.nan_to_num(p_mixture, nan=eps, posinf=1-eps, neginf=eps)
        
        # JSD: JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5*(P+Q)
        # Use manual KL for gradient stability
        
        # Safe log computation
        log_p_mixture = torch.log(p_mixture + eps)
        log_p_clean = torch.log(p_clean + eps) 
        log_p_aug1 = torch.log(p_aug1 + eps)
        log_p_aug2 = torch.log(p_aug2 + eps)
        
        # KL(P||M) = sum(P * log(P/M)) = sum(P * (log(P) - log(M)))
        kl1 = torch.sum(p_clean * (log_p_clean - log_p_mixture), dim=1).mean()
        kl2 = torch.sum(p_aug1 * (log_p_aug1 - log_p_mixture), dim=1).mean()
        kl3 = torch.sum(p_aug2 * (log_p_aug2 - log_p_mixture), dim=1).mean()
        
        # Jensen-Shannon Divergence
        loss = (kl1 + kl2 + kl3) / 3.0
        
        # Final NaN/Inf check
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.tensor(1e-6, device=pred.device, requires_grad=True)

        # apply weights and do the reduction
        if weight is not None:
            weight, _, _ = torch.chunk(weight, 3)
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss
        
    except Exception as e:
        # Fallback: return safe dummy loss
        print(f"JSD Loss Error: {e}")
        return torch.tensor(1e-6, device=pred.device, requires_grad=True)


def jsdv1_3_loss(pred,
                 label,
                 weight=None,
                 reduction='mean',
                 avg_factor=None,
                 use_cls_weight=False,
                 **kwargs):
    """Calculate the jsdv1.3 loss.
    JSD loss (sigmoid, 1-sigmoid) for rpn head, softmax for roi head
    divided by batchmean, divided by 768 (256*3) for rpn, 1056 (352*3) for roi
    reduction parameter does not affect the loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        use_cls_weight (bool): Whether to use class weights.

    Returns:
        torch.Tensor: The calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    # Clamp inputs for numerical stability
    pred_orig = torch.clamp(pred_orig, min=-10.0, max=10.0)
    pred_aug1 = torch.clamp(pred_aug1, min=-10.0, max=10.0)
    pred_aug2 = torch.clamp(pred_aug2, min=-10.0, max=10.0)
    
    # Replace NaNs/Infs
    pred_orig = torch.nan_to_num(pred_orig, nan=0.0, posinf=10.0, neginf=-10.0)
    pred_aug1 = torch.nan_to_num(pred_aug1, nan=0.0, posinf=10.0, neginf=-10.0)
    pred_aug2 = torch.nan_to_num(pred_aug2, nan=0.0, posinf=10.0, neginf=-10.0)

    eps = 1e-8
    if pred_orig.shape[-1] == 1:  # if rpn
        sig_orig = torch.clamp(torch.sigmoid(pred_orig), min=eps, max=1-eps)
        sig_aug1 = torch.clamp(torch.sigmoid(pred_aug1), min=eps, max=1-eps)
        sig_aug2 = torch.clamp(torch.sigmoid(pred_aug2), min=eps, max=1-eps)
        
        p_clean = torch.cat((sig_orig, 1 - sig_orig), dim=1)
        p_aug1 = torch.cat((sig_aug1, 1 - sig_aug1), dim=1)
        p_aug2 = torch.cat((sig_aug2, 1 - sig_aug2), dim=1)
    else:  # else roi
        p_clean = F.softmax(pred_orig, dim=1)
        p_aug1 = F.softmax(pred_aug1, dim=1)
        p_aug2 = F.softmax(pred_aug2, dim=1)
    
    # Stabilize probabilities
    p_clean = torch.clamp(p_clean, min=eps, max=1-eps)
    p_aug1 = torch.clamp(p_aug1, min=eps, max=1-eps)
    p_aug2 = torch.clamp(p_aug2, min=eps, max=1-eps)
    
    # Replace NaNs/Infs
    p_clean = torch.nan_to_num(p_clean, nan=eps, posinf=1-eps, neginf=eps)
    p_aug1 = torch.nan_to_num(p_aug1, nan=eps, posinf=1-eps, neginf=eps)
    p_aug2 = torch.nan_to_num(p_aug2, nan=eps, posinf=1-eps, neginf=eps)

    # Mixture distribution (safe)
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3.0
    p_mixture = torch.clamp(p_mixture, min=eps, max=1-eps)
    p_mixture = torch.nan_to_num(p_mixture, nan=eps, posinf=1-eps, neginf=eps)
    
    # Safe log computation with manual KL divergence
    log_p_mixture = torch.log(p_mixture + eps)
    log_p_clean = torch.log(p_clean + eps)
    log_p_aug1 = torch.log(p_aug1 + eps)
    log_p_aug2 = torch.log(p_aug2 + eps)

    # Manual KL divergence: KL(P||M) = sum(P * (log(P) - log(M)))
    kl1 = p_clean * (log_p_clean - log_p_mixture)
    kl2 = p_aug1 * (log_p_aug1 - log_p_mixture)
    kl3 = p_aug2 * (log_p_aug2 - log_p_mixture)
    
    loss = (kl1 + kl2 + kl3) / 3.
            
    # Replace NaNs/Infs then reduce
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
    loss = torch.sum(loss, dim=-1)

    # [DEV] imbalance: alpha-balanced loss
    label, _, _ = torch.chunk(label, 3)
    if use_cls_weight:
        class_weight = kwargs['class_weight'] \
            if kwargs.get('add_class_weight') is None else kwargs['add_class_weight']

        for i in range(len(class_weight)):
            mask = (label == i)
            loss[mask] = loss[mask] * class_weight[i]

    loss = torch.mean(loss, dim=0)

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


# =============================================================================
# Contrastive Loss Functions
# =============================================================================

def make_matrix(input1, input2, criterion, reduction='none'):
    """Create matrix for representation analysis."""
    if reduction == 'none':
        matrix = criterion(input1.unsqueeze(1), input2.unsqueeze(0))
    elif reduction == 'sum':
        matrix = criterion(input1.unsqueeze(1), input2.unsqueeze(0)).sum(dim=-1)
    elif reduction == 'mean':
        matrix = criterion(input1.unsqueeze(1), input2.unsqueeze(0)).mean(dim=-1)
    return matrix


def supcontrast_mask(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):
    """Supervised contrastive loss with masks."""
    if logits_anchor.size(0) == 0:
        print("no foreground")
        loss = 0
    else:
        base_temper = temper

        logits_anchor, logits_contrast = F.normalize(logits_anchor, dim=1), F.normalize(logits_contrast, dim=1)

        anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * mask_contrast
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / (mask_anchor.sum(1) + 1e-8)
        loss = - (temper / base_temper) * mean_log_prob_pos
        loss = loss.mean()

    return loss


def supcontrast(logits_clean, labels=None, num_views=2, lambda_weight=0.1, temper=0.07, min_samples=10):
    """Supervised contrastive loss.
    
    Pull same fg class, push diff class fg, push fg vs bg, push diff instance bg
    Create mask anchor and masks contrast
    Input: only clean logit
    
    Args:
        logits_clean: Input features
        labels: Target labels
        num_views: Number of views/augmentations
        lambda_weight: Weight for the loss
        temper: Temperature parameter
        min_samples: Minimum samples required
        
    Returns:
        torch.Tensor: Calculated loss
    """
    device = logits_clean.device
    batch_size = logits_clean.size()[0]

    assert num_views == 2, "Only num_views 2 and batch_size 2 case are supported."
    ori_size = 512 * num_views   # 1024
    rp_total_size = batch_size % ori_size  # num_proposals
    rp_size = rp_total_size // num_views     # num proposals 10 case, num_view 2, batch_size 2 -> rp_size 20

    targets = labels
    targets = targets.contiguous().view(-1, 1)

    inds, _ = (targets != targets.max()).nonzero(as_tuple=True)

    fg = (targets != targets.max()).float()
    bg = (targets == targets.max()).float()
    mask_bg = torch.matmul(bg, bg.T)
    mask_same_instance = torch.zeros([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_eye_ori = torch.eye(ori_size, dtype=torch.float32).to(device)
    mask_eye_rp = torch.eye(rp_size, dtype=torch.float32).to(device)
    mask_same_instance[:ori_size, ori_size:ori_size*2] = mask_eye_ori
    mask_same_instance[ori_size:ori_size*2, :ori_size] = mask_eye_ori
    mask_same_instance[ori_size*2+rp_size:ori_size*2+rp_size*2, ori_size*2:ori_size*2+rp_size] = mask_eye_rp
    mask_same_instance[ori_size*2:ori_size*2+rp_size, ori_size*2+rp_size:ori_size*2+rp_size*2] = mask_eye_rp
    mask_anchor_bg = mask_same_instance * mask_bg

    if inds.size(0) > min_samples:
        mask_fg = torch.matmul(fg, fg.T)
        mask_eye = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
        mask_anchor_except_eye = mask_anchor - mask_eye

        mask_anchor_fg = mask_anchor_except_eye * mask_fg
        mask_anchor = mask_anchor_fg + mask_anchor_bg
        mask_anchor = mask_anchor.detach()

        assert ((mask_anchor!=0)&(mask_anchor!=1)).float().sum().item() == 0, "mask_anchor error"
        mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
        mask_contrast_except_eye = mask_contrast - mask_eye
        mask_contrast_except_eye = mask_contrast_except_eye.detach()

        loss = supcontrast_mask(logits_clean, logits_clean, targets,
                                mask_anchor, mask_contrast_except_eye, lambda_weight, temper)
    else:
        loss = torch.tensor(0., device=device, dtype=torch.float32)

    return loss


# =============================================================================
# Loss Classes
# =============================================================================

# NOTE: D-FINE already has L1, Smooth L1, and Cross Entropy losses
# We only implement the unique OA-DG domain generalization losses


@register()
class ContrastiveLossPlus(nn.Module):
    """Contrastive Loss with supervised learning support."""

    def __init__(self,
                 loss_weight=1,
                 temperature=0.07,
                 num_views=2,
                 normalized_input=True,
                 min_samples=10,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.num_views = num_views
        self.normalized_input = normalized_input
        self.min_samples = min_samples
        self.kwargs = kwargs

    def forward(self, cont_feats, labels):
        """Forward function.

        Args:
            cont_feats: Contrastive features
            labels: Target labels
            
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(cont_feats) == 0:
            return torch.zeros(1, device=cont_feats.device if hasattr(cont_feats, 'device') else 'cpu')
            
        if self.normalized_input:
            cont_feats = F.normalize(cont_feats, dim=1)

        # cont_feats: [2048+random_feats, 256], labels_: [2048+random_feats, ]
        if len(cont_feats) != len(labels): # random proposal case
            random_proposal_len = len(cont_feats) - len(labels)
            random_proposal_targets = labels[-1, :].repeat(random_proposal_len, 1)
            labels = torch.cat([labels, random_proposal_targets], dim=0)

        loss = supcontrast(cont_feats, labels, num_views=self.num_views, 
                          temper=self.temperature, min_samples=self.min_samples)
        return self.loss_weight * loss


@register()
class JSDLoss(nn.Module):
    """Jensen-Shannon Divergence Loss for domain generalization."""
    
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else 'mean')
        loss = self.loss_weight * jsd_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss 