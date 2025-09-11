"""
Robust Training Engine for D-FINE
Integrates robust training methods with the existing detection engine.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from ..nn.criterion.robust_loss import RobustLossWrapper
from .validator import Validator, scale_boxes


def robust_train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    # Robust training parameters
    enable_robust_loss: bool = True,
    enable_oadg: bool = True,
    robust_loss_weight: float = 0.3,
    oadg_weight: float = 0.1,
    oadg_loss_types: List[str] = ['jsd', 'contrastive'],
    oadg_start_epoch: int = 1,
    **kwargs,
):
    """
    Training function with robust training capabilities.
    
    Args:
        enable_robust_loss: Enable calibration and bbox consistency losses
        enable_oadg: Enable OA-DG (JSD/Contrastive) losses  
        robust_loss_weight: Weight for robust loss components
        oadg_weight: Weight for OA-DG losses
        oadg_loss_types: Types of OA-DG losses to use
        oadg_start_epoch: Epoch to start OA-DG training
    """
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Robust Epoch: [{}]".format(epoch) if epochs is None else "Robust Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    # Get OA-DG augmentation config from kwargs
    oadg_aug_config = kwargs.get('oadg_aug_config', None)
    
    # Initialize robust training components
    robust_loss_wrapper = RobustLossWrapper(
        base_criterion=criterion,
        use_oadg_losses=enable_oadg and epoch >= oadg_start_epoch,
        oadg_weight=oadg_weight,
        oadg_loss_types=oadg_loss_types if oadg_loss_types else ['jsd', 'contrastive'],
        oadg_start_epoch=oadg_start_epoch,
        oadg_aug_config=oadg_aug_config,  # Pass config-based augmentation settings
        use_calibration_loss=enable_robust_loss,
        calibration_weight=robust_loss_weight if enable_robust_loss else 0.0,
        use_bbox_consistency_loss=enable_robust_loss,
        bbox_consistency_weight=robust_loss_weight if enable_robust_loss else 0.0
    )
    
    # Adversarial training is disabled - not implemented

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "robust_train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                # Standard forward pass (adversarial training disabled)
                outputs = model(samples, targets=targets)

            # NaN/Inf detection
            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print("❌ NaN/Inf detected in pred_boxes!")
                print(f"   pred_boxes shape: {outputs['pred_boxes'].shape}")
                print(f"   NaN count: {torch.isnan(outputs['pred_boxes']).sum()}")
                print(f"   Inf count: {torch.isinf(outputs['pred_boxes']).sum()}")
                
                state = model.state_dict()
                new_state = {}
                for key, value in state.items():
                    new_key = key.replace("module.", "")
                    new_state[new_key] = value
                new_state = {"model": new_state}
                dist_utils.save_on_master(new_state, "./robust_NaN.pth")
                
                raise RuntimeError("NaN detected in robust model outputs - training stopped")

            if "pred_logits" in outputs:
                if torch.isnan(outputs["pred_logits"]).any() or torch.isinf(outputs["pred_logits"]).any():
                    print("❌ NaN/Inf detected in pred_logits!")
                    raise RuntimeError("NaN detected in robust model outputs - training stopped")

            with torch.autocast(device_type=str(device), enabled=False):
                # Convert outputs to float32 to avoid dtype mismatch
                outputs_float32 = {}
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                        outputs_float32[k] = v.float()
                    else:
                        outputs_float32[k] = v
                
                # Convert targets to float32 as well if needed
                targets_float32 = []
                for target in targets:
                    target_float32 = {}
                    for k, v in target.items():
                        if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                            target_float32[k] = v.float()
                        else:
                            target_float32[k] = v
                    targets_float32.append(target_float32)
                
                # Use robust loss wrapper
                loss_dict = robust_loss_wrapper(outputs_float32, targets_float32, **metas)

            for loss_name, loss_value in loss_dict.items():
                if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
                    print(f"❌ NaN/Inf detected in robust loss '{loss_name}': {loss_value}")
                    raise RuntimeError(f"NaN detected in robust loss computation - training stopped")

            loss = sum(loss_dict.values())
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"❌ NaN/Inf detected in total robust loss: {loss}")
                raise RuntimeError("NaN detected in total robust loss - training stopped")
            
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            # Without mixed precision - standard forward pass
            outputs = model(samples, targets=targets)
                
            loss_dict = robust_loss_wrapper(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # EMA
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Robust loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log robust training specific metrics
        if use_wandb and dist_utils.is_main_process():
            log_dict = {
                "train/robust_loss": loss_value,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/global_step": global_step,
            }
            
            # Add individual loss components
            for loss_name, loss_val in loss_dict_reduced.items():
                log_dict[f"train/{loss_name}"] = loss_val
                
            wandb.log(log_dict, step=global_step)

        if writer is not None and dist_utils.is_main_process():
            writer.add_scalar("train/robust_loss", loss_value, global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            
            for loss_name, loss_val in loss_dict_reduced.items():
                writer.add_scalar(f"train/{loss_name}", loss_val, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Robust training stats:", {k: meter.global_avg for k, meter in metric_logger.meters.items()})
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 