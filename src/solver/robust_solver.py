"""
Robust Solver for D-FINE
Extends the base solver with robust training methods.
"""

import torch
from ._solver import BaseSolver
from .robust_engine import robust_train_one_epoch
from .det_engine import train_one_epoch, evaluate
from ..core import register
from ..misc import dist_utils


@register()
class RobustSolver(BaseSolver):
    """
    Robust Solver that integrates robust training methods with D-FINE.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Robust training configuration - read from cfg.yaml_cfg
        self.enable_robust_training = cfg.yaml_cfg.get('enable_robust_training', True)
        self.enable_robust_loss = cfg.yaml_cfg.get('enable_robust_loss', True)
        self.enable_oadg = cfg.yaml_cfg.get('enable_oadg', True)
        
        self.robust_loss_weight = cfg.yaml_cfg.get('robust_loss_weight', 0.3)
        
        # OA-DG configuration
        self.oadg_weight = cfg.yaml_cfg.get('oadg_weight', 0.1)
        self.oadg_loss_types = cfg.yaml_cfg.get('oadg_loss_types', ['jsd', 'contrastive'])
        self.oadg_start_epoch = cfg.yaml_cfg.get('oadg_start_epoch', 1)
        self.oadg_aug_config = cfg.yaml_cfg.get('oadg_aug_config', None)
        
        # Schedule for switching between normal and robust training
        self.robust_start_epoch = cfg.yaml_cfg.get('robust_start_epoch', 5)
        
        # Evaluation frequency
        self.eval_freq = cfg.yaml_cfg.get('eval_freq', 3)
        
    def fit(self):
        """Entry point for training - delegates to train method."""
        self.train()
        
    def train(self):
        """Override the train method to use robust training."""
        # Call parent's setup
        super().train()
        args = self.cfg
        
        # Initialize best statistics tracking like DetSolver
        best_metric_mode = getattr(args, 'best_metric', 'f1')  # Default to F1 for robust training
        print(f"Using {best_metric_mode.upper()} metric for best model selection")
        
        best_f1 = 0
        best_stat = {"epoch": -1}
        top1 = 0
        
        # Resume from checkpoint if available
        if self.last_epoch > 0:
            print(f"Resuming robust training from epoch {self.last_epoch}")
            # TODO: Add evaluation on resume to get current best scores
        
        print(f"🔒 Robust training enabled: {self.enable_robust_training}")
        print(f"   - Robust loss: {self.enable_robust_loss}")
        print(f"   - OA-DG losses: {self.enable_oadg}")
        print(f"   - Robust weight: {self.robust_loss_weight}")
        print(f"   - OA-DG weight: {self.oadg_weight}")
        print(f"   - OA-DG types: {self.oadg_loss_types}")
        print(f"   - Start epoch: {self.robust_start_epoch}")
        print(f"   - OA-DG start: {self.oadg_start_epoch}")
        
        # Training loop
        epochs = getattr(self.cfg, 'epochs', 100)
        start_epoch = self.last_epoch + 1
        
        for epoch in range(start_epoch, epochs):
            # Set epoch for dataloader (important for distributed training and data sampling)
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            # Use robust training after warmup epochs
            use_robust = self.enable_robust_training and epoch >= self.robust_start_epoch
            
            if use_robust:
                # Robust training
                train_stats = robust_train_one_epoch(
                    model=self.model,
                    criterion=self.criterion,
                    data_loader=self.train_dataloader,
                    optimizer=self.optimizer,
                    device=self.device,
                    epoch=epoch,
                    use_wandb=getattr(self.cfg, 'use_wandb', False),
                    max_norm=getattr(self.cfg, 'max_norm', 0.1),
                    # Robust training parameters
                    enable_robust_loss=self.enable_robust_loss,
                    enable_oadg=self.enable_oadg,
                    robust_loss_weight=self.robust_loss_weight,
                    # OA-DG parameters
                    oadg_weight=self.oadg_weight,
                    oadg_loss_types=self.oadg_loss_types,
                    oadg_start_epoch=self.oadg_start_epoch,
                    oadg_aug_config=self.oadg_aug_config,  # Pass augmentation config
                    # Other parameters
                    print_freq=getattr(self.cfg, 'print_freq', 10),
                    writer=getattr(self, 'writer', None),
                    ema=getattr(self, 'ema', None),
                    scaler=getattr(self, 'scaler', None),
                    lr_warmup_scheduler=getattr(self, 'lr_warmup_scheduler', None),
                    output_dir=getattr(self.cfg, 'output_dir', None),
                    epochs=epochs
                )
                print(f"✅ Robust training epoch {epoch+1} completed")
            else:
                # Standard training for warmup
                train_stats = train_one_epoch(
                    model=self.model,
                    criterion=self.criterion,
                    data_loader=self.train_dataloader,
                    optimizer=self.optimizer,
                    device=self.device,
                    epoch=epoch,
                    use_wandb=getattr(self.cfg, 'use_wandb', False),
                    max_norm=getattr(self.cfg, 'max_norm', 0.1),
                    print_freq=getattr(self.cfg, 'print_freq', 10),
                    writer=getattr(self, 'writer', None),
                    ema=getattr(self, 'ema', None),
                    scaler=getattr(self, 'scaler', None),
                    lr_warmup_scheduler=getattr(self, 'lr_warmup_scheduler', None),
                    output_dir=getattr(self.cfg, 'output_dir', None),
                    epochs=epochs
                )
                print(f"✅ Standard training epoch {epoch+1} completed")
            
            # Learning rate scheduling  
            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            # Update last_epoch for checkpoint saving
            self.last_epoch += 1
            
            # Save checkpoint (following DetSolver pattern)
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # Extra checkpoint based on checkpoint_freq  
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    print(f"💾 Saving checkpoint: {checkpoint_path}")
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)
            
            # Evaluation (following DetSolver pattern)
            if hasattr(self, 'evaluator') and self.evaluator is not None:
                if epoch % self.eval_freq == 0:
                    print(f"📊 Running evaluation at epoch {epoch+1}")
                    
                    module = self.ema.module if self.ema else self.model
                    test_stats, coco_evaluator = evaluate(
                        module,
                        self.criterion,
                        self.postprocessor,
                        self.val_dataloader,
                        self.evaluator,
                        self.device,
                        epoch,
                        self.use_wandb,
                        output_dir=self.output_dir,
                    )
                    
                    # Get F1 score from validator metrics
                    current_f1 = 0
                    if "validator_metrics" in test_stats and "f1" in test_stats["validator_metrics"]:
                        current_f1 = test_stats["validator_metrics"]["f1"]
                        print(f"🔍 Current F1 = {current_f1:.4f}, Best F1 = {best_f1:.4f} [Epoch {epoch+1}]")
                    else:
                        print("Warning: F1 score not available in validation results")
                    
                    # Save best model based on F1 score
                    if best_metric_mode == "f1" and current_f1 > best_f1:
                        print(f"🎉 F1 improved from {best_f1:.4f} to {current_f1:.4f} at epoch {epoch+1}!")
                        best_f1 = current_f1
                        
                        if self.output_dir:
                            print(f"💾 Saving new best F1 model (epoch {epoch+1})")
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_f1_robust.pth"
                            )
                    else:
                        print(f"📊 F1 did not improve: {current_f1:.4f} <= {best_f1:.4f} [Epoch {epoch+1}]")
        
        print("🎉 Robust training completed!") 