"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        
        # Choose metric for best model selection  
        best_metric_mode = args.yaml_cfg.get('best_metric', 'ap')
        print(f"Using {best_metric_mode.upper()} metric for best model selection")
        print(f"🔍 DEBUG: best_metric_mode = '{best_metric_mode}'")  # 디버깅 추가

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_f1 = 0  # Track best F1 score separately
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                if k != "validator_metrics":  # Skip validator metrics in this loop
                    best_stat["epoch"] = self.last_epoch
                    best_stat[k] = test_stats[k][0]
                    top1 = test_stats[k][0]
            
            # Initialize F1 score from resume
            if "validator_metrics" in test_stats and "f1" in test_stats["validator_metrics"]:
                best_f1 = test_stats["validator_metrics"]["f1"]
            
            print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_name = "best_f1_stg1.pth" if best_metric_mode == "f1" else "best_stg1.pth"
                self.load_resume_state(str(self.output_dir / checkpoint_name))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

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
                print(f"🔍 DEBUG: Current F1 = {current_f1:.4f}, Best F1 = {best_f1:.4f}")  # 디버깅 추가
            else:
                print("Warning: F1 score not available in validation results")
                print(f"🔍 DEBUG: Available keys in test_stats: {list(test_stats.keys())}")  # 디버깅 추가
                if "validator_metrics" in test_stats:
                    print(f"🔍 DEBUG: Available keys in validator_metrics: {list(test_stats['validator_metrics'].keys())}")  # 디버깅 추가

            # Determine which metric to use for best model selection
            save_best_model = False
            current_metric_value = 0
            metric_name = ""
            
            if best_metric_mode == "f1":
                current_metric_value = current_f1
                metric_name = "F1"
                if current_f1 > best_f1:
                    print(f"🔍 DEBUG: F1 improved! {current_f1:.4f} > {best_f1:.4f}")  # 디버깅 추가
                    best_f1 = current_f1
                    save_best_model = True
                else:
                    print(f"🔍 DEBUG: F1 did not improve: {current_f1:.4f} <= {best_f1:.4f}")  # 디버깅 추가
            else:  # AP mode
                # Use the first COCO AP metric (AP50:95) as before
                for k in test_stats:
                    if k in best_stat:
                        best_stat["epoch"] = (
                            epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                        )
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat[k] > top1:
                        top1 = best_stat[k]
                        save_best_model = True
                        current_metric_value = test_stats[k][0]
                        metric_name = "AP"
                    break  # Only use first metric for comparison

            # Save best model based on selected metric
            if save_best_model and self.output_dir:
                print(f"New best {metric_name} score: {current_metric_value:.4f} at epoch {epoch}")
                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if best_metric_mode == "f1":
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_f1_stg2.pth"
                        )
                    else:
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg2.pth"
                        )
                else:
                    if best_metric_mode == "f1":
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_f1_stg1.pth"
                        )
                    else:
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

            # Update logging for tensorboard
            for k in test_stats:
                if k == "validator_metrics":
                    # Log validator metrics separately
                    if self.writer and dist_utils.is_main_process():
                        for metric_name, metric_value in test_stats[k].items():
                            if isinstance(metric_value, (int, float)):
                                self.writer.add_scalar(f"Test/Validator_{metric_name}", metric_value, epoch)
                else:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                    if k in best_stat:
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat[k] = test_stats[k][0]

            # Print best statistics
            best_stat_print = best_stat.copy()
            if best_metric_mode == "f1":
                best_stat_print["best_f1"] = best_f1
                best_stat_print["current_f1"] = current_f1
            print(f"best_stat: {best_stat_print}")

            # Handle stage transition logic
            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                if not save_best_model:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        checkpoint_name = "best_f1_stg1.pth" if best_metric_mode == "f1" else "best_stg1.pth"
                        self.load_resume_state(str(self.output_dir / checkpoint_name))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                
                # Add F1 score and other validator metrics to wandb
                if "validator_metrics" in test_stats:
                    for metric_name, metric_value in test_stats["validator_metrics"].items():
                        if isinstance(metric_value, (int, float)):
                            wandb_logs[f"validator/{metric_name}"] = metric_value
                
                wandb_logs["epoch"] = epoch
                wandb_logs[f"best_{best_metric_mode}"] = best_f1 if best_metric_mode == "f1" else top1
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
