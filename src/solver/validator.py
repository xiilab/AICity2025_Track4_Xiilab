import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchvision.ops import box_iou

def compute_average_precision(score_tps, num_gt):
    if len(score_tps) == 0 or num_gt == 0:
        return 0.0

    score_tps.sort(reverse=True, key=lambda x: x[0])
    tps = np.array([int(tp) for _, tp in score_tps])
    fps = 1 - tps

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)

    recalls = tp_cumsum / (num_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap


def filter_preds(preds, conf_thresh):
    for pred in preds:
        keep_idxs = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep_idxs]
        pred["boxes"] = pred["boxes"][keep_idxs]
        pred["labels"] = pred["labels"][keep_idxs]
    return preds


def scale_boxes(boxes, orig_shape, resized_shape):
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes

class Validator:
    def __init__(
        self,
        gt: List[Dict[str, torch.Tensor]],
        preds: List[Dict[str, torch.Tensor]],
        conf_thresh=0.5,
        iou_thresh=0.5,
    ) -> None:
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.conf_matrix = None

    def compute_metrics(self, extended=False) -> Dict[str, float]:
        filtered_preds = filter_preds(copy.deepcopy(self.preds), self.conf_thresh)
        metrics = self._compute_main_metrics(filtered_preds)
        if not extended:
            metrics.pop("extended_metrics", None)
        return metrics

    def _compute_main_metrics(self, preds):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds)
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}
        cls_aps = {}

        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            num_gt = value["TPs"] + value["FNs"]
            score_tps = value.get("score_tps", [])
            cls_aps[key] = compute_average_precision(score_tps, num_gt)

            extended_metrics[f"precision_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )
            extended_metrics[f"iou_{key}"] = np.mean(value["IoUs"])
            extended_metrics[f"ap_{key}"] = cls_aps[key]

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = np.mean(ious).item() if ious else 0
        mAP = np.mean(list(cls_aps.values())) if cls_aps else 0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "mAP": mAP,
            "APs": cls_aps,
            "extended_metrics": extended_metrics,
        }

    def _compute_metrics_and_confusion_matrix(self, preds):
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": [], "score_tps": []})
        all_classes = set()
        for pred in preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())
        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        for pred, gt in zip(preds, self.gt):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_preds = len(pred_boxes)
            n_gts = len(gt_boxes)

            ious = box_iou(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])
            matched_pred_indices = set()
            matched_gt_indices = set()

            if ious.numel() > 0:
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if pred_idx.item() in matched_pred_indices or gt_idx.item() in matched_gt_indices:
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()
                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                        metrics_per_class[gt_label]["score_tps"].append((pred_scores[pred_idx].item(), True))
                    else:
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["score_tps"].append((pred_scores[pred_idx].item(), False))

            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                conf_matrix[n_classes, pred_cls_idx] += 1
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)
                metrics_per_class[pred_label]["score_tps"].append((pred_scores[pred_idx].item(), False))

            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                conf_matrix[gt_cls_idx, n_classes] += 1
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return metrics_per_class, conf_matrix, class_to_idx