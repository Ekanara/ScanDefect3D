from __future__ import annotations

import numpy as np
import torch


def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    target_np = target.detach().cpu().numpy().astype(np.int64)
    mask = target_np != ignore_index
    pred_np = pred_np[mask]
    target_np = target_np[mask]
    hist = np.bincount(
        num_classes * target_np + pred_np,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def metrics_from_hist(hist: np.ndarray) -> dict[str, float]:
    acc = np.diag(hist).sum() / max(hist.sum(), 1)
    per_class_acc = np.divide(
        np.diag(hist),
        hist.sum(axis=1),
        out=np.zeros(hist.shape[0], dtype=np.float64),
        where=hist.sum(axis=1) != 0,
    )
    iu = np.divide(
        np.diag(hist),
        hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist),
        out=np.zeros(hist.shape[0], dtype=np.float64),
        where=(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) != 0,
    )
    return {
        "overall_acc": float(acc),
        "mean_class_acc": float(np.nanmean(per_class_acc)),
        "mean_iou": float(np.nanmean(iu)),
    }

