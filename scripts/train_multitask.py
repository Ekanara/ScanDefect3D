from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from scandefect3d.data.multitask_dataset import OpenTrenchMultiTaskDataset
from scandefect3d.models.multitask_factory import build_multitask_model
from scandefect3d.utils.io import ensure_dir
from scandefect3d.utils.metrics import fast_hist
from scandefect3d.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpenTrench3D multi-task model: semantic + defect.")
    parser.add_argument("--data-root", type=str, default="data/opentrench3d_defect_multitask")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet2_transformer",
        choices=[
            "pointnet",
            "pointnet2",
            "pointnet2_transformer",
            "pointnet++",
            "pointnet2-attn",
            "pointtransformer",
            "transformer",
        ],
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--semantic-classes", type=int, default=5)
    parser.add_argument("--semantic-ignore-index", type=int, default=4)
    parser.add_argument("--defect-supervise-class", type=int, default=0, help="Defect loss/metrics computed on this semantic class.")
    parser.add_argument(
        "--defect-task-level",
        type=str,
        default="scene",
        choices=["point", "scene"],
        help="point: per-point defect segmentation on supervise class, scene: defect-vs-normal classification per scene.",
    )
    parser.add_argument(
        "--defect-scene-threshold",
        type=float,
        default=0.001,
        help="Only used when --defect-task-level=scene. Predict defect if P(defect) >= threshold.",
    )
    parser.add_argument(
        "--defect-scene-pooling",
        type=str,
        default="topk",
        choices=["mean", "max", "topk", "lse"],
        help="Pooling method from per-point defect logits to scene defect logits.",
    )
    parser.add_argument("--defect-scene-topk-ratio", type=float, default=0.1)
    parser.add_argument("--defect-scene-lse-temp", type=float, default=8.0)
    parser.add_argument("--scene-defect-min-ratio-train", type=float, default=0.0)
    parser.add_argument("--scene-defect-min-ratio-val", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-use-class-weight", action="store_true")
    parser.add_argument("--semantic-weight-mode", type=str, default="inverse_sqrt", choices=["inverse", "inverse_sqrt"])
    parser.add_argument("--train-semantic-aware-sampling", action="store_true")
    parser.add_argument("--semantic-sampling-power", type=float, default=1.0)
    parser.add_argument("--defect-loss-weight", type=float, default=1.2)
    parser.add_argument("--defect-pos-weight", type=float, default=3.0)
    parser.add_argument("--defect-use-focal", action="store_true")
    parser.add_argument("--defect-focal-gamma", type=float, default=2.0)
    parser.add_argument("--defect-label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-defect-sampling-ratio", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--hard-stop-epoch", type=int, default=5, help="Hard stop at this epoch for quick iterations. Set 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--exp-name", type=str, default="opentrench_multitask_pointnet2")
    return parser.parse_args()


def semantic_miou_from_hist(hist: np.ndarray, semantic_ignore_index: int) -> float:
    denom = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = np.divide(np.diag(hist), denom, out=np.zeros(hist.shape[0], dtype=np.float64), where=denom != 0)
    valid_classes = [i for i in range(hist.shape[0]) if i != semantic_ignore_index]
    if not valid_classes:
        return 0.0
    return float(np.mean(iou[valid_classes]))


def compute_semantic_class_weight(
    files: list[Path],
    semantic_classes: int,
    semantic_ignore_index: int,
    mode: str,
) -> torch.Tensor | None:
    counts = np.zeros(semantic_classes, dtype=np.float64)
    for npz_file in files:
        with np.load(npz_file) as f:
            labels = f["semantic_labels"].astype(np.int64)
        for cls_id in range(semantic_classes):
            if cls_id == semantic_ignore_index:
                continue
            counts[cls_id] += float((labels == cls_id).sum())

    valid_ids = [i for i in range(semantic_classes) if i != semantic_ignore_index and counts[i] > 0]
    if not valid_ids:
        return None

    weights = np.zeros(semantic_classes, dtype=np.float32)
    if mode == "inverse":
        for cls_id in valid_ids:
            weights[cls_id] = 1.0 / float(counts[cls_id])
    else:
        for cls_id in valid_ids:
            weights[cls_id] = 1.0 / float(np.sqrt(counts[cls_id]))

    mean_w = float(np.mean([weights[i] for i in valid_ids]))
    if mean_w > 0:
        weights = weights / mean_w
    weights[semantic_ignore_index] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    semantic_classes: int,
    semantic_ignore_index: int,
    semantic_class_weight: torch.Tensor | None,
    defect_supervise_class: int,
    defect_task_level: str,
    defect_scene_threshold: float,
    defect_scene_pooling: str,
    defect_scene_topk_ratio: float,
    defect_scene_lse_temp: float,
    semantic_loss_weight: float,
    defect_loss_weight: float,
    defect_pos_weight: float,
    defect_use_focal: bool,
    defect_focal_gamma: float,
    defect_label_smoothing: float,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    semantic_hist = np.zeros((semantic_classes, semantic_classes), dtype=np.int64)
    defect_tp = defect_fp = defect_fn = defect_tn = 0

    defect_weight = torch.tensor([1.0, defect_pos_weight], dtype=torch.float32, device=device)

    def scene_pool_logits(logits: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
        # logits: [2, N], class_mask: [N]
        if class_mask.any():
            cls_logits = logits[:, class_mask]
        else:
            cls_logits = logits

        if defect_scene_pooling == "max":
            return cls_logits.max(dim=1).values
        if defect_scene_pooling == "topk":
            npts = cls_logits.size(1)
            k = max(1, int(npts * float(np.clip(defect_scene_topk_ratio, 1e-4, 1.0))))
            topk_vals = torch.topk(cls_logits, k=k, dim=1).values
            return topk_vals.mean(dim=1)
        if defect_scene_pooling == "lse":
            temp = max(float(defect_scene_lse_temp), 1e-3)
            return torch.logsumexp(cls_logits * temp, dim=1) / temp
        return cls_logits.mean(dim=1)

    def defect_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not defect_use_focal:
            return F.cross_entropy(logits, targets, weight=defect_weight, label_smoothing=defect_label_smoothing)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=defect_weight,
            reduction="none",
            label_smoothing=defect_label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** defect_focal_gamma) * ce
        return focal.mean()

    for batch in tqdm(loader, leave=False):
        points = batch["points"].to(device, non_blocking=True)
        semantic_labels = batch["semantic_labels"].to(device, non_blocking=True)
        defect_labels = batch["defect_labels"].to(device, non_blocking=True)
        scene_defect_labels = batch["scene_defect_label"].to(device, non_blocking=True)

        scene_logits: torch.Tensor | None = None
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                semantic_logits, defect_logits = model(points)
                semantic_loss = F.cross_entropy(
                    semantic_logits,
                    semantic_labels,
                    ignore_index=semantic_ignore_index,
                    weight=semantic_class_weight,
                )

                if defect_task_level == "point":
                    defect_logits_flat = defect_logits.transpose(1, 2).reshape(-1, 2)
                    defect_labels_flat = defect_labels.reshape(-1)
                    semantic_labels_flat = semantic_labels.reshape(-1)
                    defect_mask = semantic_labels_flat == defect_supervise_class
                    if defect_mask.any():
                        defect_loss = defect_ce(defect_logits_flat[defect_mask], defect_labels_flat[defect_mask])
                    else:
                        defect_loss = torch.zeros((), dtype=semantic_loss.dtype, device=device)
                else:
                    pooled_logits = []
                    for bidx in range(points.size(0)):
                        class_mask = semantic_labels[bidx] == defect_supervise_class
                        pooled_logits.append(scene_pool_logits(defect_logits[bidx], class_mask))
                    scene_logits = torch.stack(pooled_logits, dim=0)
                    defect_loss = defect_ce(scene_logits, scene_defect_labels)
                loss = semantic_loss_weight * semantic_loss + defect_loss_weight * defect_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * points.size(0)

        semantic_preds = torch.argmax(semantic_logits, dim=1)
        semantic_hist += fast_hist(semantic_preds, semantic_labels, num_classes=semantic_classes, ignore_index=semantic_ignore_index)

        if defect_task_level == "point":
            defect_preds = torch.argmax(defect_logits, dim=1).reshape(-1)
            gt_np = defect_labels.reshape(-1)
            semantic_np = semantic_labels.reshape(-1)
            metric_mask = semantic_np == defect_supervise_class
            pred_np = defect_preds[metric_mask].detach().cpu().numpy().astype(np.int64)
            gt_np = gt_np[metric_mask].detach().cpu().numpy().astype(np.int64)
        else:
            assert scene_logits is not None
            scene_prob = torch.softmax(scene_logits, dim=1)[:, 1]
            pred_np = (scene_prob >= defect_scene_threshold).detach().cpu().numpy().astype(np.int64)
            gt_np = scene_defect_labels.detach().cpu().numpy().astype(np.int64)
        defect_tp += int(np.logical_and(pred_np == 1, gt_np == 1).sum())
        defect_fp += int(np.logical_and(pred_np == 1, gt_np == 0).sum())
        defect_fn += int(np.logical_and(pred_np == 0, gt_np == 1).sum())
        defect_tn += int(np.logical_and(pred_np == 0, gt_np == 0).sum())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    semantic_acc = float(np.diag(semantic_hist).sum() / max(semantic_hist.sum(), 1))
    semantic_miou = semantic_miou_from_hist(semantic_hist, semantic_ignore_index)

    precision = defect_tp / max(defect_tp + defect_fp, 1)
    recall = defect_tp / max(defect_tp + defect_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = defect_tp / max(defect_tp + defect_fp + defect_fn, 1)
    defect_acc = (defect_tp + defect_tn) / max(defect_tp + defect_tn + defect_fp + defect_fn, 1)

    metrics = {
        "semantic_acc": semantic_acc,
        "semantic_miou": float(semantic_miou),
        "defect_precision": float(precision),
        "defect_recall": float(recall),
        "defect_f1": float(f1),
        "defect_iou": float(iou),
        "defect_acc": float(defect_acc),
    }
    return epoch_loss, metrics


def compute_scene_positive_ratio(files: list[Path], min_defect_ratio: float) -> float:
    if not files:
        return 0.0
    pos = 0
    for npz_file in files:
        with np.load(npz_file) as f:
            defect = f["defect_labels"].astype(np.int64) > 0
        ratio = float(defect.mean()) if defect.size > 0 else 0.0
        pos += int(defect.any() and ratio >= min_defect_ratio)
    return float(pos / len(files))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ensure_dir(Path(args.save_dir) / args.exp_name)

    train_ds = OpenTrenchMultiTaskDataset(
        root=args.data_root,
        split=args.train_split,
        num_points=args.num_points,
        defect_aware_sampling=True,
        defect_sampling_ratio=args.train_defect_sampling_ratio,
        semantic_aware_sampling=args.train_semantic_aware_sampling,
        semantic_sampling_power=args.semantic_sampling_power,
        scene_defect_min_ratio=args.scene_defect_min_ratio_train,
    )
    val_ds = OpenTrenchMultiTaskDataset(
        root=args.data_root,
        split=args.val_split,
        num_points=args.num_points,
        defect_aware_sampling=False,
        semantic_aware_sampling=False,
        scene_defect_min_ratio=args.scene_defect_min_ratio_val,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_multitask_model(model_name=args.model, num_semantic_classes=args.semantic_classes, input_dim=args.input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

    semantic_class_weight = None
    if args.semantic_use_class_weight:
        semantic_class_weight = compute_semantic_class_weight(
            files=train_ds.files,
            semantic_classes=args.semantic_classes,
            semantic_ignore_index=args.semantic_ignore_index,
            mode=args.semantic_weight_mode,
        )
        if semantic_class_weight is not None:
            semantic_class_weight = semantic_class_weight.to(device)
            print(f"Semantic class weight ({args.semantic_weight_mode}): {semantic_class_weight.detach().cpu().tolist()}")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Defect task level: {args.defect_task_level}")
    print(f"Defect scene pooling: {args.defect_scene_pooling}")
    train_scene_pos_ratio = compute_scene_positive_ratio(train_ds.files, args.scene_defect_min_ratio_train)
    val_scene_pos_ratio = compute_scene_positive_ratio(val_ds.files, args.scene_defect_min_ratio_val)
    print(
        f"Scene-defect positive ratio (train/val) with min_ratio="
        f"{args.scene_defect_min_ratio_train:.4f}/{args.scene_defect_min_ratio_val:.4f}: "
        f"{train_scene_pos_ratio:.4f}/{val_scene_pos_ratio:.4f}"
    )

    history: list[dict[str, float]] = []
    best_score = -1.0
    best_defect_f1 = -1.0
    epochs_without_improve = 0
    t0 = time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            semantic_classes=args.semantic_classes,
            semantic_ignore_index=args.semantic_ignore_index,
            semantic_class_weight=semantic_class_weight,
            defect_supervise_class=args.defect_supervise_class,
            defect_task_level=args.defect_task_level,
            defect_scene_threshold=args.defect_scene_threshold,
            defect_scene_pooling=args.defect_scene_pooling,
            defect_scene_topk_ratio=args.defect_scene_topk_ratio,
            defect_scene_lse_temp=args.defect_scene_lse_temp,
            semantic_loss_weight=args.semantic_loss_weight,
            defect_loss_weight=args.defect_loss_weight,
            defect_pos_weight=args.defect_pos_weight,
            defect_use_focal=args.defect_use_focal,
            defect_focal_gamma=args.defect_focal_gamma,
            defect_label_smoothing=args.defect_label_smoothing,
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                semantic_classes=args.semantic_classes,
                semantic_ignore_index=args.semantic_ignore_index,
                semantic_class_weight=semantic_class_weight,
                defect_supervise_class=args.defect_supervise_class,
                defect_task_level=args.defect_task_level,
                defect_scene_threshold=args.defect_scene_threshold,
                defect_scene_pooling=args.defect_scene_pooling,
                defect_scene_topk_ratio=args.defect_scene_topk_ratio,
                defect_scene_lse_temp=args.defect_scene_lse_temp,
                semantic_loss_weight=args.semantic_loss_weight,
                defect_loss_weight=args.defect_loss_weight,
                defect_pos_weight=args.defect_pos_weight,
                defect_use_focal=args.defect_use_focal,
                defect_focal_gamma=args.defect_focal_gamma,
                defect_label_smoothing=args.defect_label_smoothing,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        score = val_metrics["semantic_miou"] + val_metrics["defect_f1"]
        if score > best_score:
            best_score = score
            torch.save(checkpoint, run_dir / "best.pt")

        if val_metrics["defect_f1"] > best_defect_f1 + 1e-6:
            best_defect_f1 = val_metrics["defect_f1"]
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch >= args.min_epochs and epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: no val defect F1 improvement for "
                f"{args.early_stop_patience} epochs."
            )
            break
        if args.hard_stop_epoch > 0 and epoch >= args.hard_stop_epoch:
            print(f"Hard stop at epoch {epoch} (configured --hard-stop-epoch={args.hard_stop_epoch}).")
            break

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Training done in {time() - t0:.1f}s, best score={best_score:.4f}")
    print(f"Checkpoint directory: {run_dir.resolve().as_posix()}")


if __name__ == "__main__":
    main()
