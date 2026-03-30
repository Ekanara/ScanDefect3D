from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from scandefect3d.data.factory import build_dataloader
from scandefect3d.models.factory import build_model
from scandefect3d.utils.io import ensure_dir
from scandefect3d.utils.metrics import fast_hist, metrics_from_hist
from scandefect3d.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet/PointNet++ for 3D defect segmentation.")
    parser.add_argument("--dataset-type", type=str, default="npz", choices=["npz", "semantickitti"])
    parser.add_argument("--data-root", type=str, default="data/synthetic")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--model", type=str, default="pointnet", choices=["pointnet", "pointnet2"])
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--exp-name", type=str, default="pointnet_synthetic")
    return parser.parse_args()


def clean_labels(labels: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = (labels >= 0) & (labels < num_classes)
    return torch.where(valid, labels, torch.full_like(labels, ignore_index))


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(loader, leave=False):
        points = batch["points"].to(device, non_blocking=True)
        labels = clean_labels(batch["labels"].to(device, non_blocking=True), num_classes, ignore_index)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(points)
                loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * points.size(0)
        preds = torch.argmax(logits, dim=1)
        hist += fast_hist(preds, labels, num_classes=num_classes, ignore_index=ignore_index)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    metrics = metrics_from_hist(hist)
    return epoch_loss, metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ensure_dir(Path(args.save_dir) / args.exp_name)

    train_loader = build_dataloader(
        dataset_type=args.dataset_type,
        root=args.data_root,
        split=args.train_split,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        dataset_type=args.dataset_type,
        root=args.data_root,
        split=args.val_split,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(args.model, num_classes=args.num_classes, input_dim=args.input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best_miou = -1.0
    start = time()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            scaler=scaler,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                scaler=scaler,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_miou": train_metrics["mean_iou"],
            "val_miou": val_metrics["mean_iou"],
            "train_acc": train_metrics["overall_acc"],
            "val_acc": val_metrics["overall_acc"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        latest_path = run_dir / "last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": row,
            },
            latest_path,
        )

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": row,
                },
                run_dir / "best.pt",
            )

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    elapsed = time() - start
    print(f"Training done in {elapsed:.1f}s. Best mIoU={best_miou:.4f}")
    print(f"Checkpoint directory: {run_dir.resolve().as_posix()}")


if __name__ == "__main__":
    main()
