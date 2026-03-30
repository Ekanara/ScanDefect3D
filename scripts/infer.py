from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scandefect3d.models.factory import build_model
from scandefect3d.utils.io import ensure_dir
from scandefect3d.utils.pointcloud import (
    load_point_cloud,
    normalize_points,
    remove_statistical_outliers,
    save_colored_point_cloud,
    voxel_downsample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for ScanDefect3D.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input point cloud path (.ply/.pcd/.xyz/.npz)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-votes", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="0 means no voxel downsample")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--defect-class-ids", type=str, default="4,5")
    parser.add_argument("--save-colored-ply", action="store_true")
    return parser.parse_args()


def make_palette(num_classes: int, defect_ids: set[int]) -> np.ndarray:
    rng = np.random.default_rng(7)
    colors = rng.uniform(0.2, 0.95, size=(num_classes, 3))
    for i in defect_ids:
        if 0 <= i < num_classes:
            colors[i] = np.array([0.95, 0.1, 0.1])  # defect class in red
    return colors


def batched_vote_predict(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    num_classes: int,
    num_points: int,
    num_votes: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    votes = np.zeros((n, num_classes), dtype=np.float32)
    points_norm = normalize_points(points)
    chunks_per_vote = max(1, int(np.ceil(n / num_points)))

    with torch.no_grad():
        for _ in range(num_votes):
            perm = np.random.permutation(n)
            for chunk in np.array_split(perm, chunks_per_vote):
                if chunk.shape[0] < num_points:
                    pad = np.random.choice(n, num_points - chunk.shape[0], replace=True)
                    idx = np.concatenate([chunk, pad], axis=0)
                else:
                    idx = chunk
                sample = points_norm[idx]
                x = torch.from_numpy(sample.T).float().unsqueeze(0).to(device)
                logits = model(x)[0].transpose(0, 1)  # [num_points, C]
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                votes[idx] += probs
    preds = votes.argmax(axis=1).astype(np.int64)
    return preds, votes


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]

    model_name = ckpt_args["model"]
    num_classes = int(ckpt_args["num_classes"])
    input_dim = int(ckpt_args.get("input_dim", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name=model_name, num_classes=num_classes, input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    points = load_point_cloud(args.input)
    if args.voxel_size > 0:
        points = voxel_downsample(points, voxel_size=args.voxel_size)
    if args.denoise:
        points = remove_statistical_outliers(points)
    xyz = points[:, :3].astype(np.float32)

    preds, scores = batched_vote_predict(
        model=model,
        points=points[:, :input_dim].astype(np.float32),
        device=device,
        num_classes=num_classes,
        num_points=args.num_points,
        num_votes=args.num_votes,
    )

    defect_ids = {int(x.strip()) for x in args.defect_class_ids.split(",") if x.strip()}
    defect_mask = np.isin(preds, list(defect_ids))

    base_name = Path(args.input).stem
    npz_path = out_dir / f"{base_name}_pred.npz"
    np.savez_compressed(
        npz_path,
        points=xyz,
        pred_labels=preds,
        scores=scores,
        defect_mask=defect_mask.astype(np.uint8),
    )

    if args.save_colored_ply:
        palette = make_palette(num_classes=num_classes, defect_ids=defect_ids)
        colors = palette[preds]
        save_colored_point_cloud(out_dir / f"{base_name}_pred.ply", xyz=xyz, colors=colors)

    summary = {
        "input": str(Path(args.input).resolve()),
        "output_npz": str(npz_path.resolve()),
        "num_points": int(xyz.shape[0]),
        "defect_points": int(defect_mask.sum()),
        "defect_ratio": float(defect_mask.mean()),
        "device": str(device),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
