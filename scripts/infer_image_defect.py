from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

from scandefect3d.models.multitask_factory import build_multitask_model
from scandefect3d.utils.io import ensure_dir
from scandefect3d.utils.pointcloud import normalize_points, save_colored_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference pipeline: image (URL or local) -> pseudo point cloud -> defect prediction."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Multi-task checkpoint (.pt).")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--image-url", type=str, help="Public image URL.")
    src_group.add_argument("--image-path", type=str, help="Local image path.")
    parser.add_argument("--output-dir", type=str, default="outputs/image_infer")
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--max-points", type=int, default=60000, help="Downsample pseudo cloud before model inference.")
    parser.add_argument("--num-points", type=int, default=4096, help="Points per model forward pass.")
    parser.add_argument("--num-votes", type=int, default=4)
    parser.add_argument("--depth-mode", type=str, default="inverse_luma", choices=["inverse_luma", "luma"])
    parser.add_argument("--depth-scale", type=float, default=0.6)
    parser.add_argument(
        "--defect-threshold",
        type=float,
        default=-1.0,
        help="Override threshold; use <0 to read from checkpoint args.",
    )
    parser.add_argument(
        "--defect-scene-pooling",
        type=str,
        default="auto",
        choices=["auto", "mean", "max", "topk", "lse"],
        help="auto: reuse pooling from checkpoint args when available.",
    )
    parser.add_argument("--defect-scene-topk-ratio", type=float, default=-1.0, help="Override top-k ratio; <0 uses checkpoint.")
    parser.add_argument("--defect-scene-lse-temp", type=float, default=-1.0, help="Override lse temperature; <0 uses checkpoint.")
    parser.add_argument("--save-colored-ply", action="store_true")
    return parser.parse_args()


def safe_stem(name: str) -> str:
    base = Path(name).stem or "image"
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
    return cleaned[:80] if cleaned else "image"


def download_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "ScanDefect3D/1.0"})
    with urlopen(req, timeout=30) as resp:
        content = resp.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    return image


def load_local_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.BILINEAR)


def image_to_pseudo_point_cloud(
    image_rgb: np.ndarray,
    depth_mode: str = "inverse_luma",
    depth_scale: float = 0.6,
) -> np.ndarray:
    # image_rgb: [H, W, 3] in [0, 1]
    h, w, _ = image_rgb.shape
    gray = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    if depth_mode == "inverse_luma":
        depth = 1.0 - gray
    else:
        depth = gray

    depth = depth - depth.min()
    depth = depth / max(float(depth.max()), 1e-8)
    z = (depth - 0.5) * float(depth_scale)

    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    xyz = np.stack([xg, -yg, z.astype(np.float32)], axis=-1).reshape(-1, 3)
    rgb = image_rgb.reshape(-1, 3).astype(np.float32)
    points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return points


def pool_scene_logits(
    defect_logits_np: np.ndarray,  # [N, 2]
    semantic_pred_np: np.ndarray,  # [N]
    supervise_class: int,
    pooling: str,
    topk_ratio: float,
    lse_temp: float,
) -> np.ndarray:
    class_mask = semantic_pred_np == supervise_class
    chosen = defect_logits_np[class_mask] if class_mask.any() else defect_logits_np
    if chosen.shape[0] == 0:
        chosen = defect_logits_np

    if pooling == "max":
        return chosen.max(axis=0)
    if pooling == "topk":
        k = max(1, int(chosen.shape[0] * float(np.clip(topk_ratio, 1e-4, 1.0))))
        part0 = np.partition(chosen[:, 0], -k)[-k:]
        part1 = np.partition(chosen[:, 1], -k)[-k:]
        return np.array([part0.mean(), part1.mean()], dtype=np.float32)
    if pooling == "lse":
        t = max(float(lse_temp), 1e-3)
        m0 = float(chosen[:, 0].max())
        m1 = float(chosen[:, 1].max())
        l0 = m0 + np.log(np.exp((chosen[:, 0] - m0) * t).sum()) / t
        l1 = m1 + np.log(np.exp((chosen[:, 1] - m1) * t).sum()) / t
        return np.array([l0, l1], dtype=np.float32)
    return chosen.mean(axis=0).astype(np.float32)


def softmax2(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    ex = np.exp(x)
    return ex / max(float(ex.sum()), 1e-8)


def infer_multitask(
    model: torch.nn.Module,
    points: np.ndarray,  # [N, 6], normalized by caller
    device: torch.device,
    num_semantic_classes: int,
    input_dim: int,
    num_points: int,
    num_votes: int,
    defect_supervise_class: int,
    defect_scene_pooling: str,
    defect_scene_topk_ratio: float,
    defect_scene_lse_temp: float,
    defect_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    n = points.shape[0]
    chunks_per_vote = max(1, int(np.ceil(n / float(num_points))))

    sem_sum = np.zeros((n, num_semantic_classes), dtype=np.float32)
    defect_sum = np.zeros((n, 2), dtype=np.float32)
    counts = np.zeros((n,), dtype=np.float32)

    with torch.no_grad():
        for _ in range(num_votes):
            perm = np.random.permutation(n)
            for chunk in np.array_split(perm, chunks_per_vote):
                if chunk.shape[0] < num_points:
                    pad = np.random.choice(n, num_points - chunk.shape[0], replace=True)
                    idx = np.concatenate([chunk, pad], axis=0)
                else:
                    idx = chunk

                sample = points[idx, :input_dim]
                x = torch.from_numpy(sample.T).float().unsqueeze(0).to(device)
                sem_logits, defect_logits = model(x)

                sem_np = sem_logits[0].transpose(0, 1).detach().cpu().numpy().astype(np.float32)  # [num_points, C]
                defect_np = defect_logits[0].transpose(0, 1).detach().cpu().numpy().astype(np.float32)  # [num_points, 2]

                sem_sum[idx] += sem_np
                defect_sum[idx] += defect_np
                counts[idx] += 1.0

    counts = np.maximum(counts, 1.0)
    sem_avg = sem_sum / counts[:, None]
    defect_avg = defect_sum / counts[:, None]

    sem_pred = sem_avg.argmax(axis=1).astype(np.int64)
    defect_point_prob = np.apply_along_axis(lambda x: softmax2(x)[1], 1, defect_avg)
    defect_point_pred = (defect_point_prob >= 0.5).astype(np.uint8)

    scene_logits = pool_scene_logits(
        defect_logits_np=defect_avg,
        semantic_pred_np=sem_pred,
        supervise_class=defect_supervise_class,
        pooling=defect_scene_pooling,
        topk_ratio=defect_scene_topk_ratio,
        lse_temp=defect_scene_lse_temp,
    )
    scene_prob = float(softmax2(scene_logits)[1])
    scene_pred = int(scene_prob >= defect_threshold)

    return {
        "semantic_pred": sem_pred,
        "defect_point_prob": defect_point_prob.astype(np.float32),
        "defect_point_pred": defect_point_pred,
        "scene_prob": scene_prob,
        "scene_pred": scene_pred,
        "scene_logits": scene_logits.astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    model_name = ckpt_args.get("model", "")
    if not model_name:
        raise ValueError("Invalid checkpoint: missing args.model for multitask model build.")
    num_semantic_classes = int(ckpt_args.get("semantic_classes", 5))
    input_dim = int(ckpt_args.get("input_dim", 6))
    defect_supervise_class = int(ckpt_args.get("defect_supervise_class", 0))

    defect_threshold = (
        float(args.defect_threshold)
        if args.defect_threshold >= 0.0
        else float(ckpt_args.get("defect_scene_threshold", 0.5))
    )
    defect_scene_pooling = (
        ckpt_args.get("defect_scene_pooling", "mean")
        if args.defect_scene_pooling == "auto"
        else args.defect_scene_pooling
    )
    defect_scene_topk_ratio = (
        float(ckpt_args.get("defect_scene_topk_ratio", 0.1))
        if args.defect_scene_topk_ratio < 0.0
        else float(args.defect_scene_topk_ratio)
    )
    defect_scene_lse_temp = (
        float(ckpt_args.get("defect_scene_lse_temp", 8.0))
        if args.defect_scene_lse_temp < 0.0
        else float(args.defect_scene_lse_temp)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_multitask_model(
        model_name=model_name,
        num_semantic_classes=num_semantic_classes,
        input_dim=input_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    source_value = args.image_url if args.image_url else args.image_path
    if source_value is None:
        raise ValueError("One of --image-url or --image-path must be provided.")

    if args.image_url:
        image = resize_keep_aspect(download_image(args.image_url), args.image_max_side)
        source_kind = "url"
    else:
        image = resize_keep_aspect(load_local_image(args.image_path), args.image_max_side)
        source_kind = "local_path"
    image_np = np.asarray(image).astype(np.float32) / 255.0

    points = image_to_pseudo_point_cloud(
        image_rgb=image_np,
        depth_mode=args.depth_mode,
        depth_scale=args.depth_scale,
    )
    if args.max_points > 0 and points.shape[0] > args.max_points:
        keep_idx = np.random.choice(points.shape[0], args.max_points, replace=False)
        points = points[keep_idx]

    points_norm = normalize_points(points)

    result = infer_multitask(
        model=model,
        points=points_norm,
        device=device,
        num_semantic_classes=num_semantic_classes,
        input_dim=input_dim,
        num_points=args.num_points,
        num_votes=args.num_votes,
        defect_supervise_class=defect_supervise_class,
        defect_scene_pooling=defect_scene_pooling,
        defect_scene_topk_ratio=defect_scene_topk_ratio,
        defect_scene_lse_temp=defect_scene_lse_temp,
        defect_threshold=defect_threshold,
    )

    stem = safe_stem(source_value)
    image_path = Path(out_dir) / f"{stem}_input.jpg"
    npz_path = Path(out_dir) / f"{stem}_pseudo_pointcloud_pred.npz"
    image.save(image_path)

    np.savez_compressed(
        npz_path,
        points=points[:, :3].astype(np.float32),
        colors=points[:, 3:6].astype(np.float32),
        semantic_pred=result["semantic_pred"].astype(np.int64),
        defect_point_prob=result["defect_point_prob"].astype(np.float32),
        defect_point_pred=result["defect_point_pred"].astype(np.uint8),
    )

    if args.save_colored_ply:
        pred_mask = result["defect_point_pred"].astype(bool)
        colors = points[:, 3:6].copy()
        colors[pred_mask] = np.array([0.95, 0.1, 0.1], dtype=np.float32)
        save_colored_point_cloud(Path(out_dir) / f"{stem}_defect_highlight.ply", points[:, :3], colors)

    summary = {
        "image_source": source_value,
        "image_source_kind": source_kind,
        "saved_image": str(image_path.resolve()),
        "saved_npz": str(npz_path.resolve()),
        "num_points_used": int(points.shape[0]),
        "defect_scene_probability": float(result["scene_prob"]),
        "defect_scene_threshold": float(defect_threshold),
        "defect_scene_prediction": int(result["scene_pred"]),
        "defect_scene_label": "defect" if int(result["scene_pred"]) == 1 else "normal",
        "defect_point_ratio": float(result["defect_point_pred"].mean()),
        "device": str(device),
        "model_name": model_name,
        "scene_pooling": defect_scene_pooling,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
