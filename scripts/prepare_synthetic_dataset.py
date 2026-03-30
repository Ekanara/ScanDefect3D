from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


LABEL_MAP = {
    0: "wall",
    1: "floor",
    2: "column",
    3: "beam",
    4: "crack_defect",
    5: "misalignment_defect",
}


def sample_plane(n: int, axis: str, value: float, span_a: tuple[float, float], span_b: tuple[float, float], noise: float = 0.005) -> np.ndarray:
    a = np.random.uniform(span_a[0], span_a[1], n)
    b = np.random.uniform(span_b[0], span_b[1], n)
    pts = np.zeros((n, 3), dtype=np.float32)
    if axis == "x":
        pts[:, 0] = value
        pts[:, 1] = a
        pts[:, 2] = b
    elif axis == "y":
        pts[:, 0] = a
        pts[:, 1] = value
        pts[:, 2] = b
    elif axis == "z":
        pts[:, 0] = a
        pts[:, 1] = b
        pts[:, 2] = value
    else:
        raise ValueError(axis)
    pts += np.random.normal(0.0, noise, size=pts.shape).astype(np.float32)
    return pts


def sample_cylinder(n: int, center: tuple[float, float], radius: float, z_range: tuple[float, float], noise: float = 0.004) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n)
    z = np.random.uniform(z_range[0], z_range[1], n)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    pts += np.random.normal(0.0, noise, size=pts.shape).astype(np.float32)
    return pts


def sample_beam(n: int, start: tuple[float, float, float], end: tuple[float, float, float], width: float = 0.05) -> np.ndarray:
    t = np.random.uniform(0, 1, size=(n, 1)).astype(np.float32)
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    line = start + t * (end - start)
    jitter = np.random.normal(0, width, size=(n, 3)).astype(np.float32)
    return line + jitter


def sample_crack(n: int, base_x: float = 0.0, z_range: tuple[float, float] = (0.2, 2.6)) -> np.ndarray:
    z = np.linspace(z_range[0], z_range[1], n, dtype=np.float32)
    y = np.random.uniform(0.6, 3.2, n).astype(np.float32)
    x = np.full_like(y, base_x)
    x += np.sin(z * 22.0) * 0.015
    pts = np.stack([x, y, z], axis=1)
    pts += np.random.normal(0.0, 0.0025, size=pts.shape).astype(np.float32)
    return pts


def sample_misalignment(n: int, center: tuple[float, float, float], scale: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = scale
    pts = np.random.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    pts[:, 0] = cx + pts[:, 0] * sx
    pts[:, 1] = cy + pts[:, 1] * sy
    pts[:, 2] = cz + pts[:, 2] * sz
    pts += np.random.normal(0.0, 0.0025, size=pts.shape).astype(np.float32)
    return pts


def make_scene(total_points: int) -> tuple[np.ndarray, np.ndarray]:
    n_floor = int(total_points * 0.30)
    n_walls = int(total_points * 0.35)
    n_column = int(total_points * 0.15)
    n_beam = int(total_points * 0.10)
    n_crack = int(total_points * 0.06)
    n_misaligned = total_points - (n_floor + n_walls + n_column + n_beam + n_crack)

    floor = sample_plane(n_floor, axis="z", value=0.0, span_a=(0.0, 4.0), span_b=(0.0, 3.5))
    wall_1 = sample_plane(n_walls // 2, axis="x", value=0.0, span_a=(0.0, 3.5), span_b=(0.0, 2.8))
    wall_2 = sample_plane(n_walls - n_walls // 2, axis="y", value=0.0, span_a=(0.0, 4.0), span_b=(0.0, 2.8))
    walls = np.concatenate([wall_1, wall_2], axis=0)
    column = sample_cylinder(n_column, center=(1.5, 1.2), radius=0.12, z_range=(0.0, 2.7))
    beam = sample_beam(n_beam, start=(0.0, 2.8, 2.5), end=(3.8, 2.8, 2.5), width=0.03)
    crack = sample_crack(n_crack, base_x=0.0)
    misaligned = sample_misalignment(n_misaligned, center=(2.8, 0.15, 1.4), scale=(0.08, 0.04, 0.45))

    points = np.concatenate([walls, floor, column, beam, crack, misaligned], axis=0).astype(np.float32)
    labels = np.concatenate(
        [
            np.full(walls.shape[0], 0, dtype=np.int64),
            np.full(floor.shape[0], 1, dtype=np.int64),
            np.full(column.shape[0], 2, dtype=np.int64),
            np.full(beam.shape[0], 3, dtype=np.int64),
            np.full(crack.shape[0], 4, dtype=np.int64),
            np.full(misaligned.shape[0], 5, dtype=np.int64),
        ],
        axis=0,
    )

    shuffle_idx = np.random.permutation(points.shape[0])
    return points[shuffle_idx], labels[shuffle_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 3D construction defect dataset.")
    parser.add_argument("--output-root", type=str, default="data/synthetic")
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--points-per-scene", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_split(root: Path, split: str, count: int, points_per_scene: int) -> None:
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(count), desc=f"Generating {split}"):
        points, labels = make_scene(points_per_scene)
        np.savez_compressed(out_dir / f"{split}_{i:04d}.npz", points=points, labels=labels)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    root = Path(args.output_root)
    write_split(root, "train", args.train_count, args.points_per_scene)
    write_split(root, "val", args.val_count, args.points_per_scene)
    write_split(root, "test", args.test_count, args.points_per_scene)
    with (root / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2, ensure_ascii=False)
    print(f"Synthetic dataset generated at: {root.resolve().as_posix()}")


if __name__ == "__main__":
    main()
