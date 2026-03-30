from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFECT_TYPES = [
    "missing_segment",
    "misalignment",
    "broken_pipe",
    "occlusion",
    "wrong_depth",
]

SKIP_DIR_PATTERN = re.compile(r"Heating_Area_1_Finetuning_(1|5|10|20)$")

CFG_BAND_SCALE_MIN = 0.15
CFG_BAND_SCALE_MAX = 0.25
CFG_LABEL_DILATION = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenTrench3D defect dataset (semantic + defect labels).")
    parser.add_argument("--input-root", type=str, default="data/OpenTrench3D/OpenTrench3D")
    parser.add_argument("--output-root", type=str, default="data/opentrench3d_defect_multitask")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means use all canonical scenes")
    parser.add_argument("--variants-per-scene", type=int, default=2)
    parser.add_argument("--max-points-per-scene", type=int, default=200000)
    parser.add_argument("--defect-band-scale-min", type=float, default=0.15)
    parser.add_argument("--defect-band-scale-max", type=float, default=0.25)
    parser.add_argument("--label-dilation", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--include-normal-sample", action="store_true")
    return parser.parse_args()


def canonical_scene_files(root: Path) -> list[Path]:
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not SKIP_DIR_PATTERN.search(d.name)]
    files: list[Path] = []
    for d in dirs:
        files.extend(sorted(d.glob("*.ply")))
    return files


def load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        skip_rows = 0
        for line in f:
            skip_rows += 1
            if line.strip() == "end_header":
                break
    arr = np.loadtxt(path, skiprows=skip_rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 7:
        raise ValueError(f"Invalid OpenTrench3D row format in {path}")
    points = arr[:, :6].copy()
    points[:, 3:6] = points[:, 3:6] / 255.0
    semantic_labels = arr[:, 6].astype(np.int64)
    return points, semantic_labels


def maybe_downsample(points: np.ndarray, semantic_labels: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, semantic_labels
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx], semantic_labels[idx]


def pca_axes(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_0 = vh[0] / (np.linalg.norm(vh[0]) + 1e-8)
    axis_1 = vh[1] / (np.linalg.norm(vh[1]) + 1e-8)
    axis_2 = vh[2] / (np.linalg.norm(vh[2]) + 1e-8)
    return axis_0, axis_1, axis_2


def choose_main_utility_band(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    main_idx = np.where(semantic_labels == 0)[0]
    if main_idx.size < 120:
        return None
    main_xyz = points[main_idx, :3]
    axis_0, axis_1, _ = pca_axes(main_xyz)
    centered = main_xyz - main_xyz.mean(axis=0, keepdims=True)
    proj = centered @ axis_0
    lo, hi = np.quantile(proj, [0.15, 0.85])
    center = float(np.random.uniform(lo, hi))
    width_ratio = float(np.random.uniform(CFG_BAND_SCALE_MIN, CFG_BAND_SCALE_MAX))
    width = max(0.08, width_ratio * float(hi - lo))
    band_mask = np.abs(proj - center) <= width
    if band_mask.sum() < 60:
        return None
    return main_idx, axis_1, center, width


def apply_missing_segment(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    band = choose_main_utility_band(points, semantic_labels)
    if band is None:
        return None
    main_idx, _, center, width = band
    main_xyz = points[main_idx, :3]
    axis_0, _, _ = pca_axes(main_xyz)
    proj = (main_xyz - main_xyz.mean(axis=0, keepdims=True)) @ axis_0
    remove_mask = np.abs(proj - center) <= width
    if remove_mask.sum() < 80:
        return None

    remove_idx = main_idx[remove_mask]
    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[remove_idx] = False
    kept_points = points[keep_mask]
    kept_semantic = semantic_labels[keep_mask]
    defect_labels = np.zeros(kept_points.shape[0], dtype=np.int64)

    remaining_main_idx = main_idx[~remove_mask]
    if remaining_main_idx.size > 0:
        rem_xyz = points[remaining_main_idx, :3]
        rem_proj = (rem_xyz - main_xyz.mean(axis=0, keepdims=True)) @ axis_0
        edge_mask = (np.abs(rem_proj - center) > width) & (np.abs(rem_proj - center) <= CFG_LABEL_DILATION * width)
        if edge_mask.any():
            old_to_new = np.cumsum(keep_mask) - 1
            defect_labels[old_to_new[remaining_main_idx[edge_mask]]] = 1

    return kept_points, kept_semantic, defect_labels


def apply_misalignment(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    band = choose_main_utility_band(points, semantic_labels)
    if band is None:
        return None
    main_idx, _, center, width = band
    main_xyz = points[main_idx, :3]
    axis_0, axis_1, axis_2 = pca_axes(main_xyz)
    proj = (main_xyz - main_xyz.mean(axis=0, keepdims=True)) @ axis_0
    move_mask = np.abs(proj - center) <= width
    influence_mask = np.abs(proj - center) <= (width * 1.4)
    move_idx = main_idx[influence_mask]
    if move_idx.size < 120:
        return None

    shift_scale = float(np.random.uniform(0.03, 0.08))
    shift = axis_1 * shift_scale + axis_2 * float(np.random.uniform(-0.02, 0.02))
    out_points = points.copy()
    out_points[move_idx, :3] += shift[None, :]
    defect_labels = np.zeros(points.shape[0], dtype=np.int64)
    defect_labels[move_idx] = 1
    return out_points, semantic_labels.copy(), defect_labels


def apply_broken_pipe(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    band = choose_main_utility_band(points, semantic_labels)
    if band is None:
        return None
    main_idx, axis_1, center, width = band
    main_xyz = points[main_idx, :3]
    axis_0, _, _ = pca_axes(main_xyz)
    centered = main_xyz - main_xyz.mean(axis=0, keepdims=True)
    proj = centered @ axis_0
    split_candidates = np.abs(proj - center) <= (width * 1.35)
    split_idx = main_idx[split_candidates]
    if split_idx.size < 160:
        return None

    side_values = centered[split_candidates] @ axis_1
    median_side = np.median(side_values)
    side_a = split_idx[side_values >= median_side]
    side_b = split_idx[side_values < median_side]
    if side_a.size < 40 or side_b.size < 40:
        return None

    out_points = points.copy()
    out_points[side_a, :3] += axis_1[None, :] * 0.025 + axis_0[None, :] * 0.015
    out_points[side_b, :3] -= axis_1[None, :] * 0.025 + axis_0[None, :] * 0.015
    defect_labels = np.zeros(points.shape[0], dtype=np.int64)
    defect_labels[split_idx] = 1
    return out_points, semantic_labels.copy(), defect_labels


def apply_occlusion(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    band = choose_main_utility_band(points, semantic_labels)
    if band is None:
        return None
    main_idx, axis_1, center, width = band
    main_xyz = points[main_idx, :3]
    axis_0, _, _ = pca_axes(main_xyz)
    centered = main_xyz - main_xyz.mean(axis=0, keepdims=True)
    proj = centered @ axis_0
    band_mask = np.abs(proj - center) <= width * 1.1
    band_idx = main_idx[band_mask]
    if band_idx.size < 150:
        return None

    side_values = centered[band_mask] @ axis_1
    remove_mask = side_values >= np.quantile(side_values, 0.5)
    remove_idx = band_idx[remove_mask]
    if remove_idx.size < 60:
        return None

    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[remove_idx] = False
    kept_points = points[keep_mask]
    kept_semantic = semantic_labels[keep_mask]
    defect_labels = np.zeros(kept_points.shape[0], dtype=np.int64)

    remaining_band_idx = band_idx[~remove_mask]
    if remaining_band_idx.size > 0:
        old_to_new = np.cumsum(keep_mask) - 1
        defect_labels[old_to_new[remaining_band_idx]] = 1
    return kept_points, kept_semantic, defect_labels


def apply_wrong_depth(points: np.ndarray, semantic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    band = choose_main_utility_band(points, semantic_labels)
    if band is None:
        return None
    main_idx, _, center, width = band
    main_xyz = points[main_idx, :3]
    axis_0, _, _ = pca_axes(main_xyz)
    proj = (main_xyz - main_xyz.mean(axis=0, keepdims=True)) @ axis_0
    move_mask = np.abs(proj - center) <= (width * 1.4)
    move_idx = main_idx[move_mask]
    if move_idx.size < 120:
        return None

    out_points = points.copy()
    lift = float(np.random.uniform(0.05, 0.14))
    out_points[move_idx, 2] += lift
    defect_labels = np.zeros(points.shape[0], dtype=np.int64)
    defect_labels[move_idx] = 1
    return out_points, semantic_labels.copy(), defect_labels


def apply_defect(points: np.ndarray, semantic_labels: np.ndarray, defect_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if defect_type == "missing_segment":
        return apply_missing_segment(points, semantic_labels)
    if defect_type == "misalignment":
        return apply_misalignment(points, semantic_labels)
    if defect_type == "broken_pipe":
        return apply_broken_pipe(points, semantic_labels)
    if defect_type == "occlusion":
        return apply_occlusion(points, semantic_labels)
    if defect_type == "wrong_depth":
        return apply_wrong_depth(points, semantic_labels)
    raise ValueError(defect_type)


def write_npz(
    out_file: Path,
    points: np.ndarray,
    semantic_labels: np.ndarray,
    defect_labels: np.ndarray,
    source_scene: str,
    defect_type: str,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        points=points.astype(np.float32),
        semantic_labels=semantic_labels.astype(np.int64),
        defect_labels=defect_labels.astype(np.int64),
        source_scene=np.array(source_scene),
        defect_type=np.array(defect_type),
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    global CFG_BAND_SCALE_MIN, CFG_BAND_SCALE_MAX, CFG_LABEL_DILATION
    CFG_BAND_SCALE_MIN = args.defect_band_scale_min
    CFG_BAND_SCALE_MAX = args.defect_band_scale_max
    CFG_LABEL_DILATION = args.label_dilation

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = canonical_scene_files(input_root)
    if args.max_scenes > 0:
        files = files[: args.max_scenes]
    if not files:
        raise FileNotFoundError(f"No canonical .ply files found in {input_root}")

    rng = np.random.default_rng(args.seed)
    files = list(files)
    rng.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    split_map = {}
    for i, scene_file in enumerate(files):
        if i < n_train:
            split_map[scene_file] = "train"
        elif i < n_train + n_val:
            split_map[scene_file] = "val"
        else:
            split_map[scene_file] = "test"

    summary = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "total_scenes": n_total,
        "variants_per_scene": args.variants_per_scene,
        "max_points_per_scene": args.max_points_per_scene,
        "splits": {"train": 0, "val": 0, "test": 0},
        "defect_type_counts": {k: 0 for k in DEFECT_TYPES},
    }

    for scene_idx, scene_file in enumerate(tqdm(files, desc="Building OpenTrench defect dataset")):
        split = split_map[scene_file]
        points, semantic_labels = load_ascii_ply(scene_file)
        points, semantic_labels = maybe_downsample(points, semantic_labels, args.max_points_per_scene)

        base_stem = scene_file.stem
        if args.include_normal_sample:
            normal_defect = np.zeros(points.shape[0], dtype=np.int64)
            normal_file = output_root / split / f"{base_stem}__normal.npz"
            write_npz(normal_file, points, semantic_labels, normal_defect, source_scene=scene_file.as_posix(), defect_type="none")
            summary["splits"][split] += 1

        for variant_idx in range(args.variants_per_scene):
            defect_type = DEFECT_TYPES[(scene_idx + variant_idx) % len(DEFECT_TYPES)]
            out = apply_defect(points, semantic_labels, defect_type)
            if out is None:
                fallback = apply_misalignment(points, semantic_labels)
                if fallback is None:
                    continue
                out_points, out_sem, out_def = fallback
                defect_type = "misalignment"
            else:
                out_points, out_sem, out_def = out

            defect_file = output_root / split / f"{base_stem}__defect_{variant_idx}_{defect_type}.npz"
            write_npz(defect_file, out_points, out_sem, out_def, source_scene=scene_file.as_posix(), defect_type=defect_type)
            summary["splits"][split] += 1
            summary["defect_type_counts"][defect_type] += 1

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
