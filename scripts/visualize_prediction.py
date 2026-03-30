from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize defect prediction from npz file.")
    parser.add_argument("--pred-npz", type=str, required=True, help="Output .npz from scripts/infer.py")
    parser.add_argument("--defect-class-ids", type=str, default="4,5")
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def make_palette(num_classes: int, defect_ids: set[int]) -> np.ndarray:
    rng = np.random.default_rng(17)
    colors = rng.uniform(0.2, 0.95, size=(num_classes, 3))
    for i in defect_ids:
        if 0 <= i < num_classes:
            colors[i] = np.array([0.95, 0.1, 0.1])
    return colors


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_npz)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    with np.load(pred_path) as f:
        points = f["points"].astype(np.float32)
        pred_labels = f["pred_labels"].astype(np.int64)

    defect_ids = {int(x.strip()) for x in args.defect_class_ids.split(",") if x.strip()}
    num_classes = int(pred_labels.max()) + 1
    palette = make_palette(num_classes=max(num_classes, max(defect_ids) + 1), defect_ids=defect_ids)
    colors = palette[pred_labels]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Prediction: {pred_path.name}", width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0.03, 0.03, 0.03])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

