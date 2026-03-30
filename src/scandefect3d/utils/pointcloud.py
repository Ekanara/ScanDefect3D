from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points.copy()
    centered[:, :3] = centered[:, :3] - centered[:, :3].mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered[:, :3], axis=1).max()
    if scale > 1e-9:
        centered[:, :3] = centered[:, :3] / scale
    return centered


def random_sample(points: np.ndarray, labels: np.ndarray | None, num_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if points.shape[0] >= num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        idx = np.random.choice(points.shape[0], num_points, replace=True)
    sampled_points = points[idx]
    sampled_labels = labels[idx] if labels is not None else None
    return sampled_points, sampled_labels


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.03) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points)
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors)
        return np.concatenate([xyz, rgb], axis=1)
    return xyz


def remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    xyz = np.asarray(filtered.points)
    if filtered.has_colors():
        rgb = np.asarray(filtered.colors)
        return np.concatenate([xyz, rgb], axis=1)
    return xyz


def load_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors)
            return np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        return xyz.astype(np.float32)
    if suffix in {".xyz", ".txt"}:
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
    if suffix == ".npz":
        with np.load(path) as f:
            return f["points"].astype(np.float32)
    raise ValueError(f"Unsupported point cloud format: {suffix}")


def save_colored_point_cloud(path: str | Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.io.write_point_cloud(str(path), pcd)

