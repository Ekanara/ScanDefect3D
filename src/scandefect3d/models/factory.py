from __future__ import annotations

import torch.nn as nn

from scandefect3d.models.pointnet import PointNetSeg
from scandefect3d.models.pointnet2 import PointNet2Seg


def build_model(model_name: str, num_classes: int, input_dim: int = 3) -> nn.Module:
    name = model_name.lower()
    if name == "pointnet":
        return PointNetSeg(num_classes=num_classes, input_dim=input_dim)
    if name in {"pointnet2", "pointnet++"}:
        return PointNet2Seg(num_classes=num_classes, input_dim=input_dim)
    raise ValueError(f"Unsupported model_name: {model_name}")

