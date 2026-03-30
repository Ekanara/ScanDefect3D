from __future__ import annotations

from torch import nn

from scandefect3d.models.multitask_pointnet import MultiTaskPointNet
from scandefect3d.models.multitask_pointnet2 import MultiTaskPointNet2
from scandefect3d.models.multitask_pointnet2_transformer import MultiTaskPointNet2Transformer


def build_multitask_model(model_name: str, num_semantic_classes: int, input_dim: int) -> nn.Module:
    name = model_name.lower()
    if name == "pointnet":
        return MultiTaskPointNet(num_semantic_classes=num_semantic_classes, input_dim=input_dim)
    if name in {"pointnet2", "pointnet++"}:
        return MultiTaskPointNet2(num_semantic_classes=num_semantic_classes, input_dim=input_dim)
    if name in {"pointnet2_transformer", "pointnet2-attn", "pointtransformer", "transformer"}:
        return MultiTaskPointNet2Transformer(num_semantic_classes=num_semantic_classes, input_dim=input_dim)
    raise ValueError(f"Unsupported multitask model: {model_name}")
