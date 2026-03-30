from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scandefect3d.models.pointnet2 import PointNetFeaturePropagation, PointNetSetAbstraction


class MultiTaskPointNet2Transformer(nn.Module):
    """PointNet++ backbone with a lightweight transformer block at SA2 resolution."""

    def __init__(
        self,
        num_semantic_classes: int = 5,
        input_dim: int = 6,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.sa1 = PointNetSetAbstraction(npoint=1024, nsample=32, in_channel=input_dim + 3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.l2_pre = nn.Conv1d(256, 256, 1)
        self.l2_pre_bn = nn.BatchNorm1d(256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=transformer_heads,
            dim_feedforward=512,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.l2_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.l2_post = nn.Conv1d(256, 256, 1)
        self.l2_post_bn = nn.BatchNorm1d(256)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + input_dim, mlp=[128, 128, 128])

        self.shared_conv = nn.Conv1d(128, 128, 1)
        self.shared_bn = nn.BatchNorm1d(128)
        self.shared_drop = nn.Dropout(0.3)

        self.semantic_head = nn.Conv1d(128, num_semantic_classes, 1)
        self.defect_head = nn.Conv1d(128, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l0_xyz = x[:, :3, :]
        l0_points = x

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # Local-global context mixing at SA2 tokens (256 points).
        l2_attn = F.relu(self.l2_pre_bn(self.l2_pre(l2_points)))
        l2_attn = self.l2_transformer(l2_attn.transpose(1, 2)).transpose(1, 2)
        l2_points = l2_points + F.relu(self.l2_post_bn(self.l2_post(l2_attn)))

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        feat = F.relu(self.shared_bn(self.shared_conv(l0_points)))
        feat = self.shared_drop(feat)

        semantic_logits = self.semantic_head(feat)
        defect_logits = self.defect_head(feat)
        return semantic_logits, defect_logits
