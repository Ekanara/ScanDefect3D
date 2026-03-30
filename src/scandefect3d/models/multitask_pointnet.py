from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scandefect3d.models.pointnet import TNet


class MultiTaskPointNet(nn.Module):
    def __init__(self, num_semantic_classes: int = 5, input_dim: int = 6, feature_transform: bool = True) -> None:
        super().__init__()
        self.input_transform = TNet(k=input_dim)
        self.feature_transform = feature_transform
        self.feat_transform = TNet(k=64) if feature_transform else None

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.shared_conv4 = nn.Conv1d(1088, 512, 1)
        self.shared_conv5 = nn.Conv1d(512, 256, 1)
        self.shared_conv6 = nn.Conv1d(256, 128, 1)
        self.shared_bn4 = nn.BatchNorm1d(512)
        self.shared_bn5 = nn.BatchNorm1d(256)
        self.shared_bn6 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        self.semantic_head = nn.Conv1d(128, num_semantic_classes, 1)
        self.defect_head = nn.Conv1d(128, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_pts = x.size(2)

        transform = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        point_features = x

        if self.feature_transform and self.feat_transform is not None:
            feat_transform = self.feat_transform(point_features)
            point_features = point_features.transpose(2, 1)
            point_features = torch.bmm(point_features, feat_transform).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(point_features)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, n_pts)

        x = torch.cat([point_features, global_feature], dim=1)
        x = F.relu(self.shared_bn4(self.shared_conv4(x)))
        x = F.relu(self.shared_bn5(self.shared_conv5(x)))
        x = F.relu(self.shared_bn6(self.shared_conv6(x)))
        x = self.dropout(x)

        semantic_logits = self.semantic_head(x)
        defect_logits = self.defect_head(x)
        return semantic_logits, defect_logits

