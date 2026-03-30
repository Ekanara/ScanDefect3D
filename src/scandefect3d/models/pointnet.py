from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        nn.init.constant_(self.fc3.weight, 0)
        init_bias = torch.eye(k).view(-1)
        with torch.no_grad():
            self.fc3.bias.copy_(init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetSeg(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 3, feature_transform: bool = True) -> None:
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

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, N]
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
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        logits = self.conv7(x)
        return logits

