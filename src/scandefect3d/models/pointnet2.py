from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)[1]
    return idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int | None, nsample: int | None, in_channel: int, mlp: list[int], group_all: bool = False) -> None:
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        convs = []
        bns = []
        for out_channel in mlp:
            convs.append(nn.Conv2d(last_channel, out_channel, 1))
            bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: [B, 3, N], points: [B, D, N] or None
        xyz_t = xyz.transpose(2, 1)
        points_t = points.transpose(2, 1) if points is not None else None

        if self.group_all:
            new_xyz = torch.zeros(xyz_t.shape[0], 1, 3, device=xyz.device)
            grouped_xyz = xyz_t.view(xyz_t.shape[0], 1, xyz_t.shape[1], 3)
            if points_t is not None:
                grouped_points = points_t.view(points_t.shape[0], 1, points_t.shape[1], points_t.shape[2])
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            assert self.npoint is not None and self.nsample is not None
            fps_idx = farthest_point_sample(xyz_t, self.npoint)
            new_xyz = index_points(xyz_t, fps_idx)
            idx = knn_point(self.nsample, xyz_t, new_xyz)
            grouped_xyz = index_points(xyz_t, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
            if points_t is not None:
                grouped_points = index_points(points_t, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm

        # [B, npoint, nsample, C] -> [B, C, nsample, npoint]
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        return new_xyz.transpose(2, 1), new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list[int]) -> None:
        super().__init__()
        convs = []
        bns = []
        last_channel = in_channel
        for out_channel in mlp:
            convs.append(nn.Conv1d(last_channel, out_channel, 1))
            bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        # xyz1: [B,3,N], xyz2: [B,3,S], points2: [B,D,S]
        xyz1_t = xyz1.transpose(2, 1)
        xyz2_t = xyz2.transpose(2, 1)
        points2_t = points2.transpose(2, 1)
        b, n, _ = xyz1_t.shape
        s = xyz2_t.shape[1]

        if s == 1:
            interpolated_points = points2_t.repeat(1, n, 1)
        else:
            dists = square_distance(xyz1_t, xyz2_t)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2_t, idx) * weight[:, :, :, None], dim=2)

        if points1 is not None:
            points1_t = points1.transpose(2, 1)
            new_points = torch.cat([points1_t, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(2, 1)
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.sa1 = PointNetSetAbstraction(npoint=1024, nsample=32, in_channel=input_dim + 3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + input_dim, mlp=[128, 128, 128])

        self.cls_conv1 = nn.Conv1d(128, 128, 1)
        self.cls_bn1 = nn.BatchNorm1d(128)
        self.cls_drop = nn.Dropout(0.3)
        self.cls_conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l0_xyz = x[:, :3, :]
        l0_points = x

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.cls_bn1(self.cls_conv1(l0_points)))
        x = self.cls_drop(x)
        x = self.cls_conv2(x)
        return x
