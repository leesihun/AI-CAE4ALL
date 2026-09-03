"""PointNet geometry/state encoder for Point-DeepONet's branch (section 8.1),
following the published shared-weight Conv1D + BatchNorm + ReLU + global max
pool design (Qi et al. 2017 as adapted by Park & Kang 2026). No T-Net: a
learned canonicalizing rotation would transform coordinate axes and vector
targets without applying the same transform to the physical vector channels
(displacement) carried alongside them, breaking equivariance.

A shared Conv1D with kernel_size=1 over the point dimension is mathematically
identical to a Linear layer applied independently to every point; this module
uses `nn.Linear` + `nn.BatchNorm1d` on flattened [*, C] tensors, which keeps
the dense (fixed M sensors) and segmented (all-nodes ablation) code paths
sharing one block stack.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

_ACTIVATIONS = {'relu': nn.ReLU}


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 depth: int = 3, activation: str = 'relu', norm: str = 'batch'):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"pointnet_activation must be 'relu' (baseline), got '{activation}'.")
        if norm != 'batch':
            raise ValueError(f"pointnet_norm must be 'batch' (baseline), got '{norm}'.")
        act_cls = _ACTIVATIONS[activation]

        widths = [hidden_channels] * max(depth - 1, 0) + [out_channels]
        blocks = []
        prev = in_channels
        for w in widths:
            blocks.append(nn.Linear(prev, w))
            blocks.append(nn.BatchNorm1d(w))
            blocks.append(act_cls())
            prev = w
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, M, C_in] fixed-size sensor sets -> [B, out_channels]."""
        b, m, c = x.shape
        feat = self.blocks(x.reshape(b * m, c)).reshape(b, m, self.out_channels)
        pooled, _ = feat.max(dim=1)
        return pooled

    def forward_segmented(self, x: torch.Tensor, batch: torch.Tensor,
                          num_graphs: int) -> torch.Tensor:
        """x: [sum_N, C_in], batch: [sum_N] -> [num_graphs, out_channels].

        Used only by the `point_sensor_count 0` all-points ablation (section
        8.1): segmented max pooling via `torch_geometric.utils.scatter`,
        never assumed to be numerically identical to the dense path.
        """
        feat = self.blocks(x)
        return scatter(feat, batch, dim=0, dim_size=num_graphs, reduce='max')
