"""Native graph-neural-operator kernel integral layer (IMPLEMENTATION_PLAN.md
section 8.4, A.5). Pure torch (`index_add_` + count division); torch-scatter
is never required. Empty-neighbor queries return zero, which the mandatory
coverage preflight (`MeshGINO.coverage_preflight`) is meant to catch before
training rather than silently degrade results.
"""

import torch
import torch.nn as nn

from model.mlp import build_deep_mlp


class GNOLayer(nn.Module):
    def __init__(self, query_dim: int, source_dim: int, source_feat_dim: int,
                 hidden: int, out_dim: int, depth: int = 2):
        super().__init__()
        kernel_in = query_dim + source_dim + source_feat_dim
        self.kernel_mlp = build_deep_mlp(kernel_in, hidden, out_dim, depth, activation='silu')
        self.out_dim = out_dim

    def forward(self, q_pos: torch.Tensor, s_pos: torch.Tensor, s_feat: torch.Tensor,
                edge_index: torch.Tensor, num_queries: int) -> torch.Tensor:
        """edge_index: [2, E], row 0 = query index, row 1 = source index."""
        with torch.autocast(device_type=q_pos.device.type, enabled=False):
            q_pos_f = q_pos.float()
            s_pos_f = s_pos.float()
            s_feat_f = s_feat.float()

            out = torch.zeros(num_queries, self.out_dim, device=q_pos.device, dtype=torch.float32)
            if edge_index.shape[1] == 0:
                return out.to(s_feat.dtype)

            q_e, s_e = edge_index[0], edge_index[1]
            msg_in = torch.cat([q_pos_f[q_e], s_pos_f[s_e], s_feat_f[s_e]], dim=1)
            msg = self.kernel_mlp(msg_in)

            cnt = torch.zeros(num_queries, device=q_pos.device, dtype=torch.float32)
            out.index_add_(0, q_e, msg.float())
            cnt.index_add_(0, q_e, torch.ones_like(q_e, dtype=torch.float32))
            out = out / cnt.clamp(min=1.0).unsqueeze(-1)
        return out.to(s_feat.dtype)
