import numpy as np
import torch


EDGE_FEATURE_DIM = 8

# Guards the gradient of ||r|| at r == 0 (coincident nodes). d/dr sqrt(r.r) is
# undefined there and produces NaN, which would poison an AR-RT unroll the
# first time two nodes collapse onto each other. The numpy path never needed
# this because it is only ever evaluated, never differentiated.
_DIST_EPS = 1e-12


def compute_edge_attr(reference_pos: np.ndarray, deformed_pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Build 8-D edge features from reference and deformed positions.

    Feature order:
        [deformed_dx, deformed_dy, deformed_dz, deformed_dist,
         ref_dx,      ref_dy,      ref_dz,      ref_dist]
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    deformed_rel = deformed_pos[dst_idx] - deformed_pos[src_idx]
    deformed_dist = np.linalg.norm(deformed_rel, axis=1, keepdims=True)

    ref_rel = reference_pos[dst_idx] - reference_pos[src_idx]
    ref_dist = np.linalg.norm(ref_rel, axis=1, keepdims=True)

    return np.concatenate([deformed_rel, deformed_dist, ref_rel, ref_dist], axis=1).astype(np.float32)


def deformed_edge_attr_torch(deformed_pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Differentiable GPU counterpart of `compute_edge_attr`'s first 4 channels.

    During an AR-RT unroll only the deformed half of the 8-D edge feature
    changes -- reference geometry is fixed for the whole trajectory -- so the
    rollout recomputes these 4 channels per step and reuses the reference half
    that the dataloader already produced.

    Returns [E, 4] = [deformed_dx, deformed_dy, deformed_dz, deformed_dist].
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    rel = deformed_pos[dst_idx] - deformed_pos[src_idx]
    dist = torch.sqrt((rel * rel).sum(dim=1, keepdim=True) + _DIST_EPS)
    return torch.cat([rel, dist], dim=1)


def compute_edge_attr_torch(reference_pos: torch.Tensor, deformed_pos: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
    """Full 8-D edge features in torch; same feature order as `compute_edge_attr`.

    Used for world edges, whose connectivity is rebuilt from scratch each
    rollout step, so no reference half survives from the previous step.
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    ref_rel = reference_pos[dst_idx] - reference_pos[src_idx]
    ref_dist = torch.sqrt((ref_rel * ref_rel).sum(dim=1, keepdim=True) + _DIST_EPS)
    return torch.cat([deformed_edge_attr_torch(deformed_pos, edge_index), ref_rel, ref_dist], dim=1)
