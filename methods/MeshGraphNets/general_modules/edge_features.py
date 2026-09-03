import numpy as np
import torch


_POS_DIM = 3  # dx, dy, dz

# An edge feature is two structurally identical halves, each a relative-position
# vector plus its norm:
#
#   [deformed_dx, deformed_dy, deformed_dz, deformed_dist | ref_dx, ref_dy, ref_dz, ref_dist]
#    <---------------- DEFORMED ---------------->  <---------------- REFERENCE ------------->
#
# The split matters because the two halves have different lifetimes: the
# deformed half moves every timestep, the reference half is fixed for a whole
# trajectory. AR-RT and inference exploit that by rebuilding only the deformed
# half per rollout step and reusing the reference half from the dataloader
# (see deformed_edge_attr_torch below). Both halves are named here so that
# split does not appear as a bare `4` at each call site.
DEFORMED_FEATURE_DIM = _POS_DIM + 1
REFERENCE_FEATURE_DIM = _POS_DIM + 1
EDGE_FEATURE_DIM = DEFORMED_FEATURE_DIM + REFERENCE_FEATURE_DIM  # 8

# Ready-made slices for the two halves, so callers index by meaning:
#   edge_attr[:, DEFORMED_SLICE]   edge_attr[:, REFERENCE_SLICE]
DEFORMED_SLICE = slice(0, DEFORMED_FEATURE_DIM)
REFERENCE_SLICE = slice(DEFORMED_FEATURE_DIM, EDGE_FEATURE_DIM)

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
    """Differentiable GPU counterpart of `compute_edge_attr`'s DEFORMED_SLICE.

    During an AR-RT unroll only the deformed half of the edge feature changes
    -- reference geometry is fixed for the whole trajectory -- so the rollout
    recomputes these channels per step and reuses the reference half that the
    dataloader already produced.

    Returns [E, DEFORMED_FEATURE_DIM] =
        [deformed_dx, deformed_dy, deformed_dz, deformed_dist].
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
