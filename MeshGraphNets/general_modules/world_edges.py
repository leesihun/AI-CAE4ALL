"""
Shared world-edge (radius-based collision) computation used by both
`general_modules/mesh_dataset.py` (training / preprocessing) and
`inference_profiles/rollout.py` (inference).

Consolidates the previously duplicated torch_cluster / scipy.KDTree backends
into one function with a proper `device` argument.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr

try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except (ImportError, OSError):
    HAS_TORCH_CLUSTER = False


def compute_world_edges(
    reference_pos: np.ndarray,
    deformed_pos: np.ndarray,
    mesh_edges: np.ndarray,
    radius: float,
    max_num_neighbors: int,
    backend: str = 'torch_cluster',
    device: Optional[torch.device] = None,
    edge_mean: Optional[np.ndarray] = None,
    edge_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute world edges (radius-based collision) for a single timestep.

    Supports two backends:
        - 'torch_cluster': GPU-accelerated (5-10x faster for 68k-node meshes)
        - any other value (e.g. 'scipy_kdtree'): CPU fallback via scipy.KDTree

    Edges present in the mesh topology (as encoded by `mesh_edges`) are filtered
    out of the result — the two edge sets are kept disjoint.

    If `edge_mean` and `edge_std` are provided, the returned `world_edge_attr`
    is z-score normalized; otherwise it is raw 8-D features.

    Args:
        reference_pos:      (N, 3) reference node positions.
        deformed_pos:       (N, 3) current deformed node positions.
        mesh_edges:         (2, E_mesh) existing bidirectional mesh edges.
        radius:             world-edge radius cutoff.
        max_num_neighbors:  cap on neighbors per node (torch_cluster only).
        backend:            'torch_cluster' or any other string (CPU fallback).
        device:             torch device for torch_cluster backend; defaults to
                            CUDA if available, else CPU.
        edge_mean, edge_std: optional (8,) arrays for inline z-score
                             normalization of the output edge attrs.

    Returns:
        (world_edge_index, world_edge_attr) where:
            world_edge_index: (2, E_world) int64 (possibly empty)
            world_edge_attr:  (E_world, 8)  float32 (possibly empty)
    """
    empty_ei = np.zeros((2, 0), dtype=np.int64)
    empty_ea = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    if backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pos_tensor = torch.from_numpy(deformed_pos).float().to(device)
        world_edges = radius_graph(
            x=pos_tensor, r=radius, batch=None,
            loop=False, max_num_neighbors=max_num_neighbors,
        )
        world_edges_np = world_edges.cpu().numpy()
    else:
        tree = KDTree(deformed_pos)
        pairs = tree.query_pairs(r=radius, output_type='ndarray')
        if len(pairs) == 0:
            return empty_ei, empty_ea
        # scipy returns unordered pairs; expand to directed edges, both ways.
        pairs = pairs.T.astype(np.int64)  # [2, P]
        candidates = np.concatenate([pairs, pairs[[1, 0]]], axis=1)
        wei = _drop_mesh_edges(candidates, mesh_edges, deformed_pos.shape[0])
        if wei.shape[1] == 0:
            return empty_ei, empty_ea
        wea = compute_edge_attr(reference_pos, deformed_pos, wei)
        if edge_mean is not None and edge_std is not None:
            wea = (wea - edge_mean) / edge_std
        return wei, wea.astype(np.float32)

    if world_edges_np.shape[1] == 0:
        return empty_ei, empty_ea

    we = _drop_mesh_edges(world_edges_np.astype(np.int64), mesh_edges,
                          deformed_pos.shape[0])
    if we.shape[1] == 0:
        return empty_ei, empty_ea

    wea = compute_edge_attr(reference_pos, deformed_pos, we)
    if edge_mean is not None and edge_std is not None:
        wea = (wea - edge_mean) / edge_std
    return we, wea.astype(np.float32)


def _drop_mesh_edges(candidates: np.ndarray, mesh_edges: np.ndarray,
                     num_nodes: int) -> np.ndarray:
    """Remove candidate edges that already exist in the mesh topology.

    Both edge sets are encoded as `src * num_nodes + dst` int64 keys so the
    membership test runs in numpy. The previous implementation materialized a
    Python set of one tuple per mesh edge, which cost ~0.9 s per __getitem__ on
    a 200k-node / 1.5M-edge mesh and dominated dataloader time.

    Returns the surviving edges; column order is not preserved relative to the
    old tuple-loop version, but the edge set is identical and downstream
    aggregation is order-invariant.
    """
    if candidates.shape[1] == 0:
        return candidates
    mesh_keys = (mesh_edges[0].astype(np.int64) * num_nodes
                 + mesh_edges[1].astype(np.int64))
    cand_keys = candidates[0] * num_nodes + candidates[1]
    return candidates[:, ~np.isin(cand_keys, mesh_keys)]
