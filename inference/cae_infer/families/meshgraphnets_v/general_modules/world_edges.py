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

from general_modules.edge_features import (
    EDGE_FEATURE_DIM,
    compute_edge_attr,
    compute_edge_attr_torch,
)

try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
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


def compute_world_edges_torch(
    reference_pos: torch.Tensor,
    deformed_pos: torch.Tensor,
    mesh_edges: torch.Tensor,
    radius: float,
    max_num_neighbors: int,
    batch: Optional[torch.Tensor] = None,
    ptr: Optional[torch.Tensor] = None,
    edge_mean: Optional[torch.Tensor] = None,
    edge_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch/GPU world edges for one AR-RT rollout step.

    Connectivity is a discrete function of position and carries no useful
    gradient, so the radius search runs under `no_grad`; the returned edge
    *features* are differentiable in `deformed_pos`, which is what lets a
    contact that forms mid-rollout influence the loss.

    `batch`/`ptr` come from the PyG Batch, so a batched graph never grows
    world edges between two different samples.

    Returns (world_edge_index [2, E_world] long, world_edge_attr [E_world, 8]).
    """
    device = deformed_pos.device
    empty_ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    empty_ea = torch.zeros((0, EDGE_FEATURE_DIM), dtype=deformed_pos.dtype, device=device)

    with torch.no_grad():
        pos_detached = deformed_pos.detach()
        if HAS_TORCH_CLUSTER and device.type == 'cuda':
            candidates = radius_graph(
                x=pos_detached.float(), r=radius, batch=batch,
                loop=False, max_num_neighbors=max_num_neighbors,
            ).long()
        else:
            candidates = _radius_graph_kdtree(pos_detached, ptr, radius, device)

        if candidates.shape[1] == 0:
            return empty_ei, empty_ea

        world_edge_index = _drop_mesh_edges_torch(candidates, mesh_edges,
                                                  deformed_pos.shape[0])

    if world_edge_index.shape[1] == 0:
        return empty_ei, empty_ea

    attr = compute_edge_attr_torch(reference_pos, deformed_pos, world_edge_index)
    if edge_mean is not None and edge_std is not None:
        attr = (attr - edge_mean) / edge_std
    return world_edge_index, attr


def _radius_graph_kdtree(pos: torch.Tensor, ptr: Optional[torch.Tensor],
                         radius: float, device) -> torch.Tensor:
    """CPU scipy fallback for `compute_world_edges_torch`, one graph at a time.

    Per-graph so that a batch never produces edges between separate samples;
    `ptr` is the PyG batch boundary vector (None => a single graph).
    """
    pos_np = pos.cpu().numpy()
    if ptr is None:
        bounds = [(0, pos_np.shape[0])]
    else:
        ptr_list = ptr.tolist()
        bounds = list(zip(ptr_list[:-1], ptr_list[1:]))

    per_graph = []
    for start, end in bounds:
        tree = KDTree(pos_np[start:end])
        pairs = tree.query_pairs(r=radius, output_type='ndarray')
        if len(pairs) == 0:
            continue
        pairs = pairs.T.astype(np.int64) + start
        per_graph.append(np.concatenate([pairs, pairs[[1, 0]]], axis=1))

    if not per_graph:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.from_numpy(np.concatenate(per_graph, axis=1)).to(device)


def _drop_mesh_edges_torch(candidates: torch.Tensor, mesh_edges: torch.Tensor,
                           num_nodes: int) -> torch.Tensor:
    """Torch port of `_drop_mesh_edges` (same src*N+dst key encoding)."""
    mesh_keys = mesh_edges[0] * num_nodes + mesh_edges[1]
    cand_keys = candidates[0] * num_nodes + candidates[1]
    return candidates[:, ~torch.isin(cand_keys, mesh_keys)]


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
