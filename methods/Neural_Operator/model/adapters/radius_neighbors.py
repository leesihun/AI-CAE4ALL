"""Radius-neighbor search for GINO's input/output graph-neural-operator kernels
(IMPLEMENTATION_PLAN.md section 7.4). scipy cKDTree is the baseline backend;
torch_cluster.radius is an optional accelerator, enabled only after a parity
test on random fixtures (test_radius_neighbors.py) proves it returns the same
edge set. Both operate on a single graph at a time -- GINO loops over `ptr`
per graph anyway (section 8.4), so no batched/ragged radius search is needed.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from torch_cluster import radius as _tc_radius
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


def radius_neighbors_scipy(queries: np.ndarray, sources: np.ndarray, r: float) -> np.ndarray:
    """For each query point, all source points within distance r.

    Args:
        queries: [Nq, d] numpy array
        sources: [Ns, d] numpy array
        r: search radius

    Returns:
        edge_index [2, E] int64, row 0 = query index, row 1 = source index,
        sorted by (query index, source index) for determinism.
    """
    if sources.shape[0] == 0 or queries.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    tree = cKDTree(sources)
    # return_sorted keeps each query's neighbors ascending, so the assembled
    # edge list is (query, source)-sorted without a per-query Python sort.
    neighbor_lists = tree.query_ball_point(queries, r, return_sorted=True)
    counts = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64,
                         count=len(neighbor_lists))
    total = int(counts.sum())
    if total == 0:
        return np.zeros((2, 0), dtype=np.int64)
    # Vectorized assembly: np.repeat + np.concatenate over the per-query lists
    # (O(num_queries) Python, not O(num_edges)) -- the old double for-loop built
    # the edge list with one .append() per edge, which is tens of millions of
    # Python ops per GINO forward on real meshes.
    q_idx = np.repeat(np.arange(queries.shape[0], dtype=np.int64), counts)
    s_idx = np.concatenate(
        [np.asarray(nb, dtype=np.int64) for nb in neighbor_lists if len(nb) > 0])
    return np.stack([q_idx, s_idx], axis=0)


def radius_neighbors_torch_cluster(queries: torch.Tensor, sources: torch.Tensor, r: float,
                                   max_num_neighbors: int = 0,
                                   query_chunk: int = 0) -> torch.Tensor:
    """Same contract as radius_neighbors_scipy but via torch_cluster.radius,
    on tensors directly (no CPU round trip). Row 0 = query index, row 1 =
    source index, matching torch_cluster's `radius(x=sources, y=queries, ...)`
    convention (edges point from queries in `y` to sources in `x`).

    `max_num_neighbors` is the per-query cap torch_cluster's CUDA kernel
    preallocates against: too small silently DROPS neighbors (changing the
    kernel-integral result), too large wastes O(num_queries * cap) memory.
    <= 0 means "no cap" (= number of sources) -- callers that know the true max
    (e.g. MeshGINO's auto-growing cap) should pass a snug value. Truncation
    detection is the caller's job (this function does not grow the cap).

    `query_chunk` > 0 splits the queries into contiguous blocks and searches
    each separately, bounding the preallocation to `query_chunk * cap` rows.
    Results are identical to an unchunked search (edges only get a query-index
    offset), just with lower peak memory -- essential when the query side is
    the large mesh (the output-decode direction).
    """
    if not HAS_TORCH_CLUSTER:
        raise RuntimeError("torch_cluster is not installed; use radius_neighbors_scipy.")
    nq, ns = queries.shape[0], sources.shape[0]
    if nq == 0 or ns == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=queries.device)
    cap = int(max_num_neighbors)
    if cap <= 0 or cap > ns:
        cap = ns
    step = nq if (query_chunk is None or query_chunk <= 0) else int(query_chunk)
    if step >= nq:
        return _tc_radius(sources, queries, r, max_num_neighbors=cap)
    parts = []
    for lo in range(0, nq, step):
        ei = _tc_radius(sources, queries[lo:lo + step], r, max_num_neighbors=cap)
        if ei.shape[1]:
            ei[0] += lo  # local query index -> global
            parts.append(ei)
    if not parts:
        return torch.zeros((2, 0), dtype=torch.long, device=queries.device)
    return torch.cat(parts, dim=1)


def radius_neighbor_count_sum(queries: np.ndarray, sources: np.ndarray, r: float) -> int:
    """Total number of (query, source) pairs within distance r, without
    materializing the edge list. Used by the model-split stage partitioner's
    cost model (parallelism/stages.py) -- `query_ball_point(...,
    return_length=True)` is vectorized and avoids the O(E) python loop of
    `radius_neighbors_scipy` on million-node probes."""
    if sources.shape[0] == 0 or queries.shape[0] == 0:
        return 0
    tree = cKDTree(sources)
    return int(np.sum(tree.query_ball_point(queries, r, return_length=True)))


def neighbor_stats(edge_index: np.ndarray, num_queries: int) -> dict:
    """min/median/max neighbor count per query and the empty-query fraction,
    used by GINO's mandatory coverage preflight (section 8.4)."""
    if num_queries == 0:
        return {'min': 0, 'median': 0.0, 'max': 0, 'empty_fraction': 1.0}
    counts = np.zeros(num_queries, dtype=np.int64)
    if edge_index.shape[1] > 0:
        idx, cnt = np.unique(edge_index[0], return_counts=True)
        counts[idx] = cnt
    return {
        'min': int(counts.min()),
        'median': float(np.median(counts)),
        'max': int(counts.max()),
        'empty_fraction': float(np.mean(counts == 0)),
    }


def min_reachable_radius(resolution, dim: int) -> float:
    """Half the diagonal of one grid cell in [0,1]^d -- the smallest radius
    that provably reaches every grid point from its nearest neighbors
    (section 8.4's coverage arithmetic)."""
    cell_sizes = [1.0 / max(r - 1, 1) for r in resolution]
    diag = float(np.sqrt(sum(c ** 2 for c in cell_sizes[:dim])))
    return diag / 2.0
