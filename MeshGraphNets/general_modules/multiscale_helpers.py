"""
Shared helpers for multiscale graph coarsening used by both the dataset loader
(`general_modules/mesh_dataset.py`) and the rollout loop
(`inference_profiles/rollout.py`).

Factors out two duplicated blocks:
  1. The per-level coarsening loop that produces a list of
     {ftc, c_ei, n_c[, up_ei]} entries.
  2. The per-timestep centroid-chaining pass that attaches normalized
     coarse edge features, centroids, and bipartite unpool edges onto a PyG
     Data object.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from model.coarsening import (
    build_unpool_edges,
    coarsen_graph,
    compute_coarse_centroids,
)


def _normalize_method(method: str) -> str:
    return method.strip().lower()


def _is_inherit(method: str) -> bool:
    """Return True for the seed-anchored (variant C) voronoi mode."""
    return _normalize_method(method) == 'voronoi_inherit'


def _mode_for(method_norm: str) -> str:
    if _is_inherit(method_norm):
        return 'inherit'
    if method_norm == 'voronoi_seedmean':
        return 'seedmean'
    return 'centroid'


def _coarsen_one(current_ei, current_n, method_norm, n_clusters, level_ref_pos, seed_start=0):
    """Build one partition's entry dict (shared by the unbranched and
    per-branch code paths in build_multiscale_hierarchy)."""
    ftc, c_ei, n_c, seeds = coarsen_graph(
        current_ei, current_n, method=method_norm,
        num_clusters=n_clusters, ref_pos=level_ref_pos, seed_start=seed_start,
    )
    return {
        'ftc': ftc, 'c_ei': c_ei, 'n_c': n_c, 'seeds': seeds, 'mode': _mode_for(method_norm),
        'up_ei': build_unpool_edges(ftc, c_ei, n_c),
    }


def build_multiscale_hierarchy(
    edge_index: np.ndarray,
    num_nodes: int,
    ref_pos: np.ndarray,
    multiscale_levels: int,
    coarsening_types: Sequence[str],
    voronoi_clusters: Sequence[int],
    voronoi_branches: Optional[Sequence[int]] = None,
) -> List[dict]:
    """
    Build coarsening topology for a single sample.

    Returns a list of per-level entries — len(result) may be less than
    `multiscale_levels` when the coarsening saturates (n_c <= 1 or empty edges).

    Each entry is normally a single-partition dict:
        'ftc':   [N_level]       fine-to-coarse mapping (np.int64)
        'c_ei':  [2, E_coarse]   coarse edge index (np.int64)
        'n_c':   int             number of coarse nodes
        'seeds': [n_c]           fine-node index per coarse cluster (np.int64)
        'mode':  'centroid' | 'inherit' | 'seedmean'   per-level pool/position mode
        'up_ei': [2, E_up]       bipartite unpool edges

    When `voronoi_branches[level] > 1` (multi-partition coarsening,
    ATTENTION_TRANSFER_DESIGN.md Part II), that level's entry is instead
    `{'branches': [entry_0, entry_1, ...]}`, one dict per branch as above.
    Branching is only supported on the *last* level actually reached (an
    earlier saturation break still short-circuits normally): each branch is a
    different Voronoi partition of the SAME node set (varying the FPS start
    point, not the cluster count), giving complementary coverage instead of
    one partition's fixed cluster boundaries. A branched level is always
    terminal -- nothing chains a "next level" from multiple parallel
    partitions, so this is checked up front, not discovered mid-build.
    """
    if voronoi_branches is None:
        voronoi_branches = [1] * multiscale_levels
    for lvl, kb in enumerate(voronoi_branches):
        if kb > 1 and lvl != multiscale_levels - 1:
            raise ValueError(
                f"voronoi_branches > 1 is only supported on the last configured "
                f"level ({multiscale_levels - 1}); got {kb} branches at level {lvl}. "
                f"Branches don't chain into a further level (see build_multiscale_hierarchy docstring)."
            )

    hierarchy: List[dict] = []
    current_ei, current_n = edge_index, num_nodes
    level_ref_pos = ref_pos.astype(np.float32)

    for level in range(multiscale_levels):
        method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
        method_norm = _normalize_method(method)
        n_clusters = voronoi_clusters[level] if level < len(voronoi_clusters) else 0
        k_branches = voronoi_branches[level] if level < len(voronoi_branches) else 1

        if k_branches <= 1:
            entry = _coarsen_one(current_ei, current_n, method_norm, n_clusters, level_ref_pos)
            hierarchy.append(entry)

            c_ei, n_c = entry['c_ei'], entry['n_c']
            if n_c <= 1 or c_ei.shape[1] == 0:
                break

            # Chain to next level using the same seed-vs-centroid selection
            # stats accumulation and graph attachment use, so all three can
            # never diverge on which position anchors a coarse level.
            level_ref_pos, _ = coarse_positions_for_entry(entry, level_ref_pos)
            level_ref_pos = level_ref_pos.astype(np.float32)
            current_ei, current_n = c_ei, n_c
        else:
            if method_norm == 'bfs':
                raise ValueError(
                    f"voronoi_branches > 1 at level {level} requires a voronoi_* "
                    f"coarsening_type: 'bfs' has no seed-choice axis to vary "
                    f"across branches, so K identical bfs partitions would add "
                    f"nothing. Got coarsening_type='{method_norm}'."
                )
            branches = []
            for b in range(k_branches):
                seed_start = int(b * current_n / k_branches) % current_n if current_n > 0 else 0
                branches.append(_coarsen_one(
                    current_ei, current_n, method_norm, n_clusters, level_ref_pos,
                    seed_start=seed_start,
                ))
            hierarchy.append({'branches': branches})
            break  # branched level is always terminal (validated above too)

    return hierarchy


def lift_world_edges(world_ei: np.ndarray, ftc: np.ndarray) -> np.ndarray:
    """Lift fine world edge index to coarse level via fine_to_coarse mapping.
    Drops intra-cluster edges (self-loops at coarse level) and deduplicates.
    Returns [2, E_coarse_world] int64, possibly empty.
    """
    if world_ei.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src_c = ftc[world_ei[0]]
    dst_c = ftc[world_ei[1]]
    mask = src_c != dst_c
    if not mask.any():
        return np.zeros((2, 0), dtype=np.int64)
    return np.unique(np.stack([src_c[mask], dst_c[mask]], axis=0), axis=1).astype(np.int64)


def coarse_positions_for_entry(entry, cur_ref, cur_def=None):
    """Return the coarse positions defined by one hierarchy entry.

    Keeping this selection shared by hierarchy construction, preprocessing,
    and graph attachment is important: seed-anchored modes must compute their
    normalization statistics from the same FPS seed positions that produce
    the forward edge features. `cur_def` is optional so hierarchy-building
    (which only tracks reference positions) can share this same selection
    logic instead of duplicating the mode branch.
    """
    ftc = entry['ftc']
    n_c = entry['n_c']
    seeds = entry.get('seeds')
    mode = entry.get('mode', 'centroid')

    if mode in ('inherit', 'seedmean') and seeds is not None:
        coarse_ref = cur_ref[seeds].astype(np.float64)
        coarse_def = cur_def[seeds].astype(np.float64) if cur_def is not None else None
        return coarse_ref, coarse_def
    coarse_ref = compute_coarse_centroids(cur_ref, ftc, n_c)
    coarse_def = compute_coarse_centroids(cur_def, ftc, n_c) if cur_def is not None else None
    return coarse_ref, coarse_def


def _attach_one_partition(graph, key, entry, cur_ref, cur_def, mean, std,
                          device, world_edge_index, expose_anchors):
    """Attach one partition's attributes to `graph` under `_{key}` names.

    `key` is `str(level)` for an unbranched level or `f'{level}_{branch}'`
    for one branch of a multi-partition level (ATTENTION_TRANSFER_DESIGN.md
    Part II). Returns (coarse_ref, coarse_def, lifted_world_edge_index) so an
    unbranched caller can chain to the next level; branched callers ignore
    the return value since a branched level is always terminal.
    """
    ftc = entry['ftc']
    c_ei = entry['c_ei']
    n_c = entry['n_c']
    seeds = entry.get('seeds')
    mode = entry.get('mode', 'centroid')
    coarse_ref, coarse_def = coarse_positions_for_entry(entry, cur_ref, cur_def)

    if c_ei.shape[1] > 0:
        c_ea_raw = compute_edge_attr(
            coarse_ref.astype(np.float32),
            coarse_def.astype(np.float32),
            c_ei,
        )
    else:
        c_ea_raw = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    if c_ea_raw.shape[0] > 0 and mean is not None:
        c_ea_norm = (c_ea_raw - mean) / std
    else:
        c_ea_norm = c_ea_raw

    ftc_t = torch.from_numpy(ftc.astype(np.int64))
    c_ei_t = torch.from_numpy(c_ei)
    c_ea_t = torch.from_numpy(c_ea_norm.astype(np.float32))
    n_c_t = torch.tensor([n_c], dtype=torch.long)
    # Note: under seed-anchored modes (voronoi_inherit, voronoi_seedmean)
    # this attribute holds seed-anchor positions (real fine-mesh
    # coordinates), not arithmetic centroids. Name retained for
    # backward-compat with reader code in MeshGraphNets.py and
    # parallelism/model_split.py.
    cent_t = torch.from_numpy(coarse_ref.astype(np.float32))

    if device is not None:
        ftc_t = ftc_t.to(device)
        c_ei_t = c_ei_t.to(device)
        c_ea_t = c_ea_t.to(device)
        n_c_t = n_c_t.to(device)
        cent_t = cent_t.to(device)

    graph[f'fine_to_coarse_{key}']    = ftc_t
    graph[f'coarse_edge_index_{key}'] = c_ei_t
    graph[f'coarse_edge_attr_{key}']  = c_ea_t
    graph[f'num_coarse_{key}']        = n_c_t
    graph[f'coarse_centroid_{key}']   = cent_t

    # Inherit-mode levels expose seed indices so the model's pool step can
    # gather features at the seeds (variant C). seedmean levels must NOT
    # write this attribute — its absence makes the model mean-pool.
    if mode == 'inherit' and seeds is not None:
        seed_idx_t = torch.from_numpy(seeds.astype(np.int64))
        if device is not None:
            seed_idx_t = seed_idx_t.to(device)
        graph[f'coarse_seed_idx_{key}'] = seed_idx_t

    # AR-RT needs to re-derive coarse positions from predicted fine
    # positions at every unrolled step. Centroid levels can do that from
    # fine_to_coarse alone; seed-anchored levels cannot, so the anchor
    # indices are exported under a name the model never reads (writing
    # coarse_seed_idx here would silently switch seedmean levels from
    # mean-pooling to gather-pooling).
    if expose_anchors and mode in ('inherit', 'seedmean') and seeds is not None:
        anchor_idx_t = torch.from_numpy(seeds.astype(np.int64))
        if device is not None:
            anchor_idx_t = anchor_idx_t.to(device)
        graph[f'coarse_anchor_idx_{key}'] = anchor_idx_t

    if 'up_ei' in entry:
        up_t = torch.from_numpy(entry['up_ei'])
        if device is not None:
            up_t = up_t.to(device)
        graph[f'unpool_edge_index_{key}'] = up_t

    next_world_ei = None
    if world_edge_index is not None:
        cw_ei = lift_world_edges(world_edge_index, ftc)
        if cw_ei.shape[1] > 0:
            cw_ea_raw = compute_edge_attr(
                coarse_ref.astype(np.float32),
                coarse_def.astype(np.float32),
                cw_ei,
            )
            cw_ea_norm = (cw_ea_raw - mean) / std if mean is not None else cw_ea_raw
        else:
            cw_ea_norm = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

        cw_ei_t = torch.from_numpy(cw_ei)
        cw_ea_t = torch.from_numpy(cw_ea_norm.astype(np.float32))
        if device is not None:
            cw_ei_t = cw_ei_t.to(device)
            cw_ea_t = cw_ea_t.to(device)
        graph[f'coarse_world_edge_index_{key}'] = cw_ei_t
        graph[f'coarse_world_edge_attr_{key}']  = cw_ea_t
        next_world_ei = cw_ei  # lift further for next level

    return coarse_ref, coarse_def, next_world_ei


def attach_coarse_levels_to_graph(
    graph,
    hierarchy: List[dict],
    ref_pos: np.ndarray,
    deformed_pos: np.ndarray,
    coarse_edge_means: Sequence,
    coarse_edge_stds: Sequence,
    device: Optional[torch.device] = None,
    world_edge_index: Optional[np.ndarray] = None,
    expose_anchors: bool = False,
) -> None:
    """
    Compute per-level centroids and coarse edge features for a single timestep,
    normalize with the provided per-level stats, and attach as graph attributes.

    Mutates `graph` in place, setting for each level `i` in `hierarchy`:
        fine_to_coarse_{i}, coarse_edge_index_{i}, coarse_edge_attr_{i},
        num_coarse_{i}, coarse_centroid_{i},
        unpool_edge_index_{i} (only if the entry carries 'up_ei')

    If level `i` is a multi-partition level (`hierarchy[i] == {'branches':
    [...]}`, see build_multiscale_hierarchy), the same attributes are written
    per branch instead, suffixed `_{i}_{b}`. `coarse_edge_means[i]` /
    `coarse_edge_stds[i]` are then expected to be a *list* of per-branch
    arrays rather than a single array (general_modules/mesh_dataset.py's
    `_compute_coarse_edge_stats` produces exactly this shape) -- world edges
    are not lifted into a branched level (coarse_world_edges + branching is
    rejected at config-validation time, since which branch would receive the
    lifted contacts is not defined), so `world_edge_index` is not threaded
    into the branch loop.

    If `device` is provided, tensors are moved to that device; otherwise they
    stay on CPU (the DataLoader / .to() will handle the transfer).
    """
    cur_ref = ref_pos.astype(np.float32)
    cur_def = deformed_pos.astype(np.float32)

    for level, entry in enumerate(hierarchy):
        mean = coarse_edge_means[level] if level < len(coarse_edge_means) else None
        std = coarse_edge_stds[level] if level < len(coarse_edge_stds) else None

        if 'branches' in entry:
            for b, sub in enumerate(entry['branches']):
                m = mean[b] if isinstance(mean, list) else mean
                s = std[b] if isinstance(std, list) else std
                _attach_one_partition(
                    graph, f'{level}_{b}', sub, cur_ref, cur_def, m, s,
                    device, None, expose_anchors,
                )
            # A branched level is always terminal (build_multiscale_hierarchy
            # never chains a next level from it), so no cur_ref/cur_def update.
        else:
            coarse_ref, coarse_def, world_edge_index = _attach_one_partition(
                graph, str(level), entry, cur_ref, cur_def, mean, std,
                device, world_edge_index, expose_anchors,
            )
            cur_ref, cur_def = coarse_ref, coarse_def
