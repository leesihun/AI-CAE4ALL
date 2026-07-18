"""Normalization + coordinate-domain statistics for MeshGraphDataset (section 4.4).

Extends the MeshGraphNets/transolver pattern (element-weighted first/second
moments over the train split, optionally parallel across processes) with the
coordinate-domain statistics this repository's grid/latent adapters need:
per-axis extents (for active-axis resolution), the isotropic `position_scale`,
and the rotation-invariant in-plane radius used to keep augmented samples
inside a fixed grid box (IMPLEMENTATION_PLAN.md section 4.5).

No edge features are computed here: none of the four operator cores consume
MGN edge attributes (section 3).
"""

import multiprocessing as mp
from typing import Dict, List, Tuple

import h5py
import numpy as np

from general_modules.positional_features import compute_positional_features

# Sub-sample long trajectories: statistics converge well before 500 timesteps.
MAX_TIMESTEPS_FOR_STATS = 500

# Don't parallelize tiny datasets; spawn overhead dominates.
_MIN_SAMPLES_FOR_PARALLEL = 10


def finalize_moments(feature_sum, feature_sumsq, count):
    """Finalize element-weighted mean/std from sums and squared sums."""
    if count <= 0:
        raise ValueError("Cannot finalize normalization statistics with count <= 0")
    mean = (feature_sum / count).astype(np.float32)
    meansq = (feature_sumsq / count).astype(np.float32)
    var = np.maximum(meansq - mean ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, np.maximum(std, 1e-8)


def finalize_position_scale(sqnorm_sum: float, count: int) -> float:
    """RMS radius of per-sample-centered reference geometry (section 4.4)."""
    if count <= 0:
        raise ValueError("Cannot finalize position_scale with count <= 0")
    return float(np.sqrt(max(sqnorm_sum / count, 0.0)))


def finalize_axis_bounds(axis_min_raw: np.ndarray, axis_max_raw: np.ndarray,
                          position_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw (unscaled, per-sample-centered) axis bounds into
    pos_normalized units: dividing every node's centered coordinate by the
    same positive scalar `position_scale` preserves per-axis min/max order.
    """
    scale = max(position_scale, 1e-8)
    return axis_min_raw.astype(np.float64) / scale, axis_max_raw.astype(np.float64) / scale


def resolve_active_axes(axis_min_norm: np.ndarray, axis_max_norm: np.ndarray,
                         dimension_tolerance: float) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Axis k is active iff extent_k >= dimension_tolerance * max(extent) (section 4.4)."""
    extent = axis_max_norm - axis_min_norm
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        raise ValueError("Degenerate geometry: all coordinate axes have zero extent.")
    active = tuple(k for k in range(3) if extent[k] >= dimension_tolerance * max_extent)
    if len(active) not in (2, 3):
        raise ValueError(
            f"Resolved {len(active)} active axes (extents={extent.tolist()}); "
            "expected 2 or 3. Check dimension_tolerance."
        )
    return active, extent


def finalize_rot_invariant_radius(xy_radius_max_raw: float, position_scale: float) -> float:
    """Max in-plane (x, y) radius of any training node, in pos_normalized units.

    A Z-axis rotation/reflection preserves every node's distance from the
    (centered) origin in the x-y plane, so this value upper-bounds the
    in-plane radius of every augmented sample too (section 4.5).
    """
    scale = max(position_scale, 1e-8)
    return float(xy_radius_max_raw) / scale


def _empty_accumulators(node_dim: int, output_dim: int) -> Dict:
    return {
        'node_sum': np.zeros(node_dim, dtype=np.float64),
        'node_sumsq': np.zeros(node_dim, dtype=np.float64),
        'node_count': 0,
        'delta_sum': np.zeros(output_dim, dtype=np.float64),
        'delta_sumsq': np.zeros(output_dim, dtype=np.float64),
        'delta_count': 0,
        'delta_min': np.full(output_dim, np.inf, dtype=np.float64),
        'delta_max': np.full(output_dim, -np.inf, dtype=np.float64),
        'pos_sqnorm_sum': 0.0,
        'pos_count': 0,
        'axis_min_raw': np.full(3, np.inf, dtype=np.float64),
        'axis_max_raw': np.full(3, -np.inf, dtype=np.float64),
        'xy_radius_max_raw': 0.0,
        'num_samples_processed': 0,
    }


def _process_sample_chunk(h5_file: str, sample_ids: List[int], input_dim: int,
                          output_dim: int, num_timesteps: int,
                          num_pos_features: int = 0) -> Dict:
    """Accumulate stats over one chunk of samples. Runs in a pool worker."""
    acc = _empty_accumulators(input_dim + num_pos_features, output_dim)

    with h5py.File(h5_file, 'r') as f:
        for sid in sample_ids:
            try:
                data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

                if num_timesteps > 1:
                    n_t = min(MAX_TIMESTEPS_FOR_STATS, num_timesteps)
                    timesteps = np.linspace(0, num_timesteps - 1, n_t, dtype=int)
                else:
                    timesteps = [0]

                ref_pos_0 = data[:3, 0, :].T  # [N, 3]

                pos_feat = None
                if num_pos_features > 0:
                    mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]
                    edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
                    pos_feat = compute_positional_features(ref_pos_0, edge_idx, num_pos_features)

                # Coordinate-domain statistics depend on reference geometry only.
                center = ref_pos_0.mean(axis=0)
                pos_centered = ref_pos_0 - center
                acc['pos_sqnorm_sum'] += float(np.sum(pos_centered ** 2))
                acc['pos_count'] += pos_centered.shape[0]
                np.minimum(acc['axis_min_raw'], pos_centered.min(axis=0), out=acc['axis_min_raw'])
                np.maximum(acc['axis_max_raw'], pos_centered.max(axis=0), out=acc['axis_max_raw'])
                xy_radius = np.sqrt(pos_centered[:, 0] ** 2 + pos_centered[:, 1] ** 2)
                acc['xy_radius_max_raw'] = max(acc['xy_radius_max_raw'], float(xy_radius.max()))

                for t in timesteps:
                    if num_timesteps == 1:
                        # Static: physical features are zeros (model sees zeros)
                        node_feat = np.zeros((data.shape[2], input_dim), dtype=np.float64)
                    else:
                        node_feat = data[3:3 + input_dim, t, :].T  # [N, input_dim]

                    if pos_feat is not None:
                        node_feat = np.concatenate([node_feat, pos_feat], axis=1)

                    acc['node_sum'] += np.sum(node_feat, axis=0)
                    acc['node_sumsq'] += np.sum(node_feat ** 2, axis=0)
                    acc['node_count'] += node_feat.shape[0]

                # Target deltas
                if num_timesteps > 1:
                    n_d = min(MAX_TIMESTEPS_FOR_STATS, num_timesteps - 1)
                    delta_timesteps = np.linspace(0, num_timesteps - 2, n_d, dtype=int)
                    deltas = (
                        (data[3:3 + output_dim, t + 1, :] - data[3:3 + output_dim, t, :]).T
                        for t in delta_timesteps
                    )
                else:
                    deltas = (data[3:3 + output_dim, 0, :].T,)

                for delta in deltas:
                    acc['delta_sum'] += np.sum(delta, axis=0)
                    acc['delta_sumsq'] += np.sum(delta ** 2, axis=0)
                    acc['delta_min'] = np.minimum(acc['delta_min'], np.min(delta, axis=0))
                    acc['delta_max'] = np.maximum(acc['delta_max'], np.max(delta, axis=0))
                    acc['delta_count'] += delta.shape[0]

                acc['num_samples_processed'] += 1

            except Exception as e:
                print(f"Warning: Failed to process sample {sid}: {e}")
                continue

    return acc


def _merge_accumulators(results: List[Dict], node_dim: int, output_dim: int) -> Dict:
    merged = _empty_accumulators(node_dim, output_dim)
    for r in results:
        if r['num_samples_processed'] == 0:
            continue
        for key in ('node_sum', 'node_sumsq', 'delta_sum', 'delta_sumsq'):
            merged[key] += r[key]
        merged['pos_sqnorm_sum'] += r['pos_sqnorm_sum']
        for key in ('node_count', 'delta_count', 'pos_count', 'num_samples_processed'):
            merged[key] += r[key]
        np.minimum(merged['delta_min'], r['delta_min'], out=merged['delta_min'])
        np.maximum(merged['delta_max'], r['delta_max'], out=merged['delta_max'])
        np.minimum(merged['axis_min_raw'], r['axis_min_raw'], out=merged['axis_min_raw'])
        np.maximum(merged['axis_max_raw'], r['axis_max_raw'], out=merged['axis_max_raw'])
        merged['xy_radius_max_raw'] = max(merged['xy_radius_max_raw'], r['xy_radius_max_raw'])
    return merged


def compute_normalization_stats(h5_file: str, sample_ids: List[int], input_dim: int,
                                output_dim: int, num_timesteps: int,
                                num_pos_features: int, use_parallel: bool = True) -> Dict:
    """Compute raw stat sums over a sample split, in parallel when worthwhile.

    Returns the accumulator dict (see `_empty_accumulators`); use
    `finalize_moments` on the sum/sumsq/count triples for mean/std,
    `finalize_position_scale` on pos_sqnorm_sum/pos_count for position_scale,
    `finalize_axis_bounds`+`resolve_active_axes` for the coordinate domain,
    and `finalize_rot_invariant_radius` for the augmentation-safe grid radius.
    """
    n = len(sample_ids)
    if use_parallel and n >= _MIN_SAMPLES_FOR_PARALLEL:
        # Cap at 8: spawn-context pool workers each re-import torch, so a
        # large pool risks OOM-killed workers and a permanent starmap hang.
        num_workers = max(1, min(8, int(mp.cpu_count() * 0.45)))
    else:
        num_workers = 1

    if num_workers <= 1:
        reason = ('disabled (use_parallel_stats=False)' if not use_parallel
                  else f'serial ({n} samples < {_MIN_SAMPLES_FOR_PARALLEL})')
        print(f'  Normalization stats: {reason}')
        return _process_sample_chunk(h5_file, sample_ids, input_dim, output_dim,
                                     num_timesteps, num_pos_features)

    print(f'  Normalization stats: {num_workers} parallel workers for {n} samples')
    chunk_size = max(1, n // num_workers)
    chunks = [sample_ids[i:i + chunk_size] for i in range(0, n, chunk_size)]
    try:
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(_process_sample_chunk, [
                (h5_file, chunk, input_dim, output_dim, num_timesteps, num_pos_features)
                for chunk in chunks
            ])
        merged = _merge_accumulators(results, input_dim + num_pos_features, output_dim)
        print(f"  Successfully processed {merged['num_samples_processed']}/{n} samples")
        if merged['num_samples_processed'] == 0:
            raise RuntimeError("No samples were successfully processed in parallel mode")
        return merged
    except Exception as e:
        print(f'  Warning: Parallel processing failed ({e}), falling back to serial')
        return _process_sample_chunk(h5_file, sample_ids, input_dim, output_dim,
                                     num_timesteps, num_pos_features)
