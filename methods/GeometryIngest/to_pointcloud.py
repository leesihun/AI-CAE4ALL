"""Node set -> fixed-size point cloud for the operator models.

MeshGraphNets consumes the full node set + edges. DeepONet/FNO/GINO/Transolver
consume the nodes as a *point cloud* and several want a consistent point count,
so this offers a deterministic resample. Farthest-point sampling (FPS) gives an
even geometric spread; random subsampling is faster for large N.

FPS here is the plain O(num_samples * N) greedy loop (same idea as the FPS used
in MeshGraphNets multiscale). Fine for a few-thousand-point target; swap for a
KD-tree/torch_cluster variant if N grows large.
"""

from __future__ import annotations

import numpy as np


def farthest_point_sample(points: np.ndarray, num_samples: int, seed: int = 0) -> np.ndarray:
    """Return row indices of an FPS subset of ``points`` (deterministic given seed)."""
    n = points.shape[0]
    if num_samples <= 0 or num_samples >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = np.empty(num_samples, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    min_dist = np.full(n, np.inf)
    last = idx[0]
    for i in range(1, num_samples):
        d = np.sum((points - points[last]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        last = int(np.argmax(min_dist))
        idx[i] = last
    return idx


def resample(coords: np.ndarray, num_points: int, method: str = "fps",
             seed: int = 0) -> np.ndarray:
    """Return coords subsampled to ``num_points`` (0 = keep all)."""
    if num_points <= 0 or num_points >= coords.shape[0]:
        return coords
    if method == "fps":
        idx = farthest_point_sample(coords, num_points, seed=seed)
    elif method == "random":
        idx = np.sort(np.random.default_rng(seed).choice(
            coords.shape[0], size=num_points, replace=False))
    else:
        raise ValueError(f"unknown resample method: {method}")
    return coords[idx]
