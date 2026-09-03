"""Independent fp64 numpy reference for the 2D grid splat (section 7.2 / 16),
used to catch axis-order and weighting bugs in model/adapters/grid.py without
sharing any code with it.
"""

import numpy as np


def reference_splat_2d(values: np.ndarray, c01: np.ndarray, resolution) -> np.ndarray:
    """Pure fp64 loop implementation of the bilinear splat for a single graph.

    values: [N, C], c01: [N, 2] in [0, 1], resolution: (Rx, Ry)
    returns grid [C, Rx, Ry] (weighted mean, zero in empty cells) -- same
    axis convention as model.adapters.grid.splat: dim0=axis0(x), dim1=axis1(y).
    """
    values = values.astype(np.float64)
    c01 = c01.astype(np.float64)
    Rx, Ry = resolution
    C = values.shape[1]
    val_acc = np.zeros((Rx, Ry, C), dtype=np.float64)
    w_acc = np.zeros((Rx, Ry), dtype=np.float64)

    tx = c01[:, 0] * (Rx - 1)
    ty = c01[:, 1] * (Ry - 1)
    ix0 = np.clip(np.floor(tx).astype(int), 0, Rx - 2)
    iy0 = np.clip(np.floor(ty).astype(int), 0, Ry - 2)
    fx = np.clip(tx - ix0, 0.0, 1.0)
    fy = np.clip(ty - iy0, 0.0, 1.0)

    for cx in (0, 1):
        for cy in (0, 1):
            wx = fx if cx == 1 else (1 - fx)
            wy = fy if cy == 1 else (1 - fy)
            w = wx * wy
            xi = ix0 + cx
            yi = iy0 + cy
            for n in range(values.shape[0]):
                val_acc[xi[n], yi[n]] += values[n] * w[n]
                w_acc[xi[n], yi[n]] += w[n]

    grid = np.zeros((Rx, Ry, C), dtype=np.float64)
    nonzero = w_acc > 1e-10
    grid[nonzero] = val_acc[nonzero] / w_acc[nonzero, None]
    return np.transpose(grid, (2, 0, 1))  # [C, Rx, Ry]
