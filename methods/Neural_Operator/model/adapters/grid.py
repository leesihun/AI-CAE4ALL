"""Deterministic multilinear splat/sample grid adapter (IMPLEMENTATION_PLAN.md
section 7.2). Shared by normal DeepONet (fixed-sensor branch) and FNO (grid
core). Pure torch, differentiable w.r.t. `values`, permutation-invariant in
node order, never mixes nodes across graphs.

Axis-order convention (documented once, here, because this is the single
most bug-prone part of the whole repository):

  * `resolution` and the grid tensors this module returns store spatial axes
    in ascending *active-axis* order: tensor dim 2 <-> active axis 0 (x),
    dim 3 <-> active axis 1 (y), dim 4 (if d==3) <-> active axis 2 (z).
  * `F.grid_sample` uses the opposite convention: coordinate channel 0 (x)
    addresses the LAST input spatial dim, channel d-1 addresses the FIRST
    (see torch docs for `grid_sample`). `sample()` below reverses the
    *input tensor's* spatial dims before the call so that the coordinate
    channel order can stay untouched (x, y, [z]) end to end.
"""

import itertools
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_strides(resolution: Sequence[int]) -> list:
    """Row-major (C-order) strides over `resolution`, last axis fastest."""
    d = len(resolution)
    strides = [1] * d
    for k in range(d - 2, -1, -1):
        strides[k] = strides[k + 1] * resolution[k + 1]
    return strides


def splat(values: torch.Tensor, c01: torch.Tensor, batch: torch.Tensor,
          num_graphs: int, resolution: Sequence[int]
          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multilinear (bilinear/trilinear) splat of ragged node values onto a
    batch of regular grids. O(N * 2**d), fp32 accumulation.

    Args:
        values: [sum_N, C] float, per-node channels to splat.
        c01:    [sum_N, d] float in [0, 1], d == len(resolution).
        batch:  [sum_N] long in [0, num_graphs).
        resolution: length-d sequence of per-axis grid sizes (each >= 2).

    Returns:
        grid:      [B, C, *resolution] weighted-mean splat (zero in empty cells)
        occupancy: [B, 1, *resolution] 1.0 where any node contributed, else 0.0
        density:   [B, 1, *resolution] log1p(total splat weight) per cell
    """
    device = values.device
    d = len(resolution)
    if c01.shape[1] != d:
        raise ValueError(f"c01 has {c01.shape[1]} columns, resolution has {d} axes.")
    res = list(resolution)
    if any(r < 2 for r in res):
        raise ValueError(f"Every grid resolution axis must be >= 2, got {res}.")
    C = values.shape[1]

    res_t = torch.tensor(res, device=device, dtype=torch.float32)
    zeros_d = torch.zeros(d, device=device, dtype=torch.float32)
    max_idx = torch.tensor([r - 2 for r in res], device=device, dtype=torch.float32)

    t = c01.float() * (res_t - 1)
    i0 = t.floor().clamp(min=zeros_d, max=max_idx).long()
    frac = (t - i0.float()).clamp(0.0, 1.0)

    strides = compute_strides(res)
    strides_t = torch.tensor(strides, device=device, dtype=torch.long)
    numel_grid = 1
    for r in res:
        numel_grid *= r

    val_acc = torch.zeros(num_graphs * numel_grid, C, device=device, dtype=torch.float32)
    w_acc = torch.zeros(num_graphs * numel_grid, device=device, dtype=torch.float32)
    batch_l = batch.long()

    for corner in itertools.product([0, 1], repeat=d):
        corner_t = torch.tensor(corner, device=device, dtype=torch.long)
        idx = i0 + corner_t  # [N, d]; i0 in [0, R-2] so idx in [0, R-1] always
        w = torch.ones(values.shape[0], device=device, dtype=torch.float32)
        for k in range(d):
            w = w * (frac[:, k] if corner[k] == 1 else (1.0 - frac[:, k]))
        flat_local = (idx * strides_t).sum(dim=1)  # [N]
        flat_global = batch_l * numel_grid + flat_local
        val_acc.index_add_(0, flat_global, values.float() * w.unsqueeze(-1))
        w_acc.index_add_(0, flat_global, w)

    grid_flat = val_acc / w_acc.clamp_min(1e-10).unsqueeze(-1)  # [B*numel, C]
    occupancy_flat = (w_acc > 0).float()
    density_flat = torch.log1p(w_acc)

    grid = grid_flat.view(num_graphs, *res, C)
    perm = [0, d + 1] + list(range(1, d + 1))
    grid = grid.permute(*perm).contiguous()  # [B, C, *res]

    occupancy = occupancy_flat.view(num_graphs, *res).unsqueeze(1).contiguous()
    density = density_flat.view(num_graphs, *res).unsqueeze(1).contiguous()

    return grid, occupancy, density


def sample(grid: torch.Tensor, c01: torch.Tensor, batch: torch.Tensor,
           num_graphs: int) -> torch.Tensor:
    """Grid -> ragged per-node values via bilinear/trilinear `F.grid_sample`.

    Args:
        grid:  [B, C, *resolution], spatial dims in ascending active-axis
               order (dim2=axis0, dim3=axis1, (dim4=axis2)) -- the same
               convention `splat()` produces.
        c01:   [Nq, d] in [0, 1], same active-axis order as `grid`.
        batch: [Nq] long in [0, num_graphs).

    Returns:
        values: [Nq, C]
    """
    d = grid.dim() - 2
    C = grid.shape[1]
    out = torch.zeros(c01.shape[0], C, device=grid.device, dtype=grid.dtype)
    if c01.shape[0] == 0:
        return out

    spatial_perm = [0, 1] + list(reversed(range(2, 2 + d)))
    batch_l = batch.long()

    for g in range(num_graphs):
        mask = batch_l == g
        if not mask.any():
            continue
        pts = c01[mask].float()
        grid_coords = 2 * pts - 1  # [-1, 1], channel order unchanged (x, y, [z])

        inp = grid[g:g + 1].permute(*spatial_perm)  # reverse spatial dims only
        if d == 2:
            gcoord = grid_coords.view(1, 1, -1, 2)
        elif d == 3:
            gcoord = grid_coords.view(1, 1, 1, -1, 3)
        else:
            raise ValueError(f"grid adapter only supports d in (2, 3), got {d}.")

        out_g = F.grid_sample(inp, gcoord, mode='bilinear', align_corners=True,
                              padding_mode='border')
        out_g = out_g.reshape(C, -1).T
        out[mask] = out_g.to(out.dtype)

    return out
