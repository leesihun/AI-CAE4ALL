"""Shared coordinate-domain adapter (IMPLEMENTATION_PLAN.md section 7.1).

Wraps the train-derived active axes and global grid bounds (computed once in
general_modules/dataset_stats.py / mesh_dataset.py) into the runtime
`[0,1]^d` mapping used by every grid/point/GINO adapter. Deterministic,
checkpointed, and never recomputed from a single sample: per-sample geometry
uses `pos_normalized` (from the dataset), this module owns only the shared
*domain* those coordinates are mapped into.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class CoordinateDomain:
    active_axes: Tuple[int, ...]      # subset of (0, 1, 2) meaning (x, y, z)
    grid_bound_min: torch.Tensor      # [d] float32, in pos_normalized units
    grid_bound_max: torch.Tensor      # [d] float32
    out_of_bounds_policy: str = 'error'   # 'error' | 'clamp'

    @property
    def dim(self) -> int:
        return len(self.active_axes)

    @classmethod
    def from_dataset(cls, dataset, out_of_bounds_policy: str = 'error') -> "CoordinateDomain":
        active = tuple(dataset.active_axes)
        gmin = torch.tensor([dataset.grid_bound_min[k] for k in active], dtype=torch.float32)
        gmax = torch.tensor([dataset.grid_bound_max[k] for k in active], dtype=torch.float32)
        return cls(active_axes=active, grid_bound_min=gmin, grid_bound_max=gmax,
                   out_of_bounds_policy=out_of_bounds_policy)

    def to_dict(self) -> dict:
        return {
            'active_axes': list(self.active_axes),
            'grid_bound_min': self.grid_bound_min.tolist(),
            'grid_bound_max': self.grid_bound_max.tolist(),
            'out_of_bounds_policy': self.out_of_bounds_policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoordinateDomain":
        return cls(
            active_axes=tuple(d['active_axes']),
            grid_bound_min=torch.tensor(d['grid_bound_min'], dtype=torch.float32),
            grid_bound_max=torch.tensor(d['grid_bound_max'], dtype=torch.float32),
            out_of_bounds_policy=d.get('out_of_bounds_policy', 'error'),
        )

    def select_active(self, pos_normalized: torch.Tensor) -> torch.Tensor:
        """[N, 3] -> [N, d], keeping only active-axis columns in axis order."""
        idx = torch.tensor(self.active_axes, device=pos_normalized.device, dtype=torch.long)
        return pos_normalized.index_select(1, idx)

    def to_unit_box(self, pos_normalized: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Map active-axis pos_normalized coordinates into [0, 1]^d.

        Returns (c01 [N, d], out_of_bounds_count). Honors `out_of_bounds_policy`:
        'error' raises when any coordinate falls (measurably) outside [0, 1]
        after mapping; 'clamp' clips into range and still reports the count.
        """
        coords = self.select_active(pos_normalized)
        gmin = self.grid_bound_min.to(coords.device)
        gmax = self.grid_bound_max.to(coords.device)
        extent = (gmax - gmin).clamp_min(1e-8)
        c01 = (coords - gmin) / extent

        eps = 1e-5
        oob_mask = (c01 < -eps) | (c01 > 1 + eps)
        oob_count = int(oob_mask.any(dim=1).sum().item())

        if oob_count > 0:
            if self.out_of_bounds_policy == 'error':
                raise ValueError(
                    f"{oob_count} point(s) fall outside the training coordinate "
                    f"domain [0,1]^{self.dim} (grid_bound_min={gmin.tolist()}, "
                    f"grid_bound_max={gmax.tolist()}). Set out_of_bounds_policy "
                    "clamp to tolerate this, or increase grid_padding."
                )
            c01 = c01.clamp(0.0, 1.0)

        return c01, oob_count
