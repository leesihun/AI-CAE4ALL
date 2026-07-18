"""Deterministic fixed-size sensor sampling for Point-DeepONet (section 7.3).

Sampling affects only the PointNet branch input; every query node is still
supervised by the shared loss regardless of which nodes were sampled.
"""

from typing import Optional

import torch


def stable_hash(*parts: int) -> int:
    """Deterministic 63-bit hash, independent of Python's per-process salted
    `hash()` (which must never be used for reproducible seeding, since it
    changes between interpreter runs)."""
    h = 1469598103934665603  # FNV offset basis (truncated to fit int63 math below)
    for p in parts:
        h = (h ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) * 1099511628211
        h &= (1 << 63) - 1
    return h


class PointSampler:
    """Fixed-size sensor selection, section 7.3.

    `point_sensor_count 0` is handled by callers as an explicit "use all
    nodes" ablation -- this class is only invoked when count > 0.
    """

    def __init__(self, sensor_count: int, base_seed: int = 0,
                 resample_each_epoch: bool = True):
        if sensor_count <= 0:
            raise ValueError("PointSampler requires sensor_count > 0; "
                             "point_sensor_count 0 is handled by the caller as an ablation.")
        self.sensor_count = sensor_count
        self.base_seed = base_seed
        self.resample_each_epoch = resample_each_epoch
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def sample_indices(self, num_nodes: int, sample_id: int, time_idx: Optional[int],
                       training: bool, device=None) -> torch.Tensor:
        """Return [min(M, ...)] long indices into a graph with `num_nodes` nodes.

        Without replacement when num_nodes >= M; with replacement otherwise.
        Deterministic: validation/inference always use epoch=-1 regardless of
        `resample_each_epoch`; training uses the sampler's current epoch
        unless `resample_each_epoch` is False, in which case epoch is pinned
        to 0. DDP rank is never part of the seed (section 7.3).
        """
        epoch_for_seed = 0
        if training and self.resample_each_epoch:
            epoch_for_seed = self._epoch
        elif not training:
            epoch_for_seed = -1

        time_key = -1 if time_idx is None else int(time_idx)
        seed = stable_hash(self.base_seed, epoch_for_seed, int(sample_id), time_key)
        seed = seed % (2 ** 31 - 1)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)

        m = self.sensor_count
        if num_nodes >= m:
            idx = torch.randperm(num_nodes, generator=gen)[:m]
        else:
            idx = torch.randint(0, num_nodes, (m,), generator=gen)
        if device is not None:
            idx = idx.to(device)
        return idx
