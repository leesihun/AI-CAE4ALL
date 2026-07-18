"""Native N-dimensional spectral convolution (IMPLEMENTATION_PLAN.md section
8.3, A.4). Implements the standard FNO corner-truncation scheme (Li et al.
2021) without depending on the external `neuraloperator` package (section 0):
real FFT over the spatial dims, `2**(d-1)` learned weight blocks covering
every sign combination of the non-final axes (the final axis is halved by
the real FFT itself, so it never needs a negative-frequency corner).

Weights are stored as real tensors with a trailing size-2 dimension and
viewed as complex only inside `forward`, entirely so `torch.optim.AdamW(...,
fused=True)` keeps working (fused AdamW rejects complex parameters).
"""

import itertools
from typing import Sequence

import torch
import torch.nn as nn


def validate_fno_modes(modes: Sequence[int], resolution: Sequence[int]) -> None:
    d = len(resolution)
    if len(modes) != d:
        raise ValueError(f"modes has {len(modes)} entries, expected {d} (operator_dim).")
    for k in range(d):
        if modes[k] < 1:
            raise ValueError(f"modes[{k}] must be >= 1, got {modes[k]}.")
        if k < d - 1:
            limit = resolution[k] // 2
            if modes[k] > limit:
                raise ValueError(
                    f"modes[{k}]={modes[k]} exceeds the Nyquist limit {limit} for "
                    f"resolution[{k}]={resolution[k]} (non-final axis needs +/- corners "
                    "that must not overlap)."
                )
        else:
            limit = resolution[k] // 2 + 1
            if modes[k] > limit:
                raise ValueError(
                    f"modes[{k}]={modes[k]} exceeds the real-FFT limit {limit} for "
                    f"resolution[{k}]={resolution[k]} (final/rfft axis)."
                )


class SpectralConvNd(nn.Module):
    """N-dimensional (d in {2, 3}) spectral convolution, one learned mode
    truncation per corner of the low/high-frequency lattice.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: Sequence[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(int(m) for m in modes)
        self.d = len(self.modes)
        if self.d not in (2, 3):
            raise ValueError(f"SpectralConvNd only supports d in (2, 3), got {self.d}.")

        self._corners = list(itertools.product([1, -1], repeat=self.d - 1))
        self.n_blocks = len(self._corners)

        scale = 1.0 / (in_channels * out_channels)
        weight_shape = (self.n_blocks, in_channels, out_channels, *self.modes, 2)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape))

    def _corner_slices(self, signs) -> tuple:
        slices = []
        for k, s in enumerate(signs):
            m = self.modes[k]
            slices.append(slice(0, m) if s == 1 else slice(-m, None))
        slices.append(slice(0, self.modes[-1]))  # final axis: rfft, low-frequency only
        return tuple(slices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, Cin, *R] real -> [B, Cout, *R] real."""
        spatial_dims = tuple(range(2, 2 + self.d))
        resolution = x.shape[2:]
        orig_dtype = x.dtype

        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            X = torch.fft.rfftn(xf, dim=spatial_dims, norm='ortho')
            freq_shape = X.shape[2:]
            Y = torch.zeros(x.shape[0], self.out_channels, *freq_shape,
                            dtype=torch.cfloat, device=x.device)
            for b_idx, signs in enumerate(self._corners):
                sl = self._corner_slices(signs)
                idx = (slice(None), slice(None)) + sl
                X_corner = X[idx]                                            # [B,Cin,*modes]
                W = torch.view_as_complex(self.weight[b_idx].contiguous())   # [Cin,Cout,*modes]
                Y[idx] = torch.einsum('bi...,io...->bo...', X_corner, W)
            out = torch.fft.irfftn(Y, s=resolution, dim=spatial_dims, norm='ortho')

        return out.to(orig_dtype)
