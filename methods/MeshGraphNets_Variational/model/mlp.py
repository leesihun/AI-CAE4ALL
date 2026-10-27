import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


class AmpLayerNorm(nn.LayerNorm):
    """nn.LayerNorm that, with `in_amp_dtype` set, normalizes a bf16/fp16 input in that dtype.

    Autocast puts layer_norm on its fp32 list, so every MLP output is upcast,
    normalized in fp32, and kept for backward in fp32: about a fifth of the
    MGN-V step's kernel time on ex3 (2026-10-27 profile). The half-precision
    CUDA kernel still accumulates its statistics in fp32; what changes is that
    the output and the tensor saved for backward are bf16/fp16 (measured on a
    3090: step 1.20x with checkpointing, 1.06x without; memory -19%). Set by the owning model from `amp_layernorm`.
    The state dict is nn.LayerNorm's.
    """

    in_amp_dtype = False

    def forward(self, x):
        if (self.in_amp_dtype and x.dtype in (torch.bfloat16, torch.float16)
                and torch.is_autocast_enabled(x.device.type)):
            with torch.autocast(x.device.type, enabled=False):
                return F.layer_norm(x, self.normalized_shape,
                                    None if self.weight is None else self.weight.to(x.dtype),
                                    None if self.bias is None else self.bias.to(x.dtype),
                                    self.eps)
        return super().forward(x)


def build_mlp(in_size, hidden_size, out_size, layer_norm=True):
    """Two-hidden-layer SiLU MLP following the MeshGraphNets architecture.

    LayerNorm (AmpLayerNorm) is appended on the output when layer_norm=True
    (all non-decoder uses).
    """
    layers = [
        nn.Linear(in_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, out_size),
    ]
    if layer_norm:
        layers.append(AmpLayerNorm(normalized_shape=out_size))
    return nn.Sequential(*layers)
