"""The MLP surrogate: a plain fully-connected N-in -> M-out regressor."""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}

_OUTPUT_ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


def _norm_layer(kind: str, width: int) -> nn.Module | None:
    if kind == "batch":
        return nn.BatchNorm1d(width)
    if kind == "layer":
        return nn.LayerNorm(width)
    return None


class MLP(nn.Module):
    """Linear -> [Norm] -> Activation -> [Dropout] blocks, then a linear head."""

    def __init__(
        self,
        input_var: int,
        output_var: int,
        hidden_layers: list[int],
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = input_var
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            norm_layer = _norm_layer(norm, width)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, output_var))
        layers.append(_OUTPUT_ACTIVATIONS[output_activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(arch: dict) -> MLP:
    """Rebuild a model from the architecture block stored in a checkpoint/config."""
    return MLP(
        input_var=int(arch["input_var"]),
        output_var=int(arch["output_var"]),
        hidden_layers=list(arch["hidden_layers"]),
        activation=str(arch.get("activation", "gelu")),
        dropout=float(arch.get("dropout", 0.0)),
        norm=str(arch.get("norm", "none")),
        output_activation=str(arch.get("output_activation", "none")),
    )
