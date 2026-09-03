"""Paper-faithful DeepONet for the 2D fractional-Laplacian benchmark.

This module is intentionally isolated from ``model.factory`` and the normal
mesh runtime.  The paper benchmark has a fixed 225-value branch vector and a
three-component trunk query ``(x, y, alpha)``; forcing it through the mesh
splat/temporal contracts would change the operator being validated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _truncated_xavier_(linear: nn.Linear, generator: torch.Generator) -> None:
    """Match the released TensorFlow truncated-normal Xavier initializer."""
    std = math.sqrt(2.0 / (linear.in_features + linear.out_features))
    nn.init.trunc_normal_(
        linear.weight,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
        generator=generator,
    )
    nn.init.zeros_(linear.bias)


class FractionalLaplacianDeepONet(nn.Module):
    """Released 2D DeepONet topology: linear branch basis, tanh trunk basis."""

    def __init__(
        self,
        branch_dim: int = 225,
        query_dim: int = 3,
        width: int = 60,
        seed: int = 12345,
    ) -> None:
        super().__init__()
        self.branch_dim = int(branch_dim)
        self.query_dim = int(query_dim)
        self.width = int(width)

        # The released ``layers_u=[225]+[60]*3`` path applies tanh to the
        # first two branch layers and leaves the final basis layer linear.
        self.branch_layers = nn.ModuleList([
            nn.Linear(self.branch_dim, self.width),
            nn.Linear(self.width, self.width),
            nn.Linear(self.width, self.width),
        ])

        # Its trunk uses neural_net2, which applies tanh after all three
        # layers, including the final basis layer.
        self.trunk_layers = nn.ModuleList([
            nn.Linear(self.query_dim, self.width),
            nn.Linear(self.width, self.width),
            nn.Linear(self.width, self.width),
        ])
        self.bias = nn.Parameter(torch.zeros(1))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for layer in [*self.branch_layers, *self.trunk_layers]:
            _truncated_xavier_(layer, generator)

    def encode_branch(self, branch_values: torch.Tensor) -> torch.Tensor:
        hidden = branch_values
        hidden = torch.tanh(self.branch_layers[0](hidden))
        hidden = torch.tanh(self.branch_layers[1](hidden))
        return self.branch_layers[2](hidden)

    def encode_trunk(self, queries: torch.Tensor) -> torch.Tensor:
        hidden = queries
        for layer in self.trunk_layers:
            hidden = torch.tanh(layer(hidden))
        return hidden

    def decode_encoded(
        self, branch_code: torch.Tensor, trunk_code: torch.Tensor
    ) -> torch.Tensor:
        if branch_code.shape != trunk_code.shape:
            raise ValueError(
                "Paired branch and trunk codes must have identical shapes, got "
                f"{tuple(branch_code.shape)} and {tuple(trunk_code.shape)}"
            )
        return torch.sum(branch_code * trunk_code, dim=-1, keepdim=True) + self.bias

    def forward(
        self, branch_values: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        return self.decode_encoded(
            self.encode_branch(branch_values), self.encode_trunk(queries)
        )

    def export_model_config(self) -> dict[str, int | str]:
        return {
            "model_name": "deeponet_fractional_laplacian_2d",
            "branch_dim": self.branch_dim,
            "query_dim": self.query_dim,
            "width": self.width,
            "branch_activation_pattern": "tanh,tanh,linear",
            "trunk_activation_pattern": "tanh,tanh,tanh",
        }
