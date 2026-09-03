"""SIREN (sinusoidal representation network) trunk layers with the
paper-correct initialization (IMPLEMENTATION_PLAN.md section 8.1, following
Sitzmann et al. 2020). Used only by Point-DeepONet's query trunk.
"""

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """One `sin(omega0 * (W x + b))` layer with sine-specific init:

    - first layer: W ~ U(-1/fan_in, 1/fan_in)
    - hidden layers: W ~ U(-sqrt(6/fan_in)/omega0, sqrt(6/fan_in)/omega0)

    Excluded from the repo's generic `init_weights` (Kaiming) pass -- SIREN
    initialization is load-bearing for the sine activations to behave.
    """

    def __init__(self, in_features: int, out_features: int, omega0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features: int) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega0
            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class Siren(nn.Module):
    """Stack of `depth` SineLayers: in_features -> hidden -> ... -> hidden.

    The output width equals `hidden_features` (matching the branch context
    width so Point-DeepONet's early fusion `branch * trunk` is well-defined).
    """

    def __init__(self, in_features: int, hidden_features: int, depth: int,
                 omega0: float = 30.0):
        super().__init__()
        if depth < 1:
            raise ValueError(f"Siren depth must be >= 1, got {depth}.")
        layers = [SineLayer(in_features, hidden_features, omega0=omega0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega0=omega0, is_first=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
