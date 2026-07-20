"""Isolated Transolver-v1 profile for the paper's Elasticity benchmark.

The normal suite wrapper intentionally keeps the shared MeshGraphNets HDF5
contract.  The released Elasticity experiment instead consumes raw XY only,
uses unclamped slice temperature, and trains on decoded relative L2.  This
module exercises the repository's Physics-Attention implementation without
changing the default wrapper or its forward hot path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.blocks import FFN
from model.physics_attention import PhysicsAttentionIrregular


class PaperPhysicsAttentionIrregular(PhysicsAttentionIrregular):
    """Official irregular-mesh v1 temperature has no clamp."""

    def _clamped_temperature(self) -> torch.Tensor:
        return self.temperature.view(self.heads)


class PaperElasticityBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
        last_layer: bool,
    ) -> None:
        super().__init__()
        self.last_layer = bool(last_layer)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PaperPhysicsAttentionIrregular(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            slice_num=slice_num,
            dropout=dropout,
            temperature_init=0.5,
            temperature_min=0.1,
            temperature_max=5.0,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, mlp_ratio)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.ln_1(hidden), ptr, "naive", 0, False
        )
        hidden = hidden + self.ffn(self.ln_2(hidden))
        if self.last_layer:
            return self.head(self.ln_3(hidden))
        return hidden


class PaperElasticityTransolver(nn.Module):
    """Paper topology operating on raw ``[B,972,2]`` XY coordinates."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.slice_num = int(slice_num)
        self.mlp_ratio = int(mlp_ratio)
        self.dropout = float(dropout)

        self.preprocess = nn.Sequential(
            nn.Linear(2, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            PaperElasticityBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                slice_num=self.slice_num,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                last_layer=index == self.num_layers - 1,
            )
            for index in range(self.num_layers)
        ])

        # This ordering deliberately matches the release: attention constructors
        # first initialize slice projectors orthogonally, then the model-wide
        # truncated-normal pass overwrites every Linear, including those
        # projectors.
        self.apply(self._init_module)
        self.placeholder = nn.Parameter(
            (1.0 / self.hidden_dim) * torch.rand(self.hidden_dim)
        )

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"Expected raw XY [B,N,2], got {tuple(xy.shape)}")
        batch_size, num_points, _ = xy.shape
        hidden = self.preprocess(xy.reshape(batch_size * num_points, 2))
        hidden = hidden + self.placeholder[None, :]
        ptr = torch.arange(
            0,
            (batch_size + 1) * num_points,
            num_points,
            dtype=torch.long,
            device=xy.device,
        )
        for block in self.blocks:
            hidden = block(hidden, ptr)
        return hidden.reshape(batch_size, num_points)

    def export_model_config(self) -> dict[str, object]:
        return {
            "model_name": "transolver_paper_elasticity",
            "coordinate_input": "raw_xy",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "slice_num": self.slice_num,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "temperature": "learned_unclamped_initial_0.5",
            "slice_projector_final_init": "truncated_normal_std_0.02",
        }
