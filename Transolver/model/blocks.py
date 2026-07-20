"""Transformer block and head (IMPLEMENTATION_PLAN.md section 6.5, Appendix A.2)."""

import torch
import torch.nn as nn

from model.physics_attention import PhysicsAttentionIrregular, make_tile_ranges


class FFN(nn.Module):
    """Linear(C, C*mlp_ratio) -> GELU -> Linear(C*mlp_ratio, C). No dropout:
    the official MLP class has none on this path (section 6.5)."""

    def __init__(self, dim: int, mlp_ratio: int = 1):
        super().__init__()
        hidden = dim * mlp_ratio
        self.linear_pre = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.linear_post = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.linear_post(self.act(self.linear_pre(x)))


class TransolverBlock(nn.Module):
    """Pre-norm block: h = h + Attn(LN(h)); h = h + FFN(LN(h)).

    The last block additionally carries the output head (ln_3 + Linear to
    out_dim) so the final prediction is produced inline, matching the official
    v1 structure -- functionally "after the final block" but literally part of
    it, not a separate module owned by the top-level model.
    """

    def __init__(self, num_heads: int, hidden_dim: int, slice_num: int,
                 dropout: float, mlp_ratio: int, last_layer: bool, out_dim: int,
                 temperature_init: float, temperature_min: float, temperature_max: float):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregular(
            dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            slice_num=slice_num, dropout=dropout,
            temperature_init=temperature_init, temperature_min=temperature_min,
            temperature_max=temperature_max,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, mlp_ratio)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, ptr, attention_kernel: str = 'naive', chunk_size: int = 0,
                use_checkpointing: bool = False):
        fx = fx + self.attn(self.ln_1(fx), ptr, attention_kernel, chunk_size, use_checkpointing)
        fx = fx + self.ffn(self.ln_2(fx))
        if self.last_layer:
            return self.head(self.ln_3(fx))
        return fx

    # ------------------------------------------------------------------
    # decoupled two-stage inference (section 11, Appendix A.4)
    # ------------------------------------------------------------------

    def compute_tokens(self, fx, ptr, chunk_size: int = 0):
        """Stage 1: LayerNorm + per-graph token accumulation only -- no
        attend, no deslice, no FFN. Returns a list of [H, M, D] token tensors,
        one per graph in ptr."""
        u = self.ln_1(fx)
        tokens_per_graph = []
        for i in range(ptr.shape[0] - 1):
            s, e = int(ptr[i].item()), int(ptr[i + 1].item())
            tile_ranges = make_tile_ranges(e - s, chunk_size)
            tokens_per_graph.append(self.attn.compute_layer_tokens(u[s:e], tile_ranges))
        return tokens_per_graph

    def forward_with_tokens(self, fx, ptr, tokens_per_graph, chunk_size: int = 0):
        """Stage 2: attend on a precomputed token cache + deslice + residual
        + FFN (+ head if last_layer). fx's graphs must align 1:1 with
        tokens_per_graph (same ptr length), but may hold different node
        coordinates than whatever produced the cache."""
        u = self.ln_1(fx)
        outs = []
        for i in range(ptr.shape[0] - 1):
            s, e = int(ptr[i].item()), int(ptr[i + 1].item())
            tile_ranges = make_tile_ranges(e - s, chunk_size)
            attn_out = self.attn.decode_with_tokens(u[s:e], tokens_per_graph[i], tile_ranges)
            outs.append(fx[s:e] + attn_out)
        fx2 = torch.cat(outs, dim=0)
        fx2 = fx2 + self.ffn(self.ln_2(fx2))
        if self.last_layer:
            return self.head(self.ln_3(fx2))
        return fx2
