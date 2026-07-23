"""Physics-Attention for irregular meshes: packed, ptr-segmented, dual-kernel.

Implements IMPLEMENTATION_PLAN.md section 6.2/6.3 and Appendix A.1: one
architecture (v1's per-head slicing) with two numerically exact kernels
sharing a single v1-layout state dict.

- `naive`: project-then-aggregate, matching the official v1 order. Fewer
  N-scaled FLOPs; default at small-to-medium mesh sizes.
- `slice_space`: aggregate-then-project (Transolver-3's reformulation), in its
  general two-pass chunked form. `chunk_size <= 0` degenerates to a single
  tile (the whole graph), which is mathematically identical to `chunk_size`
  covering the graph in one pass.

Both kernels operate per graph (segmented by `ptr`) so no computation ever
mixes nodes across graphs in a batch.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

EPS = 1e-5


def _shard_all_reduce_sum(num: torch.Tensor, den: torch.Tensor, group):
    """Differentiable SUM all-reduce of the slice aggregates across a node-shard
    group (IMPLEMENTATION_PLAN.md 6.6 Phase 7).

    Under node sharding each rank holds a disjoint node subset of ONE graph, so
    its (num, den) are partial sums over the reduction dimension. Because both
    are additive over nodes, summing them across ranks reproduces the exact
    whole-graph aggregates -- tokens then match single-process bit-for-bit
    (the EPS in num/(den+EPS) is applied once to the global den, as it is
    single-process). The reduce is autograd-aware so the token pathway's
    cross-shard gradients flow; the remaining per-rank parameter gradients are
    SUM-reduced once after backward by the shard launcher (see
    training_profiles/sharded_training.py). This function is only reached when a
    group is set; the single-process kernel path is byte-unchanged.
    """
    import torch.distributed.nn as dist_nn
    # Pack into one collective so num and den reduce in a single call.
    H, M, D = num.shape
    flat = torch.cat([num.reshape(H * M * D), den.reshape(H * M)], dim=0)
    flat = dist_nn.all_reduce(flat, group=group)  # op defaults to SUM
    num = flat[:H * M * D].reshape(H, M, D)
    den = flat[H * M * D:].reshape(H, M)
    return num, den


def _stable(x: torch.Tensor) -> torch.Tensor:
    """Promote fp16/bf16 to fp32 for numerically sensitive reductions
    (section 6.4). fp32 and fp64 inputs pass through unchanged, so this is a
    no-op for the fp64 parity harness and only "widens" autocast activations.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def make_tile_ranges(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Split [0, n) into contiguous tiles of at most chunk_size nodes.
    chunk_size <= 0 (or >= n) yields a single tile: the whole range."""
    if chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    ranges = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        ranges.append((start, end))
        start = end
    return ranges


class PhysicsAttentionIrregular(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, slice_num: int,
                 dropout: float = 0.0, temperature_init: float = 0.5,
                 temperature_min: float = 0.1, temperature_max: float = 5.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.in_project_x = nn.Linear(dim, inner_dim)      # assignment features
        self.in_project_fx = nn.Linear(dim, inner_dim)     # value features
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)  # re-applied by the
        # model-wide init pass too (Appendix A.3) -- this call makes the layer
        # sane when constructed/tested standalone.

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * temperature_init)
        self.dropout = nn.Dropout(dropout)

        # Node-shard process group (Phase 7). None -> single-process kernel,
        # byte-identical to before. Set by Transolver.set_shard_group() only
        # under parallel_mode node_shard.
        self.shard_group = None

    def _clamped_temperature(self) -> torch.Tensor:
        # [1, H, 1, 1] -> [H]
        t = torch.clamp(self.temperature, self.temperature_min, self.temperature_max)
        return t.view(self.heads)

    # ------------------------------------------------------------------
    # shared building blocks (Appendix A.1)
    # ------------------------------------------------------------------

    def _fused_slice_weights(self):
        """Fuse in_project_x with in_project_slice into one C -> [H, M] map,
        N-independent so it is computed once per forward, not per tile."""
        H, D, C = self.heads, self.dim_head, self.dim
        w_in = self.in_project_x.weight.view(H, D, C)
        b_in = self.in_project_x.bias.view(H, D)
        fused_w = torch.einsum('md,hdc->hmc', self.in_project_slice.weight, w_in)  # [H, M, C]
        fused_b = (torch.einsum('md,hd->hm', self.in_project_slice.weight, b_in)
                   + self.in_project_slice.bias[None, :])                          # [H, M]
        return fused_w, fused_b

    def _slice_weights(self, x_g: torch.Tensor, fused_w: torch.Tensor,
                        fused_b: torch.Tensor) -> torch.Tensor:
        """x_g: [Ng, C] -> W: [H, Ng, M], fp32-stabilized softmax."""
        logits = torch.einsum('nc,hmc->hnm', _stable(x_g), _stable(fused_w)) \
            + _stable(fused_b)[:, None, :]
        temp = self._clamped_temperature()
        logits = logits / temp[:, None, None]
        return torch.softmax(logits, dim=-1)  # stable dtype throughout

    def _chunk_stats(self, x_g: torch.Tensor, fused_w: torch.Tensor,
                      fused_b: torch.Tensor):
        """x_g: [Ng, C] -> (num [H, M, D], den [H, M]), both stable-dtype.

        THE bias convention proven exact vs the naive v1 kernel (section 6.3):
        normalize the raw aggregate through in_project_fx with the bias term
        scaled by the density `den`, not applied before normalizing.
        """
        H, D, C = self.heads, self.dim_head, self.dim
        W = self._slice_weights(x_g, fused_w, fused_b)          # [H, Ng, M] stable
        den = W.sum(dim=1)                                      # [H, M] stable
        agg = torch.einsum('nc,hnm->hmc', _stable(x_g), W)      # [H, M, C] stable
        w_fx = _stable(self.in_project_fx.weight).view(H, D, C)
        b_fx = _stable(self.in_project_fx.bias).view(H, D)
        num = torch.einsum('hmc,hdc->hmd', agg, w_fx) + den[:, :, None] * b_fx[:, None, :]
        return num, den, W

    def _slice_attend(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [H, M, D] -> [H, M, D], fp32-stabilized softmax."""
        q = self.to_q(tokens)
        k = self.to_k(tokens)
        v = self.to_v(tokens)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(_stable(dots), dim=-1).to(v.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def _deslice(self, x_g: torch.Tensor, token_out: torch.Tensor,
                 W: torch.Tensor, fused_w: torch.Tensor, fused_b: torch.Tensor) -> torch.Tensor:
        """token_out: [H, M, D], W: [H, Ng, M] or None -> [Ng, C].

        Bias added once, outside the deslice sum: exact because softmax rows
        of W sum to 1 (section 6.3), so no ordering caveat like the value path.
        """
        H, D, C = self.heads, self.dim_head, self.dim
        if W is None:
            W = self._slice_weights(x_g, fused_w, fused_b)
        w_out = _stable(self.to_out.weight).view(C, H, D).permute(1, 2, 0)  # [H, D, C]
        proj = torch.einsum('hmd,hdc->hmc', _stable(token_out), w_out)
        out = torch.einsum('hnm,hmc->nc', W, proj) + _stable(self.to_out.bias)
        return self.dropout(out.to(x_g.dtype))

    # ------------------------------------------------------------------
    # per-graph kernels (Appendix A.1 methods 6-7)
    # ------------------------------------------------------------------

    def _forward_naive(self, x_g: torch.Tensor) -> torch.Tensor:
        """v1 order: project in node space, then aggregate. One tile only."""
        H, D, C = self.heads, self.dim_head, self.dim
        Ng = x_g.shape[0]

        fx_mid = self.in_project_fx(x_g).view(Ng, H, D).permute(1, 0, 2)  # [H, Ng, D]
        x_mid = self.in_project_x(x_g).view(Ng, H, D).permute(1, 0, 2)    # [H, Ng, D]

        logits = torch.einsum('hnd,md->hnm', _stable(x_mid), _stable(self.in_project_slice.weight)) \
            + _stable(self.in_project_slice.bias)[None, None, :]
        temp = self._clamped_temperature()
        logits = logits / temp[:, None, None]
        W = torch.softmax(logits, dim=-1)                                  # [H, Ng, M] stable

        den = W.sum(dim=1)                                                 # [H, M] stable
        tokens = torch.einsum('hnd,hnm->hmd', _stable(fx_mid), W)          # [H, M, D] stable
        tokens = tokens / (den[:, :, None] + EPS)

        out_tok = self._slice_attend(tokens)                               # [H, M, D]
        out_x = torch.einsum('hmd,hnm->hnd', _stable(out_tok), W)          # [H, Ng, D] stable
        out_x = out_x.permute(1, 0, 2).reshape(Ng, H * D).to(x_g.dtype)

        return self.dropout(self.to_out(out_x))

    def _forward_slice_space(self, x_g: torch.Tensor, tile_ranges: List[Tuple[int, int]],
                              use_checkpointing: bool = False) -> torch.Tensor:
        """Transolver-3's two-pass form. tile_ranges == [(0, Ng)] is the exact
        single-tile special case (section 6.3)."""
        fused_w, fused_b = self._fused_slice_weights()
        do_checkpoint = use_checkpointing and self.training and x_g.requires_grad

        num_acc = den_acc = None
        for (s, e) in tile_ranges:
            xt = x_g[s:e]
            if do_checkpoint:
                num_t, den_t, _ = checkpoint(
                    self._chunk_stats, xt, fused_w, fused_b, use_reentrant=False)
            else:
                num_t, den_t, _ = self._chunk_stats(xt, fused_w, fused_b)
            num_acc = num_t if num_acc is None else num_acc + num_t
            den_acc = den_t if den_acc is None else den_acc + den_t

        if self.shard_group is not None:
            # This rank's num_acc/den_acc are partial sums over its node shard;
            # reduce to the whole-graph aggregates before normalizing.
            num_acc, den_acc = _shard_all_reduce_sum(num_acc, den_acc, self.shard_group)

        tokens = num_acc / (den_acc[:, :, None] + EPS)   # stable dtype
        tokens = tokens.to(x_g.dtype)
        out_tok = self._slice_attend(tokens)              # [H, M, D]

        outs = []
        for (s, e) in tile_ranges:
            xt = x_g[s:e]
            if do_checkpoint:
                out_t = checkpoint(
                    self._deslice, xt, out_tok, None, fused_w, fused_b, use_reentrant=False)
            else:
                out_t = self._deslice(xt, out_tok, None, fused_w, fused_b)
            outs.append(out_t)
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # decoupled two-stage inference (section 11, Appendix A.4): split the
    # slice_space kernel's own two passes across a cache-building call and a
    # decode call, so the physics tokens for one layer can be computed from
    # one set of chunks and applied to a different set of query coordinates.
    # ------------------------------------------------------------------

    def compute_layer_tokens(self, x_g: torch.Tensor,
                              tile_ranges: List[Tuple[int, int]]) -> torch.Tensor:
        """Stage 1 (pass 1 only): x_g -> tokens [H, M, D], residual dtype."""
        fused_w, fused_b = self._fused_slice_weights()
        num_acc = den_acc = None
        for (s, e) in tile_ranges:
            num_t, den_t, _ = self._chunk_stats(x_g[s:e], fused_w, fused_b)
            num_acc = num_t if num_acc is None else num_acc + num_t
            den_acc = den_t if den_acc is None else den_acc + den_t
        tokens = num_acc / (den_acc[:, :, None] + EPS)
        return tokens.to(x_g.dtype)

    def decode_with_tokens(self, x_g: torch.Tensor, cached_tokens: torch.Tensor,
                            tile_ranges: List[Tuple[int, int]]) -> torch.Tensor:
        """Stage 2: attend on an already-built token cache, then deslice
        x_g's own chunks against it. x_g may be a different coordinate set
        than the one that produced cached_tokens."""
        out_tok = self._slice_attend(cached_tokens)
        fused_w, fused_b = self._fused_slice_weights()
        outs = [self._deslice(x_g[s:e], out_tok, None, fused_w, fused_b) for (s, e) in tile_ranges]
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # packed, ptr-segmented entry point
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, ptr: torch.Tensor,
                attention_kernel: str = 'naive', chunk_size: int = 0,
                use_checkpointing: bool = False) -> torch.Tensor:
        """x: [sum_N, C] packed nodes, ptr: [B+1] graph boundaries -> [sum_N, C].

        Every reduction is scoped to x[s:e] for one graph at a time: no tensor
        operation ever spans two graphs, so batching cannot leak information
        and graph B's output is independent of graph A's contents.
        """
        outs = []
        for i in range(ptr.shape[0] - 1):
            s, e = int(ptr[i].item()), int(ptr[i + 1].item())
            x_g = x[s:e]
            if attention_kernel == 'naive':
                out_g = self._forward_naive(x_g)
            elif attention_kernel == 'slice_space':
                tile_ranges = make_tile_ranges(e - s, chunk_size)
                out_g = self._forward_slice_space(x_g, tile_ranges, use_checkpointing)
            else:
                raise ValueError(f"Unknown attention_kernel '{attention_kernel}'")
            outs.append(out_g)
        return torch.cat(outs, dim=0)
