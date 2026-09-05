"""
SDF-VAE: point-cloud encoder -> latent token(s) with KL -> SDF decoder.

Tier-1 default is a single global latent token with a DeepSDF-style MLP
decoder; `latent_tokens > 1` with `decoder_type attention` is the VecSet-style
upgrade path (same trainer, same FM stage).
"""

import contextlib
import warnings

import numpy as np
import torch
import torch.nn as nn

from model.mlp import init_weights

ENCODER_QUERY_TYPES = ('learned', 'fps')

# State-dict entries whose presence depends on a config flag rather than on the
# architecture, so a checkpoint and a freshly built model can legitimately
# disagree about them:
#   `mu_spread`        exists only with posterior_min_std_rel > 0
#   `encoder.queries`  exists only with encoder_query_type learned
FLAG_DEPENDENT_STATE_KEYS = {
    'mu_spread': 'posterior_min_std_rel (the posterior std floor)',
    'encoder.queries': 'encoder_query_type (learned nn.Parameter queries vs fps)',
}


def load_vae_state_dict(model, state, source=''):
    """`load_state_dict` that tolerates ONLY the flag-dependent keys above.

    Warm starts (`init_vae_modelpath`) cross config flags all the time: a v3 arm
    started from an ex1/ex2 checkpoint changes `posterior_min_std_rel` and/or
    `encoder_query_type`, which adds or removes a state-dict entry. A strict
    load then dies with a bare torch error naming an internal buffer. This
    loader accepts exactly those differences and re-raises everything else, with
    the responsible config key named.

    Returns (missing, unexpected) as lists of the tolerated key names.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    hard_missing = [k for k in missing if k not in FLAG_DEPENDENT_STATE_KEYS]
    hard_unexpected = [k for k in unexpected if k not in FLAG_DEPENDENT_STATE_KEYS]
    if hard_missing or hard_unexpected:
        where = f' from {source}' if source else ''
        raise RuntimeError(
            f'Incompatible VAE state dict{where}: missing {sorted(hard_missing)}, '
            f'unexpected {sorted(hard_unexpected)}. The checkpoint architecture does not '
            'match this config (latent_tokens / latent_dim / decoder_type / widths).')
    return list(missing), list(unexpected)


def describe_state_key_flag(keys):
    """Human-readable 'key (config flag)' list for the tolerated keys above."""
    return ', '.join(f'{k} <- {FLAG_DEPENDENT_STATE_KEYS.get(k, "?")}' for k in sorted(keys))


def farthest_point_sample(points, k):
    """Deterministic farthest-point sampling: `points` [B, N, 3] -> indices [B, k].

    Starts at the point farthest from the per-sample centroid and then
    iteratively picks the point with the largest distance to the set already
    chosen. Iterative O(B * k * N); fine for N <= 16384. Deterministic given
    the point set (`argmax` returns the first maximal index on ties). Requires
    k <= N -- the caller decides what to do otherwise.
    """
    if points.dim() != 3:
        raise ValueError(f'farthest_point_sample expects [B, N, 3], got {tuple(points.shape)}')
    batch, num_points = points.shape[0], points.shape[1]
    k = int(k)
    if k < 1 or k > num_points:
        raise ValueError(f'farthest_point_sample needs 1 <= k <= N, got k={k}, N={num_points}')
    with torch.no_grad():
        pts = points.detach().float()
        centroid = pts.mean(dim=1, keepdim=True)
        current = (pts - centroid).pow(2).sum(dim=-1).argmax(dim=1)  # [B]
        indices = torch.empty(batch, k, dtype=torch.long, device=pts.device)
        min_dist = torch.full((batch, num_points), float('inf'), dtype=pts.dtype, device=pts.device)
        batch_ar = torch.arange(batch, device=pts.device)
        for i in range(k):
            indices[:, i] = current
            chosen = pts[batch_ar, current].unsqueeze(1)  # [B, 1, 3]
            min_dist = torch.minimum(min_dist, (pts - chosen).pow(2).sum(dim=-1))
            current = min_dist.argmax(dim=1)
    return indices


def sign_accuracy(sdf_pred, sdf_target, eps=0.001):
    """Fraction of query points with |target| > eps whose predicted SDF sign
    matches the target sign. Returns a python float (NaN if no point passes the
    |target| > eps filter). Accepts torch tensors or array-likes of any
    (matching) shape; used by the VAE trainer's validation and by `evaluate`.
    """
    pred = torch.as_tensor(sdf_pred).detach().float().flatten()
    target = torch.as_tensor(sdf_target).detach().float().flatten().to(pred.device)
    if pred.numel() != target.numel():
        raise ValueError(
            f'sign_accuracy: shape mismatch, pred has {pred.numel()} values, target {target.numel()}')
    mask = target.abs() > float(eps)
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return float('nan')
    matches = torch.sign(pred[mask]) == torch.sign(target[mask])
    return float(matches.double().mean().item())


def _math_attention_ctx():
    """Force the math scaled-dot-product-attention backend.

    The fused flash / mem-efficient SDPA kernels have no double-backward, so the
    eikonal/normal gradient penalties (which need create_graph=True) fail when
    the decoder uses attention. The math backend is decomposed and supports
    second-order gradients on both CPU and CUDA.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return contextlib.nullcontext()


class FourierFeatures(nn.Module):
    """NeRF-style positional encoding: [x, sin(2^i pi x), cos(2^i pi x)]."""

    def __init__(self, num_bands=8, input_dim=3):
        super().__init__()
        self.register_buffer('freqs', (2.0 ** torch.arange(num_bands)) * np.pi)
        self.out_dim = input_dim * (1 + 2 * num_bands)

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs  # (..., 3, B)
        enc = torch.cat([x.unsqueeze(-1), torch.sin(proj), torch.cos(proj)], dim=-1)
        return enc.flatten(-2)


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention + feed-forward, residual."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, queries, context):
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.ff(self.norm_ff(queries))
        return queries


class SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention + feed-forward among the latent tokens (residual).

    Lets the latent tokens exchange information so a VecSet latent can
    specialize spatially (Dora-style dual attention). A no-op for a single
    global token, but harmless there.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, tokens):
        h = self.norm_attn(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm_ff(tokens))
        return tokens


class PointCloudEncoder(nn.Module):
    """Surface point cloud (+normals) -> latent tokens (mu, logvar).

    `encoder_query_type` selects the initial cross-attention queries:
      * `learned` (default): a learned `[1, latent_tokens, d_model]` parameter
        (`queries`), shared by every shape -- the original behaviour.
      * `fps`: 3DShape2VecSet-style geometry-anchored tokens. `latent_tokens`
        input points are farthest-point-sampled per shape (deterministic given
        the point set), embedded with the SAME `point_proj` features
        (fourier(xyz) + normals) as the context, and used as the queries. No
        `queries` parameter exists in this mode.
    """

    def __init__(self, config):
        super().__init__()
        d_model = int(config.get('encoder_dim', 256))
        num_heads = int(config.get('encoder_heads', 4))
        num_blocks = int(config.get('encoder_blocks', 2))
        num_bands = int(config.get('fourier_bands', 8))
        self.latent_tokens = int(config.get('latent_tokens', 1))
        self.latent_dim = int(config.get('latent_dim', 256))

        self.self_attention = bool(config.get('encoder_self_attention', False))
        self.query_type = str(config.get('encoder_query_type', 'learned')).lower()
        if self.query_type not in ENCODER_QUERY_TYPES:
            raise ValueError(
                f"encoder_query_type must be one of {ENCODER_QUERY_TYPES}, got '{self.query_type}'")
        self._fps_fallback_warned = False

        self.fourier = FourierFeatures(num_bands)
        self.point_proj = nn.Linear(self.fourier.out_dim + 3, d_model)
        if self.query_type == 'learned':
            self.queries = nn.Parameter(torch.randn(1, self.latent_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        if self.self_attention:
            self.self_blocks = nn.ModuleList(
                SelfAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        self.out_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * self.latent_dim)

    def _fps_queries(self, surface_points, context):
        """Gather `latent_tokens` farthest-point-sampled context features per shape.

        The degenerate `latent_tokens > num_points` case cannot farthest-point
        sample at all (FPS needs k <= N), so it falls back to a plain
        DETERMINISTIC tiling of the available points: token i takes point
        i % num_points. Determinism is the requirement here -- the fallback
        originally drew the extra indices from an unseeded `torch.randint`,
        which made the encoder stochastic (two `encode()` calls on identical
        input returned different `mu`) and silently broke the deterministic
        encode contract that `dataset.deterministic`, `train_fm._encode_split`
        and `evaluate` all depend on. The tiling is a guard, not a feature: the
        tokens repeat and the extra latent capacity is wasted, so the config is
        what should be fixed. Preflight rejects the configuration outright
        (SDF-QUERY-003); this branch only guards a direct native run.
        """
        num_points = surface_points.shape[1]
        if self.latent_tokens <= num_points:
            idx = farthest_point_sample(surface_points, self.latent_tokens)  # [B, T]
        else:
            if not self._fps_fallback_warned:
                warnings.warn(
                    f'encoder_query_type=fps: latent_tokens ({self.latent_tokens}) exceeds the '
                    f'number of input points ({num_points}), so farthest-point sampling is '
                    'impossible; falling back to a deterministic tiling of the input points '
                    '(token i uses point i % num_points), which repeats tokens and wastes the '
                    'extra latent capacity. Fix the config: lower latent_tokens or raise '
                    'num_encoder_points.', RuntimeWarning)
                self._fps_fallback_warned = True
            tile = torch.arange(self.latent_tokens, device=surface_points.device) % num_points
            idx = tile.unsqueeze(0).expand(surface_points.shape[0], -1)  # [B, T]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, context.shape[-1])
        return torch.gather(context, 1, gather_idx)  # [B, T, d_model]

    def forward(self, surface_points, surface_normals):
        feats = torch.cat([self.fourier(surface_points), surface_normals], dim=-1)
        context = self.point_proj(feats)
        if self.query_type == 'fps':
            queries = self._fps_queries(surface_points, context)
        else:
            queries = self.queries.expand(surface_points.shape[0], -1, -1)
        for i, block in enumerate(self.blocks):
            queries = block(queries, context)
            if self.self_attention:
                queries = self.self_blocks[i](queries)
        mu, logvar = self.to_latent(self.out_norm(queries)).chunk(2, dim=-1)
        return mu, logvar  # (B, tokens, latent_dim)


class SDFDecoderMLP(nn.Module):
    """DeepSDF-style MLP: [fourier(x), z_flat] -> sdf, skip connection mid-way.

    No LayerNorm on the output (decoder convention).
    """

    def __init__(self, config, latent_flat_dim):
        super().__init__()
        hidden = int(config.get('decoder_hidden', 512))
        num_layers = int(config.get('decoder_layers', 8))
        num_bands = int(config.get('fourier_bands', 8))
        self.fourier = FourierFeatures(num_bands)
        in_dim = self.fourier.out_dim + latent_flat_dim
        self.skip_layer = num_layers // 2

        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            if i == self.skip_layer:
                d_in = hidden + in_dim
            layers.append(nn.Linear(d_in, hidden))
        self.layers = nn.ModuleList(layers)
        self.act = nn.SiLU()
        self.out = nn.Linear(hidden, 1)
        self.apply(init_weights)
        # Keep the initial SDF inside the truncation band.  A full Kaiming
        # initialization on this scalar head produces values far outside
        # [-clamp_dist, clamp_dist], where a truncated loss can become
        # effectively flat before the decoder learns any geometry.
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_flat, query_points):
        # z_flat: (B, D), query_points: (B, Q, 3)
        z_exp = z_flat.unsqueeze(1).expand(-1, query_points.shape[1], -1)
        x_in = torch.cat([self.fourier(query_points), z_exp], dim=-1)
        h = x_in
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, x_in], dim=-1)
            h = self.act(layer(h))
        return self.out(h).squeeze(-1)  # (B, Q)


class SDFDecoderAttention(nn.Module):
    """VecSet-style decoder: query points cross-attend to latent tokens."""

    def __init__(self, config, latent_dim):
        super().__init__()
        d_model = int(config.get('decoder_hidden', 512))
        num_heads = int(config.get('decoder_heads', 4))
        num_blocks = int(config.get('decoder_layers', 4))
        num_bands = int(config.get('fourier_bands', 8))
        self.fourier = FourierFeatures(num_bands)
        self.query_proj = nn.Linear(self.fourier.out_dim, d_model)
        self.token_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, z_tokens, query_points):
        # z_tokens: (B, T, latent_dim), query_points: (B, Q, 3)
        q = self.query_proj(self.fourier(query_points))
        context = self.token_proj(z_tokens)
        for block in self.blocks:
            q = block(q, context)
        return self.out(q).squeeze(-1)


class SDFVAE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.latent_tokens = int(config.get('latent_tokens', 1))
        self.latent_dim = int(config.get('latent_dim', 256))
        self.latent_flat_dim = self.latent_tokens * self.latent_dim
        self.decoder_type = str(config.get('decoder_type', 'mlp'))

        self.encoder = PointCloudEncoder(config)
        if self.decoder_type == 'mlp':
            self.decoder = SDFDecoderMLP(config, self.latent_flat_dim)
        elif self.decoder_type == 'attention':
            self.decoder = SDFDecoderAttention(config, self.latent_dim)
        else:
            raise ValueError(f"decoder_type must be 'mlp' or 'attention', got '{self.decoder_type}'")

        # Posterior std floor (`posterior_min_std_rel`): std_eff = max(std, rel *
        # mu_spread), where `mu_spread` is a per-latent-dim running estimate of
        # the batch std of mu over (batch, tokens). The buffer is registered
        # ONLY when the floor is configured so checkpoints trained without it
        # keep loading with strict=True. Warm starts that CROSS the flag go
        # through `load_vae_state_dict`, which tolerates exactly this key.
        self.posterior_min_std_rel = float(config.get('posterior_min_std_rel', 0.0) or 0.0)
        if self.posterior_min_std_rel < 0:
            raise ValueError(
                f'posterior_min_std_rel must be >= 0, got {self.posterior_min_std_rel}')
        # `mu_spread_initialized` makes the FIRST update COPY the observed batch
        # std instead of blending 1% of it into the `ones` initialization: until
        # the momentum-0.99 EMA converges (~100 steps) the "relative" floor
        # would otherwise be an absolute `rel * 1.0`. It is registered
        # `persistent=False`, so the persistent state-dict key set stays exactly
        # what the pre-flag code wrote and old checkpoints load unchanged.
        self._mu_spread_skip_warned = False
        if self.posterior_min_std_rel > 0:
            self.register_buffer('mu_spread', torch.ones(self.latent_dim))
            self.register_buffer('mu_spread_initialized', torch.zeros(()), persistent=False)

    def encode(self, surface_points, surface_normals):
        return self.encoder(surface_points, surface_normals)

    @staticmethod
    def reparameterize(mu, logvar, noise_scale=1.0, min_std=None):
        """z = mu + noise_scale * eps * std, with std = exp(0.5 * logvar).

        `min_std` (a tensor broadcastable to `mu`, or None) floors the std
        elementwise: std = max(std, min_std). None reproduces the legacy path.
        """
        std = torch.exp(0.5 * logvar)
        if min_std is not None:
            std = torch.maximum(std, min_std.to(dtype=std.dtype, device=std.device))
        return mu + float(noise_scale) * torch.randn_like(mu) * std

    def _update_mu_spread(self, mu):
        """Running per-dim estimate of the spread of mu (momentum 0.99).

        Called during training only. The FIRST update copies the observed batch
        std instead of blending it into the `ones` initialization, so the
        "relative" floor is relative to the real spread from step one rather
        than to an absolute 1.0 for the ~100 steps of the EMA horizon.

        Skipped (with a one-time stdout warning per model instance) when
        batch * tokens < 2, where a std over a single vector is undefined: with
        `latent_tokens 1` and `batch_size 1` the buffer would otherwise stay
        pinned at its initialization forever and silently turn the relative
        floor into an absolute one, with nothing in the training log to say so.
        """
        initialized = getattr(self, 'mu_spread_initialized', None)
        seen = initialized is not None and float(initialized) != 0.0
        if mu.shape[0] * mu.shape[1] < 2:
            if not self._mu_spread_skip_warned:
                state = 'last updated' if seen else 'uninitialized, 1.0'
                print(
                    f'WARNING: posterior_min_std_rel {self.posterior_min_std_rel:g} is on but '
                    f'batch ({mu.shape[0]}) x latent_tokens ({mu.shape[1]}) < 2, so the '
                    'mu_spread estimate cannot be updated (the std of a single vector is '
                    f'undefined). The posterior std floor stays at rel x its {state} value, '
                    'i.e. an absolute floor rather than a relative one. Raise batch_size or '
                    'latent_tokens.', flush=True)
                self._mu_spread_skip_warned = True
            return
        with torch.no_grad():
            batch_std = mu.detach().float().reshape(-1, mu.shape[-1]).std(dim=0)  # [latent_dim]
            batch_std = batch_std.to(self.mu_spread.dtype)
            if not seen:
                self.mu_spread.copy_(batch_std)
                if initialized is not None:
                    initialized.fill_(1.0)
            else:
                self.mu_spread.mul_(0.99).add_(0.01 * batch_std)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Mark `mu_spread` as initialized when a checkpoint supplies it.

        `mu_spread_initialized` is deliberately non-persistent, so a resumed or
        warm-started run rebuilds it as 0 next to a `mu_spread` that already
        carries a converged estimate. Without this hook the first update after
        the load would overwrite that estimate with a single batch's std.
        """
        if (getattr(self, 'mu_spread_initialized', None) is not None
                and prefix + 'mu_spread' in state_dict):
            self.mu_spread_initialized.fill_(1.0)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _posterior_min_std(self, posterior_min_std_rel):
        """`rel * mu_spread` as a [1, 1, latent_dim] tensor, or None when the
        floor is off or the buffer was never registered (rel was 0 at build)."""
        rel = float(posterior_min_std_rel or 0.0)
        if rel <= 0 or not hasattr(self, 'mu_spread'):
            return None
        return (rel * self.mu_spread).view(1, 1, -1)

    @staticmethod
    def kl_divergence(mu, logvar):
        """Standard diagonal-Gaussian KL to N(0, I), summed over the latent and
        averaged over the batch."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2)).mean()

    def decode(self, z_tokens, query_points):
        if self.decoder_type == 'mlp':
            return self.decoder(z_tokens.flatten(1), query_points)
        return self.decoder(z_tokens, query_points)

    def decode_flat(self, z_flat, query_points):
        """Decode from a flattened latent (as produced by the FM stage)."""
        z_tokens = z_flat.view(-1, self.latent_tokens, self.latent_dim)
        return self.decode(z_tokens, query_points)

    def forward(self, surface_points, surface_normals, query_points, query_sdf,
                posterior_noise_scale=1.0, clamp_dist=0.1,
                surface_weight=0.0, normal_weight=0.0, eikonal_weight=0.0,
                hybrid_grad_points=0, posterior_min_std_rel=0.0):
        """Compute every training loss in a single pass and return them as a dict.

        Routing the whole step through `forward` (rather than calling encode/
        decode separately) is what lets a DDP/FSDP wrapper install its gradient
        hooks correctly — the wrapper only instruments `forward`.

        `posterior_min_std_rel` > 0 floors the posterior std at
        rel * mu_spread (see `reparameterize`); it is ignored unless the model
        was built with `posterior_min_std_rel > 0` (which registers the buffer).
        """
        mu, logvar = self.encode(surface_points, surface_normals)
        if self.training and hasattr(self, 'mu_spread'):
            self._update_mu_spread(mu)
        min_std = self._posterior_min_std(posterior_min_std_rel)
        z = self.reparameterize(mu, logvar, posterior_noise_scale, min_std)
        sdf_pred = self.decode(z, query_points)
        losses = {
            'recon': sdf_loss(sdf_pred.float(), query_sdf, clamp_dist),
            'kl': self.kl_divergence(mu.float(), logvar.float()),
        }
        if surface_weight > 0 or normal_weight > 0 or eikonal_weight > 0:
            surface_l, normal_l, eikonal_l = hybrid_geometry_losses(
                self, z, surface_points, surface_normals, query_points,
                subsample=hybrid_grad_points)
            losses['surface'] = surface_l
            losses['normal'] = normal_l
            losses['eikonal'] = eikonal_l
        return losses


def sdf_loss(sdf_pred, sdf_target, clamp_dist=0.1):
    """L1 against a truncated SDF target, with gradients for every prediction.

    Clamping the prediction as well as the target creates a zero-gradient
    region whenever an untrained decoder emits values outside the truncation
    band.  Only the supervision is truncated here so saturated predictions are
    always pulled back toward a valid SDF.
    """
    target = torch.clamp(sdf_target, -clamp_dist, clamp_dist)
    return (sdf_pred - target).abs().mean()


def _sdf_gradient(sdf, points):
    """d(sdf)/d(points) with a retained graph for second-order backprop."""
    grad = torch.autograd.grad(
        sdf, points, grad_outputs=torch.ones_like(sdf),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad


def hybrid_geometry_losses(vae, z, surface_points, surface_normals, query_points,
                           subsample=0):
    """Extra SDF-VAE losses beyond plain reconstruction (TripoSG-style).

    Returns (surface, normal, eikonal) scalar losses:
      * surface: |f(x_surface)| -> 0  (the level set passes through the surface)
      * normal:  1 - cos(<grad f, n>) at the surface (correct surface orientation)
      * eikonal: (||grad f|| - 1)^2 over query space (a true metric SDF)

    Must run outside autocast: the gradient terms need a stable fp32 graph.
    `subsample` (>0) caps the number of surface / query points used for the
    gradient terms to bound memory and the double-backward cost.
    """
    if subsample and subsample > 0:
        if surface_points.shape[1] > subsample:
            idx = torch.randperm(surface_points.shape[1], device=surface_points.device)[:subsample]
            surface_points = surface_points[:, idx]
            surface_normals = surface_normals[:, idx]
        if query_points.shape[1] > subsample:
            idx = torch.randperm(query_points.shape[1], device=query_points.device)[:subsample]
            query_points = query_points[:, idx]

    with _math_attention_ctx():
        surf = surface_points.detach().requires_grad_(True)
        sdf_surf = vae.decode(z, surf)
        grad_surf = _sdf_gradient(sdf_surf, surf)
        surface_l = sdf_surf.abs().mean()
        grad_surf_n = grad_surf / (grad_surf.norm(dim=-1, keepdim=True) + 1e-8)
        normal_l = (1.0 - (grad_surf_n * surface_normals).sum(dim=-1)).mean()

        qp = query_points.detach().requires_grad_(True)
        sdf_q = vae.decode(z, qp)
        grad_q = _sdf_gradient(sdf_q, qp)
        eikonal_l = (grad_q.norm(dim=-1) - 1.0).pow(2).mean()

    return surface_l, normal_l, eikonal_l
