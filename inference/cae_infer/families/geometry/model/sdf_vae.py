"""
SDF-VAE: point-cloud encoder -> latent token(s) with KL -> SDF decoder.

Tier-1 default is a single global latent token with a DeepSDF-style MLP
decoder; `latent_tokens > 1` with `decoder_type attention` is the VecSet-style
upgrade path (same trainer, same FM stage).
"""

import contextlib

import numpy as np
import torch
import torch.nn as nn

from model.mlp import init_weights


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
    """Surface point cloud (+normals) -> latent tokens (mu, logvar)."""

    def __init__(self, config):
        super().__init__()
        d_model = int(config.get('encoder_dim', 256))
        num_heads = int(config.get('encoder_heads', 4))
        num_blocks = int(config.get('encoder_blocks', 2))
        num_bands = int(config.get('fourier_bands', 8))
        self.latent_tokens = int(config.get('latent_tokens', 1))
        self.latent_dim = int(config.get('latent_dim', 256))

        self.self_attention = bool(config.get('encoder_self_attention', False))

        self.fourier = FourierFeatures(num_bands)
        self.point_proj = nn.Linear(self.fourier.out_dim + 3, d_model)
        self.queries = nn.Parameter(torch.randn(1, self.latent_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        if self.self_attention:
            self.self_blocks = nn.ModuleList(
                SelfAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        self.out_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * self.latent_dim)

    def forward(self, surface_points, surface_normals):
        feats = torch.cat([self.fourier(surface_points), surface_normals], dim=-1)
        context = self.point_proj(feats)
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

    def encode(self, surface_points, surface_normals):
        return self.encoder(surface_points, surface_normals)

    @staticmethod
    def reparameterize(mu, logvar, noise_scale=1.0):
        return mu + float(noise_scale) * torch.randn_like(mu) * torch.exp(0.5 * logvar)

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
                hybrid_grad_points=0):
        """Compute every training loss in a single pass and return them as a dict.

        Routing the whole step through `forward` (rather than calling encode/
        decode separately) is what lets a DDP/FSDP wrapper install its gradient
        hooks correctly — the wrapper only instruments `forward`.
        """
        mu, logvar = self.encode(surface_points, surface_normals)
        z = self.reparameterize(mu, logvar, posterior_noise_scale)
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
