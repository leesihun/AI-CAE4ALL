"""
SDF-VAE: point-cloud encoder -> latent token(s) with KL -> SDF decoder.

Tier-1 default is a single global latent token with a DeepSDF-style MLP
decoder; `latent_tokens > 1` with `decoder_type attention` is the VecSet-style
upgrade path (same trainer, same FM stage).
"""

import numpy as np
import torch
import torch.nn as nn

from model.mlp import init_weights


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

        self.fourier = FourierFeatures(num_bands)
        self.point_proj = nn.Linear(self.fourier.out_dim + 3, d_model)
        self.queries = nn.Parameter(torch.randn(1, self.latent_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, num_heads) for _ in range(num_blocks))
        self.out_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * self.latent_dim)

    def forward(self, surface_points, surface_normals):
        feats = torch.cat([self.fourier(surface_points), surface_normals], dim=-1)
        context = self.point_proj(feats)
        queries = self.queries.expand(surface_points.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
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

    def decode(self, z_tokens, query_points):
        if self.decoder_type == 'mlp':
            return self.decoder(z_tokens.flatten(1), query_points)
        return self.decoder(z_tokens, query_points)

    def decode_flat(self, z_flat, query_points):
        """Decode from a flattened latent (as produced by the FM stage)."""
        z_tokens = z_flat.view(-1, self.latent_tokens, self.latent_dim)
        return self.decode(z_tokens, query_points)

    def forward(self, surface_points, surface_normals, query_points,
                posterior_noise_scale=1.0):
        mu, logvar = self.encode(surface_points, surface_normals)
        z = self.reparameterize(mu, logvar, posterior_noise_scale)
        sdf_pred = self.decode(z, query_points)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2)).mean()
        return sdf_pred, kl, mu, logvar


def sdf_loss(sdf_pred, sdf_target, clamp_dist=0.1):
    """L1 against a truncated SDF target, with gradients for every prediction.

    Clamping the prediction as well as the target creates a zero-gradient
    region whenever an untrained decoder emits values outside the truncation
    band.  Only the supervision is truncated here so saturated predictions are
    always pulled back toward a valid SDF.
    """
    target = torch.clamp(sdf_target, -clamp_dist, clamp_dist)
    return (sdf_pred - target).abs().mean()
