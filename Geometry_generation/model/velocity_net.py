"""
Flow-matching velocity network over flattened shape latents.

Rectified flow convention: z_t = (1 - t) * noise + t * data, target velocity
v = data - noise. AdaLN-Zero residual MLP blocks conditioned on (t, cond);
condition dropout at train time enables classifier-free guidance.

For latent_tokens == 1 this is the Tier-1 MLP velocity net; a DiT over token
sets can be added later behind the same interface.
"""

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):

    def __init__(self, dim=128, out_dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(emb)


class AdaLNBlock(nn.Module):
    """Residual MLP block with AdaLN-Zero modulation (gate zero-initialized)."""

    def __init__(self, hidden, cond_hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden))
        self.modulation = nn.Linear(cond_hidden, 3 * hidden)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h, c):
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class VelocityNet(nn.Module):

    def __init__(self, config, latent_flat_dim, cond_dim=0):
        super().__init__()
        hidden = int(config.get('fm_hidden', 512))
        num_blocks = int(config.get('fm_blocks', 6))
        cond_hidden = int(config.get('fm_cond_hidden', 256))
        self.cond_dim = cond_dim

        self.in_proj = nn.Linear(latent_flat_dim, hidden)
        self.t_embed = TimestepEmbedding(out_dim=cond_hidden)
        if cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, cond_hidden))
            self.null_cond = nn.Parameter(torch.zeros(1, cond_hidden))
        self.blocks = nn.ModuleList(AdaLNBlock(hidden, cond_hidden) for _ in range(num_blocks))
        self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden, latent_flat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, t, cond=None, cond_mask=None):
        """z: (B, D) noisy latent, t: (B,), cond: (B, cond_dim) normalized.

        cond_mask: (B,) bool; False entries use the learned null condition
        (condition dropout during training, unconditional branch of CFG).
        """
        c = self.t_embed(t)
        if self.cond_dim > 0:
            if cond is None:
                c_emb = self.null_cond.expand(z.shape[0], -1)
            else:
                c_emb = self.cond_embed(cond)
                if cond_mask is not None:
                    c_emb = torch.where(cond_mask[:, None], c_emb, self.null_cond)
            c = c + c_emb

        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(self.out_norm(h))


def flow_matching_loss(model, z_data, cond=None, cond_dropout=0.1):
    """Rectified-flow objective with condition dropout."""
    noise = torch.randn_like(z_data)
    t = torch.rand(z_data.shape[0], device=z_data.device)
    z_t = (1 - t[:, None]) * noise + t[:, None] * z_data
    target = z_data - noise

    cond_mask = None
    if cond is not None and cond_dropout > 0:
        cond_mask = torch.rand(z_data.shape[0], device=z_data.device) >= cond_dropout

    v_pred = model(z_t, t, cond=cond, cond_mask=cond_mask)
    return (v_pred - target).pow(2).mean()


@torch.no_grad()
def sample_latents(model, num_samples, latent_flat_dim, device,
                   cond=None, cfg_scale=1.0, ode_steps=50, generator=None):
    """Integrate the learned ODE from noise (t=0) to data (t=1) with Euler steps.

    cond: (num_samples, cond_dim) normalized conditions or None.
    cfg_scale: 1.0 = plain conditional; > 1.0 = classifier-free guidance.
    """
    z = torch.randn(num_samples, latent_flat_dim, device=device, generator=generator)
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        if cond is not None and cfg_scale != 1.0:
            v_cond = model(z, t, cond=cond)
            v_uncond = model(z, t, cond=None)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(z, t, cond=cond)
        z = z + v * dt
    return z
