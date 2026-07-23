"""
Flow-matching velocity network over shape latents.

Rectified flow convention: z_t = (1 - t) * noise + t * data, target velocity
v = data - noise. Condition dropout at train time enables classifier-free
guidance.

Two architectures behind one interface (config `fm_arch`):
  - `mlp` (default): AdaLN-Zero residual MLP over the *flattened* latent. This
    is the Tier-1 net and the historical default for a single global token.
  - `dit`: an AdaLN-Zero Diffusion Transformer over the latent *token set*
    (self-attention among tokens). Use with `latent_tokens > 1`; it is the
    architecture that actually exploits a VecSet latent. Falls back gracefully
    to a 1-token sequence.

Both expose forward(z_flat, t, cond, cond_mask) -> velocity of the same flat
shape, so the trainer, sampler, and inference paths are architecture-agnostic.
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


class DiTBlock(nn.Module):
    """Diffusion-Transformer block: token self-attention + MLP, both AdaLN-Zero
    modulated by the shared (timestep, condition) embedding.

    The modulation Linear is zero-initialized so the block starts as identity,
    matching the AdaLN-Zero convention used by the MLP path.
    """

    def __init__(self, hidden, num_heads, cond_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden))
        self.modulation = nn.Linear(cond_hidden, 6 * hidden)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h, c):
        # h: (B, T, hidden), c: (B, cond_hidden)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=-1)
        a = self.norm1(h) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(a, a, a, need_weights=False)
        h = h + gate1.unsqueeze(1) * attn_out
        m = self.norm2(h) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return h + gate2.unsqueeze(1) * self.mlp(m)


class VelocityNet(nn.Module):

    def __init__(self, config, latent_flat_dim, cond_dim=0):
        super().__init__()
        hidden = int(config.get('fm_hidden', 512))
        num_blocks = int(config.get('fm_blocks', 6))
        cond_hidden = int(config.get('fm_cond_hidden', 256))
        self.cond_dim = cond_dim
        self.latent_flat_dim = latent_flat_dim
        self.arch = str(config.get('fm_arch', 'mlp')).lower()
        if self.arch not in ('mlp', 'dit'):
            raise ValueError(f"fm_arch must be 'mlp' or 'dit', got '{self.arch}'")

        self.t_embed = TimestepEmbedding(out_dim=cond_hidden)
        if cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, cond_hidden))
            self.null_cond = nn.Parameter(torch.zeros(1, cond_hidden))

        if self.arch == 'mlp':
            self.in_proj = nn.Linear(latent_flat_dim, hidden)
            self.blocks = nn.ModuleList(AdaLNBlock(hidden, cond_hidden) for _ in range(num_blocks))
            self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False)
            self.out_proj = nn.Linear(hidden, latent_flat_dim)
        else:
            # DiT over the latent token set. Token count/width come from the VAE
            # latent geometry so the flat latent can be reshaped to (B, T, C).
            self.latent_tokens = int(config.get('latent_tokens', 1))
            if latent_flat_dim % self.latent_tokens != 0:
                raise ValueError(
                    f'latent_flat_dim {latent_flat_dim} is not divisible by '
                    f'latent_tokens {self.latent_tokens}')
            self.token_dim = latent_flat_dim // self.latent_tokens
            num_heads = int(config.get('fm_heads', 8))
            self.in_proj = nn.Linear(self.token_dim, hidden)
            self.pos_embed = nn.Parameter(torch.randn(1, self.latent_tokens, hidden) * 0.02)
            self.blocks = nn.ModuleList(
                DiTBlock(hidden, num_heads, cond_hidden) for _ in range(num_blocks))
            self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False)
            self.out_proj = nn.Linear(hidden, self.token_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _embed_condition(self, batch_size, t, cond, cond_mask):
        """Shared (timestep + optional condition) embedding with CFG dropout."""
        c = self.t_embed(t)
        if self.cond_dim > 0:
            if cond is None:
                c_emb = self.null_cond.expand(batch_size, -1)
            else:
                c_emb = self.cond_embed(cond)
                if cond_mask is not None:
                    c_emb = torch.where(cond_mask[:, None], c_emb, self.null_cond)
            c = c + c_emb
        return c

    def forward(self, z, t, cond=None, cond_mask=None):
        """z: (B, D) noisy flat latent, t: (B,), cond: (B, cond_dim) normalized.

        cond_mask: (B,) bool; False entries use the learned null condition
        (condition dropout during training, unconditional branch of CFG).
        """
        c = self._embed_condition(z.shape[0], t, cond, cond_mask)

        if self.arch == 'mlp':
            h = self.in_proj(z)
            for block in self.blocks:
                h = block(h, c)
            return self.out_proj(self.out_norm(h))

        # DiT: (B, T*C) -> (B, T, C) -> tokens -> (B, T*C)
        h = z.view(z.shape[0], self.latent_tokens, self.token_dim)
        h = self.in_proj(h) + self.pos_embed
        for block in self.blocks:
            h = block(h, c)
        out = self.out_proj(self.out_norm(h))
        return out.reshape(z.shape[0], self.latent_flat_dim)


def _sample_time(batch_size, device, time_sampling='uniform', logit_mean=0.0, logit_std=1.0):
    """Draw flow-matching timesteps.

    `uniform` is the classic rectified-flow schedule. `logit_normal` follows
    Stable Diffusion 3: sample u ~ N(logit_mean, logit_std) and set t = sigmoid(u),
    concentrating supervision on the harder middle timesteps.
    """
    if time_sampling == 'logit_normal':
        u = logit_mean + logit_std * torch.randn(batch_size, device=device)
        return torch.sigmoid(u)
    return torch.rand(batch_size, device=device)


def flow_matching_loss(model, z_data, cond=None, cond_dropout=0.1,
                       time_sampling='uniform', logit_mean=0.0, logit_std=1.0):
    """Rectified-flow objective with condition dropout and a choice of time schedule."""
    noise = torch.randn_like(z_data)
    t = _sample_time(z_data.shape[0], z_data.device, time_sampling, logit_mean, logit_std)
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
