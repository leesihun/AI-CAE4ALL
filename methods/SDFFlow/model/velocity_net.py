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

Condition dropout modes (config `cond_dropout_mode`, read by `VelocityNet`
and stored with the rest of the config in the FM checkpoint):

  - `all` (default, legacy): one Bernoulli mask per *sample*. A dropped sample
    swaps its whole condition embedding for the learned `null_cond` vector.
    Parameter names, shapes, registration order, and forward outputs are
    bit-identical to the pre-`cond_dropout_mode` file, so every existing FM
    checkpoint loads and samples unchanged.
  - `per_dim`: an independent Bernoulli mask per *condition dimension*. The
    condition MLP sees `concat([cond_filled, mask.float()])` (width
    `2 * cond_dim`) where `cond_filled = where(mask, cond, null_values)` and
    `null_values` is a learned `nn.Parameter` of shape `[cond_dim]`. The
    all-masked row is the unconditional branch (there is no separate
    `null_cond` parameter in this mode). This is what lets inference request a
    *partial* condition -- some entries specified, the rest `nan`/unspecified --
    from a single network. With independent per-entry dropout the fully
    unconditional row would only appear with probability `cond_dropout **
    cond_dim` (6e-5 at 0.2 over six conditions), so `flow_matching_loss` draws
    an EXPLICIT drop-all term at `cond_dropout_all_prob` (config
    `cond_dropout_all_prob`, default 0.1) on top of it; set that to 0 to get
    the chance-only behaviour, and then keep `cfg_scale 1.0` because the CFG
    branch is untrained.

`cond_mask` accepted by `forward` in either mode: None (everything observed),
`[B]` bool (per-sample; False = unconditional row), or `[B, cond_dim]` bool
(per-dimension; False = that entry unspecified). In mode `all` a
`[B, cond_dim]` mask is reduced with `.all(dim=1)` and a warning, because that
network cannot represent a partially observed condition.
"""

import math
import warnings

import torch
import torch.nn as nn


COND_DROPOUT_MODES = ('all', 'per_dim')


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


def _parse_cond_dropout_mode(value):
    mode = str(value if value is not None else 'all').lower()
    if mode not in COND_DROPOUT_MODES:
        raise ValueError(
            f"cond_dropout_mode must be one of {COND_DROPOUT_MODES}, got '{mode}'")
    return mode


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
        # 'all' (legacy, per-sample null embedding) or 'per_dim' (per-dimension
        # mask + learned null values). Read from the FM config, which the
        # checkpoint stores, so inference rebuilds the right parameter set.
        self.cond_dropout_mode = _parse_cond_dropout_mode(config.get('cond_dropout_mode', 'all'))

        self.t_embed = TimestepEmbedding(out_dim=cond_hidden)
        if cond_dim > 0:
            if self.cond_dropout_mode == 'per_dim':
                # Input = [cond_filled | mask] so the network can tell "value 0"
                # from "unspecified". No null_cond: the all-masked row *is* the
                # unconditional branch.
                self.cond_input_dim = 2 * cond_dim
                self.cond_embed = nn.Sequential(
                    nn.Linear(self.cond_input_dim, cond_hidden), nn.SiLU(),
                    nn.Linear(cond_hidden, cond_hidden))
                self.null_values = nn.Parameter(torch.zeros(cond_dim))
            else:
                # Legacy parameter set and registration order -- do not reorder.
                self.cond_input_dim = cond_dim
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, cond_hidden))
                self.null_cond = nn.Parameter(torch.zeros(1, cond_hidden))
        else:
            self.cond_input_dim = 0

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

    # ------------------------------------------------------------------ masks
    @staticmethod
    def _check_mask(cond_mask, batch_size, cond_dim):
        """Validate a user mask and return it as bool. Accepts [B] or [B, cond_dim]."""
        if not torch.is_tensor(cond_mask):
            raise TypeError(f'cond_mask must be a bool tensor, got {type(cond_mask).__name__}')
        if cond_mask.dtype != torch.bool:
            cond_mask = cond_mask.bool()
        if cond_mask.dim() == 1 and cond_mask.shape[0] == batch_size:
            return cond_mask
        if cond_mask.dim() == 2 and tuple(cond_mask.shape) == (batch_size, cond_dim):
            return cond_mask
        raise ValueError(
            f'cond_mask must have shape [{batch_size}] or [{batch_size}, {cond_dim}], '
            f'got {tuple(cond_mask.shape)}')

    def expand_cond_mask(self, cond_mask, batch_size, device=None):
        """Return the mask in this network's native layout.

        mode 'all'    -> [B] bool (a [B, cond_dim] mask is reduced with .all(dim=1)
                         and a warning: this network cannot express a partial condition)
        mode 'per_dim'-> [B, cond_dim] bool (a [B] mask is broadcast over the dims)
        None          -> all True in the native layout (everything observed).
        """
        if device is None:
            device = next(self.parameters()).device
        if self.cond_dropout_mode == 'per_dim':
            if cond_mask is None:
                return torch.ones(batch_size, self.cond_dim, dtype=torch.bool, device=device)
            cond_mask = self._check_mask(cond_mask, batch_size, self.cond_dim)
            if cond_mask.dim() == 1:
                cond_mask = cond_mask[:, None].expand(batch_size, self.cond_dim)
            return cond_mask
        if cond_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        cond_mask = self._check_mask(cond_mask, batch_size, self.cond_dim)
        if cond_mask.dim() == 2:
            partial_rows = int((cond_mask.any(dim=1) & ~cond_mask.all(dim=1)).sum().item())
            warnings.warn(
                "cond_dropout_mode 'all' received a per-dimension [B, cond_dim] cond_mask; "
                "reducing it with .all(dim=1) (a row is conditional only if every entry is "
                f"specified). {partial_rows} of {batch_size} rows were partially observed and "
                "are treated as unconditional"
                + (" -- train with cond_dropout_mode per_dim to support partial conditions."
                   if partial_rows else "."),
                RuntimeWarning, stacklevel=3)
            cond_mask = cond_mask.all(dim=1)
        return cond_mask

    def unconditional_cond_mask(self, batch_size, device=None):
        """All-False mask in the native layout: the CFG unconditional branch."""
        if device is None:
            device = next(self.parameters()).device
        if self.cond_dropout_mode == 'per_dim':
            return torch.zeros(batch_size, self.cond_dim, dtype=torch.bool, device=device)
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # -------------------------------------------------------------- embedding
    def _embed_condition_per_dim(self, batch_size, cond, cond_mask):
        null = self.null_values
        if cond is None:
            # Unconditional branch: every entry unspecified.
            mask = torch.zeros(batch_size, self.cond_dim, dtype=torch.bool, device=null.device)
            cond_filled = null.unsqueeze(0).expand(batch_size, self.cond_dim)
        else:
            if tuple(cond.shape) != (batch_size, self.cond_dim):
                raise ValueError(
                    f'cond must have shape [{batch_size}, {self.cond_dim}], got {tuple(cond.shape)}')
            mask = self.expand_cond_mask(cond_mask, batch_size, device=cond.device)
            # where() copies null_values into masked slots, so a masked entry's
            # value (even nan) neither reaches the network nor receives gradient.
            cond_filled = torch.where(
                mask, cond, null.to(cond.dtype).unsqueeze(0).expand(batch_size, self.cond_dim))
        x = torch.cat([cond_filled, mask.to(cond_filled.dtype)], dim=-1)
        return self.cond_embed(x)

    def _embed_condition(self, batch_size, t, cond, cond_mask):
        """Shared (timestep + optional condition) embedding with CFG dropout."""
        c = self.t_embed(t)
        if self.cond_dim > 0:
            if self.cond_dropout_mode == 'per_dim':
                c_emb = self._embed_condition_per_dim(batch_size, cond, cond_mask)
            elif cond is None:
                c_emb = self.null_cond.expand(batch_size, -1)
            else:
                c_emb = self.cond_embed(cond)
                if cond_mask is not None:
                    cond_mask = self.expand_cond_mask(cond_mask, batch_size, device=cond.device)
                    c_emb = torch.where(cond_mask[:, None], c_emb, self.null_cond)
            c = c + c_emb
        return c

    def forward(self, z, t, cond=None, cond_mask=None):
        """z: (B, D) noisy flat latent, t: (B,), cond: (B, cond_dim) normalized.

        cond_mask: None, (B,) bool, or (B, cond_dim) bool. False entries use the
        learned null condition -- the whole row's `null_cond` embedding in mode
        'all', the per-entry `null_values` in mode 'per_dim' (condition dropout
        during training, unconditional branch of CFG, unspecified entries of a
        partial request). See the module docstring for the mode semantics.
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


def unwrap_velocity_net(model):
    """Strip DDP/FSDP (or any `.module`-holding) wrappers down to the VelocityNet."""
    seen = 0
    while not isinstance(model, VelocityNet) and hasattr(model, 'module') and seen < 8:
        model = model.module
        seen += 1
    return model


def resolve_cond_dropout_mode(model, cond_dropout_mode=None):
    """The dropout mode a loss/sampler should use for `model`.

    An explicit `cond_dropout_mode` wins; otherwise the (possibly wrapped)
    network's own attribute is used, defaulting to 'all' for objects that do
    not carry one (mocks, foreign modules).
    """
    if cond_dropout_mode is not None:
        return _parse_cond_dropout_mode(cond_dropout_mode)
    return _parse_cond_dropout_mode(getattr(unwrap_velocity_net(model), 'cond_dropout_mode', 'all'))


def _velocity(model, z, t, cond=None, cond_mask=None):
    """`model(z, t, cond=cond[, cond_mask=cond_mask])` -- the keyword is passed
    only when a mask is given, so the maskless call is literally the legacy one
    and a velocity model that predates `cond_mask` still integrates."""
    if cond_mask is None:
        return model(z, t, cond=cond)
    return model(z, t, cond=cond, cond_mask=cond_mask)


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
                       time_sampling='uniform', logit_mean=0.0, logit_std=1.0,
                       cond_dropout_mode=None, cond_dropout_all_prob=0.1):
    """Rectified-flow objective with condition dropout and a choice of time schedule.

    The dropout mask follows `cond_dropout_mode` (resolved from the model when
    not given): mode 'all' draws one Bernoulli(1 - cond_dropout) keep-flag per
    sample, `[B]`; mode 'per_dim' draws an independent keep-flag per condition
    entry, `[B, cond_dim]`. The RNG consumption of mode 'all' is exactly the
    legacy `torch.rand(B) >= cond_dropout`, so seeded legacy runs reproduce and
    `cond_dropout_all_prob` is ignored there.

    `cond_dropout_all_prob` (per_dim only) is the EXPLICIT drop-all term. With
    independent per-entry dropout the fully unconditional row -- which is what
    `sample_latents`' CFG branch evaluates -- appears by chance with probability
    `cond_dropout ** cond_dim`: 6.4e-5 at the ex5 settings (p 0.2, six
    conditions), i.e. about 111 rows in a whole 1000-epoch run, so CFG would
    extrapolate from a branch still near initialization and the resulting
    quality loss would be blamed on CFG rather than on the starved branch.
    Composer/CoLay use ~0.1 for the same reason; 0.0 restores the
    chance-only behaviour. The term is applied ON TOP of the per-entry draw, so
    the effective marginal drop rate of one entry is `cond_dropout +
    (1 - cond_dropout) * cond_dropout_all_prob` (0.28 at the ex5 settings,
    measured 0.284 over 4096 rows) -- the partial-condition training signal is
    slightly stronger than `cond_dropout` alone, not weaker.
    """
    noise = torch.randn_like(z_data)
    t = _sample_time(z_data.shape[0], z_data.device, time_sampling, logit_mean, logit_std)
    z_t = (1 - t[:, None]) * noise + t[:, None] * z_data
    target = z_data - noise

    cond_mask = None
    if cond is not None and cond_dropout > 0:
        mode = resolve_cond_dropout_mode(model, cond_dropout_mode)
        if mode == 'per_dim':
            cond_mask = torch.rand(z_data.shape[0], cond.shape[1], device=z_data.device) >= cond_dropout
            if cond_dropout_all_prob > 0:
                drop_all = torch.rand(z_data.shape[0], device=z_data.device) < cond_dropout_all_prob
                cond_mask = cond_mask & ~drop_all[:, None]
        else:
            cond_mask = torch.rand(z_data.shape[0], device=z_data.device) >= cond_dropout

    v_pred = model(z_t, t, cond=cond, cond_mask=cond_mask)
    return (v_pred - target).pow(2).mean()


@torch.no_grad()
def sample_latents(model, num_samples, latent_flat_dim, device,
                   cond=None, cfg_scale=1.0, ode_steps=50, generator=None, noise=None,
                   cond_mask=None, guidance_fn=None):
    """Integrate the learned ODE from noise (t=0) to data (t=1) with Euler steps.

    cond: (num_samples, cond_dim) normalized conditions or None.
    cfg_scale: 1.0 = plain conditional; > 1.0 = classifier-free guidance. The
        unconditional branch is `model(z, t, cond=None)`: the `null_cond`
        embedding in mode 'all', the all-masked row in mode 'per_dim'.
    noise: optional (num_samples, latent_flat_dim) tensor used (cloned) as the
        t=0 state instead of drawing from `generator`. The internal draw is
        exactly `torch.randn(num_samples, latent_flat_dim, device=device,
        generator=generator)`, so a caller that makes that same call itself and
        passes selected rows back in reproduces those rows' samples -- this is
        what lets `interpolate` re-integrate two rows of a seeded batch plus a
        slerp'd mixture without re-running the whole batch.
    cond_mask: optional bool mask passed through to every conditional model
        call -- (num_samples,) per-sample or (num_samples, cond_dim) per-entry
        (the latter needs a `cond_dropout_mode per_dim` network to mean
        anything; see VelocityNet). Requires `cond`. A partial request builds it
        as `~isnan(cond_values)`.
    guidance_fn: optional callable `(z_next, t_next, dt) -> delta_z` invoked
        AFTER each Euler update; the state becomes `z_next + delta_z`. `z_next`
        is the updated (num_samples, latent_flat_dim) state, `t_next` is a
        (num_samples,) tensor holding the time of that state, `(i + 1) /
        ode_steps` (exactly 1.0 after the last step; same convention as the
        model's `t` argument, so it can be fed straight back into the model),
        and `dt = 1 / ode_steps`. The callback owns its eta / dt scaling and
        must open `torch.enable_grad()` itself if it differentiates -- this
        function runs under `no_grad`. Returning None means no correction.

    With `cond_mask=None` and `guidance_fn=None` the integration is
    bit-identical to the pre-guidance sampler.
    """
    if cond_mask is not None:
        if cond is None:
            raise ValueError('cond_mask requires cond; pass cond=None to sample unconditionally')
        if not torch.is_tensor(cond_mask) or cond_mask.shape[0] != int(num_samples):
            raise ValueError(
                f'cond_mask must have {int(num_samples)} rows, got '
                f'{tuple(cond_mask.shape) if torch.is_tensor(cond_mask) else type(cond_mask).__name__}')
        cond_mask = cond_mask.to(device=device)
        if cond_mask.dtype != torch.bool:
            cond_mask = cond_mask.bool()
    if noise is not None:
        expected = (int(num_samples), int(latent_flat_dim))
        if tuple(noise.shape) != expected:
            raise ValueError(f'noise must have shape {expected}, got {tuple(noise.shape)}')
        z = noise.detach().to(device=device).clone()
    else:
        z = torch.randn(num_samples, latent_flat_dim, device=device, generator=generator)
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        if cond is not None and cfg_scale != 1.0:
            v_cond = _velocity(model, z, t, cond=cond, cond_mask=cond_mask)
            # null_cond ('all') / all-masked row ('per_dim'): the unconditional branch.
            v_uncond = model(z, t, cond=None)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = _velocity(model, z, t, cond=cond, cond_mask=cond_mask)
        z = z + v * dt
        if guidance_fn is not None:
            t_next = torch.full((num_samples,), (i + 1) / ode_steps, device=device)
            delta = guidance_fn(z, t_next, dt)
            if delta is not None:
                if tuple(delta.shape) != tuple(z.shape):
                    raise ValueError(
                        f'guidance_fn must return a delta of shape {tuple(z.shape)}, '
                        f'got {tuple(delta.shape)}')
                z = z + delta.detach().to(device=z.device, dtype=z.dtype)
    return z
