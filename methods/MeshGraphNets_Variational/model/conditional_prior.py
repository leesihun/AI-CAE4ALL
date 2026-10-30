import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation

from model.blocks import AdaLNZero, apply_adaln
from model.checkpointing import checkpoint_active, run_mlp, run_plain_blocks
from model.coarsening import pool_features
from model.encoder_decoder import GnBlock
from model.latent import Latent, broadcast_rows, coarsest_view
from model.mlp import build_mlp, init_weights


class _ConditionalPriorBase(nn.Module):
    """Shared graph trunk for conditional priors on the VAE latent z.

    Encodes the input graph (node/edge features + message passing + attention
    pooling) into one conditioning vector per graph. Subclasses attach either
    an explicit density head (Gaussian mixture) or a velocity head (flow
    matching). Attribute names (node_encoder / edge_encoder / mp_layers / pool)
    are load-bearing: mixture checkpoints saved before this refactor must keep
    identical state_dict keys.
    """

    def __init__(self, config, build_pool=True):
        super().__init__()
        self.config = dict(config)
        self.z_dim = int(config.get('vae_latent_dim', 32))
        # Keep prior_hidden_dim as the backwards-compatible default, but allow
        # the graph conditioner and density head to be sized independently.
        # The old sweep changed both together, so a wider "prior" could not say
        # whether graph encoding or the velocity field had improved.
        self.hidden_dim = int(config.get(
            'prior_condition_hidden_dim',
            config.get('prior_hidden_dim', config.get('latent_dim', 128)),
        ))
        self.num_mp_layers = int(config.get('prior_mp_layers', 3))
        # The conditioner trunk runs on the fine mesh; checkpoint it like the
        # simulator's processor (training only, see checkpoint_active).
        # Set by the owning model from `use_checkpointing` (training only);
        # build_prior_config does not carry the key.
        self.use_checkpointing = False
        # Per-level z slots (matches MeshGraphNets.num_z). Default 1 for back-compat.
        if config.get('use_multiscale', False):
            default_num_z = int(config.get('multiscale_levels', 1)) + 1
        else:
            default_num_z = 1
        self.num_z = int(config.get('num_z', default_num_z))

        # graph.x layout: [state | conditions | positional | node-type one-hot]
        base_input_size = int(config.get('input_var'))
        base_input_size += int(config.get('cond_var', 0) or 0)
        base_input_size += int(config.get('positional_features', 0))
        if config.get('use_node_types', False):
            base_input_size += int(config.get('num_node_types', 0))
        edge_input_size = int(config.get('edge_var'))

        self.node_encoder = build_mlp(base_input_size, self.hidden_dim, self.hidden_dim)
        self.edge_encoder = build_mlp(edge_input_size, self.hidden_dim, self.hidden_dim)
        self.mp_layers = nn.ModuleList([
            GnBlock(self.hidden_dim, use_world_edges=False)
            for _ in range(self.num_mp_layers)
        ])
        # State-dict compatible with the deprecated GlobalAttention (same gate_nn keys).
        # Not built when nothing reads it (latent_type local), so no parameter
        # sits unused under DDP.
        if build_pool:
            self.pool = AttentionalAggregation(nn.Linear(self.hidden_dim, 1))

    def condition(self, graph):
        """Encode graph → pooled conditioning vector [B, hidden_dim]."""
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
        ckpt = checkpoint_active(self)
        h = run_mlp(self.node_encoder, graph.x, ckpt)
        e = run_mlp(self.edge_encoder, graph.edge_attr, ckpt)
        g = Data(x=h, edge_attr=e, edge_index=graph.edge_index)
        g = run_plain_blocks(self.mp_layers, g, ckpt)
        return self.pool(g.x, batch)


class _FiLMResidualBlock(nn.Module):
    """Pre-norm residual block modulated by time and graph condition."""

    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, context):
        scale, shift = self.film(context).chunk(2, dim=-1)
        # Bounding the scale prevents an early, untrained conditioner from
        # amplifying the residual stream by orders of magnitude.
        h = self.norm(x) * (1.0 + 0.1 * torch.tanh(scale)) + shift
        return (x + self.net(h)) / math.sqrt(2.0)


class _ResidualFiLMVelocity(nn.Module):
    """Higher-capacity FM velocity with explicit time/condition modulation."""

    def __init__(self, flat_dim, condition_dim, time_dim, hidden_dim, num_blocks):
        super().__init__()
        self.z_in = nn.Linear(flat_dim, hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(condition_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            _FiLMResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, flat_dim)

    def forward(self, z_t, time_embedding, condition):
        context = self.context(torch.cat([time_embedding, condition], dim=-1))
        h = self.z_in(z_t)
        for block in self.blocks:
            h = block(h, context)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


class ConditionalMixturePrior(_ConditionalPriorBase):
    """Graph-conditioned Gaussian mixture prior for VAE latent z (legacy family).

    Kept as the `prior_family gmm` fallback and for loading pre-FM checkpoints.
    At inference it replaces the global latent sampler with p(z | graph).
    """

    family = 'gmm'

    def __init__(self, config):
        super().__init__(config)
        self.num_components = int(config.get('prior_mixture_components', 10))
        self.min_std = float(config.get('prior_min_std', 0.05))
        # Low-rank covariance per mixture component: Sigma_k = L_k L_k^T + diag(psi_k).
        # 0 (default) = diagonal components (back-compat, no cov_factor emitted).
        # r > 0 lets a component capture correlated latent directions — the
        # manufacturing-spread axis a diagonal prior misses (see diag_prior_spread).
        self.cov_rank = int(config.get('prior_cov_rank', 0))
        # Start near-diagonal (weak correlations) and let training grow them.
        self.cov_factor_scale = 0.1
        # Per component: 1 logit + D mean + D diagonal log-std (+ D*rank cov factor).
        self.params_per_comp = 1 + (2 + self.cov_rank) * self.z_dim
        self.head = build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            self.num_z * self.num_components * self.params_per_comp,
            layer_norm=False,
        )
        self.apply(init_weights)

    def forward(self, graph):
        pooled = self.condition(graph)
        raw = self.head(pooled)
        bsz = raw.shape[0]
        D = self.z_dim
        # [B, num_z, K, params_per_comp]
        raw = raw.view(bsz, self.num_z, self.num_components, self.params_per_comp)

        logits = raw[..., 0]                          # [B, num_z, K]
        mu = raw[..., 1:1 + D]                        # [B, num_z, K, D]
        log_std = raw[..., 1 + D:1 + 2 * D]           # [B, num_z, K, D]
        log_std = torch.clamp(log_std, min=math.log(self.min_std), max=5.0)
        out = {'logits': logits, 'mu': mu, 'log_std': log_std}
        if self.cov_rank > 0:
            # Low-rank covariance factor L_k: [B, num_z, K, D, rank]
            factor = raw[..., 1 + 2 * D:].view(
                bsz, self.num_z, self.num_components, D, self.cov_rank)
            out['cov_factor'] = factor * self.cov_factor_scale
        return out

    @torch.no_grad()
    def sample(self, graph, temperature=1.0, inflation=1.0):
        # `inflation` is accepted for interface parity with the FM prior and
        # ignored: for a mixture, `temperature` already scales the component
        # stds directly, which is the same knob done properly.
        params = self.forward(graph)
        return sample_from_mixture(params, temperature=temperature)


class ConditionalFMPrior(_ConditionalPriorBase):
    """Graph-conditioned flow-matching prior for VAE latent z (default family).

    Instead of emitting an explicit density, learns a velocity field
    v(z_t, t, c) that transports N(0, I) onto the aggregate posterior of z for
    each graph (conditional flow matching, Lipman et al. 2023; same recipe as
    the LFMGN sampler in tum-pbs/dgn4cfd). Training is plain MSE regression on
    straight-line interpolation paths — no logsumexp, no components, hence none
    of the mixture's collapse machinery (min-std floors, KL anchor, Gumbel
    reparameterization). Sampling integrates the ODE with `prior_fm_steps`
    fixed steps of `prior_fm_solver` (Heun by default, Euler optional).

    The num_z slots are modeled jointly as one flat vector, so cross-level
    correlations between per-level latents are captured — the mixture treated
    each slot as an independent mixture.

    z is consumed at its native scale unless the run has a prior-only tail
    (`prior_freeze_epoch`): at the freeze, the z_shift / z_scale buffers are
    fitted once from the frozen posterior (fit_standardization) and the flow
    runs on (z - shift)/scale from then on. They are fitted, not running
    statistics, so EMA/DDP snapshots stay exact; the freeze copies them into
    the EMA shadow explicitly. Without a freeze (or with the conditional
    moment head) they stay at the identity.
    """

    family = 'fm'
    # Small path-endpoint noise floor from the conditional-FM objective;
    # sigma_min = 0 would make t=1 a Dirac endpoint.
    sigma_min = 1e-4

    def __init__(self, config):
        super().__init__(config)
        self.num_steps = int(config.get('prior_fm_steps', 20))
        self.solver = str(config.get('prior_fm_solver', 'heun')).lower().strip()
        if self.solver not in ('heun', 'euler'):
            raise ValueError(
                f"prior_fm_solver must be 'heun' or 'euler', got '{self.solver}'")
        self.flat_dim = self.num_z * self.z_dim
        self.velocity_hidden_dim = int(config.get(
            'prior_velocity_hidden_dim',
            config.get('prior_hidden_dim', self.hidden_dim),
        ))
        self.velocity_arch = str(config.get('prior_fm_velocity_arch', 'mlp')).lower().strip()
        if self.velocity_arch not in ('mlp', 'residual_film'):
            raise ValueError(
                "prior_fm_velocity_arch must be 'mlp' or 'residual_film', "
                f"got '{self.velocity_arch}'")
        self.velocity_blocks = int(config.get('prior_fm_blocks', 4))
        if self.velocity_blocks < 1:
            raise ValueError('prior_fm_blocks must be >= 1')
        # Fourier features of t: [sin(2^k π t), cos(2^k π t)], k = 0..15.
        freqs = (2.0 ** torch.arange(16, dtype=torch.float32)) * math.pi
        self.register_buffer('t_freqs', freqs, persistent=False)
        t_emb_dim = 2 * freqs.numel()
        if self.velocity_arch == 'mlp':
            # Exact legacy module layout when the new options are absent.
            self.velocity_net = build_mlp(
                self.flat_dim + t_emb_dim + self.hidden_dim,
                self.velocity_hidden_dim,
                self.flat_dim,
                layer_norm=False,
            )
        else:
            self.velocity_net = _ResidualFiLMVelocity(
                self.flat_dim, self.hidden_dim, t_emb_dim,
                self.velocity_hidden_dim, self.velocity_blocks,
            )

        # Optional condition-dependent affine base. It handles conditional
        # location and diagonal scale; FM transports the remaining joint,
        # non-Gaussian residual across every latent slot at once.
        self.use_conditional_moments = bool(config.get('prior_fm_moments', False))
        self.moment_loss_weight = float(config.get('prior_fm_moment_weight', 1.0))
        self.moment_min_scale = float(config.get('prior_fm_moment_min_scale', 0.03))
        self.moment_max_scale = float(config.get('prior_fm_moment_max_scale', 20.0))
        if not 0.0 < self.moment_min_scale <= self.moment_max_scale:
            raise ValueError('prior_fm_moment scales must satisfy 0 < min <= max')
        if self.moment_loss_weight < 0.0:
            raise ValueError('prior_fm_moment_weight must be >= 0')
        if self.use_conditional_moments:
            moment_hidden = int(config.get('prior_fm_moment_hidden_dim', self.hidden_dim))
            self.moment_head = build_mlp(
                self.hidden_dim, moment_hidden, 2 * self.flat_dim,
                layer_norm=False,
            )
        else:
            self.moment_head = None
        # Latent standardization. The FM path runs on (z - shift)/scale and
        # integration output is mapped back, so the velocity net always works
        # in a unit-scale space while callers keep decoder units. Identity by
        # default: a prior fitted without it behaves exactly as before.
        #
        # It matters because MMD pins the AGGREGATE q(z) to N(0,I), and with
        # many realizations of a few parts that aggregate is a mixture: total
        # variance 1 is compatible with every per-part cloud -- what the prior
        # must hit -- being far smaller and off-origin.
        self.register_buffer('z_shift', torch.zeros(self.flat_dim))
        self.register_buffer('z_scale', torch.ones(self.flat_dim))
        self.apply(init_weights)
        if self.moment_head is not None:
            # Start from mu(c)=0, scale(c)=1 so the initial flow-space contract
            # matches the baseline arm exactly.
            nn.init.zeros_(self.moment_head[4].weight)
            nn.init.zeros_(self.moment_head[4].bias)

    @torch.no_grad()
    def fit_standardization(self, mu, logvar, eps=1e-6):
        """Set z_shift / z_scale from posterior parameters over a dataset.

        mu, logvar: [N, num_z, D] (or already flat [N, flat_dim]).

        A posterior SAMPLE is mu + sigma*eps, so across the set its mean is
        E[mu] and its variance Var(mu) + E[sigma^2]. Standardizing by those is
        what puts the FM path's endpoint at unit scale -- the target the
        velocity net actually regresses -- while sample_n undoes it, so callers
        keep decoder units. Returns (shift, scale) for logging.
        """
        flat_mu = mu.reshape(mu.shape[0], -1).double()
        flat_var = logvar.reshape(logvar.shape[0], -1).double().exp()
        shift = flat_mu.mean(dim=0)
        scale = (flat_mu.var(dim=0, unbiased=False) + flat_var.mean(dim=0)).sqrt()
        scale = scale.clamp(min=eps)
        self.z_shift.copy_(shift.to(self.z_shift))
        self.z_scale.copy_(scale.to(self.z_scale))
        return shift.float(), scale.float()

    def _t_embed(self, t):
        ang = t * self.t_freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def velocity(self, z_t, t, cond):
        """v_θ(z_t, t, c): [B, flat_dim] × [B, 1] × [B, hidden] → [B, flat_dim]."""
        t_emb = self._t_embed(t)
        if self.velocity_arch == 'mlp':
            return self.velocity_net(torch.cat([z_t, t_emb, cond], dim=-1))
        return self.velocity_net(z_t, t_emb, cond)

    def moment_parameters(self, cond):
        """Return condition-dependent diagonal base mean and scale."""
        if self.moment_head is None:
            shape = (cond.shape[0], self.flat_dim)
            return cond.new_zeros(shape), cond.new_ones(shape)
        mean, raw_log_scale = self.moment_head(cond).chunk(2, dim=-1)
        log_scale = raw_log_scale.clamp(
            min=math.log(self.moment_min_scale),
            max=math.log(self.moment_max_scale),
        )
        return mean, log_scale.exp()

    def _flow_to_standardized_z(self, flow_z, cond):
        """Map residual-flow coordinates to globally standardized decoder z."""
        if self.moment_head is None:
            return flow_z
        mean, scale = self.moment_parameters(cond)
        return mean + scale * flow_z

    def fm_loss(self, cond, target_z, target_mu=None, target_logvar=None,
                objective='all'):
        """Conditional flow-matching MSE on a detached posterior sample.

        Path: z_t = (1 − (1−σ_min)·t)·z0 + t·z1,  target v* = z1 − (1−σ_min)·z0.
        The regression optimum at (z_t, t) is the marginal velocity
        E[v* | z_t, c], whose ODE flow transports N(0,I) exactly onto the
        distribution of z1 — single-sample noise averages out instead of
        sculpting spurious sharp modes (the mixture-NLL failure).

        Args:
            cond:     [B, hidden_dim] pooled condition. Gradient flows through
                      it into the trunk, so the trunk trains jointly.
            target_z: [B, num_z, D] fresh detached posterior sample.
        Returns scalar loss (fp32).
        """
        if objective not in ('all', 'moment', 'flow'):
            raise ValueError("FM objective must be 'all', 'moment' or 'flow'")
        if objective == 'moment' and self.moment_head is None:
            raise ValueError('moment-only FM objective requires prior_fm_moments=True')
        with _autocast_disabled_for(cond):
            z1 = target_z.reshape(target_z.shape[0], -1).float()
            z1 = (z1 - self.z_shift) / self.z_scale
            c = cond.float()
            moment_loss = z1.new_zeros(())
            if self.moment_head is not None:
                moment_mean, moment_scale = self.moment_parameters(c)
                # The moment head has its own proper likelihood objective. Do
                # not let the residual FM redefine its coordinate system.
                residual_target = (
                    (z1 - moment_mean.detach()) / moment_scale.detach()
                )
                if objective != 'flow' and target_mu is not None and target_logvar is not None:
                    q_mean = target_mu.reshape(target_mu.shape[0], -1).float()
                    q_var = target_logvar.reshape(target_logvar.shape[0], -1).float().exp()
                    q_mean = (q_mean - self.z_shift) / self.z_scale
                    q_var = q_var / self.z_scale.square()
                    second_moment = q_var + (q_mean - moment_mean).square()
                    moment_loss = 0.5 * (
                        second_moment / moment_scale.square()
                        + 2.0 * moment_scale.log()
                    ).mean()
                elif objective != 'flow':
                    moment_loss = 0.5 * (
                        (z1 - moment_mean).square() / moment_scale.square()
                        + 2.0 * moment_scale.log()
                    ).mean()
                z1 = residual_target
            if objective == 'moment':
                return self.moment_loss_weight * moment_loss
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], 1, device=z1.device)
            s = 1.0 - self.sigma_min
            z_t = (1.0 - s * t) * z0 + t * z1
            target_v = z1 - s * z0
            pred_v = self.velocity(z_t, t, c)
            flow_loss = torch.nn.functional.mse_loss(pred_v, target_v)
            if objective == 'flow':
                return flow_loss
            return flow_loss + self.moment_loss_weight * moment_loss

    def _integrate(self, z, c, steps):
        """Transport z ~ N(0,I) along v_theta from t=0 to t=1.

        'heun' is the second-order explicit trapezoid (predictor-corrector):
        local error O(dt^3) against Euler's O(dt^2), for two velocity
        evaluations per step. The FM velocity is largest and most curved near
        t=1, which is exactly where first-order Euler overshoots into the z
        tails — the "Gen_max spike" that forced prior_fm_steps from 20 up to
        100. Heun at ~25 steps is more accurate than Euler at 100 and half the
        cost. 'euler' keeps the old integrator for comparison.
        """
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((z.shape[0], 1), k * dt, device=z.device)
            v1 = self.velocity(z, t, c)
            if self.solver == 'euler':
                z = z + dt * v1
                continue
            v2 = self.velocity(z + dt * v1, t + dt, c)
            z = z + 0.5 * dt * (v1 + v2)
        return z

    @torch.no_grad()
    def sample(self, graph, temperature=1.0, inflation=1.0):
        """One z per graph: [B, num_z, D]. See sample_n for `inflation`."""
        return self.sample_n(graph, 1, temperature=temperature,
                             inflation=inflation)[:, 0]

    @torch.no_grad()
    def sample_n_from_pooled(self, cond, n, steps=None):
        """sample_n but taking the pooled conditioning vector directly.

        The training loop already computed it on the forward pass; recomputing
        condition(graph) would rerun the whole prior trunk. A reduced `steps`
        (e.g. 8) is fine for training-time regularizers; inference keeps the
        full num_steps.
        """
        steps = int(steps or self.num_steps)
        c = cond.float().repeat_interleave(n, dim=0)
        z = torch.randn(c.shape[0], self.flat_dim, device=c.device)
        z = self._integrate(z, c, steps)
        z = self._flow_to_standardized_z(z, c)
        z = z * self.z_scale + self.z_shift
        return z.view(cond.shape[0], n, self.num_z, self.z_dim).to(cond.dtype)

    @torch.no_grad()
    def sample_n(self, graph, n, temperature=1.0, inflation=1.0):
        """Draw n z samples per graph via ODE integration: [B, n, num_z, D].

        The trunk runs once per graph; only the (tiny) velocity net is called
        per step. Temperature scales the initial noise std by sqrt(temperature)
        — the nearest analog of mixture covariance scaling. Note that it is NOT
        a clean width knob: the velocity field is only trained along the
        trajectories N(0,I) produces, so a wider start leaves them.

        `inflation` is the clean width knob. After integration every sample is
        pulled away from the per-graph center by that factor,

            z <- c + inflation * (z - c),    c = the ODE image of z0 = 0,

        so the decoder sees a latent cloud `inflation` times wider around the
        same center and the field distribution widens with it. 1.0 is the
        identity. This is the standard ensemble-inflation fix for a sampler
        whose distribution has the right shape and location but too little
        spread -- which is what MGN-V's spread histograms show, uniformly, on
        every arm and eval set (sd_ratio ~ 0.5 while |dmean|/sd stays < 0.3).
        """
        cond = self.condition(graph)
        inflation = float(inflation)
        with _autocast_disabled_for(cond):
            c = cond.float().repeat_interleave(n, dim=0)     # [B*n, hidden]
            bn = c.shape[0]
            z = torch.randn(bn, self.flat_dim, device=c.device)
            z = z * math.sqrt(max(float(temperature), 1e-6))
            z = self._integrate(z, c, self.num_steps)
            z = self._flow_to_standardized_z(z, c)
            if inflation != 1.0:
                center = self._integrate(torch.zeros_like(z), c, self.num_steps)
                center = self._flow_to_standardized_z(center, c)
                z = center + inflation * (z - center)
            z = z * self.z_scale + self.z_shift
        B = cond.shape[0]
        return z.view(B, n, self.num_z, self.z_dim).to(cond.dtype)


class CoarseFMPrior(_ConditionalPriorBase):
    """Flow-matching prior for latent_type local / hybrid.

    p(z_l | graph) lives on the coarsest V-cycle level, one C-vector per
    coarsest node, so its velocity field is a GNN over that level's graph
    rather than an MLP over a flat vector:

      trunk:  the shared prior trunk on the fine mesh, node features
              mean-pooled into the coarsest clusters  -> cond_x [n_c, H]
      v_l:    [z_l_t | cond_x] -> in_proj -> prior_fm_blocks GnBlocks, each
              AdaLN-Zero-modulated by the time embedding -> out_head -> [n_c, C]

    hybrid models (z_g, z_l) JOINTLY: a per-graph token starts from
    [z_g_t | attention-pooled trunk], and before every block it reads the
    mean of the node stream and writes back into it, so the two parts see
    each other through the whole integration. g_head reads [token | node mean]
    and emits v_g [B, num_z*D].

    The path, solver, temperature and `inflation` follow ConditionalFMPrior.
    z_l is standardized per channel (zl_shift / zl_scale), z_g per flat
    coordinate (z_shift / z_scale) -- fitted at the prior freeze, identity
    otherwise.

    condition(graph) returns a context dict, not a pooled vector; every other
    method takes that dict where the global prior takes `cond`.
    """

    family = 'fm'
    sigma_min = 1e-4
    use_conditional_moments = False
    velocity_arch = 'coarse_gnn'

    def __init__(self, config):
        latent_type = str(config.get('latent_type', 'local')).lower().strip()
        if latent_type not in ('local', 'hybrid'):
            raise ValueError(f"CoarseFMPrior is for latent_type local/hybrid, got '{latent_type}'")
        super().__init__(config, build_pool=(latent_type == 'hybrid'))
        self.latent_type = latent_type
        self.hybrid = latent_type == 'hybrid'
        self.latent_ch = int(config.get('latent_ch', 4))
        self.levels = int(config.get('multiscale_levels', 1))
        self.num_steps = int(config.get('prior_fm_steps', 20))
        self.solver = str(config.get('prior_fm_solver', 'heun')).lower().strip()
        if self.solver not in ('heun', 'euler'):
            raise ValueError(
                f"prior_fm_solver must be 'heun' or 'euler', got '{self.solver}'")
        self.flat_dim = self.num_z * self.z_dim
        H = self.hidden_dim
        Hv = int(config.get('prior_velocity_hidden_dim',
                            config.get('prior_hidden_dim', H)))
        self.velocity_hidden_dim = Hv
        self.velocity_blocks = int(config.get('prior_fm_blocks', 4))
        if self.velocity_blocks < 1:
            raise ValueError('prior_fm_blocks must be >= 1')
        freqs = (2.0 ** torch.arange(16, dtype=torch.float32)) * math.pi
        self.register_buffer('t_freqs', freqs, persistent=False)
        t_dim = 2 * freqs.numel()

        self.coarse_edge_encoder = build_mlp(int(config.get('edge_var')), Hv, Hv)
        self.in_proj = nn.Linear(self.latent_ch + H, Hv)
        self.blocks = nn.ModuleList([
            GnBlock(Hv, use_world_edges=False) for _ in range(self.velocity_blocks)
        ])
        self.time_mod = nn.ModuleList([
            AdaLNZero(Hv, t_dim) for _ in range(self.velocity_blocks)
        ])
        self.out_head = nn.Linear(Hv, self.latent_ch)
        if self.hybrid:
            self.g_in = nn.Linear(self.flat_dim + H, Hv)
            self.g_update = nn.ModuleList([
                build_mlp(2 * Hv + t_dim, Hv, Hv) for _ in range(self.velocity_blocks)
            ])
            self.g_out = nn.ModuleList([
                nn.Linear(Hv, Hv) for _ in range(self.velocity_blocks)
            ])
            self.g_head = nn.Linear(2 * Hv, self.flat_dim)
            self.register_buffer('z_shift', torch.zeros(self.flat_dim))
            self.register_buffer('z_scale', torch.ones(self.flat_dim))
        self.register_buffer('zl_shift', torch.zeros(self.latent_ch))
        self.register_buffer('zl_scale', torch.ones(self.latent_ch))

        self.apply(init_weights)
        with torch.no_grad():
            for m in self.time_mod:
                m.reset_identity()
            # Near-zero velocity at init; the token's write-back starts closed.
            self.out_head.weight.mul_(0.01)
            if self.hybrid:
                self.g_head.weight.mul_(0.01)
                for lin in self.g_out:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)

    # -- conditioning ---------------------------------------------------------

    def condition(self, graph):
        """Trunk on the fine mesh -> coarsest-level context dict."""
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
        ckpt = checkpoint_active(self)
        g = Data(x=run_mlp(self.node_encoder, graph.x, ckpt),
                 edge_attr=run_mlp(self.edge_encoder, graph.edge_attr, ckpt),
                 edge_index=graph.edge_index)
        g = run_plain_blocks(self.mp_layers, g, ckpt)
        cv = coarsest_view(graph, self.levels)
        B = int(getattr(graph, 'num_graphs', 0) or (int(batch.max().item()) + 1))
        ctx = {
            'x': pool_features(g.x, cv['parent'], cv['n_c']),
            'e': self.coarse_edge_encoder(cv['c_ea']),
            'c_ei': cv['c_ei'],
            'batch': cv['batch'],
            'B': B,
        }
        if self.hybrid:
            ctx['pooled'] = self.pool(g.x, batch)
        return ctx

    @staticmethod
    def _replicate(ctx, n):
        """n disjoint copies of the coarse graph, draw-major (copy k = rows k*n_c...)."""
        if n == 1:
            return ctx
        n_c = ctx['x'].shape[0]
        B = ctx['B']
        E = ctx['c_ei'].shape[1]
        dev = ctx['x'].device
        node_off = (torch.arange(n, device=dev) * n_c).repeat_interleave(E)
        graph_off = (torch.arange(n, device=dev) * B).repeat_interleave(n_c)
        out = {
            'x': ctx['x'].repeat(n, 1),
            'e': ctx['e'].repeat(n, 1),
            'c_ei': ctx['c_ei'].repeat(1, n) + node_off.unsqueeze(0),
            'batch': ctx['batch'].repeat(n) + graph_off,
            'B': B * n,
        }
        if 'pooled' in ctx:
            out['pooled'] = ctx['pooled'].repeat(n, 1)
        return out

    @staticmethod
    def _float_ctx(ctx):
        return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in ctx.items()}

    # -- velocity -------------------------------------------------------------

    def _t_embed(self, t):
        ang = t * self.t_freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def velocity(self, zl_t, zg_t, t, ctx):
        """(v_l [n_c, C], v_g [B, flat] or None) at per-graph time t [B, 1]."""
        batch, B = ctx['batch'], ctx['B']
        t_emb = self._t_embed(t)
        t_node = broadcast_rows(t_emb, batch)
        x = self.in_proj(torch.cat([zl_t, ctx['x']], dim=-1))
        tok = self.g_in(torch.cat([zg_t, ctx['pooled']], dim=-1)) if self.hybrid else None
        graph = Data(x=x, edge_attr=ctx['e'], edge_index=ctx['c_ei'])
        for j, block in enumerate(self.blocks):
            if self.hybrid:
                x_mean = pool_features(graph.x, batch, B)
                tok = tok + self.g_update[j](torch.cat([tok, x_mean, t_emb], dim=-1))
                graph.x = graph.x + broadcast_rows(self.g_out[j](tok), batch)
            graph = apply_adaln(block, graph, self.time_mod[j], t_node)
        v_l = self.out_head(graph.x)
        if not self.hybrid:
            return v_l, None
        x_mean = pool_features(graph.x, batch, B)
        return v_l, self.g_head(torch.cat([tok, x_mean], dim=-1))

    # -- standardization ------------------------------------------------------

    @torch.no_grad()
    def fit_standardization(self, mu, logvar, eps=1e-6):
        """Fit the shift/scale buffers from posterior parameters over a dataset.

        mu, logvar: Latent with l [rows, C] (and g [N, num_z, D] for hybrid).
        Same moment matching as ConditionalFMPrior; z_l per channel, pooling
        every coarsest node of every graph. Returns (shift, scale) flattened
        [z_g coords..., z_l channels] for logging.
        """
        l_mu = mu.l.double()
        shift_l = l_mu.mean(dim=0)
        scale_l = (l_mu.var(dim=0, unbiased=False)
                   + logvar.l.double().exp().mean(dim=0)).sqrt().clamp(min=eps)
        self.zl_shift.copy_(shift_l.to(self.zl_shift))
        self.zl_scale.copy_(scale_l.to(self.zl_scale))
        shifts, scales = [shift_l], [scale_l]
        if self.hybrid:
            g_mu = mu.g.reshape(mu.g.shape[0], -1).double()
            g_var = logvar.g.reshape(logvar.g.shape[0], -1).double().exp()
            shift_g = g_mu.mean(dim=0)
            scale_g = (g_mu.var(dim=0, unbiased=False) + g_var.mean(dim=0)).sqrt().clamp(min=eps)
            self.z_shift.copy_(shift_g.to(self.z_shift))
            self.z_scale.copy_(scale_g.to(self.z_scale))
            shifts.insert(0, shift_g)
            scales.insert(0, scale_g)
        return torch.cat(shifts).float(), torch.cat(scales).float()

    # -- training -------------------------------------------------------------

    def fm_loss(self, cond, target_z, target_mu=None, target_logvar=None,
                objective='all'):
        """Conditional flow-matching MSE on a detached posterior sample Latent.

        One t per graph, shared by z_g and every coarsest node of that graph.
        hybrid sums the z_l and z_g regressions.
        """
        if objective == 'moment':
            raise ValueError('the coarse FM prior has no moment head')
        if objective not in ('all', 'flow'):
            raise ValueError("FM objective must be 'all', 'moment' or 'flow'")
        with _autocast_disabled_for(cond['x']):
            ctx = self._float_ctx(cond)
            s = 1.0 - self.sigma_min
            t = torch.rand(ctx['B'], 1, device=ctx['x'].device)
            t_l = t[ctx['batch']]
            z1_l = (target_z.l.float() - self.zl_shift) / self.zl_scale
            z0_l = torch.randn_like(z1_l)
            zl_t = (1.0 - s * t_l) * z0_l + t_l * z1_l
            zg_t = tgt_g = None
            if self.hybrid:
                z1_g = target_z.g.reshape(target_z.g.shape[0], -1).float()
                z1_g = (z1_g - self.z_shift) / self.z_scale
                z0_g = torch.randn_like(z1_g)
                zg_t = (1.0 - s * t) * z0_g + t * z1_g
                tgt_g = z1_g - s * z0_g
            v_l, v_g = self.velocity(zl_t, zg_t, t, ctx)
            loss = torch.nn.functional.mse_loss(v_l, z1_l - s * z0_l)
            if self.hybrid:
                loss = loss + torch.nn.functional.mse_loss(v_g, tgt_g)
            return loss

    # -- sampling -------------------------------------------------------------

    def _integrate(self, zl, zg, ctx, steps):
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((ctx['B'], 1), k * dt, device=zl.device)
            v1_l, v1_g = self.velocity(zl, zg, t, ctx)
            if self.solver == 'euler':
                zl = zl + dt * v1_l
                zg = None if zg is None else zg + dt * v1_g
                continue
            v2_l, v2_g = self.velocity(
                zl + dt * v1_l, None if zg is None else zg + dt * v1_g, t + dt, ctx)
            zl = zl + 0.5 * dt * (v1_l + v2_l)
            zg = None if zg is None else zg + 0.5 * dt * (v1_g + v2_g)
        return zl, zg

    @torch.no_grad()
    def sample_n_from_pooled(self, cond, n, steps=None, temperature=1.0, inflation=1.0):
        """n draws from a context dict: Latent(g [B, n, num_z, D], l [n_c, n, C])."""
        steps = int(steps or self.num_steps)
        dtype = cond['x'].dtype
        n_c, B = cond['x'].shape[0], cond['B']
        with _autocast_disabled_for(cond['x']):
            ctx = self._replicate(self._float_ctx(cond), n)
            dev = ctx['x'].device
            amp = math.sqrt(max(float(temperature), 1e-6))
            zl = torch.randn(ctx['x'].shape[0], self.latent_ch, device=dev) * amp
            zg = (torch.randn(ctx['B'], self.flat_dim, device=dev) * amp
                  if self.hybrid else None)
            zl, zg = self._integrate(zl, zg, ctx, steps)
            inflation = float(inflation)
            if inflation != 1.0:
                cl, cg = self._integrate(torch.zeros_like(zl),
                                         None if zg is None else torch.zeros_like(zg),
                                         ctx, steps)
                zl = cl + inflation * (zl - cl)
                zg = None if zg is None else cg + inflation * (zg - cg)
            zl = zl * self.zl_scale + self.zl_shift
            l = zl.view(n, n_c, self.latent_ch).transpose(0, 1).to(dtype)
            g = None
            if self.hybrid:
                zg = zg * self.z_scale + self.z_shift
                g = zg.view(n, B, self.num_z, self.z_dim).transpose(0, 1).to(dtype)
        return Latent(g, l)

    @torch.no_grad()
    def sample_n(self, graph, n, temperature=1.0, inflation=1.0):
        """n draws per graph: Latent(g [B, n, num_z, D] or None, l [n_c, n, C])."""
        return self.sample_n_from_pooled(self.condition(graph), n,
                                         temperature=temperature, inflation=inflation)

    @torch.no_grad()
    def sample(self, graph, temperature=1.0, inflation=1.0):
        """One draw per graph: Latent(g [B, num_z, D] or None, l [n_c, C])."""
        z = self.sample_n(graph, 1, temperature=temperature, inflation=inflation)
        return Latent(None if z.g is None else z.g[:, 0], z.l[:, 0])


def build_conditional_prior(config):
    """Instantiate the conditional prior selected by `prior_family`.

    'fm' (default) → ConditionalFMPrior; 'gmm' → ConditionalMixturePrior.
    latent_type local/hybrid → CoarseFMPrior.
    """
    prior_config = build_prior_config(config)
    family = prior_config.get('prior_family', 'fm')
    if prior_config.get('latent_type', 'global') != 'global':
        # local / hybrid: z has a coarse-node part, so the prior is a GNN on the
        # coarsest level. Only the fm family exists for it.
        if family != 'fm':
            raise ValueError(f"latent_type '{prior_config['latent_type']}' needs prior_family fm")
        return CoarseFMPrior(prior_config)
    if family == 'gmm':
        return ConditionalMixturePrior(prior_config)
    if family != 'fm':
        raise ValueError(f"Unknown prior_family '{family}' (expected 'fm' or 'gmm')")
    return ConditionalFMPrior(prior_config)


def _autocast_disabled_for(tensor):
    device_type = tensor.device.type
    if device_type in ('cuda', 'cpu'):
        return torch.amp.autocast(device_type, enabled=False)
    return nullcontext()


def _lowrank_mvn(mu, log_std, cov_factor):
    """Batched low-rank-plus-diagonal Gaussian: Sigma = L L^T + diag(exp(2*log_std)).

    Construct and consume this distribution with autocast disabled.
    LowRankMultivariateNormal uses a capacitance Cholesky internally, and CUDA
    does not implement Cholesky for bfloat16.
    """
    return torch.distributions.LowRankMultivariateNormal(
        loc=mu.float(),
        cov_factor=cov_factor.float(),
        cov_diag=torch.exp(2.0 * log_std.float()),
        validate_args=False,
    )


def _lowrank_log_prob(mu, log_std, cov_factor, value):
    with _autocast_disabled_for(mu):
        return _lowrank_mvn(mu, log_std, cov_factor).log_prob(value.float())


def _lowrank_sample(mu, log_std, cov_factor, *, temperature=1.0, reparameterized=False):
    temp = max(float(temperature), 1e-6)
    with _autocast_disabled_for(mu):
        mvn = torch.distributions.LowRankMultivariateNormal(
            loc=mu.float(),
            cov_factor=cov_factor.float() * math.sqrt(temp),
            cov_diag=torch.exp(2.0 * log_std.float()) * temp,
            validate_args=False,
        )
        return mvn.rsample() if reparameterized else mvn.sample()


def mixture_nll(params, target_z):
    """Negative log likelihood of target_z under a Gaussian mixture.

    Components are diagonal, or low-rank-plus-diagonal when params has
    'cov_factor' (Sigma_k = L_k L_k^T + diag(exp(2*log_std_k))).

    Shapes:
        target_z:  [B, num_z, D]            or [MC, B, num_z, D] for MC stacks
        logits:    [B, num_z, K]
        mu:        [B, num_z, K, D]
        log_std:   [B, num_z, K, D]
        cov_factor:[B, num_z, K, D, rank]   (optional)
    """
    if target_z.dim() == 4:
        losses = [mixture_nll(params, target_z[i]) for i in range(target_z.shape[0])]
        return torch.stack(losses).mean()

    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']
    cov_factor = params.get('cov_factor')

    z = target_z.unsqueeze(2)  # [B, num_z, 1, D]
    if cov_factor is not None:
        comp_log_prob = _lowrank_log_prob(
            mu, log_std, cov_factor, z,
        )  # [B, num_z, K]
    else:
        var_term = ((z - mu) / torch.exp(log_std)).pow(2)
        comp_log_prob = -0.5 * (
            var_term.sum(dim=-1)
            + 2.0 * log_std.sum(dim=-1)
            + target_z.shape[-1] * math.log(2.0 * math.pi)
        )  # [B, num_z, K]
    log_mix = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_mix + comp_log_prob, dim=-1).mean()


def analytical_prior_kl_loss(params, q_mu, q_logvar):
    """Variational upper bound on KL(q(z|y) || prior_mixture(z|graph)).

    Uses Jensen's inequality on log Σ_k π_k N_k(z):
        log p(z) ≥ Σ_k π_k log N_k(z) + H(π)
    Hence H(q, p) = -E_q[log p] ≤ -Σ_k π_k E_q[log N_k(z)] - H(π)

    E_q[log N(z | μ_k, σ_k²)] is closed-form when q is diagonal Gaussian:
        = -½ [D log 2π + Σ_d log σ_k_d² + Σ_d (σ_q_d² + (μ_q_d - μ_k_d)²) / σ_k_d²]

    Critically, this loss is computed against the full posterior distribution
    (μ_q, σ_q), not against single Monte Carlo samples. That eliminates the
    component-overfitting failure of MC NLL where prior components collapse to
    individual posterior samples rather than covering the posterior's spread.

    Args:
        params:    dict with 'logits' [B, K], 'mu' [B, K, D], 'log_std' [B, K, D]
        q_mu:      posterior mean [B, D]
        q_logvar:  posterior log variance [B, D]

    Returns:
        Scalar loss (upper bound on KL(q || p) up to the constant H(q)).
    """
    # Per-level shapes:
    #   q_mu, q_logvar: [B, num_z, D]
    #   logits:         [B, num_z, K]
    #   mu, log_std:    [B, num_z, K, D]
    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']

    q_mu_b = q_mu.unsqueeze(2)                # [B, num_z, 1, D]
    q_var_b = torch.exp(q_logvar).unsqueeze(2)
    D = q_mu.shape[-1]

    var_k = torch.exp(2.0 * log_std)
    cov_factor = params.get('cov_factor')
    if cov_factor is not None:
        # Small stability anchor: match the prior's per-dim MARGINAL variance
        # (diagonal of L L^T + diag); cross-correlations are left to mc_nll.
        var_k = var_k + cov_factor.pow(2).sum(dim=-1)
    log_var_k = torch.log(var_k.clamp_min(1e-8))

    expected_log_pk = -0.5 * (
        D * math.log(2.0 * math.pi)
        + log_var_k.sum(dim=-1)
        + ((q_var_b + (q_mu_b - mu).pow(2)) / var_k).sum(dim=-1)
    )  # [B, num_z, K]

    log_pi = torch.log_softmax(logits, dim=-1)
    pi = log_pi.exp()

    weighted_term = -(pi * expected_log_pk).sum(dim=-1)   # [B, num_z]
    entropy_pi = -(pi * log_pi).sum(dim=-1)               # [B, num_z]

    return (weighted_term - entropy_pi).mean()


def sample_from_mixture(params, temperature=1.0):
    """Sample z of shape [B, num_z, D]."""
    logits = params['logits']        # [B, num_z, K]
    mu = params['mu']                # [B, num_z, K, D]
    log_std = params['log_std']      # [B, num_z, K, D]

    temp = max(float(temperature), 1e-6)
    cat = torch.distributions.Categorical(logits=logits / temp)
    component = cat.sample()         # [B, num_z]

    B, num_z = component.shape
    b_idx = torch.arange(B, device=logits.device).view(B, 1).expand(B, num_z)
    z_idx = torch.arange(num_z, device=logits.device).view(1, num_z).expand(B, num_z)

    chosen_mu = mu[b_idx, z_idx, component]                  # [B, num_z, D]
    chosen_log_std = log_std[b_idx, z_idx, component]        # [B, num_z, D]

    cov_factor = params.get('cov_factor')
    if cov_factor is not None:
        chosen_factor = cov_factor[b_idx, z_idx, component]  # [B, num_z, D, rank]
        # Temperature scales the covariance by `temp` (std by sqrt(temp)).
        return _lowrank_sample(
            chosen_mu, chosen_log_std, chosen_factor, temperature=temp,
        ).to(chosen_mu.dtype)

    chosen_std = torch.exp(chosen_log_std) * math.sqrt(temp)
    return chosen_mu + chosen_std * torch.randn_like(chosen_std)


def build_prior_config(config):
    use_multiscale = bool(config.get('use_multiscale', False))
    default_num_z = (int(config.get('multiscale_levels', 1)) + 1) if use_multiscale else 1
    return {
        'input_var': config.get('input_var'),
        'cond_var': config.get('cond_var', 0),
        'edge_var': config.get('edge_var'),
        'latent_dim': config.get('latent_dim'),
        'vae_latent_dim': config.get('vae_latent_dim'),
        'positional_features': config.get('positional_features', 0),
        'use_node_types': config.get('use_node_types', False),
        'num_node_types': config.get('num_node_types', 0),
        'use_multiscale': use_multiscale,
        'multiscale_levels': config.get('multiscale_levels', 1),
        'num_z': int(config.get('num_z', default_num_z)),
        'latent_type': str(config.get('latent_type', 'global')).lower().strip(),
        'latent_ch': int(config.get('latent_ch', 4)),
        'prior_family': str(config.get('prior_family', 'fm')).lower().strip(),
        'prior_hidden_dim': config.get('prior_hidden_dim', config.get('latent_dim')),
        'prior_condition_hidden_dim': config.get(
            'prior_condition_hidden_dim',
            config.get('prior_hidden_dim', config.get('latent_dim')),
        ),
        'prior_mp_layers': config.get('prior_mp_layers', 10),
        # fm family only:
        'prior_fm_steps': config.get('prior_fm_steps', 20),
        'prior_fm_solver': config.get('prior_fm_solver', 'heun'),
        'prior_velocity_hidden_dim': config.get(
            'prior_velocity_hidden_dim',
            config.get('prior_hidden_dim', config.get('latent_dim')),
        ),
        'prior_fm_velocity_arch': str(
            config.get('prior_fm_velocity_arch', 'mlp')
        ).lower().strip(),
        'prior_fm_blocks': config.get('prior_fm_blocks', 4),
        'prior_fm_moments': config.get('prior_fm_moments', False),
        'prior_fm_moment_hidden_dim': config.get(
            'prior_fm_moment_hidden_dim',
            config.get('prior_condition_hidden_dim',
                       config.get('prior_hidden_dim', config.get('latent_dim'))),
        ),
        'prior_fm_moment_weight': config.get('prior_fm_moment_weight', 1.0),
        'prior_fm_moment_min_scale': config.get('prior_fm_moment_min_scale', 0.03),
        'prior_fm_moment_max_scale': config.get('prior_fm_moment_max_scale', 20.0),
        # gmm family only:
        'prior_mixture_components': config.get('prior_mixture_components', 50),
        'prior_min_std': config.get('prior_min_std', 0.1),
        'prior_cov_rank': config.get('prior_cov_rank', 0),
    }
