"""cHI-MGNflow -- an LDGN-style latent diffusion graph net, built on HI-MGN.

Two-stage architecture, after Lino, Pfaff & Thuerey, "Learning Distributions
of Complex Fluid Simulations with Diffusion Graph Networks" (ICLR 2025,
arXiv:2504.02843; "LDGN" is that paper's own name for its compressed-latent
variant): their winning MGN+flow design,
adopted directly, with HI-MGN's hierarchical V-cycle standing in for their own
multiscale backbone everywhere a graph network is needed -- both the
compressor and the latent-space prior. See model/autoencoder.py for the full
rationale (the paper's own VGAE ablation, why hierarchy alone does not explain
the win, and the two-pass shared-encoder fix for skip-connection leakage).

    Stage 1 (mode 'train_ae'):    HierarchyEncoder + MultiscaleDecoder learn to
                                   compress the true field onto the coarsest
                                   mesh level and back, near-losslessly
                                   (KL against N(0,I) at `ae_kl_weight`, near
                                   zero, matching the paper's ~1e-6).
    Stage 2 (mode 'train_prior'): LatentFlowPrior learns a flow-matching
                                   velocity field OVER THAT COARSE LATENT ONLY,
                                   conditioned on geometry/BC features that
                                   exist even when the field is unknown. The
                                   compressor is loaded from `ae_checkpoint`
                                   and frozen.
    mode 'train':                 back-compat convenience -- runs stage 1 to
                                   completion, checkpoints it, then runs stage
                                   2, mirroring SimulGenVAE's combined pipeline.
                                   See training_profiles/single_training.py.

Node feature layout is unchanged from the rest of the suite:

    [ state | y (true field, or zeros) | conditions | positional | node-type ]

`self.forward(task, ...)` is the single dispatch point so a DDP-wrapped model
always routes through `nn.Module.__call__` -> `forward`, regardless of stage
-- required for DDP's gradient reducer to be armed; see training_loop.py.
Anything that never backprops (`generate`, conditioning, sampling diagnostics)
may call the corresponding method directly on the unwrapped module instead.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.autoencoder import (HierarchyEncoder, MultiscaleDecoder, LatentFlowPrior,
                               extract_level_data, _run_plain)
from model.encoder_decoder import GnBlock
from model.flow import integrate, predict_mean, resolve_flow_config
from model.mlp import init_weights


class CHiMGNFlow(nn.Module):
    """Thin wrapper: construction, weight init, stage freezing, task dispatch."""

    def __init__(self, config, device: str):
        super().__init__()
        self.config = config
        self.device = device

        self.model = LatentDiffusionGraphNet(config).to(device)
        self.model.apply(init_weights)
        # init_weights kaiming-fills every nn.Linear, the near-zero-init output
        # heads (decoder, prior velocity) and the AdaLN-Zero heads included --
        # and their whole point is to start at/near the identity. Restoring
        # them AFTER the global init is what makes that guarantee hold.
        self.model.reset_zero_init_heads()

        mode = str(config.get('mode', 'train')).lower().strip()
        if mode == 'train_prior':
            self._load_frozen_ae(config, device)

        print('cHI-MGNflow model created successfully')

    def _load_frozen_ae(self, config, device):
        ae_ckpt_path = config.get('ae_checkpoint')
        if not ae_ckpt_path:
            raise ValueError(
                "mode 'train_prior' requires ae_checkpoint: path to a checkpoint "
                "saved by a completed train_ae run."
            )
        checkpoint = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
        raw_state = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
        # An 'ema_state_dict' comes from torch.optim.swa_utils.AveragedModel, which
        # wraps every key in a 'module.' prefix and adds an 'n_averaged' buffer.
        # Strip both before matching, exactly as rollout.py does.
        if any(k.startswith('module.') for k in raw_state):
            raw_state = {k[len('module.'):]: v for k, v in raw_state.items()
                         if k.startswith('module.')}
        # Checkpoints store this outer CHiMGNFlow's state_dict (see setup.py),
        # so every key carries the 'model.' prefix down to the inner
        # LatentDiffusionGraphNet -- load into self, not self.model.
        ae_state = {k: v for k, v in raw_state.items() if not k.startswith('model.prior.')}
        missing, unexpected = self.load_state_dict(ae_state, strict=False)
        missing = [k for k in missing if not k.startswith('model.prior.')]
        if missing or unexpected:
            raise RuntimeError(
                f"ae_checkpoint '{ae_ckpt_path}' does not match this model's compressor "
                f"architecture: missing={missing}, unexpected={unexpected}"
            )
        self.model.freeze_ae()
        print(f"  Loaded frozen compressor from {ae_ckpt_path}")

    def set_checkpointing(self, enabled: bool):
        self.model.set_checkpointing(enabled)

    def ae_parameters(self):
        return self.model.ae_parameters()

    def prior_parameters(self):
        return self.model.prior.parameters()

    def freeze_ae(self):
        """Freeze the compressor in place (combined mode='train', after stage 1
        completes in-memory -- no checkpoint round-trip needed, unlike the
        standalone 'train_prior' path through _load_frozen_ae)."""
        self.model.freeze_ae()

    # ── stage-specific entry points (no-grad callers may use these directly
    #    on the unwrapped module; see module docstring) ─────────────────────

    def forward_ae(self, graph):
        return self.model.forward_ae(graph)

    def encode_both_auto(self, graph):
        return self.model.encode_both_auto(graph)

    def encode_condition(self, graph):
        return self.model.encode_condition(graph)

    def forward_prior_step(self, ctx, z_t, t):
        return self.model.forward_prior_step(ctx, z_t, t)

    def decode(self, z, ctx):
        return self.model.decode(z, ctx)

    def generate(self, graph, flow_cfg=None):
        return self.model.generate(graph, flow_cfg)

    # ── DDP-safe dispatch for anything that needs a backward pass ──────────

    def forward(self, task, graph, z_t=None, t=None):
        if task == 'ae':
            return self.model.forward_ae(graph)
        if task == 'prior':
            return self.model.forward_prior_step(graph, z_t, t)
        raise ValueError(f"unknown forward task '{task}' (expected 'ae' or 'prior')")


class LatentDiffusionGraphNet(nn.Module):
    """Composes the compressor (HierarchyEncoder + MultiscaleDecoder) and the
    coarse-latent LatentFlowPrior. See model/autoencoder.py for each part."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.edge_input_size = int(config['edge_var'])
        if self.edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {self.edge_input_size}")
        self.latent_dim = int(config['latent_dim'])
        self.latent_ch = int(config.get('latent_ch', 4))
        self.use_checkpointing = bool(config.get('use_checkpointing', False))
        self.use_world_edges = bool(config.get('use_world_edges', False))
        if not config.get('use_multiscale', False):
            raise ValueError(
                "cHI-MGNflow's LDGN-style architecture compresses the field onto a "
                "coarse mesh level, which requires use_multiscale=True (with "
                "multiscale_levels >= 1 and a matching mp_per_level). A flat (non-"
                "hierarchical) HI-MGN has no coarse level to place the latent on."
            )
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False)) and self.use_world_edges
        )

        self.node_output_size = int(config['output_var'])
        num_cond = int(config.get('cond_var', 0) or 0)
        num_pos_features = int(config.get('positional_features', 0))
        base_input_size = (int(config['input_var']) + self.node_output_size
                           + num_cond + num_pos_features)
        use_node_types = config.get('use_node_types', False)
        num_node_types = int(config.get('num_node_types', 0) or 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
        else:
            self.node_input_size = base_input_size
        print(f"  Model input: {config['input_var']} state + {self.node_output_size} field(y) "
              f"+ {num_cond} conditions + {num_pos_features} positional"
              + (f" + {num_node_types} node types" if self.node_input_size != base_input_size else "")
              + f" = {self.node_input_size}")

        L = int(config.get('multiscale_levels', 1))
        self.multiscale_levels = L
        mp_per_level = config.get('mp_per_level', None)
        if mp_per_level is None:
            raise ValueError(
                "use_multiscale=True requires mp_per_level "
                "(2 * multiscale_levels + 1 entries, e.g. '4, 6, 8, 6, 4' for 2 levels)"
            )
        mp_per_level = ([int(mp_per_level)] if not isinstance(mp_per_level, list)
                        else [int(x) for x in mp_per_level])
        expected_len = 2 * L + 1
        if len(mp_per_level) != expected_len:
            raise ValueError(
                f"mp_per_level must have {expected_len} entries for {L} levels, "
                f"got {len(mp_per_level)}: {mp_per_level}"
            )
        self.mp_per_level = mp_per_level
        parts = [f"pre[{i}]={mp_per_level[i]}" for i in range(L)]
        parts.append(f"coarsest={mp_per_level[L]}")
        parts += [f"post[{i}]={mp_per_level[2 * L - i]}" for i in range(L - 1, -1, -1)]
        print(f"  Compressor V-cycle ({L} levels): {', '.join(parts)}")

        self.hierarchy_encoder = HierarchyEncoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            L, mp_per_level, use_world_edges=self.use_world_edges,
            use_coarse_world_edges=self.use_coarse_world_edges,
            use_checkpointing=self.use_checkpointing,
        )
        self.latent_head = nn.Linear(self.latent_dim, 2 * self.latent_ch)
        self.z_lift = nn.Linear(self.latent_ch, self.latent_dim)
        self.z_fuse = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.post_z_blocks = nn.ModuleList([
            GnBlock(self.latent_dim, use_world_edges=self.use_coarse_world_edges)
            for _ in range(mp_per_level[L])
        ])
        self.decoder_ascend = MultiscaleDecoder(
            self.latent_dim, self.node_output_size, L, mp_per_level,
            use_world_edges=self.use_world_edges,
            use_coarse_world_edges=self.use_coarse_world_edges,
            use_checkpointing=self.use_checkpointing,
        )
        print(f"  Coarse latent: {self.latent_ch} channels/coarse-node "
              f"(paper: 1; wider here as a capacity margin)")

        flow_cfg = resolve_flow_config(config)
        self.flow_cfg = flow_cfg
        prior_blocks = int(config.get('prior_blocks', 4))
        self.prior = LatentFlowPrior(
            self.latent_ch, self.latent_dim, flow_cfg['time_freqs'], prior_blocks,
            use_checkpointing=self.use_checkpointing,
        )
        print(f"  Latent flow prior: {prior_blocks} AdaLN-Zero GnBlocks over the "
              f"coarsest level only")

        self._ae_frozen = False

    # ── construction / lifecycle ────────────────────────────────────────────

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled
        self.hierarchy_encoder.use_checkpointing = enabled
        self.decoder_ascend.use_checkpointing = enabled
        self.prior.use_checkpointing = enabled

    def reset_zero_init_heads(self):
        """Restore near-zero-init on outputs that must start as identity/no-op.

        Must run AFTER any global apply(init_weights) (see CHiMGNFlow.__init__):
        a kaiming fill would otherwise overwrite this scaling -- same reasoning
        as the AdaLN-Zero heads it also resets.
        """
        with torch.no_grad():
            last = self.decoder_ascend.decoder.decode_module[-1]
            last.weight.mul_(0.01)
            self.prior.out_head.weight.mul_(0.01)
        self.prior.reset_time_conditioning()

    def freeze_ae(self):
        """Freeze every submodule except `prior` (mode == 'train_prior')."""
        self._ae_frozen = True
        for name, module in self.named_children():
            if name == 'prior':
                continue
            for p in module.parameters():
                p.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._ae_frozen:
            for name, module in self.named_children():
                if name != 'prior':
                    module.eval()
        return self

    def ae_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith('prior.'):
                yield p

    # ── graph plumbing ──────────────────────────────────────────────────────

    def _batch_of(self, graph):
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
        return batch

    def _build_work_graph(self, graph, y_block):
        s0 = int(self.config['input_var'])
        x_in = torch.cat([
            graph.x[:, :s0],
            y_block.to(graph.x.dtype),
            graph.x[:, s0:],
        ], dim=-1)
        work = Data(x=x_in, edge_attr=graph.edge_attr, edge_index=graph.edge_index)
        if self.use_world_edges:
            work.world_edge_attr = getattr(graph, 'world_edge_attr', None)
            work.world_edge_index = getattr(graph, 'world_edge_index', None)
        return work

    # ── compressor ──────────────────────────────────────────────────────────

    def encode_both(self, graph, level_data, batch):
        """Shared two-pass encode. Returns (mu, logvar, outs_0, coarse_batch).

        outs_0 (y-blind) is what both the decoder's skip path and the prior's
        conditioning read -- see the leakage-fix rationale in autoencoder.py.
        """
        work_y = self._build_work_graph(graph, graph.y)
        outs_y, _ = self.hierarchy_encoder(work_y, level_data, batch)
        mu, logvar = self.latent_head(outs_y[-1]['x']).chunk(2, dim=-1)
        logvar = logvar.clamp(-10.0, 10.0)

        y_blind = torch.zeros_like(graph.y)
        work_0 = self._build_work_graph(graph, y_blind)
        outs_0, coarse_batch = self.hierarchy_encoder(work_0, level_data, batch)
        return mu, logvar, outs_0, coarse_batch

    def encode_both_auto(self, graph):
        """encode_both, computing level_data/batch internally -- used by both
        forward_ae and the prior's training step (on a frozen AE, see
        training_loop.py::prior_loss)."""
        level_data = extract_level_data(graph, self.multiscale_levels)
        batch = self._batch_of(graph)
        mu, logvar, outs_0, coarse_batch = self.encode_both(graph, level_data, batch)
        ctx = {'levels': outs_0, 'level_data': level_data, 'coarse_batch': coarse_batch}
        return mu, logvar, ctx

    def encode_condition(self, graph, level_data=None):
        """y-blind pass ONLY -- geometry/BC conditioning, available even when y
        is unknown (generation time)."""
        if level_data is None:
            level_data = extract_level_data(graph, self.multiscale_levels)
        batch = self._batch_of(graph)
        y_blind = torch.zeros(graph.x.shape[0], self.node_output_size,
                              device=graph.x.device, dtype=graph.x.dtype)
        work_0 = self._build_work_graph(graph, y_blind)
        outs_0, coarse_batch = self.hierarchy_encoder(work_0, level_data, batch)
        return {'levels': outs_0, 'level_data': level_data, 'coarse_batch': coarse_batch}

    def decode(self, z, ctx):
        """z: [n_coarse, latent_ch] at the coarsest level."""
        outs_0 = ctx['levels']
        level_data = ctx['level_data']
        coarsest = outs_0[-1]
        z_h = self.z_lift(z.to(coarsest['x'].dtype))
        fused = self.z_fuse(torch.cat([coarsest['x'], z_h], dim=-1))
        cg = Data(x=fused, edge_attr=coarsest['edge_attr'], edge_index=coarsest['edge_index'])
        if self.use_coarse_world_edges and coarsest['w_attr'] is not None:
            cg.world_edge_attr = coarsest['w_attr']
            cg.world_edge_index = coarsest['w_idx']
        cg = _run_plain(self.post_z_blocks, cg, self.use_checkpointing, self.training)
        return self.decoder_ascend(cg, level_data, outs_0[:-1])

    def forward_ae(self, graph):
        """Reconstruction forward: reparameterized sample -> decode. Returns
        (y_hat, mu, logvar) for train_ae/validate_ae's recon+KL loss."""
        mu, logvar, ctx = self.encode_both_auto(graph)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        y_hat = self.decode(z, ctx)
        return y_hat, mu, logvar

    # ── prior ───────────────────────────────────────────────────────────────

    def forward_prior_step(self, ctx, z_t, t):
        """One velocity evaluation at the coarsest level. Cheap: touches only
        `prior`, never the full-mesh hierarchy -- `ctx` is computed once by
        encode_condition and reused across every ODE step."""
        coarsest = ctx['levels'][-1]
        return self.prior(z_t, t, coarsest['x'], coarsest['edge_attr'],
                          coarsest['edge_index'], ctx['coarse_batch'])

    @torch.no_grad()
    def generate(self, graph, flow_cfg=None):
        """Full inference path: encode conditioning ONCE, integrate the coarse
        latent ODE, decode. This is what realizes the LDGN inference-cost
        reduction -- the full-mesh V-cycle runs once per graph, not once per
        ODE step (contrast the old field-space flow, which paid the whole
        network per step).

        Draws exactly one sample per call ('mean' or 'sample'); 'ensemble_mean'
        must be done by the caller averaging several generate() calls in FIELD
        space (decode is nonlinear, so latents cannot be averaged first) --
        see training_loop.py / rollout.py.
        """
        cfg = flow_cfg or self.flow_cfg
        predict = cfg['predict']
        if predict == 'ensemble_mean':
            raise ValueError(
                "generate() draws exactly one sample per call; average DECODED "
                "FIELDS over several generate() calls for ensemble_mean, not "
                "latents inside one call."
            )
        ctx = self.encode_condition(graph)
        coarsest = ctx['levels'][-1]
        n_c = coarsest['x'].shape[0]
        cb = ctx['coarse_batch']
        # Hoisted out of `velocity`: an int(...item()) sync paid once here, not
        # once per ODE step -- see extract_level_data's identical warning.
        num_graphs = int(cb.max().item()) + 1 if cb.numel() else 1

        def velocity(z_cur, t_scalar):
            t = torch.full((num_graphs, 1), float(t_scalar),
                           device=z_cur.device, dtype=z_cur.dtype)
            return self.forward_prior_step(ctx, z_cur, t)

        z0 = torch.randn(n_c, self.latent_ch, device=coarsest['x'].device, dtype=coarsest['x'].dtype)
        if predict == 'mean':
            z1 = predict_mean(velocity, z0)
        else:
            z1 = integrate(velocity, z0, cfg['steps'], cfg['solver'])
        return self.decode(z1, ctx)
