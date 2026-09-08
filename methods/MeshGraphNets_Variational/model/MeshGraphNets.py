import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.blocks import AdaLNZero, apply_adaln
from model.checkpointing import process_with_checkpointing
from model.coarsening import pool_features
from model.conditional_prior import build_conditional_prior
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp, init_weights
from model.vae import GNNVariationalEncoder, gather_across_ranks


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super().__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)
        # init_weights kaiming-fills every nn.Linear, including the AdaLN-Zero
        # modulation heads whose whole point is to start at the identity.
        # Restore them afterwards (no-op when z_conditioning is 'concat').
        self.model.reset_z_conditioning()

        # Scale decoder's last layer for better initial predictions.
        # T>1 (delta prediction): scale to ~0 ("predict no change" prior)
        num_timesteps = config.get('num_timesteps', None)
        if num_timesteps is None or num_timesteps > 1:
            with torch.no_grad():
                last_layer = self.model.decoder.decode_module[-1]
                last_layer.weight.mul_(0.01)

        # Joint-trained conditional prior (prior_type gnn_e2e).
        # Lives on the outer MeshGraphNets so DDP/EMA wrap both VAE and prior together.
        prior_type = str(config.get('prior_type', '')).lower().strip()
        self.use_vae = bool(config.get('use_vae', False))
        self.has_gnn_prior = self.use_vae and prior_type == 'gnn_e2e'
        if self.has_gnn_prior:
            self.prior = build_conditional_prior(config).to(device)
            print(f'MeshGraphNets: joint GNN conditional prior enabled '
                  f'(prior_type=gnn_e2e, family={self.prior.family})')
        else:
            self.prior = None

        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        self.model.set_checkpointing(enabled)

    def forward(self, graph, add_noise=None, use_posterior=None, fixed_z=None,
                compute_prior_path=False):
        """
        Forward pass of the simulator.

        Expects pre-normalized inputs from the dataloader:
            - graph.x: normalized node features [N, input_var]
            - graph.edge_attr: normalized edge features [E, edge_var]
            - graph.y: normalized target delta (y_t+1 - x_t) [N, output_var]

        compute_prior_path: when True and a GNN conditional prior exists, runs
            the prior network on the graph and returns what its density loss
            needs in the 5th tuple element — mixture parameters for the gmm
            family, the pooled conditioning vector for the fm family.

        Returns:
            predicted:     [N, output_var] posterior-path prediction
            target:        [N, output_var] graph.y (possibly noised)
            vae_losses:    dict from the posterior-path VAE encoder
            aux_loss:      scalar peak-to-valley loss on the decoded field
                           (0 if not training or use_vae is off)
            prior_outputs: None, or dict with exactly one of:
                'prior_params': {'logits', 'mu', 'log_std'[, 'cov_factor']}
                                (prior_family gmm)
                'pooled':       [B, hidden_dim] conditioning vector
                                (prior_family fm)
        """
        if add_noise is None:
            add_noise = self.training

        if add_noise:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                output_var = self.config['output_var']
                noise = torch.randn(graph.x.shape[0], output_var,
                                    device=graph.x.device, dtype=graph.x.dtype) * noise_std
                noise_padded = torch.zeros_like(graph.x)
                noise_padded[:, :output_var] = noise
                graph.x = graph.x + noise_padded
                noise_gamma = self.config.get('noise_gamma', 0.1)
                noise_std_ratio = self.config.get('noise_std_ratio', None)
                if noise_std_ratio is not None:
                    ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
                    graph.y = graph.y - noise_gamma * noise * ratio
                graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

        predicted, vae_losses, aux_loss = self.model(
            graph, use_posterior=use_posterior, fixed_z=fixed_z,
        )

        prior_outputs = None
        if compute_prior_path and self.prior is not None:
            # Prior reads raw (post-encoder MLP) graph features, NOT the encoded
            # latents — graph.x is unchanged across both forwards because the
            # simulator's encoder builds a fresh Data inside (does not mutate input).
            if self.prior.family == 'fm':
                prior_outputs = {'pooled': self.prior.condition(graph)}
            else:
                prior_outputs = {'prior_params': self.prior(graph)}

        return predicted, graph.y, vae_losses, aux_loss, prior_outputs


class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = int(config['edge_var'])
        if self.edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {self.edge_input_size}")
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)
        self.use_multiscale = config.get('use_multiscale', False)
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False))
            and self.use_world_edges
            and self.use_multiscale
        )

        # graph.x layout: [state | conditions | positional | node-type one-hot]
        num_cond = int(config.get('cond_var', 0) or 0)
        num_pos_features = int(config.get('positional_features', 0))
        base_input_size = config['input_var'] + num_cond + num_pos_features
        use_node_types = config.get('use_node_types', False)
        num_node_types = config.get('num_node_types', 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
            print(f"  Model input: {config['input_var']} physical + {num_cond} conditions + {num_pos_features} positional + {num_node_types} node types = {self.node_input_size}")
        else:
            self.node_input_size = base_input_size
            if num_pos_features > 0 or num_cond > 0:
                print(f"  Model input: {config['input_var']} physical + {num_cond} conditions + {num_pos_features} positional = {self.node_input_size}")

        self.node_output_size = config['output_var']

        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        if not self.use_multiscale:
            self.processer_list = nn.ModuleList([
                GnBlock(self.latent_dim, use_world_edges=self.use_world_edges)
                for _ in range(self.message_passing_num)
            ])
        else:
            self._build_multiscale_processor(config)

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

        self.use_vae = config.get('use_vae', False)
        if self.use_vae:
            self._build_vae_components(config)

    def _build_multiscale_processor(self, config):
        L = int(config.get('multiscale_levels', 1))
        self.multiscale_levels = L

        mp_per_level = config.get('mp_per_level', None)
        if mp_per_level is None:
            raise ValueError(
                "use_multiscale=True requires mp_per_level "
                "(2 * multiscale_levels + 1 entries, e.g. '4, 8, 12, 8, 4' for 2 levels)"
            )
        if not isinstance(mp_per_level, list):
            mp_per_level = [int(mp_per_level)]
        else:
            mp_per_level = [int(x) for x in mp_per_level]
        self.mp_per_level = mp_per_level

        expected_len = 2 * L + 1
        if len(mp_per_level) != expected_len:
            raise ValueError(
                f"mp_per_level must have {expected_len} entries for {L} levels, "
                f"got {len(mp_per_level)}: {mp_per_level}"
            )

        parts = []
        for i in range(L):
            parts.append(f"pre[{i}]={mp_per_level[i]}")
        parts.append(f"coarsest={mp_per_level[L]}")
        for i in range(L - 1, -1, -1):
            parts.append(f"post[{i}]={mp_per_level[2 * L - i]}")
        print(f"  Multiscale V-cycle ({L} levels): {', '.join(parts)}")

        self.pre_blocks = nn.ModuleList()
        self.post_blocks = nn.ModuleList()
        self.coarse_eb_encoders = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(L):
            pre_count = mp_per_level[i]
            post_count = mp_per_level[2 * L - i]
            # World edges only exist at the finest level unless coarse world
            # edges are enabled.
            use_we = self.use_world_edges if (i == 0 or self.use_coarse_world_edges) else False

            self.pre_blocks.append(nn.ModuleList([
                GnBlock(self.latent_dim, use_world_edges=use_we)
                for _ in range(pre_count)
            ]))
            self.post_blocks.append(nn.ModuleList([
                GnBlock(self.latent_dim, use_world_edges=use_we)
                for _ in range(post_count)
            ]))
            self.coarse_eb_encoders.append(
                build_mlp(self.edge_input_size, self.latent_dim, self.latent_dim)
            )
            self.skip_projs.append(nn.Linear(2 * self.latent_dim, self.latent_dim))

        # Learned bipartite unpool (coarse → fine message passing) per level.
        from model.blocks import UnpoolBlock
        self.unpool_blocks = nn.ModuleList([
            UnpoolBlock(self.latent_dim, build_mlp) for _ in range(L)
        ])

        coarsest_count = mp_per_level[L]
        self.coarsest_blocks = nn.ModuleList([
            GnBlock(self.latent_dim, use_world_edges=self.use_coarse_world_edges)
            for _ in range(coarsest_count)
        ])

    def _build_vae_components(self, config):
        self.vae_latent_dim = int(config.get('vae_latent_dim', 32))
        # MMD kernel bandwidth mode: 'fixed' (legacy constant sigmas, saturates at
        # high vae_latent_dim) or 'median' (dimension-adaptive median heuristic).
        self.mmd_bandwidth = str(config.get('mmd_bandwidth', 'fixed')).lower().strip()
        vae_mp_layers = int(config.get('vae_mp_layers', 5))
        self.vae_graph_aware = bool(config.get('vae_graph_aware', False))
        self.vae_posterior_min_std = float(config.get('posterior_min_std', 0))
        # How z reaches the processor. 'concat' is the legacy fuser
        # Linear([x, z]) (kept for checkpoint compatibility); 'adaln' is
        # AdaLN-Zero modulation, identity at init. See model/blocks.py.
        self.z_conditioning = str(config.get('z_conditioning', 'concat')).lower().strip()
        if self.z_conditioning not in ('concat', 'adaln'):
            raise ValueError(
                f"z_conditioning must be 'concat' or 'adaln', got '{self.z_conditioning}'")
        # MMD is a two-sample statistic whose effective sample count is the
        # number of z rows it sees -- without gathering, the PER-RANK batch.
        self.mmd_gather_ranks = bool(config.get('mmd_gather_ranks', True))
        # Per-level z: one z per V-cycle level (L coarse-arm + 1 coarsest) for multiscale,
        # else one global z. Allows level-specific stochastic modulation.
        if self.use_multiscale:
            self.num_z = int(config.get('num_z', self.multiscale_levels + 1))
        else:
            self.num_z = int(config.get('num_z', 1))
        self.vae_encoder = GNNVariationalEncoder(
            self.node_output_size, self.edge_input_size,
            self.latent_dim, self.vae_latent_dim, num_mp_layers=vae_mp_layers,
            node_input_size=self.node_input_size,
            graph_aware=self.vae_graph_aware,
            posterior_min_std=self.vae_posterior_min_std,
            num_z=self.num_z,
        )
        print(f"  VAE z slots: {self.num_z}")
        if self.vae_graph_aware:
            print(f"  VAE encoder: graph-aware (x [N,{self.node_input_size}] fused with y [N,{self.node_output_size}])")
        print(f"  VAE posterior σ floor: {self.vae_posterior_min_std}")
        print(f"  VAE z conditioning: {self.z_conditioning}")

        if not self.use_multiscale:
            self.z_fusers = self._build_z_conditioners(self.message_passing_num)
        else:
            L = self.multiscale_levels
            self.ms_z_fusers_pre = nn.ModuleList()
            self.ms_z_fusers_post = nn.ModuleList()
            for i in range(L):
                self.ms_z_fusers_pre.append(
                    self._build_z_conditioners(self.mp_per_level[i]))
                self.ms_z_fusers_post.append(
                    self._build_z_conditioners(self.mp_per_level[2 * L - i]))
            self.ms_z_fusers_coarsest = self._build_z_conditioners(self.mp_per_level[L])

        # Which field row the peak-to-valley term scores. 2 = z_disp in the
        # (x_disp, y_disp, z_disp) state block, the row the warpage metric reads.
        self.pv_channel = int(self.config.get('pv_channel', 2))
        if not 0 <= self.pv_channel < self.node_output_size:
            raise ValueError(
                f'pv_channel {self.pv_channel} is outside the '
                f'{self.node_output_size} output rows.')
        print(f"  VAE: ENABLED (z_dim={self.vae_latent_dim}, vae_mp_layers={vae_mp_layers})")

    # ── VAE helpers ──────────────────────────────────────────────────────────

    def _build_z_conditioners(self, count):
        """One z-conditioning module per processor block.

        Both kinds live under the same attribute names, so a checkpoint saved
        with the other kind fails with a clean missing/unexpected-key error
        rather than a silent shape mismatch (their parameter paths differ:
        `...0.weight` for concat vs `...0.net.1.weight` for adaln).
        """
        if self.z_conditioning == 'adaln':
            return nn.ModuleList([
                AdaLNZero(self.latent_dim, self.vae_latent_dim) for _ in range(count)
            ])
        return nn.ModuleList([
            nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
            for _ in range(count)
        ])

    def reset_z_conditioning(self):
        """Restore every AdaLN-Zero head to the identity. No-op for 'concat'."""
        for m in self.modules():
            if isinstance(m, AdaLNZero):
                m.reset_identity()

    def _fuse_z(self, x, z_per_node, fuse_layer):
        return fuse_layer(torch.cat([x, z_per_node], dim=-1))

    def _encode_vae(self, original_y, original_x, original_edge_index, original_edge_attr,
                    original_batch, N, device, dtype, use_posterior, fixed_z=None):
        """Produce z [B, num_z, D]: posterior sample, external z, or N(0,I)."""
        zero = torch.zeros((), device=device, dtype=torch.float32)
        empty_losses = {'mmd': zero, 'mu': None, 'logvar': None}
        if fixed_z is not None:
            z = fixed_z.to(device=device, dtype=dtype)
            # External callers (rollout with N(0,I)) may pass [B, D]; expand to
            # [B, num_z, D] by replicating the same z across all per-level slots.
            if z.dim() == 2:
                z = z.unsqueeze(1).expand(-1, self.num_z, -1).contiguous()
            return z, empty_losses
        if use_posterior and original_y is not None:
            batch = (original_batch if original_batch is not None
                     else torch.zeros(N, dtype=torch.long, device=device))
            z, mu, logvar = self.vae_encoder(
                original_y, original_edge_index, original_edge_attr, batch,
                x=(original_x if self.vae_graph_aware else None),
            )
            z_mmd = z.float()
            if self.mmd_gather_ranks:
                z_mmd = gather_across_ranks(z_mmd)
            mmd = GNNVariationalEncoder.mmd_loss(z_mmd, bandwidth=self.mmd_bandwidth)
            return z, {'mmd': mmd, 'mu': mu, 'logvar': logvar}
        B = int(original_batch.max().item()) + 1 if original_batch is not None else 1
        z = torch.randn(B, self.num_z, self.vae_latent_dim, device=device, dtype=dtype)
        return z, empty_losses

    def _pv_loss(self, predicted, original_y, batch, B):
        """MSE on the per-graph PEAK-TO-VALLEY of the DECODED field.

        The scored statistic is max(z_disp) - min(z_disp) over a part's nodes.
        This puts gradient directly on it rather than hoping a node-wise MSE
        reaches the extremes on its own.

        It replaces a head that regressed per-graph [mean, std] FROM z. That
        head could not move the measured dispersion for three reasons, and this
        term fixes each: it reads the DECODER OUTPUT (nothing tied a z-readout
        to the field the rollout writes), it uses the extreme-value statistic
        that is actually scored (not a node-axis std), and because each
        posterior sample comes from a DIFFERENT realization of the part,
        tracking that realization's own peak-to-valley is what creates the
        z -> spread sensitivity that `sd_ratio` measures.

        `graph.y` is a channel-wise z-score of the field and the mean cancels in
        a difference, so max-min here is the true peak-to-valley divided by that
        channel's sigma -- a constant, absorbed into `beta_aux`. No
        denormalization needed, and no extra forward: `predicted` is already
        materialised for the reconstruction term.

        Valid only where the model predicts the ABSOLUTE field, i.e. the static
        (num_timesteps == 1) case where mesh_dataset sets
        `target_delta = y_raw.copy()`. For T > 1 `graph.y` is a step delta and
        its peak-to-valley is not the field's; the training loop refuses that
        combination rather than scoring the wrong quantity.
        """
        if original_y is None or not self.training:
            return 0.0
        # `batch` must index the FINE level, the same rows as `predicted`. On the
        # multiscale path the caller has a variable that gets coarsened per level,
        # and passing that one silently scores the wrong nodes whenever the sizes
        # happen to agree. Say so instead.
        if batch.shape[0] != predicted.shape[0]:
            raise RuntimeError(
                f"_pv_loss got a batch index of {batch.shape[0]} for "
                f"{predicted.shape[0]} predicted nodes. It needs the FINE-level "
                f"index (batch_bc from _prepare_z), not a coarsened one.")
        c = self.pv_channel
        pred_c = predicted[:, c].float()
        true_c = original_y[:, c].float()
        pv_pred = (scatter(pred_c, batch, dim=0, dim_size=B, reduce='max')
                   - scatter(pred_c, batch, dim=0, dim_size=B, reduce='min'))
        pv_true = (scatter(true_c, batch, dim=0, dim_size=B, reduce='max')
                   - scatter(true_c, batch, dim=0, dim_size=B, reduce='min'))
        return torch.nn.functional.mse_loss(pv_pred, pv_true)

    def _prepare_z(self, graph, original_y, original_x, original_edge_index,
                   original_edge_attr, original_batch, use_posterior, fixed_z):
        """Shared VAE step for both processor paths.

        Returns (z [B, num_z, D], z_per_node for slot 0, batch index,
        vae_losses dict). The peak-to-valley term needs the decoded field, so
        it is computed by the callers after `self.decoder(...)`, not here.
        """
        N = graph.x.shape[0]
        device = graph.x.device
        batch_bc = (original_batch if original_batch is not None
                    else torch.zeros(N, dtype=torch.long, device=device))
        z, vae_losses = self._encode_vae(
            original_y, original_x, original_edge_index, original_edge_attr,
            original_batch, N, device, graph.x.dtype, use_posterior, fixed_z=fixed_z,
        )
        z_per_node = z[:, 0, :][batch_bc] if z.dim() == 3 else z[batch_bc]
        return z, z_per_node, batch_bc, vae_losses

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, graph, use_posterior=None, fixed_z=None):
        if use_posterior is None:
            use_posterior = self.training and self.use_vae

        if not self.use_multiscale:
            return self._forward_flat(graph, use_posterior, fixed_z)
        return self._forward_multiscale(graph, use_posterior, fixed_z)

    def _forward_flat(self, graph, use_posterior, fixed_z):
        original_y = getattr(graph, 'y', None)
        original_x = graph.x
        original_batch = getattr(graph, 'batch', None)
        original_edge_attr = graph.edge_attr
        original_edge_index = graph.edge_index

        graph = self.encoder(graph)

        vae_losses = {'mmd': torch.zeros((), device=graph.x.device, dtype=torch.float32)}
        aux_loss = 0.0
        z_per_node = None
        z = None
        batch_bc = None
        if self.use_vae:
            z, z_per_node, batch_bc, vae_losses = self._prepare_z(
                graph, original_y, original_x, original_edge_index,
                original_edge_attr, original_batch, use_posterior, fixed_z,
            )

        if self.use_checkpointing and self.training:
            graph = process_with_checkpointing(
                self.processer_list, graph,
                z_fusers=self.z_fusers if self.use_vae else None,
                z_per_node=z_per_node,
                adaln=(self.use_vae and self.z_conditioning == 'adaln'),
            )
        else:
            graph = self._run_blocks_eager(
                self.processer_list, graph,
                self.z_fusers if self.use_vae else None, z_per_node,
            )

        predicted = self.decoder(graph)
        if self.use_vae:
            aux_loss = self._pv_loss(predicted, original_y, batch_bc, z.shape[0])
        return predicted, vae_losses, aux_loss

    def _forward_multiscale(self, graph, use_posterior, fixed_z):
        L = self.multiscale_levels

        original_y = getattr(graph, 'y', None)
        original_x = graph.x
        original_batch = getattr(graph, 'batch', None)
        original_edge_attr = graph.edge_attr
        original_edge_index = graph.edge_index

        level_data = self._extract_level_data(graph, L)
        actual_levels = len(level_data)

        graph = self.encoder(graph)

        vae_losses = {'mmd': torch.zeros((), device=graph.x.device, dtype=torch.float32)}
        aux_loss = 0.0
        z = None
        current_z_per_node = None
        current_batch = None
        # Held separately because the descending arm COARSENS current_batch at
        # every level; the decoder output is fine-level, so anything scoring it
        # needs the index as it was before that loop ran.
        fine_batch = None
        if self.use_vae:
            z, current_z_per_node, current_batch, vae_losses = self._prepare_z(
                graph, original_y, original_x, original_edge_index,
                original_edge_attr, original_batch, use_posterior, fixed_z,
            )
            fine_batch = current_batch

        # Descending arm (fine → coarse)
        skip_states = []
        current_graph = graph

        for i in range(actual_levels):
            z_fusers_pre = self.ms_z_fusers_pre[i] if (self.use_vae and current_z_per_node is not None) else None
            current_graph = self._run_processor_blocks(
                self.pre_blocks[i], current_graph, z_fusers_pre, current_z_per_node
            )

            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            skip_states.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
                'z_per_node': current_z_per_node,
            })

            ld = level_data[i]
            if 'seeds' in ld:
                # Inherit mode: coarse node IS the seed → pool is a gather.
                h_coarse = current_graph.x[ld['seeds']]
            else:
                h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
            e_coarse = self.coarse_eb_encoders[i](ld['c_ea'])
            current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
            if self.use_coarse_world_edges and ld['c_we_idx'] is not None and ld['c_we_idx'].shape[1] > 0:
                current_graph.world_edge_attr  = ld['c_we_attr']
                current_graph.world_edge_index = ld['c_we_idx']

            if self.use_vae and z is not None:
                current_batch = scatter(current_batch, ld['ftc'], dim=0, dim_size=ld['n_c'], reduce='min')
                # Advance to the next z slot (level i+1, or the coarsest slot when i == L-1).
                next_slot = min(i + 1, z.shape[1] - 1)
                current_z_per_node = z[:, next_slot, :][current_batch]

        # Coarsest level
        z_fusers_coarsest = self.ms_z_fusers_coarsest if (self.use_vae and current_z_per_node is not None) else None
        current_graph = self._run_processor_blocks(
            self.coarsest_blocks, current_graph, z_fusers_coarsest, current_z_per_node
        )

        # Ascending arm (coarse → fine): learned bipartite unpool + skip merge
        for i in range(actual_levels - 1, -1, -1):
            ld = level_data[i]
            src, dst = ld['up_ei']
            rel_pos = ld['fine_pos'][dst] - ld['coarse_centroid'][src]
            h_up = self.unpool_blocks[i](
                h_coarse=current_graph.x,
                h_fine_skip=skip_states[i]['x'],
                unpool_edge_index=ld['up_ei'],
                rel_pos=rel_pos,
            )

            skip = skip_states[i]
            h_merged = self.skip_projs[i](torch.cat([skip['x'], h_up], dim=-1))
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            if use_we_here and skip['w_attr'] is not None:
                current_graph.world_edge_attr  = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            level_z_per_node = skip_states[i].get('z_per_node') if self.use_vae else None
            z_fusers_post = self.ms_z_fusers_post[i] if (self.use_vae and level_z_per_node is not None) else None
            current_graph = self._run_processor_blocks(
                self.post_blocks[i], current_graph, z_fusers_post, level_z_per_node
            )

        predicted = self.decoder(current_graph)
        if self.use_vae:
            aux_loss = self._pv_loss(predicted, original_y, fine_batch, z.shape[0])
        return predicted, vae_losses, aux_loss

    def _extract_level_data(self, graph, L):
        """Extract per-level coarsening topology from graph before encoder drops custom attrs."""
        level_data = {}
        for i in range(L):
            ftc_key = f'fine_to_coarse_{i}'
            if not hasattr(graph, ftc_key):
                break
            centroid = graph[f'coarse_centroid_{i}']
            ld = {
                'ftc': graph[ftc_key],
                'c_ei': graph[f'coarse_edge_index_{i}'],
                'c_ea': graph[f'coarse_edge_attr_{i}'],
                # Read the coarse node count off a shape, not off the GPU:
                # int(num_coarse_{i}.sum()) forces a CPU<->GPU sync per level on
                # every forward. coarse_centroid_{i} has exactly num_coarse rows
                # (batching concatenates both), so this is the same number.
                'n_c': centroid.shape[0],
                'c_we_idx': getattr(graph, f'coarse_world_edge_index_{i}', None),
                'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{i}', None),
                'up_ei': graph[f'unpool_edge_index_{i}'],
                'coarse_centroid': centroid,
                'fine_pos': graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}'],
            }
            seed_key = f'coarse_seed_idx_{i}'
            if hasattr(graph, seed_key) and graph[seed_key] is not None:
                ld['seeds'] = graph[seed_key]
            level_data[i] = ld
        return level_data

    def _run_processor_blocks(self, blocks, graph, z_fusers, z_per_node):
        """Run a list of GnBlocks with optional per-block z injection."""
        if self.use_checkpointing and self.training:
            return process_with_checkpointing(
                blocks, graph, z_fusers=z_fusers, z_per_node=z_per_node,
                adaln=(self.use_vae and self.z_conditioning == 'adaln'),
            )
        return self._run_blocks_eager(blocks, graph, z_fusers, z_per_node)

    def _run_blocks_eager(self, blocks, graph, z_fusers, z_per_node):
        """Non-checkpointed block loop, shared by both processor paths."""
        conditioned = z_fusers is not None and z_per_node is not None
        adaln = conditioned and self.z_conditioning == 'adaln'
        for j, block in enumerate(blocks):
            if adaln:
                graph = apply_adaln(block, graph, z_fusers[j], z_per_node)
                continue
            if conditioned:
                graph.x = self._fuse_z(graph.x, z_per_node, z_fusers[j])
            graph = block(graph)
        return graph

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled
