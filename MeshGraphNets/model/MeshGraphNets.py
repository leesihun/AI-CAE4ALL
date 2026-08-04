import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.checkpointing import process_with_checkpointing, run_checkpointed
from model.coarsening import pool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp, init_weights


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super().__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)

        # For time-transient delta prediction, start near "no change".
        num_timesteps = config.get('num_timesteps', None)
        if num_timesteps is None or num_timesteps > 1:
            with torch.no_grad():
                last_layer = self.model.decoder.decode_module[-1]
                last_layer.weight.mul_(0.01)

        # Attention pool/unpool score heads must start at exactly zero so the
        # learned transfer operators reproduce today's mean-pool / sum-unpool
        # bit-for-bit at step 0 (ATTENTION_TRANSFER_DESIGN.md section 4):
        # equal scores -> uniform softmax -> mean (pool) / plain sum (unpool).
        # self.model.apply(init_weights) above already ran Kaiming init on
        # every Linear including these score heads, so they must be re-zeroed
        # here, same ordering as the decoder scale-down above. No-op when
        # pool_type/unpool_type are 'mean'/'sum' (the ModuleLists don't exist,
        # or use_attention is False on every UnpoolBlock).
        with torch.no_grad():
            for blk in getattr(self.model, 'pool_blocks', []):
                last = blk.score_mlp[-1]
                last.weight.zero_()
                last.bias.zero_()
            for blk in getattr(self.model, 'unpool_blocks', []):
                if getattr(blk, 'use_attention', False):
                    last = blk.attn_mlp[-1]
                    last.weight.zero_()
                    last.bias.zero_()

        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        self.model.set_checkpointing(enabled)

    def forward(self, graph, add_noise=None):
        """
        Forward pass of the deterministic simulator.

        Expects pre-normalized inputs from the dataloader:
            - graph.x: normalized node features [N, input_var]
            - graph.edge_attr: normalized edge features [E, edge_var]
            - graph.y: normalized target delta [N, output_var]

        Returns:
            predicted: predicted normalized delta [N, output_var]
            target: normalized target delta [N, output_var]
        """
        if add_noise is None:
            add_noise = self.training

        if add_noise:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                output_var = self.config['output_var']
                noise = torch.randn(
                    graph.x.shape[0], output_var,
                    device=graph.x.device, dtype=graph.x.dtype
                ) * noise_std
                noise_padded = torch.zeros_like(graph.x)
                noise_padded[:, :output_var] = noise
                graph.x = graph.x + noise_padded
                noise_gamma = self.config.get('noise_gamma', 1)
                noise_std_ratio = self.config.get('noise_std_ratio', None)
                if noise_std_ratio is not None:
                    ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
                    graph.y = graph.y - noise_gamma * noise * ratio
                graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

        predicted = self.model(graph)
        return predicted, getattr(graph, 'y', None)


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
            print(
                f"  Model input: {config['input_var']} physical + {num_cond} conditions + "
                f"{num_pos_features} positional + {num_node_types} node types = {self.node_input_size}"
            )
        else:
            self.node_input_size = base_input_size
            if num_pos_features > 0 or num_cond > 0:
                print(
                    f"  Model input: {config['input_var']} physical + {num_cond} conditions + "
                    f"{num_pos_features} positional = {self.node_input_size}"
                )

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

    def _build_multiscale_processor(self, config):
        L = int(config.get('multiscale_levels', 1))
        self.multiscale_levels = L

        # Transfer operators between levels. 'mean'/'sum' are today's fixed
        # operators (see model/coarsening.py:pool_features and the plain-sum
        # branch below); 'attention' replaces them with learned, per-cluster
        # weights that reduce to the fixed operators exactly at init (zero-init
        # in MeshGraphNets.__init__ -- see ATTENTION_TRANSFER_DESIGN.md
        # section 4). Independent flags so pool-only / unpool-only / both can
        # be ablated (section 10.5 of the same doc).
        self.pool_type = str(config.get('pool_type', 'mean')).strip().lower()
        self.unpool_type = str(config.get('unpool_type', 'sum')).strip().lower()
        self.pool_heads = int(config.get('pool_heads', 4))
        if self.pool_type not in ('mean', 'attention'):
            raise ValueError(f"pool_type must be 'mean' or 'attention', got {self.pool_type!r}")
        if self.unpool_type not in ('sum', 'attention'):
            raise ValueError(f"unpool_type must be 'sum' or 'attention', got {self.unpool_type!r}")
        if self.pool_type == 'attention':
            print(f"  Pool:   attention ({self.pool_heads} heads)")
        if self.unpool_type == 'attention':
            print("  Unpool: attention")

        # Multi-partition coarsening (ATTENTION_TRANSFER_DESIGN.md Part II):
        # per-level branch count. Default 1 everywhere reproduces today's
        # single-partition V-cycle exactly -- skip_projs below only widens
        # for a level with branches > 1. Only the LAST level may branch (see
        # general_modules/multiscale_helpers.py's build_multiscale_hierarchy
        # docstring for why); validated here too since a model can be built
        # directly from a checkpoint's model_config, not only via the dataset.
        raw_vb = config.get('voronoi_branches', None)
        if raw_vb is None:
            self.voronoi_branches = [1] * L
        elif isinstance(raw_vb, list):
            self.voronoi_branches = [int(v) for v in raw_vb]
        else:
            self.voronoi_branches = [int(raw_vb)]
        if len(self.voronoi_branches) == 1 and L > 1:
            self.voronoi_branches = self.voronoi_branches * L
        for lvl, kb in enumerate(self.voronoi_branches):
            if kb > 1 and lvl != L - 1:
                raise ValueError(
                    f"voronoi_branches > 1 is only supported on the last configured "
                    f"level ({L - 1}); got {kb} branches at level {lvl}."
                )
        if any(kb > 1 for kb in self.voronoi_branches):
            print(f"  Multi-partition: voronoi_branches={self.voronoi_branches}")

        mp_per_level = config.get('mp_per_level', None)
        if mp_per_level is None:
            raise ValueError(
                "use_multiscale=True requires mp_per_level "
                "(2 * multiscale_levels + 1 entries, e.g. '4, 6, 8, 6, 4' for 2 levels)"
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
            # World edges exist only at the finest level unless coarse world
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
            # Merge width: skip + 1 unpooled branch normally, skip + K
            # unpooled branches for a multi-partition level (concat, then a
            # learned linear combiner -- ATTENTION_TRANSFER_DESIGN.md Part II
            # section 7, option 1). k_i=1 reproduces today's 2*latent_dim exactly.
            k_i = self.voronoi_branches[i] if i < len(self.voronoi_branches) else 1
            self.skip_projs.append(nn.Linear((1 + k_i) * self.latent_dim, self.latent_dim))

        if self.pool_type == 'attention':
            from model.blocks import AttentionPoolBlock
            self.pool_blocks = nn.ModuleList([
                AttentionPoolBlock(self.latent_dim, self.pool_heads, build_mlp)
                for _ in range(L)
            ])

        # Learned bipartite unpool (coarse -> fine message passing) per level.
        from model.blocks import UnpoolBlock
        self.unpool_blocks = nn.ModuleList([
            UnpoolBlock(self.latent_dim, build_mlp,
                       use_attention=(self.unpool_type == 'attention'))
            for _ in range(L)
        ])

        coarsest_count = mp_per_level[L]
        self.coarsest_blocks = nn.ModuleList([
            GnBlock(self.latent_dim, use_world_edges=self.use_coarse_world_edges)
            for _ in range(coarsest_count)
        ])

    def forward(self, graph):
        if not self.use_multiscale:
            return self._forward_flat(graph)
        return self._forward_multiscale(graph)

    def _encode(self, graph):
        """Encoder wrapped by use_checkpointing: its edge MLP internals scale
        with E and would otherwise stay resident for the whole backward.

        Only tensors cross the checkpoint boundary — the Data is rebuilt here,
        outside it. Checkpointing the module directly (Data in, Data out) makes
        Dynamo abort with "lift_tracked_freevar_to_input should not be called on
        root SubgraphTracer", which is why `use_compile` + `use_checkpointing`
        used to be mutually exclusive.
        """
        has_world = (
            self.use_world_edges and hasattr(graph, 'world_edge_attr')
            and graph.world_edge_index.shape[1] > 0
        )
        node_, edge_, world_ = run_checkpointed(
            self.encoder.encode_tensors,
            graph.x, graph.edge_attr,
            graph.world_edge_attr if has_world else None,
            enabled=self.use_checkpointing and self.training,
        )
        out = Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)
        if world_ is not None:
            out.world_edge_attr = world_
            out.world_edge_index = graph.world_edge_index
        elif self.use_world_edges:
            out.world_edge_attr = torch.zeros(0, edge_.shape[1], device=edge_.device)
            out.world_edge_index = torch.zeros(2, 0, dtype=torch.long, device=edge_.device)
        return out

    def _forward_flat(self, graph):
        graph = self._encode(graph)
        graph = self._run_processor_blocks(self.processer_list, graph)
        return self.decoder(graph)

    def _forward_multiscale(self, graph):
        L = self.multiscale_levels
        level_data = self._extract_level_data(graph, L)
        actual_levels = len(level_data)

        graph = self._encode(graph)

        skip_states = []
        current_graph = graph
        branched_last = False  # True once a multi-partition level is hit (always terminal)

        for i in range(actual_levels):
            current_graph = self._run_processor_blocks(self.pre_blocks[i], current_graph)

            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            skip_states.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
            })

            ld = level_data[i]
            if 'branches' in ld:
                # Multi-partition level (ATTENTION_TRANSFER_DESIGN.md Part
                # II): pool/encode/coarsest-process each branch independently
                # (shared weights -- self.pool_blocks[i] / coarse_eb_encoders[i]
                # / coarsest_blocks are each called once per branch, not
                # duplicated per branch). Always the terminal level
                # (build_multiscale_hierarchy enforces this), so
                # coarsest_blocks runs HERE instead of once after this loop.
                branch_xs = []
                for b_ld in ld['branches']:
                    h_coarse_b = self._pool_one(current_graph, b_ld, ld['fine_pos'], i)
                    e_coarse_b = run_checkpointed(
                        self.coarse_eb_encoders[i], b_ld['c_ea'],
                        enabled=self.use_checkpointing and self.training,
                    )
                    branch_graph = Data(x=h_coarse_b, edge_attr=e_coarse_b, edge_index=b_ld['c_ei'])
                    branch_graph = self._run_processor_blocks(self.coarsest_blocks, branch_graph)
                    branch_xs.append(branch_graph.x)
                ld['_branch_x'] = branch_xs
                branched_last = True
            else:
                h_coarse = self._pool_one(current_graph, ld, ld['fine_pos'], i)
                e_coarse = run_checkpointed(
                    self.coarse_eb_encoders[i], ld['c_ea'],
                    enabled=self.use_checkpointing and self.training,
                )
                current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
                if self.use_coarse_world_edges and ld['c_we_idx'] is not None and ld['c_we_idx'].shape[1] > 0:
                    current_graph.world_edge_attr = ld['c_we_attr']
                    current_graph.world_edge_index = ld['c_we_idx']

        if not branched_last:
            current_graph = self._run_processor_blocks(self.coarsest_blocks, current_graph)
        # else: coarsest_blocks already ran per-branch above.

        for i in range(actual_levels - 1, -1, -1):
            ld = level_data[i]
            skip = skip_states[i]
            if 'branches' in ld:
                h_merged = run_checkpointed(
                    self._unpool_merge_branched_level, i, ld['_branch_x'], skip['x'], ld,
                    enabled=self.use_checkpointing and self.training,
                )
            else:
                h_merged = run_checkpointed(
                    self._unpool_merge_level, i, current_graph.x, skip['x'], ld,
                    enabled=self.use_checkpointing and self.training,
                )
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            if use_we_here and skip['w_attr'] is not None:
                current_graph.world_edge_attr = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            current_graph = self._run_processor_blocks(self.post_blocks[i], current_graph)

        return self.decoder(current_graph)

    def _pool_one(self, current_graph, ld, fine_pos, level_idx):
        """Pool current_graph.x into one coarse representation via `ld`.

        Shared by the single-partition path and every branch of a
        multi-partition level (ATTENTION_TRANSFER_DESIGN.md Part II) -- same
        weights (self.pool_blocks[level_idx]) either way, just a different
        `ld` (fine_to_coarse mapping etc). `fine_pos` is passed explicitly
        rather than read off `ld['fine_pos']` because it's a LEVEL property
        shared across branches, not stored per-branch.
        """
        if self.pool_type == 'attention':
            # rel_pos: node position relative to its own cluster's anchor
            # (coarse_centroid holds the seed-anchor position under
            # inherit/seedmean modes, the arithmetic centroid otherwise --
            # either way it's this partition's designated reference point).
            rel_pos = fine_pos - ld['coarse_centroid'][ld['ftc']]
            log_scale = self._pool_log_scale(current_graph.edge_index, fine_pos, ld['ftc'].shape[0])
            return self.pool_blocks[level_idx](current_graph.x, ld['ftc'], ld['n_c'], rel_pos, log_scale)
        elif 'seeds' in ld:
            # Inherit mode (voronoi_inherit): coarse feature = seed's
            # feature (pure gather). Centroid mode: mean over cluster.
            return current_graph.x[ld['seeds']]
        else:
            return pool_features(current_graph.x, ld['ftc'], ld['n_c'])

    def _pool_log_scale(self, edge_index, positions, num_nodes):
        """Log local mesh-size proxy per node, for the attention pool's score.

        Computed on the fly from whichever connectivity feeds this pool step
        (fine mesh edges at level 0, coarse_edge_index_{i-1} at level i>0) --
        no dataset/cache changes needed, since that connectivity and its
        matching positions (`ld['fine_pos']`) are already available here.
        FEM restriction operators are mass-matrix weighted, not uniform, and
        cluster sizes vary 3-10x in this data; this feature is what makes that
        weighting learnable rather than hard-coding it.
        """
        if edge_index.shape[1] == 0:
            return positions.new_zeros(num_nodes, 1)
        src, dst = edge_index[0], edge_index[1]
        lengths = (positions[dst] - positions[src]).norm(dim=-1, keepdim=True)
        scale = scatter(lengths, src, dim=0, dim_size=num_nodes, reduce='mean')
        return torch.log(scale.clamp_min(1e-8))

    def _unpool_merge_level(self, i, coarse_x, skip_x, ld):
        """Unpool level-i coarse features to fine and merge with the skip state.

        One method so use_checkpointing can recompute the whole step: the
        bipartite edge MLP runs on ~(1 + coarse degree) * N_fine unpool edges,
        otherwise one of the largest saved-activation buffers in the V-cycle.
        """
        src, dst = ld['up_ei']
        rel_pos = ld['fine_pos'][dst] - ld['coarse_centroid'][src]
        h_up = self.unpool_blocks[i](
            h_coarse=coarse_x,
            h_fine_skip=skip_x,
            unpool_edge_index=ld['up_ei'],
            rel_pos=rel_pos,
        )
        return self.skip_projs[i](torch.cat([skip_x, h_up], dim=-1))

    def _unpool_merge_branched_level(self, i, branch_xs, skip_x, ld):
        """Unpool a multi-partition level's K branches to fine and merge them
        all with the skip state (ATTENTION_TRANSFER_DESIGN.md Part II section
        7, option 1: concat + a learned linear combiner -- self.skip_projs[i]
        was sized for `1 + K` inputs in _build_multiscale_processor).

        Same weights unpool every branch (self.unpool_blocks[i] is called
        once per branch, not duplicated per branch); only `ld['branches']`
        differs. Kept as one method, like _unpool_merge_level, so
        use_checkpointing recomputes the whole multi-branch step during
        backward instead of holding every branch's activations live.
        """
        h_ups = []
        for b_ld, coarse_x in zip(ld['branches'], branch_xs):
            src, dst = b_ld['up_ei']
            rel_pos = ld['fine_pos'][dst] - b_ld['coarse_centroid'][src]
            h_up = self.unpool_blocks[i](
                h_coarse=coarse_x,
                h_fine_skip=skip_x,
                unpool_edge_index=b_ld['up_ei'],
                rel_pos=rel_pos,
            )
            h_ups.append(h_up)
        return self.skip_projs[i](torch.cat([skip_x, *h_ups], dim=-1))

    def _extract_level_data(self, graph, L):
        """Extract per-level coarsening topology before the encoder drops custom attrs.

        A multi-partition level (ATTENTION_TRANSFER_DESIGN.md Part II) has no
        `fine_to_coarse_{i}` attribute at all -- only branch-suffixed
        `fine_to_coarse_{i}_{b}` for b = 0 .. K-1 -- so its entry is
        `{'branches': [ld_0, ld_1, ...], 'fine_pos': ...}` instead of a flat
        dict. `fine_pos` lives at the level (not per-branch) since every
        branch partitions the SAME upstream node set.
        """
        level_data = {}
        for i in range(L):
            ftc_key = f'fine_to_coarse_{i}'
            branch0_key = f'fine_to_coarse_{i}_0'
            if hasattr(graph, branch0_key):
                fine_pos = graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}']
                branches = []
                b = 0
                while hasattr(graph, f'fine_to_coarse_{i}_{b}'):
                    key = f'{i}_{b}'
                    centroid_b = graph[f'coarse_centroid_{key}']
                    b_ld = {
                        'ftc': graph[f'fine_to_coarse_{key}'],
                        'c_ei': graph[f'coarse_edge_index_{key}'],
                        'c_ea': graph[f'coarse_edge_attr_{key}'],
                        'n_c': centroid_b.shape[0],
                        'coarse_centroid': centroid_b,
                        'up_ei': graph[f'unpool_edge_index_{key}'],
                    }
                    seed_key = f'coarse_seed_idx_{key}'
                    if hasattr(graph, seed_key):
                        b_ld['seeds'] = graph[seed_key]
                    branches.append(b_ld)
                    b += 1
                level_data[i] = {'branches': branches, 'fine_pos': fine_pos}
            elif hasattr(graph, ftc_key):
                centroid = graph[f'coarse_centroid_{i}']
                ld = {
                    'ftc': graph[ftc_key],
                    'c_ei': graph[f'coarse_edge_index_{i}'],
                    'c_ea': graph[f'coarse_edge_attr_{i}'],
                    # Read the coarse node count off a shape, not off the GPU:
                    # int(num_coarse_{i}.sum()) forces a CPU<->GPU sync per level on
                    # every forward, stalling the pipeline the training loop is
                    # otherwise careful to keep sync-free. coarse_centroid_{i} has
                    # exactly num_coarse rows (and batching concatenates both), so
                    # this is the same number.
                    'n_c': centroid.shape[0],
                    'c_we_idx': getattr(graph, f'coarse_world_edge_index_{i}', None),
                    'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{i}', None),
                }
                # Inherit-mode (voronoi_inherit) levels expose seed indices.
                seed_key = f'coarse_seed_idx_{i}'
                if hasattr(graph, seed_key):
                    ld['seeds'] = graph[seed_key]
                ld['up_ei'] = graph[f'unpool_edge_index_{i}']
                ld['coarse_centroid'] = centroid
                ld['fine_pos'] = graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}']
                level_data[i] = ld
            else:
                break
        return level_data

    def _run_processor_blocks(self, blocks, graph):
        """Run a stack of GnBlocks on raw tensors (one Data rebuild at the end).

        The per-block Data construction of the old path was measurable Python
        overhead at batch_size=1 and breaks torch.compile graphs.
        """
        x, edge_attr = graph.x, graph.edge_attr
        edge_index = graph.edge_index
        world_edge_attr = getattr(graph, 'world_edge_attr', None)
        world_edge_index = getattr(graph, 'world_edge_index', None)

        if self.use_checkpointing and self.training:
            x, edge_attr, world_edge_attr = process_with_checkpointing(
                blocks, x, edge_attr, edge_index, world_edge_attr, world_edge_index
            )
        else:
            for block in blocks:
                x, edge_attr, world_edge_attr = block.forward_tensors(
                    x, edge_attr, edge_index, world_edge_attr, world_edge_index
                )

        out = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if world_edge_attr is not None and world_edge_index is not None:
            out.world_edge_attr = world_edge_attr
            out.world_edge_index = world_edge_index
        return out

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled
