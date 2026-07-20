"""Public Transolver wrapper (IMPLEMENTATION_PLAN.md section 6.1/6.5, Appendix A.3).

forward(graph, add_noise=None) -> (predicted_normalized_delta, target_or_none)

matching the MeshGraphNets public model contract exactly, so training/inference
launchers are architecture-agnostic.
"""

import torch
import torch.nn as nn

from model.blocks import TransolverBlock
from model.checkpointing import run_checkpointed
from model.physics_attention import PhysicsAttentionIrregular


class Transolver(nn.Module):
    def __init__(self, config: dict, device: str = None):
        super().__init__()
        self.config = config

        self.output_var = config['output_var']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.slice_num = config['slice_num']
        self.mlp_ratio = int(config.get('mlp_ratio', 1))
        self.dropout = float(config.get('dropout', 0.0))
        self.temperature_init = float(config.get('temperature_init', 0.5))
        self.temperature_min = float(config.get('temperature_min', 0.1))
        self.temperature_max = float(config.get('temperature_max', 5.0))
        self.attention_kernel = config.get('attention_kernel', 'naive')
        self.chunk_size = int(config.get('chunk_size', 0))
        self.use_checkpointing = bool(config.get('use_checkpointing', False))

        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        input_var = config['input_var']
        positional_features = int(config.get('positional_features', 0))
        use_node_types = config.get('use_node_types', False)
        num_node_types = int(config.get('num_node_types', 0)) if use_node_types else 0
        self.node_input_size = input_var + positional_features + num_node_types
        embed_input_size = self.node_input_size + 3  # + pos_normalized (section 5.4)

        self.preprocess = nn.Sequential(
            nn.Linear(embed_input_size, 2 * self.latent_dim),
            nn.GELU(),
            nn.Linear(2 * self.latent_dim, self.latent_dim),
        )

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=self.num_heads, hidden_dim=self.latent_dim, slice_num=self.slice_num,
                dropout=self.dropout, mlp_ratio=self.mlp_ratio,
                last_layer=(i == self.num_layers - 1), out_dim=self.output_var,
                temperature_init=self.temperature_init, temperature_min=self.temperature_min,
                temperature_max=self.temperature_max,
            )
            for i in range(self.num_layers)
        ])

        self._initialize_weights()
        self.placeholder = nn.Parameter((1.0 / self.latent_dim) * torch.rand(self.latent_dim))

        # For time-transient delta prediction, start near "no change" -- default
        # derived from num_timesteps (temporal -> True, static -> False),
        # overridable by an explicit small_output_init config value (section 6.5).
        num_timesteps = config.get('num_timesteps', None)
        derived_default = (num_timesteps is None) or (num_timesteps > 1)
        self.small_output_init = bool(config.get('small_output_init', derived_default))
        if self.small_output_init:
            with torch.no_grad():
                self.blocks[-1].head.weight.mul_(0.01)

        if device is not None:
            self.to(device)

        total_params = sum(p.numel() for p in self.parameters())
        print('Transolver model created successfully')
        print(f'  Total parameters: {total_params:,}')
        print(f'  attention_kernel: {self.attention_kernel}, chunk_size: {self.chunk_size}')
        print(f'  latent_dim: {self.latent_dim}, num_layers: {self.num_layers}, '
              f'num_heads: {self.num_heads}, slice_num: {self.slice_num}')
        print(f'  node_input_size: {self.node_input_size} (+3 for pos_normalized)')

    def _initialize_weights(self):
        self.apply(self._init_module)
        # Re-apply orthogonal init to every slice projector AFTER the general
        # trunc_normal_ pass, so the final state is orthogonal regardless of
        # what the general pass did to it (section 6.5).
        for m in self.modules():
            if isinstance(m, PhysicsAttentionIrregular):
                nn.init.orthogonal_(m.in_project_slice.weight)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled

    def set_shard_group(self, group):
        """Enable node-sharded Physics-Attention (Phase 7): every attention
        module all-reduces its slice aggregates (num/den) across `group` inside
        the forward. `group=None` restores single-process behavior. Only the
        slice_space kernel supports sharding (the naive kernel materializes
        node-space projections and cannot tile/shard); the shard launcher
        enforces attention_kernel == 'slice_space'."""
        for m in self.modules():
            if isinstance(m, PhysicsAttentionIrregular):
                m.shard_group = group

    def _apply_noise(self, x, y):
        """Section 9: noise on the leading output_var physical channels only,
        with the matching delta-target correction. No edge_attr (none exists)."""
        noise_std = self.config.get('std_noise', 0.0)
        if not noise_std or noise_std <= 0:
            return x, y

        output_var = self.output_var
        noise = torch.randn(x.shape[0], output_var, device=x.device, dtype=x.dtype) * noise_std
        noise_padded = torch.zeros_like(x)
        noise_padded[:, :output_var] = noise
        x = x + noise_padded

        noise_gamma = self.config.get('noise_gamma', 1)
        noise_std_ratio = self.config.get('noise_std_ratio', None)
        if noise_std_ratio is not None and y is not None:
            ratio = torch.tensor(noise_std_ratio, device=x.device, dtype=x.dtype)
            y = y - noise_gamma * noise * ratio
        return x, y

    def forward(self, graph, add_noise=None):
        """
        Expects pre-normalized inputs from the dataloader:
            graph.x               [sum_N, node_input_size]
            graph.pos_normalized  [sum_N, 3]
            graph.y (optional)    [sum_N, output_var]
            graph.ptr (optional)  [B + 1]; synthesized as [0, sum_N] if absent

        Returns:
            predicted normalized delta [sum_N, output_var]
            target normalized delta [sum_N, output_var], or None
        """
        if add_noise is None:
            add_noise = self.training

        x = graph.x
        y = getattr(graph, 'y', None)

        if add_noise:
            x, y = self._apply_noise(x, y)

        ptr = getattr(graph, 'ptr', None)
        if ptr is None:
            ptr = torch.tensor([0, x.shape[0]], device=x.device, dtype=torch.long)

        inp = torch.cat([graph.pos_normalized, x], dim=-1)
        fx = self.preprocess(inp)
        fx = fx + self.placeholder[None, :]

        checkpoint_enabled = self.use_checkpointing and self.training
        for block in self.blocks:
            fx = run_checkpointed(
                block, fx, ptr, self.attention_kernel, self.chunk_size, self.use_checkpointing,
                enabled=checkpoint_enabled,
            )

        return fx, y

    def _embed(self, graph):
        x = graph.x
        ptr = getattr(graph, 'ptr', None)
        if ptr is None:
            ptr = torch.tensor([0, x.shape[0]], device=x.device, dtype=torch.long)
        inp = torch.cat([graph.pos_normalized, x], dim=-1)
        fx = self.preprocess(inp) + self.placeholder[None, :]
        return fx, ptr

    def forward_decoupled(self, cache_graph, query_graph=None, infer_chunk_size: int = 0):
        """Section 11 / Appendix A.4: two-stage inference. Stage 1 builds a
        per-layer physics-token cache from cache_graph; Stage 2 decodes
        query_graph (defaults to cache_graph) against that cache. When
        query_graph is cache_graph this is mathematically identical to the
        ordinary slice_space forward (proven by test_decoupled_inference.py),
        at the cost of embedding the mesh twice -- acceptable for a first
        implementation whose purpose is correctness, not yet true streaming
        from disk.
        """
        if query_graph is None:
            query_graph = cache_graph

        fx_cache, ptr_cache = self._embed(cache_graph)
        fx_query, ptr_query = self._embed(query_graph)

        for block in self.blocks:
            tokens_per_graph = block.compute_tokens(fx_cache, ptr_cache, infer_chunk_size)
            fx_cache = block.forward_with_tokens(fx_cache, ptr_cache, tokens_per_graph, infer_chunk_size)
            fx_query = block.forward_with_tokens(fx_query, ptr_query, tokens_per_graph, infer_chunk_size)

        y = getattr(query_graph, 'y', None)
        return fx_query, y
