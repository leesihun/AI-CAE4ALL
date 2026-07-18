"""GINO -- Geometry-Informed Neural Operator (IMPLEMENTATION_PLAN.md section
8.4, A.6). Native input graph-neural-operator kernel integral (irregular mesh
-> regular latent grid), native latent FNO blocks, native output GNO (latent
grid -> arbitrary query points). Per-graph execution (official GINO batches
only when geometry is shared; every scene here has different geometry, so
the wrapper loops over `ptr`, section 8.4/A.6).

No dataset here has validated SDF or true quadrature weights, so the default
`gino_variant mesh_state` uses coordinates + node features with unweighted
mean-kernel reductions and makes no discretization-convergence claim.
`gino_variant paper` is gated on that data and is not implemented until it
exists (section 8.4's paper-profile gate).
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from model.base import OperatorCore
from model.mlp import build_deep_mlp, init_weights
from model.gno import GNOLayer
from model.spectral import SpectralConvNd, validate_fno_modes
from model.utils import parse_int_tuple
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.radius_neighbors import (
    radius_neighbors_scipy, radius_neighbors_torch_cluster, HAS_TORCH_CLUSTER,
    neighbor_stats, min_reachable_radius,
)


def _pointwise_conv(in_c: int, out_c: int, d: int) -> nn.Module:
    conv_cls = nn.Conv2d if d == 2 else nn.Conv3d
    return conv_cls(in_c, out_c, kernel_size=1)


_VALID_VARIANTS = ("mesh_state", "paper")


def validate_config(config, data_spec):
    variant = str(config.get('gino_variant', 'mesh_state')).lower()
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"gino_variant must be one of {_VALID_VARIANTS}, got '{variant}'.")

    d = data_spec.operator_dim
    resolution = parse_int_tuple(config.get('gino_grid_resolution'), d, 'gino_grid_resolution')
    if any(r < 2 for r in resolution):
        raise ValueError(f"gino_grid_resolution entries must be >= 2, got {resolution}.")
    modes = parse_int_tuple(config.get('gino_fno_modes'), d, 'gino_fno_modes')
    validate_fno_modes(modes, resolution)

    in_radius = float(config.get('gino_in_radius', 0.08))
    out_radius = float(config.get('gino_out_radius', 0.08))
    if in_radius <= 0 or out_radius <= 0:
        raise ValueError(f"gino_in_radius/gino_out_radius must be > 0, got {in_radius}/{out_radius}.")

    transform_type = str(config.get('gino_transform_type', 'linear')).lower()
    if transform_type != 'linear':
        raise ValueError(f"gino_transform_type must be 'linear' (baseline), got '{transform_type}'.")

    use_torch_cluster = bool(config.get('gino_use_torch_cluster', False))
    if use_torch_cluster and not HAS_TORCH_CLUSTER:
        raise ValueError("gino_use_torch_cluster=True but torch_cluster is not installed.")

    min_r = min_reachable_radius(resolution, d)
    if out_radius < min_r:
        print(
            f"[gino] WARNING: gino_out_radius={out_radius:.4f} is below the half-cell-"
            f"diagonal {min_r:.4f} of gino_grid_resolution={resolution}; some output "
            "queries may be unreachable from the latent grid. Increase gino_out_radius "
            "or resolution, and run the coverage preflight before training."
        )

    if variant == 'paper':
        if not data_spec.has_sdf:
            raise ValueError("gino_variant=paper requires sdf_source != none.")
        if not data_spec.has_integration_weights:
            raise ValueError(
                "gino_variant=paper requires integration_weight_source != none "
                "(area/volume quadrature); this repository has no weighted-kernel "
                "implementation yet -- use gino_variant mesh_state."
            )


class MeshGINO(OperatorCore):
    model_name = "gino"

    def __init__(self, config, data_spec, coordinate_domain: CoordinateDomain):
        super().__init__()
        self.data_spec = data_spec
        self.domain = coordinate_domain
        self.output_var = data_spec.output_var
        self.d = data_spec.operator_dim
        self.variant = str(config.get('gino_variant', 'mesh_state')).lower()
        self.resolution = parse_int_tuple(config.get('gino_grid_resolution'), self.d, 'gino_grid_resolution')
        self.modes = parse_int_tuple(config.get('gino_fno_modes'), self.d, 'gino_fno_modes')
        self.hidden = int(config.get('gino_fno_hidden_channels', 64))
        self.n_layers = int(config.get('gino_fno_layers', 4))
        self.in_radius = float(config.get('gino_in_radius', 0.08))
        self.out_radius = float(config.get('gino_out_radius', 0.08))
        self.kernel_hidden = int(config.get('gino_kernel_hidden', 64))
        self.max_empty_input_fraction = float(config.get('gino_max_empty_input_fraction', 0.01))
        self.query_chunk_size = int(config.get('gino_query_chunk_size', 0))
        self.use_torch_cluster = bool(config.get('gino_use_torch_cluster', False))
        self.use_checkpointing = bool(config.get('use_checkpointing', False))
        self.has_sdf = bool(data_spec.has_sdf)

        source_feat_dim = (data_spec.total_node_dim + (1 if self.has_sdf else 0)
                          + data_spec.global_condition_dim)
        self.source_feat_dim = source_feat_dim

        self.input_gno = GNOLayer(query_dim=self.d, source_dim=self.d,
                                  source_feat_dim=source_feat_dim,
                                  hidden=self.kernel_hidden, out_dim=self.hidden, depth=2)

        self.lifting = _pointwise_conv(self.hidden + self.d, self.hidden, self.d)
        self.spectral_layers = nn.ModuleList([
            SpectralConvNd(self.hidden, self.hidden, self.modes) for _ in range(self.n_layers)
        ])
        self.pointwise_layers = nn.ModuleList([
            _pointwise_conv(self.hidden, self.hidden, self.d) for _ in range(self.n_layers)
        ])
        self.activation = nn.GELU()

        axes = [torch.linspace(0.0, 1.0, r) for r in self.resolution]
        grids = torch.meshgrid(*axes, indexing='ij')
        coord_grid = torch.stack(grids, dim=0)  # [d, *resolution]
        self.register_buffer('coord_grid', coord_grid, persistent=False)
        latent_points = coord_grid.reshape(self.d, -1).T.contiguous()  # [prod(res), d]
        self.register_buffer('latent_points', latent_points, persistent=False)

        self.output_gno = GNOLayer(query_dim=self.d, source_dim=self.d, source_feat_dim=self.hidden,
                                   hidden=self.kernel_hidden, out_dim=self.hidden, depth=2)
        self.projection = build_deep_mlp(self.hidden, self.hidden, self.output_var,
                                         depth=2, activation='gelu')

        self.apply(init_weights)
        if data_spec.num_timesteps > 1:
            with torch.no_grad():
                self.projection[-1].weight.mul_(0.01)

    def _neighbors(self, queries: torch.Tensor, sources: torch.Tensor, r: float) -> torch.Tensor:
        if self.use_torch_cluster and HAS_TORCH_CLUSTER:
            return radius_neighbors_torch_cluster(queries, sources, r)
        ei_np = radius_neighbors_scipy(
            queries.detach().cpu().numpy(), sources.detach().cpu().numpy(), r)
        return torch.from_numpy(ei_np).to(queries.device)

    def _assemble_node_features(self, graph) -> torch.Tensor:
        feats = graph.x
        if self.has_sdf:
            feats = torch.cat([feats, graph.sdf], dim=1)
        if self.data_spec.global_condition_dim > 0:
            gc_per_node = graph.global_conditions[graph.batch]
            feats = torch.cat([feats, gc_per_node], dim=1)
        return feats

    def _block(self, h: torch.Tensor, i: int) -> torch.Tensor:
        out = self.spectral_layers[i](h) + self.pointwise_layers[i](h)
        return self.activation(out)

    def _encode_one_graph(self, coords_g: torch.Tensor, feats_g: torch.Tensor) -> torch.Tensor:
        """coords_g: [N,d] in [0,1]^d, feats_g: [N, source_feat_dim].

        Returns latent_feat [prod(resolution), hidden] after input GNO + FNO blocks.
        """
        device = coords_g.device
        latent_pts = self.latent_points.to(device)
        ei_in = self._neighbors(latent_pts, coords_g, self.in_radius)
        z = self.input_gno(latent_pts, coords_g, feats_g, ei_in, num_queries=latent_pts.shape[0])

        z_grid = z.T.reshape(1, self.hidden, *self.resolution)
        coord_grid = self.coord_grid.to(device).unsqueeze(0)
        grid_in = torch.cat([z_grid, coord_grid], dim=1)

        h = self.lifting(grid_in)
        for i in range(self.n_layers):
            if self.use_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(self._block, h, i, use_reentrant=False)
            else:
                h = self._block(h, i)
        prod_res = latent_pts.shape[0]
        return h.reshape(self.hidden, prod_res).T  # [prod(res), hidden]

    def _decode_one_graph(self, latent_feat: torch.Tensor, coords_slice: torch.Tensor) -> torch.Tensor:
        device = coords_slice.device
        latent_pts = self.latent_points.to(device)
        ei_out = self._neighbors(coords_slice, latent_pts, self.out_radius)
        u = self.output_gno(coords_slice, latent_pts, latent_feat, ei_out,
                            num_queries=coords_slice.shape[0])
        return self.projection(u)

    def forward(self, graph) -> torch.Tensor:
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        feats_full = self._assemble_node_features(graph)

        outputs = []
        for g in range(num_graphs):
            start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            coords_g = c01[start:end]
            feats_g = feats_full[start:end]
            latent_feat = self._encode_one_graph(coords_g, feats_g)
            outputs.append(self._decode_one_graph(latent_feat, coords_g))
        return torch.cat(outputs, dim=0)

    def supports_query_chunking(self) -> bool:
        return True

    def encode_operator(self, graph):
        """Returns a list of per-graph latent features [prod(resolution), hidden]."""
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        feats_full = self._assemble_node_features(graph)
        latents = []
        for g in range(num_graphs):
            start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            latents.append(self._encode_one_graph(c01[start:end], feats_full[start:end]))
        return latents

    def decode_queries(self, encoded, graph, start: int, end: int):
        """`encoded` is the list returned by encode_operator. [start, end) may
        cross graph boundaries; each graph's slice is decoded independently
        and results are concatenated in original node order."""
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        num_graphs = graph.ptr.numel() - 1
        outputs = []
        for g in range(num_graphs):
            g_start, g_end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            lo, hi = max(start, g_start), min(end, g_end)
            if lo >= hi:
                continue
            outputs.append(self._decode_one_graph(encoded[g], c01[lo:hi]))
        return torch.cat(outputs, dim=0)

    def coverage_preflight(self, graph) -> dict:
        """Mandatory pre-training diagnostic (section 8.4): min/median/max
        neighbor counts both directions, empty-latent-cell fraction, and
        unreachable-output-query fraction for one representative graph.
        Raises if any output query is unreachable or the empty-input
        fraction exceeds `gino_max_empty_input_fraction`.
        """
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        num_graphs = graph.ptr.numel() - 1
        reports = []
        for g in range(num_graphs):
            start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            coords_g = c01[start:end]
            device = coords_g.device
            latent_pts = self.latent_points.to(device)

            ei_in = self._neighbors(latent_pts, coords_g, self.in_radius)
            in_stats = neighbor_stats(ei_in.cpu().numpy(), latent_pts.shape[0])

            ei_out = self._neighbors(coords_g, latent_pts, self.out_radius)
            out_stats = neighbor_stats(ei_out.cpu().numpy(), coords_g.shape[0])

            report = {'graph': g, 'input_gno': in_stats, 'output_gno': out_stats}
            reports.append(report)

            if in_stats['empty_fraction'] > self.max_empty_input_fraction:
                raise ValueError(
                    f"GINO coverage preflight failed on graph {g}: input GNO empty-latent "
                    f"fraction {in_stats['empty_fraction']:.4f} exceeds "
                    f"gino_max_empty_input_fraction={self.max_empty_input_fraction}. "
                    "Increase gino_in_radius or gino_grid_resolution."
                )
            if out_stats['empty_fraction'] > 0.0:
                raise ValueError(
                    f"GINO coverage preflight failed on graph {g}: {out_stats['empty_fraction']:.4f} "
                    "fraction of output queries have no latent-grid neighbor (unreachable). "
                    "Increase gino_out_radius or gino_grid_resolution."
                )
        return {'reports': reports}

    # ---- pipeline model-split protocol (parallelism/stages.py) ------------
    # Block order: 0 = entry (input GNO + lifting), 1..n_layers = latent FNO
    # blocks, n_layers+1 = exit (output GNO + projection). The two kernel
    # integrals dominate GINO's activation memory, and this decomposition puts
    # them on different pipeline stages. The boundary between any two blocks
    # is the latent grid [B, hidden, *res]; latent blocks run batched, which
    # is numerically identical to the per-graph loop (per-sample FFT/conv).

    def pipeline_num_blocks(self) -> int:
        return self.n_layers + 2

    def pipeline_block_costs(self, probe_graph) -> list:
        """Per-block fp32 activation-byte proxies for the stage partitioner.

        Kernel-edge counts are measured on the probe graph with the fast
        count-only KDTree query -- each edge holds ~3 fp32 activation vectors
        of width kernel_hidden through the GNO kernel MLP.
        """
        from model.adapters.radius_neighbors import radius_neighbor_count_sum
        c01, _ = self.domain.to_unit_box(probe_graph.pos_normalized)
        pts = c01.detach().cpu().numpy()
        latent_np = self.latent_points.cpu().numpy()
        e_in = radius_neighbor_count_sum(latent_np, pts, self.in_radius)
        e_out = radius_neighbor_count_sum(pts, latent_np, self.out_radius)
        prod_res = latent_np.shape[0]
        latent = float(prod_res * self.hidden * 4 * 4)
        entry = float(e_in * self.kernel_hidden * 4 * 3) + latent
        exit_ = float(e_out * self.kernel_hidden * 4 * 3) + latent
        return [entry] + [latent] * self.n_layers + [exit_]

    def pipeline_entry(self, graph) -> torch.Tensor:
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        feats_full = self._assemble_node_features(graph)
        device = c01.device
        latent_pts = self.latent_points.to(device)
        coord_grid = self.coord_grid.to(device).unsqueeze(0)
        grid_ins = []
        for g in range(num_graphs):
            start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            coords_g = c01[start:end]
            ei_in = self._neighbors(latent_pts, coords_g, self.in_radius)
            z = self.input_gno(latent_pts, coords_g, feats_full[start:end], ei_in,
                               num_queries=latent_pts.shape[0])
            z_grid = z.T.reshape(1, self.hidden, *self.resolution)
            grid_ins.append(torch.cat([z_grid, coord_grid], dim=1))
        return self.lifting(torch.cat(grid_ins, dim=0))

    def pipeline_block(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        i = block_idx - 1
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._block, h, i, use_reentrant=False)
        return self._block(h, i)

    def pipeline_exit(self, h: torch.Tensor, graph) -> torch.Tensor:
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        prod_res = self.latent_points.shape[0]
        outputs = []
        for g in range(num_graphs):
            start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
            latent_feat = h[g].reshape(self.hidden, prod_res).T
            outputs.append(self._decode_one_graph(latent_feat, c01[start:end]))
        return torch.cat(outputs, dim=0)

    def prune_to_pipeline_blocks(self, owned) -> None:
        """Drop every submodule not used by the owned block set, keeping the
        survivors' state-dict keys identical to the full single-GPU core."""
        owned = set(int(b) for b in owned)
        last = self.pipeline_num_blocks() - 1
        if 0 not in owned:
            self.input_gno = None
            self.lifting = None
        if last not in owned:
            self.output_gno = None
            self.projection = None
        keep = {b - 1 for b in owned if 0 < b < last}
        for i in range(self.n_layers):
            if i not in keep:
                self.spectral_layers[i] = nn.Identity()
                self.pointwise_layers[i] = nn.Identity()

    def export_model_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'gino_variant': self.variant,
            'gino_grid_resolution': list(self.resolution),
            'gino_fno_modes': list(self.modes),
            'gino_fno_hidden_channels': self.hidden,
            'gino_fno_layers': self.n_layers,
            'gino_in_radius': self.in_radius,
            'gino_out_radius': self.out_radius,
            'gino_kernel_hidden': self.kernel_hidden,
            'gino_use_torch_cluster': self.use_torch_cluster,
            'source_feat_dim': self.source_feat_dim,
        }
