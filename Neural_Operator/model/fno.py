"""Mesh-adapted FNO (IMPLEMENTATION_PLAN.md section 8.3, A.5). A deterministic
splat/sample adapter (model/adapters/grid.py) projects ragged meshes onto a
fixed regular grid; the native `SpectralConvNd` core (model/spectral.py) then
runs standard FNO blocks; a final `grid_sample` maps predictions back to the
original node coordinates. Reported as "mesh-adapted FNO" -- the splat/sample
projection error is part of this baseline, not a native mesh operator.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from model.base import OperatorCore
from model.mlp import init_weights
from model.spectral import SpectralConvNd, validate_fno_modes
from model.utils import parse_int_tuple
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.grid import splat, sample


def _pointwise_conv(in_c: int, out_c: int, d: int) -> nn.Module:
    """1x1(x1) convolution == a per-position Linear across channels."""
    conv_cls = nn.Conv2d if d == 2 else nn.Conv3d
    return conv_cls(in_c, out_c, kernel_size=1)


def validate_config(config, data_spec):
    d = data_spec.operator_dim
    resolution = parse_int_tuple(config.get('fno_grid_resolution'), d, 'fno_grid_resolution')
    if any(r < 2 for r in resolution):
        raise ValueError(f"fno_grid_resolution entries must be >= 2, got {resolution}.")
    modes = parse_int_tuple(config.get('fno_modes'), d, 'fno_modes')
    validate_fno_modes(modes, resolution)

    layers = int(config.get('fno_layers', 4))
    if layers < 1:
        raise ValueError(f"fno_layers must be >= 1, got {layers}.")
    norm = str(config.get('fno_norm', 'none')).lower()
    if norm != 'none':
        raise ValueError(f"fno_norm must be 'none' (baseline), got '{norm}'.")

    hidden = int(config.get('fno_hidden_channels', 64))
    prod_res = math.prod(resolution)
    bytes_per_layer = prod_res * hidden * 4  # fp32 activation, per sample
    print(f"[fno] grid={resolution} modes={modes} hidden={hidden} layers={layers}: "
          f"~{bytes_per_layer / 1e6:.1f} MB/sample/layer activation (fp32 estimate).")

    n_blocks = 2 ** (d - 1)
    spectral_params = n_blocks * hidden * hidden * math.prod(modes) * 2
    print(f"[fno] estimated spectral parameters per layer: {spectral_params:,} "
          f"(n_blocks={n_blocks} x hidden^2={hidden * hidden} x prod(modes)={math.prod(modes)} x 2)")


class MeshFNO(OperatorCore):
    model_name = "fno"

    def __init__(self, config, data_spec, coordinate_domain: CoordinateDomain):
        super().__init__()
        self.data_spec = data_spec
        self.domain = coordinate_domain
        self.output_var = data_spec.output_var
        self.d = data_spec.operator_dim
        self.resolution = parse_int_tuple(config.get('fno_grid_resolution'), self.d, 'fno_grid_resolution')
        self.modes = parse_int_tuple(config.get('fno_modes'), self.d, 'fno_modes')
        self.hidden = int(config.get('fno_hidden_channels', 64))
        self.n_layers = int(config.get('fno_layers', 4))
        self.use_channel_mlp = bool(config.get('fno_use_channel_mlp', True))
        self.use_checkpointing = bool(config.get('use_checkpointing', False))
        self.has_sdf = bool(data_spec.has_sdf)

        in_channels = (data_spec.total_node_dim + 2 + self.d
                      + (1 if self.has_sdf else 0) + data_spec.global_condition_dim)
        self.in_channels = in_channels

        self.lifting = _pointwise_conv(in_channels, self.hidden, self.d)
        self.spectral_layers = nn.ModuleList([
            SpectralConvNd(self.hidden, self.hidden, self.modes) for _ in range(self.n_layers)
        ])
        self.pointwise_layers = nn.ModuleList([
            _pointwise_conv(self.hidden, self.hidden, self.d) for _ in range(self.n_layers)
        ])
        self.channel_mlps = None
        if self.use_channel_mlp:
            self.channel_mlps = nn.ModuleList([
                nn.Sequential(
                    _pointwise_conv(self.hidden, self.hidden, self.d), nn.GELU(),
                    _pointwise_conv(self.hidden, self.hidden, self.d),
                ) for _ in range(self.n_layers)
            ])
        self.activation = nn.GELU()
        self.projection = nn.Sequential(
            _pointwise_conv(self.hidden, self.hidden, self.d), nn.GELU(),
            _pointwise_conv(self.hidden, self.output_var, self.d),
        )

        axes = [torch.linspace(0.0, 1.0, r) for r in self.resolution]
        grids = torch.meshgrid(*axes, indexing='ij')
        coord_grid = torch.stack(grids, dim=0)  # [d, *resolution]
        self.register_buffer('coord_grid', coord_grid, persistent=False)

        self.apply(init_weights)  # no-op on SpectralConvNd.weight (not an nn.Linear)
        if data_spec.num_timesteps > 1:
            with torch.no_grad():
                self.projection[-1].weight.mul_(0.01)

    def _assemble_grid(self, graph) -> torch.Tensor:
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        values = graph.x
        if self.has_sdf:
            values = torch.cat([values, graph.sdf], dim=1)
        grid, occ, dens = splat(values, c01, graph.batch, num_graphs, self.resolution)

        coord = self.coord_grid.unsqueeze(0).expand(num_graphs, *self.coord_grid.shape)
        parts = [grid, occ, dens, coord]
        if self.data_spec.global_condition_dim > 0:
            gc = graph.global_conditions.view(num_graphs, -1, *([1] * self.d))
            gc = gc.expand(-1, -1, *self.resolution)
            parts.append(gc)
        return torch.cat(parts, dim=1)

    def _block(self, h: torch.Tensor, i: int) -> torch.Tensor:
        out = self.spectral_layers[i](h) + self.pointwise_layers[i](h)
        if self.channel_mlps is not None:
            out = out + self.channel_mlps[i](out)
        return self.activation(out)

    def _forward_grid_stack(self, grid_in: torch.Tensor) -> torch.Tensor:
        h = self.lifting(grid_in)
        for i in range(self.n_layers):
            if self.use_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(self._block, h, i, use_reentrant=False)
            else:
                h = self._block(h, i)
        return self.projection(h)

    def _predict_grid(self, graph) -> torch.Tensor:
        grid_in = self._assemble_grid(graph)
        return self._forward_grid_stack(grid_in)

    def _decode(self, grid_out: torch.Tensor, graph, start: int, end: int) -> torch.Tensor:
        num_graphs = graph.ptr.numel() - 1
        c01, _ = self.domain.to_unit_box(graph.pos_normalized)
        return sample(grid_out, c01[start:end], graph.batch[start:end], num_graphs)

    def forward(self, graph) -> torch.Tensor:
        grid_out = self._predict_grid(graph)
        return self._decode(grid_out, graph, 0, graph.x.shape[0])

    def supports_query_chunking(self) -> bool:
        return True

    def encode_operator(self, graph):
        return self._predict_grid(graph)

    def decode_queries(self, encoded, graph, start: int, end: int):
        return self._decode(encoded, graph, start, end)

    # ---- pipeline model-split protocol (parallelism/stages.py) ------------
    # Block order: 0 = entry (splat + lifting), 1..n_layers = latent FNO
    # blocks, n_layers+1 = exit (projection + grid_sample back to nodes).
    # The boundary between any two blocks is the latent grid [B, hidden, *res].

    def pipeline_num_blocks(self) -> int:
        return self.n_layers + 2

    def pipeline_block_costs(self, probe_graph) -> list:
        """Per-block fp32 activation-byte proxies for the stage partitioner."""
        n = int(probe_graph.x.shape[0])
        prod_res = math.prod(self.resolution)
        latent = float(prod_res * self.hidden * 4 * 4)  # ~4 resident tensors/block
        entry = float((n + prod_res) * self.in_channels * 4) + latent
        exit_ = float(prod_res * self.hidden * 4 + n * self.output_var * 4) + latent
        return [entry] + [latent] * self.n_layers + [exit_]

    def pipeline_entry(self, graph) -> torch.Tensor:
        return self.lifting(self._assemble_grid(graph))

    def pipeline_block(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        i = block_idx - 1
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._block, h, i, use_reentrant=False)
        return self._block(h, i)

    def pipeline_exit(self, h: torch.Tensor, graph) -> torch.Tensor:
        grid_out = self.projection(h)
        return self._decode(grid_out, graph, 0, graph.x.shape[0])

    def prune_to_pipeline_blocks(self, owned) -> None:
        """Drop every submodule not used by the owned block set, keeping the
        survivors' state-dict keys identical to the full single-GPU core."""
        owned = set(int(b) for b in owned)
        last = self.pipeline_num_blocks() - 1
        if 0 not in owned:
            self.lifting = None
        if last not in owned:
            self.projection = None
        keep = {b - 1 for b in owned if 0 < b < last}
        for i in range(self.n_layers):
            if i not in keep:
                self.spectral_layers[i] = nn.Identity()
                self.pointwise_layers[i] = nn.Identity()
                if self.channel_mlps is not None:
                    self.channel_mlps[i] = nn.Identity()

    def export_model_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'fno_grid_resolution': list(self.resolution),
            'fno_modes': list(self.modes),
            'fno_hidden_channels': self.hidden,
            'fno_layers': self.n_layers,
            'fno_use_channel_mlp': self.use_channel_mlp,
            'fno_norm': 'none',
            'in_channels': self.in_channels,
        }
