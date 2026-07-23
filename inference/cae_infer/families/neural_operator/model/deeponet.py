"""Canonical (fixed-sensor) DeepONet -- the mathematical/reference model used
to test dot-product semantics (IMPLEMENTATION_PLAN.md section 8.2, A.2, A.3).

Ragged meshes are projected onto a checkpointed regular sensor grid via the
shared deterministic splat adapter (model/adapters/grid.py); this is what
keeps the branch input a fixed width regardless of mesh size, which is the
whole point of calling this model "DeepONet" rather than a set encoder.
"""

import torch
import torch.nn as nn

from model.base import OperatorCore
from model.mlp import build_deep_mlp, init_weights
from model.utils import parse_int_tuple
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.grid import splat, sample


def validate_config(config, data_spec):
    branch_source = str(config.get('deeponet_branch_source', 'fixed_sensors')).lower()
    if branch_source not in ('fixed_sensors', 'global_conditions'):
        raise ValueError(
            f"deeponet_branch_source must be 'fixed_sensors' or 'global_conditions', "
            f"got '{branch_source}'."
        )

    multi_output = str(config.get('deeponet_multi_output', 'split_both')).lower()
    if multi_output != 'split_both':
        raise ValueError(
            f"deeponet_multi_output must be 'split_both' (baseline), got '{multi_output}'."
        )

    if branch_source == 'global_conditions':
        if data_spec.global_condition_dim <= 0:
            raise ValueError(
                "deeponet_branch_source=global_conditions requires "
                "global_condition_dim > 0 (declare global_condition_features)."
            )
        if data_spec.num_timesteps > 1:
            print(
                "[deeponet] WARNING: global_conditions branch source with temporal "
                "data requires conditions alone to identify the transition; run "
                "misc/audit_input_identifiability.py before trusting this profile."
            )
        return

    resolution = parse_int_tuple(
        config.get('deeponet_sensor_resolution'), data_spec.operator_dim,
        'deeponet_sensor_resolution',
    )
    if any(r < 2 for r in resolution):
        raise ValueError(f"deeponet_sensor_resolution entries must be >= 2, got {resolution}.")

    prod_res = 1
    for r in resolution:
        prod_res *= r
    channels = data_spec.total_node_dim + (1 if data_spec.has_sdf else 0)
    branch_in_dim = channels * prod_res + 2 * prod_res + data_spec.global_condition_dim
    hidden = int(config.get('deeponet_hidden_channels', 256))
    estimated_params = branch_in_dim * hidden
    max_params = float(config.get('deeponet_max_branch_params', 1e8))
    if estimated_params > max_params:
        raise ValueError(
            f"Estimated DeepONet branch first-layer parameters ({estimated_params:,.0f}) "
            f"exceed deeponet_max_branch_params ({max_params:,.0f}). branch_in_dim="
            f"{branch_in_dim} (channels={channels} x prod(resolution)={prod_res} + "
            f"2*{prod_res} + conditions={data_spec.global_condition_dim}) x hidden={hidden}. "
            "Reduce deeponet_sensor_resolution or deeponet_hidden_channels."
        )


class DeepONet(OperatorCore):
    model_name = "deeponet"

    def __init__(self, config, data_spec, coordinate_domain: CoordinateDomain):
        super().__init__()
        self.data_spec = data_spec
        self.domain = coordinate_domain
        self.output_var = data_spec.output_var
        self.basis_dim = int(config.get('deeponet_basis_dim', 128))
        self.hidden = int(config.get('deeponet_hidden_channels', 256))
        self.branch_depth = int(config.get('deeponet_branch_depth', 3))
        self.trunk_depth = int(config.get('deeponet_trunk_depth', 3))
        self.activation = str(config.get('deeponet_activation', 'silu')).lower()
        self.branch_source = str(config.get('deeponet_branch_source', 'fixed_sensors')).lower()

        self.has_sdf = bool(data_spec.has_sdf)

        if self.branch_source == 'fixed_sensors':
            self.resolution = parse_int_tuple(
                config.get('deeponet_sensor_resolution'), data_spec.operator_dim,
                'deeponet_sensor_resolution',
            )
            prod_res = 1
            for r in self.resolution:
                prod_res *= r
            channels = data_spec.total_node_dim + (1 if self.has_sdf else 0)
            branch_in_dim = channels * prod_res + 2 * prod_res + data_spec.global_condition_dim
        else:
            self.resolution = None
            branch_in_dim = data_spec.global_condition_dim

        self.branch_in_dim = branch_in_dim
        self.branch_mlp = build_deep_mlp(
            branch_in_dim, self.hidden, self.output_var * self.basis_dim,
            self.branch_depth, activation=self.activation, layer_norm=False,
        )

        query_dim = data_spec.operator_dim + data_spec.context_dim + (1 if self.has_sdf else 0)
        self.query_dim = query_dim
        self.trunk_mlp = build_deep_mlp(
            query_dim, self.hidden, self.output_var * self.basis_dim,
            self.trunk_depth, activation=self.activation, layer_norm=False,
        )
        self.bias = nn.Parameter(torch.zeros(self.output_var))

        self.apply(init_weights)
        if data_spec.num_timesteps > 1:
            with torch.no_grad():
                self.branch_mlp[-1].weight.mul_(0.01)

    def _query_features(self, graph):
        """[sum_N, query_dim]: active-axis coords + context (positional +
        one-hot) + optional SDF. Physical state never enters the trunk."""
        coords = self.domain.select_active(graph.pos_normalized)
        context = graph.x[:, self.data_spec.context_slice]
        parts = [coords, context]
        if self.has_sdf:
            parts.append(graph.sdf)
        return torch.cat(parts, dim=1)

    def _branch_context(self, graph):
        """[B, output_var, basis_dim] per-graph branch coefficients."""
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() > 0 else 1
        if self.branch_source == 'fixed_sensors':
            c01, _ = self.domain.to_unit_box(graph.pos_normalized)
            values = graph.x
            if self.has_sdf:
                values = torch.cat([values, graph.sdf], dim=1)
            grid, occ, dens = splat(values, c01, graph.batch, num_graphs, self.resolution)
            branch_in = torch.cat([
                grid.flatten(1), occ.flatten(1), dens.flatten(1),
            ], dim=1)
            if self.data_spec.global_condition_dim > 0:
                gc = graph.global_conditions.view(num_graphs, -1)
                branch_in = torch.cat([branch_in, gc], dim=1)
        else:
            branch_in = graph.global_conditions.view(num_graphs, -1)

        branch_out = self.branch_mlp(branch_in)
        return branch_out.view(num_graphs, self.output_var, self.basis_dim)

    def _decode(self, branch_out, graph, start: int, end: int):
        query = self._query_features(graph)[start:end]
        batch_slice = graph.batch[start:end]
        trunk_out = self.trunk_mlp(query).view(-1, self.output_var, self.basis_dim)
        b = branch_out[batch_slice]
        with torch.autocast(device_type=query.device.type, enabled=False):
            pred = (b.float() * trunk_out.float()).sum(-1) + self.bias.float()
        return pred.to(graph.x.dtype)

    def forward(self, graph):
        branch_out = self._branch_context(graph)
        return self._decode(branch_out, graph, 0, graph.x.shape[0])

    def supports_query_chunking(self) -> bool:
        return True

    def encode_operator(self, graph):
        return self._branch_context(graph)

    def decode_queries(self, encoded, graph, start: int, end: int):
        return self._decode(encoded, graph, start, end)

    def export_model_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'deeponet_branch_source': self.branch_source,
            'deeponet_sensor_resolution': list(self.resolution) if self.resolution else None,
            'deeponet_hidden_channels': self.hidden,
            'deeponet_branch_depth': self.branch_depth,
            'deeponet_trunk_depth': self.trunk_depth,
            'deeponet_basis_dim': self.basis_dim,
            'deeponet_activation': self.activation,
            'deeponet_multi_output': 'split_both',
            'branch_in_dim': self.branch_in_dim,
            'query_dim': self.query_dim,
        }
