"""Point-DeepONet -- the primary model (IMPLEMENTATION_PLAN.md section 8.1,
2.4, A.3). PointNet branch (geometry + current state + context), SIREN query
trunk, early multiplicative fusion, refiners, final modal inner product --
following the verified published architecture (Park & Kang 2026), adapted to
the current MGN-data contract (`point_variant mesh_state`, the default) since
no dataset here has validated SDF or declared global load conditions.
`point_variant paper` enforces the published input contract and fails
loudly when that data is missing.
"""

import torch
import torch.nn as nn

from model.base import OperatorCore
from model.mlp import build_deep_mlp, init_weights
from model.pointnet import PointNetEncoder
from model.siren import Siren
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.point_sampling import PointSampler

_VALID_VARIANTS = ("mesh_state", "paper")
_VALID_OUTPUT_ACTIVATIONS = ("identity", "tanh")


def validate_config(config, data_spec):
    variant = str(config.get('point_variant', 'mesh_state')).lower()
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"point_variant must be one of {_VALID_VARIANTS}, got '{variant}'.")

    sensor_count = int(config.get('point_sensor_count', 5000))
    if sensor_count < 0:
        raise ValueError(f"point_sensor_count must be >= 0, got {sensor_count}.")

    sampling = str(config.get('point_sampling', 'random')).lower()
    if sampling != 'random':
        raise ValueError(f"point_sampling must be 'random' (baseline), got '{sampling}'.")

    pointnet_activation = str(config.get('pointnet_activation', 'relu')).lower()
    if pointnet_activation != 'relu':
        raise ValueError(f"pointnet_activation must be 'relu' (baseline), got '{pointnet_activation}'.")
    pointnet_norm = str(config.get('pointnet_norm', 'batch')).lower()
    if pointnet_norm != 'batch':
        raise ValueError(f"pointnet_norm must be 'batch' (baseline), got '{pointnet_norm}'.")
    branch_merge = str(config.get('point_branch_merge', 'sum')).lower()
    if branch_merge != 'sum':
        raise ValueError(f"point_branch_merge must be 'sum' (baseline), got '{branch_merge}'.")

    output_activation = str(config.get('point_output_activation', 'identity')).lower()
    if output_activation not in _VALID_OUTPUT_ACTIVATIONS:
        raise ValueError(
            f"point_output_activation must be one of {_VALID_OUTPUT_ACTIVATIONS}, "
            f"got '{output_activation}'."
        )

    if variant == 'paper':
        if not data_spec.has_sdf:
            raise ValueError(
                "point_variant=paper requires sdf_source != none (finite SDF at every "
                "query); the current dataset provides none. Use point_variant mesh_state."
            )
        if data_spec.global_condition_dim <= 0:
            raise ValueError(
                "point_variant=paper requires declared global_condition_features "
                "(force/direction/mass); none are declared."
            )
        if output_activation != 'tanh':
            raise ValueError("point_variant=paper requires point_output_activation tanh.")


def _graph_scalar(graph, name, g: int, default=None):
    """Extract graph `g`'s value of a possibly-batched scalar attribute.

    After PyG batching, a per-graph int attribute becomes a 1-D tensor
    indexed by graph position; if every graph in the batch had `None` for
    that attribute (e.g. `time_idx` on static data), PyG drops the attribute
    entirely. A bare (unbatched) `Data` object (rollout path) carries the
    plain Python scalar directly.
    """
    value = getattr(graph, name, None)
    if value is None:
        return default
    if torch.is_tensor(value):
        return value[g].item() if value.dim() > 0 else value.item()
    if isinstance(value, (list, tuple)):
        return value[g]
    return value


class PointDeepONet(OperatorCore):
    model_name = "point_deeponet"

    def __init__(self, config, data_spec, coordinate_domain: CoordinateDomain):
        super().__init__()
        self.data_spec = data_spec
        self.domain = coordinate_domain
        self.output_var = data_spec.output_var
        self.variant = str(config.get('point_variant', 'mesh_state')).lower()
        self.sensor_count = int(config.get('point_sensor_count', 5000))
        self.resample_each_epoch = bool(config.get('point_resample_each_epoch', True))
        self.hidden_channels = int(config.get('point_hidden_channels', 128))
        self.H = int(config.get('point_feature_dim', 128))
        self.pointnet_depth = int(config.get('pointnet_depth', 3))
        self.condition_depth = int(config.get('point_condition_depth', 2))
        self.trunk_depth = int(config.get('point_trunk_depth', 3))
        self.refiner_depth = int(config.get('point_refiner_depth', 2))
        self.omega0 = float(config.get('point_siren_omega0', 30.0))
        self.output_activation = str(config.get('point_output_activation', 'identity')).lower()
        self.has_sdf = bool(data_spec.has_sdf)
        self.base_seed = int(config.get('split_seed', 42))

        self.use_all_points = self.sensor_count == 0
        self.sampler = (
            None if self.use_all_points else
            PointSampler(self.sensor_count, base_seed=self.base_seed,
                        resample_each_epoch=self.resample_each_epoch)
        )

        geometry_only = self.variant == 'paper'
        sensor_state_width = 0 if geometry_only else data_spec.total_node_dim
        sensor_in = data_spec.operator_dim + sensor_state_width + (1 if self.has_sdf else 0)
        self.geometry_only = geometry_only

        self.pointnet = PointNetEncoder(
            sensor_in, self.hidden_channels, self.H,
            depth=self.pointnet_depth, activation='relu', norm='batch',
        )
        self.geometry_proj = nn.Linear(self.H, self.H)

        if data_spec.global_condition_dim > 0:
            self.condition_mlp = build_deep_mlp(
                data_spec.global_condition_dim, self.hidden_channels, self.H,
                self.condition_depth, activation='silu',
            )
            self.condition_proj = nn.Linear(self.H, self.H)
        else:
            self.condition_mlp = None
            self.condition_proj = None

        query_dim = data_spec.operator_dim + data_spec.context_dim + (1 if self.has_sdf else 0)
        self.query_dim = query_dim
        self.trunk = Siren(query_dim, self.H, depth=self.trunk_depth, omega0=self.omega0)

        self.branch_refiner = build_deep_mlp(
            self.H, self.hidden_channels, self.H, self.refiner_depth, activation='relu')
        self.trunk_refiner = build_deep_mlp(
            self.H, self.hidden_channels, self.H * self.output_var,
            self.refiner_depth, activation='relu')
        self.bias = nn.Parameter(torch.zeros(self.output_var))

        # Kaiming-init everything except the SIREN trunk (already correctly
        # self-initialized in its own constructor; re-applying init_weights
        # there would destroy the sine-specific bounds, section 12.4).
        for module in [self.pointnet, self.geometry_proj, self.branch_refiner, self.trunk_refiner]:
            module.apply(init_weights)
        if self.condition_mlp is not None:
            self.condition_mlp.apply(init_weights)
            self.condition_proj.apply(init_weights)

        if data_spec.num_timesteps > 1:
            with torch.no_grad():
                self.branch_refiner[-1].weight.mul_(0.01)

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def _sensor_features(self, graph, idx):
        coords = self.domain.select_active(graph.pos_normalized[idx])
        parts = [coords]
        if not self.geometry_only:
            parts.append(graph.x[idx])
        if self.has_sdf:
            parts.append(graph.sdf[idx])
        return torch.cat(parts, dim=1)

    def _query_features(self, graph):
        coords = self.domain.select_active(graph.pos_normalized)
        context = graph.x[:, self.data_spec.context_slice]
        parts = [coords, context]
        if self.has_sdf:
            parts.append(graph.sdf)
        return torch.cat(parts, dim=1)

    def _branch_context(self, graph):
        device = graph.x.device
        num_graphs = graph.ptr.numel() - 1

        if self.use_all_points:
            all_idx = torch.arange(graph.x.shape[0], device=device)
            sensor_feat = self._sensor_features(graph, all_idx)
            geom = self.pointnet.forward_segmented(sensor_feat, graph.batch, num_graphs)
        else:
            idx_parts = []
            for g in range(num_graphs):
                start, end = int(graph.ptr[g]), int(graph.ptr[g + 1])
                n_g = end - start
                sample_id = _graph_scalar(graph, 'sample_id', g, default=0)
                time_idx = _graph_scalar(graph, 'time_idx', g, default=None)
                local_idx = self.sampler.sample_indices(
                    n_g, sample_id, time_idx, self.training, device=device)
                idx_parts.append(local_idx + start)
            idx_flat = torch.cat(idx_parts, dim=0)
            sensor_feat = self._sensor_features(graph, idx_flat)
            dense = sensor_feat.view(num_graphs, self.sensor_count, -1)
            geom = self.pointnet.forward_dense(dense)

        geom = self.geometry_proj(geom)
        if self.condition_mlp is not None:
            gc = graph.global_conditions.view(num_graphs, -1)
            cond = self.condition_proj(self.condition_mlp(gc))
            branch = geom + cond
        else:
            branch = geom
        return branch

    def _decode(self, branch, graph, start: int, end: int):
        query = self._query_features(graph)[start:end]
        batch_slice = graph.batch[start:end]

        t0 = self.trunk(query)                                   # [chunk, H]
        b0 = branch[batch_slice]                                 # [chunk, H]
        fused = b0 * t0                                          # early elementwise product

        b_beta = self.branch_refiner(fused)                      # [chunk, H]
        t_beta = self.trunk_refiner(fused).view(-1, self.H, self.output_var)

        with torch.autocast(device_type=query.device.type, enabled=False):
            pred = torch.einsum('nh,nho->no', b_beta.float(), t_beta.float()) + self.bias.float()
        if self.output_activation == 'tanh':
            pred = torch.tanh(pred)
        return pred.to(graph.x.dtype)

    def forward(self, graph):
        branch = self._branch_context(graph)
        return self._decode(branch, graph, 0, graph.x.shape[0])

    def supports_query_chunking(self) -> bool:
        return True

    def encode_operator(self, graph):
        return self._branch_context(graph)

    def decode_queries(self, encoded, graph, start: int, end: int):
        return self._decode(encoded, graph, start, end)

    def export_model_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'point_variant': self.variant,
            'point_sensor_count': self.sensor_count,
            'point_hidden_channels': self.hidden_channels,
            'point_feature_dim': self.H,
            'pointnet_depth': self.pointnet_depth,
            'point_condition_depth': self.condition_depth,
            'point_trunk_depth': self.trunk_depth,
            'point_refiner_depth': self.refiner_depth,
            'point_siren_omega0': self.omega0,
            'point_output_activation': self.output_activation,
            'query_dim': self.query_dim,
        }
