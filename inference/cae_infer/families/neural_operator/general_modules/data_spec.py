"""Immutable dataset-derived specification shared by every model (section 6.1).

Built once after the training split is scanned; stored verbatim in every
checkpoint's `data_config`. Adapters and models slice `x` using the offsets
below instead of hardcoding column indices.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataSpec:
    input_var: int              # physical input channels (e.g. 4)
    output_var: int              # physical output channels (e.g. 4)
    positional_dim: int          # P: numeric positional feature columns in x
    node_type_dim: int           # K: one-hot width (0 when use_node_types is False)
    global_condition_dim: int    # Cg (0 today: no dataset provides conditions)
    operator_dim: int            # 2 or 3, resolved from active axes
    active_axes: Tuple[int, ...]  # subset of (0, 1, 2) meaning (x, y, z)
    has_sdf: bool
    has_integration_weights: bool
    num_timesteps: int

    @property
    def context_dim(self) -> int:
        """Width of x excluding the leading physical-state block."""
        return self.positional_dim + self.node_type_dim

    @property
    def total_node_dim(self) -> int:
        """Cx: total width of graph.x."""
        return self.input_var + self.positional_dim + self.node_type_dim

    @property
    def physical_slice(self) -> slice:
        """Columns of x holding the normalized current physical state."""
        return slice(0, self.input_var)

    @property
    def positional_slice(self) -> slice:
        """Columns of x holding numeric positional features (no node type)."""
        start = self.input_var
        return slice(start, start + self.positional_dim)

    @property
    def onehot_slice(self) -> slice:
        """Columns of x holding the node-type one-hot block."""
        start = self.input_var + self.positional_dim
        return slice(start, start + self.node_type_dim)

    @property
    def context_slice(self) -> slice:
        """Columns of x excluding physical state (positional + one-hot)."""
        return slice(self.input_var, self.total_node_dim)

    def to_dict(self) -> dict:
        return {
            "input_var": self.input_var,
            "output_var": self.output_var,
            "positional_dim": self.positional_dim,
            "node_type_dim": self.node_type_dim,
            "global_condition_dim": self.global_condition_dim,
            "operator_dim": self.operator_dim,
            "active_axes": list(self.active_axes),
            "has_sdf": self.has_sdf,
            "has_integration_weights": self.has_integration_weights,
            "num_timesteps": self.num_timesteps,
        }

    @staticmethod
    def from_dict(d: dict) -> "DataSpec":
        return DataSpec(
            input_var=d["input_var"],
            output_var=d["output_var"],
            positional_dim=d["positional_dim"],
            node_type_dim=d["node_type_dim"],
            global_condition_dim=d["global_condition_dim"],
            operator_dim=d["operator_dim"],
            active_axes=tuple(d["active_axes"]),
            has_sdf=d["has_sdf"],
            has_integration_weights=d["has_integration_weights"],
            num_timesteps=d["num_timesteps"],
        )


def build_data_spec_from_dataset(train_dataset, config) -> DataSpec:
    """Construct the immutable DataSpec after a train split has been fit
    (section 6.1). `train_dataset` is a general_modules.mesh_dataset.MeshGraphDataset
    that has already run `prepare_preprocessing()` (i.e. the object returned
    as the first element of `MeshGraphDataset.split(...)`).
    """
    node_type_dim = train_dataset.num_node_types if train_dataset.use_node_types else 0
    global_condition_features = config.get('global_condition_features', 'none')
    global_condition_dim = 0
    if isinstance(global_condition_features, list):
        global_condition_dim = len(global_condition_features)
    elif isinstance(global_condition_features, str) and global_condition_features != 'none':
        global_condition_dim = 1

    return DataSpec(
        input_var=train_dataset.input_dim,
        output_var=train_dataset.output_dim,
        positional_dim=train_dataset.num_pos_features,
        node_type_dim=node_type_dim,
        global_condition_dim=global_condition_dim,
        operator_dim=train_dataset.operator_dim,
        active_axes=tuple(train_dataset.active_axes),
        has_sdf=bool(train_dataset.has_sdf),
        has_integration_weights=False,
        num_timesteps=train_dataset.num_timesteps,
    )
