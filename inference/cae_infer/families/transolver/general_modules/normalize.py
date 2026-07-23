"""Inference-only extraction of the normalization/denormalization helpers from
the source repo's general_modules/mesh_dataset.py. Verbatim logic; the rest of
mesh_dataset.py (MeshGraphDataset -- HDF5 loading, z-score stats fitting,
train/val/test split, geometric augmentation) is training machinery this
bundle never needs (it pulls in general_modules/dataset_stats.py and
general_modules/time_integration.py, neither of which is on the forward path).
"""

from typing import Dict

import numpy as np

POSITION_SCALE_EPS = 1e-8


def normalize_positions(pos_raw: np.ndarray, position_scale: float,
                         eps: float = POSITION_SCALE_EPS) -> np.ndarray:
    """Per-sample centered_isotropic geometry normalization."""
    center = pos_raw.mean(axis=0)
    pos_centered = pos_raw - center
    return pos_centered / max(position_scale, eps)


def normalize_node_features(x_raw: np.ndarray, node_mean: np.ndarray, node_std: np.ndarray,
                             node_types: np.ndarray = None, node_type_to_idx: Dict = None,
                             num_node_types: int = None) -> np.ndarray:
    """Z-score node features, then append the one-hot node-type block after
    numeric normalization."""
    x_norm = (x_raw - node_mean) / node_std
    if node_types is not None:
        if node_type_to_idx is None or num_node_types is None:
            raise RuntimeError("node_types given but node_type_to_idx/num_node_types missing.")
        node_type_indices = np.array(
            [node_type_to_idx[t] for t in node_types], dtype=np.int32)
        node_type_onehot = np.zeros((len(node_types), num_node_types), dtype=np.float32)
        node_type_onehot[np.arange(len(node_types)), node_type_indices] = 1.0
        x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)
    return x_norm


def denormalize_delta(delta_norm: np.ndarray, delta_mean: np.ndarray, delta_std: np.ndarray) -> np.ndarray:
    return delta_norm * delta_std + delta_mean
