"""HDF5-to-PyG loader shared by all four operator models (IMPLEMENTATION_PLAN.md
section 4). Adapted from MeshGraphNets' general_modules/mesh_dataset.py and
transolver's variant of the same file: split/target/noise semantics are an
exact port of the pinned MGN behavior; edge attributes are dropped (no
operator core here consumes MGN edge features, per section 3); coordinate
normalization (`pos_normalized`, `position_scale`) and the coordinate-domain
statistics needed by grid/point/GINO adapters (active axes, grid bounds,
rotation-invariant radius) are new, following transolver's precedent of
extending the MGN dataset contract for a non-message-passing model.
"""

from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from general_modules.config_validation import validate_temporal_contract
from general_modules.dataset_stats import (
    compute_normalization_stats, finalize_moments, finalize_position_scale,
    finalize_axis_bounds, resolve_active_axes, finalize_rot_invariant_radius,
)
from general_modules.positional_features import compute_positional_features
from model.adapters.sdf import sdf_available, load_sdf

POSITION_SCALE_EPS = 1e-8


def normalize_positions(pos_raw: np.ndarray, position_scale: float,
                         eps: float = POSITION_SCALE_EPS) -> np.ndarray:
    """Per-sample centered_isotropic geometry normalization (section 4.4).

    Shared by MeshGraphDataset.__getitem__ (training/direct inference) and
    inference_profiles' rollout graph builder so coordinate normalization
    cannot drift between call sites.
    """
    center = pos_raw.mean(axis=0)
    pos_centered = pos_raw - center
    return pos_centered / max(position_scale, eps)


def normalize_node_features(x_raw: np.ndarray, node_mean: np.ndarray, node_std: np.ndarray,
                             node_types: np.ndarray = None, node_type_to_idx: Dict = None,
                             num_node_types: int = None) -> np.ndarray:
    """Z-score node features, then append the one-hot node-type block after
    numeric normalization (section 4.3). Shared for the same reason as
    normalize_positions above."""
    x_norm = (x_raw - node_mean) / node_std
    if node_types is not None:
        if node_type_to_idx is None or num_node_types is None:
            raise RuntimeError("node_types given but node_type_to_idx/num_node_types missing.")
        node_type_indices = np.array(
            [node_type_to_idx[int(t)] for t in node_types], dtype=np.int32)
        node_type_onehot = np.zeros((len(node_types), num_node_types), dtype=np.float32)
        node_type_onehot[np.arange(len(node_types)), node_type_indices] = 1.0
        x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)
    return x_norm


def denormalize_delta(delta_norm: np.ndarray, delta_mean: np.ndarray, delta_std: np.ndarray) -> np.ndarray:
    return delta_norm * delta_std + delta_mean


def random_augmentation_matrix(rng: np.random.Generator = None) -> np.ndarray:
    """Random full Z-axis rotation + independent x/y reflection matrix [3, 3]
    (exact port of MeshGraphNets' _random_augmentation_matrix, section 4.5).

    Gravity-independent: full Z-rotation (0-360) and axis reflections are
    valid. Translation is skipped; nothing here consumes absolute position.
    """
    if rng is None:
        theta = np.random.uniform(0, 2 * np.pi)
        flip_x = np.random.random() < 0.5
        flip_y = np.random.random() < 0.5
    else:
        theta = rng.uniform(0, 2 * np.pi)
        flip_x = rng.random() < 0.5
        flip_y = rng.random() < 0.5
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    if flip_x:
        R[0, :] *= -1
    if flip_y:
        R[1, :] *= -1
    return R


class MeshGraphDataset(Dataset):
    """HDF5-to-PyG loader matching the MeshGraphNets on-disk schema
    (IMPLEMENTATION_PLAN.md section 4.1) with a shared operator-model runtime
    contract: no edge_attr, an added `pos_normalized` (centered_isotropic
    geometry input), and the corrected node-type row rule (last row, only
    when the file has more than 7 feature rows).
    """

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        self.input_dim = config.get('input_var')
        self.output_dim = config.get('output_var')
        self.num_pos_features = int(config.get('positional_features', 0))
        self.use_node_types = config.get('use_node_types', False)
        self.dimension_tolerance = float(config.get('dimension_tolerance', 1e-4))
        self.grid_padding = float(config.get('grid_padding', 0.05))
        configured_dim = config.get('operator_dim', 'auto')
        self.configured_operator_dim = configured_dim if configured_dim == 'auto' else int(configured_dim)
        self.sdf_source = str(config.get('sdf_source', 'none')).lower()
        self.sdf_sidecar = config.get('sdf_sidecar', 'none')

        self.num_node_types = None
        self.node_type_to_idx = None

        self.node_mean = None
        self.node_std = None
        self.delta_mean = None
        self.delta_std = None
        self.position_scale = None
        self.active_axes = None
        self.operator_dim = None
        self.grid_bound_min = None
        self.grid_bound_max = None
        self.rot_invariant_radius = None
        self.has_sdf = self.sdf_source != 'none'
        self.sdf_mean = None
        self.sdf_std = None
        self._h5_handle = None
        self._static_cache: Dict = {}
        self.is_training = False
        self.augment_geometry = False

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}")
        print(f"  positional_features: {self.num_pos_features}")
        print(f"  use_node_types: {self.use_node_types}")

        with h5py.File(h5_file, 'r') as f:
            if 'data' not in f:
                raise ValueError(f"HDF5 file missing 'data' group: {h5_file}")

            self.sample_ids = sorted([int(k) for k in f['data'].keys()])
            if not self.sample_ids:
                raise ValueError(f"HDF5 file has no samples under 'data': {h5_file}")

            sample_id = self.sample_ids[0]
            nodal_shape = f[f'data/{sample_id}/nodal_data'].shape  # [features, time, nodes]
            self.num_features = nodal_shape[0]
            self.num_timesteps = nodal_shape[1]

        if self.use_node_types and self.num_features <= 7:
            raise ValueError(
                f"use_node_types=True but nodal_data has only {self.num_features} feature "
                f"rows (need row index -1, i.e. row 7, to exist). This dataset has no node "
                f"type / part-number row; set use_node_types False or use a dataset that "
                f"provides one (IMPLEMENTATION_PLAN.md section 4.1)."
            )

        config['num_timesteps'] = self.num_timesteps
        validate_temporal_contract(config)

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}, feature rows: {self.num_features}")

        self._validate_edge_indices()
        if self.has_sdf:
            self._validate_sdf_available()

    def _validate_sdf_available(self) -> None:
        """Fail fast if sdf_source is set but SDF data is missing for any sample."""
        for sid in self.sample_ids:
            if not sdf_available(self.h5_file, sid, self.sdf_source, self.sdf_sidecar):
                raise ValueError(
                    f"sdf_source='{self.sdf_source}' but sample {sid} has no SDF data "
                    f"(checked {'dataset field' if self.sdf_source == 'dataset' else self.sdf_sidecar}). "
                    "Set sdf_source none, or provide SDF for every sample (section 4.1/7.5)."
                )

    def _validate_edge_indices(self) -> None:
        """Fail fast on malformed topology instead of surfacing a later indexing error."""
        with h5py.File(self.h5_file, 'r') as f:
            for sid in self.sample_ids:
                grp = f[f'data/{sid}']
                n_nodes = grp['nodal_data'].shape[2]
                edge_data = grp['mesh_edge'][:]
                if edge_data.shape[1] == 0:
                    raise ValueError(f"Sample {sid}: mesh_edge has 0 edges.")
                emin, emax = int(edge_data.min()), int(edge_data.max())
                if emin < 0 or emax >= n_nodes:
                    raise ValueError(
                        f"Sample {sid}: edge index out of range [{emin}, {emax}] "
                        f"for {n_nodes} nodes."
                    )
                if n_nodes == 0:
                    raise ValueError(f"Sample {sid}: 0 nodes.")

    def prepare_preprocessing(self, use_parallel_stats: bool = True) -> None:
        """Fit preprocessing statistics using this dataset's sample_ids only."""
        if self.use_node_types:
            self._compute_node_type_info()
        self._compute_zscore_and_coordinate_stats(use_parallel_stats)
        if self.has_sdf:
            self._compute_sdf_stats()

    def _compute_sdf_stats(self) -> None:
        """Element-weighted mean/std of SDF values over the train split.

        A dedicated (serial) pass: no shipped dataset has SDF, so this never
        runs on the hot path used by every training run.
        """
        total_sum, total_sumsq, total_count = 0.0, 0.0, 0
        for sid in self.sample_ids:
            arr = load_sdf(self.h5_file, sid, self.sdf_source, self.sdf_sidecar)
            total_sum += float(np.sum(arr))
            total_sumsq += float(np.sum(arr ** 2))
            total_count += arr.shape[0]
        mean = total_sum / total_count
        var = max(total_sumsq / total_count - mean ** 2, 0.0)
        self.sdf_mean = np.float32(mean)
        self.sdf_std = np.float32(max(np.sqrt(var), 1e-8))
        print(f"  SDF stats - mean: {self.sdf_mean:.6f}, std: {self.sdf_std:.6f}")

    def inherit_preprocessing_from(self, source_dataset) -> None:
        """Reuse preprocessing fit on another dataset, typically the train split."""
        self.node_mean = source_dataset.node_mean.copy()
        self.node_std = source_dataset.node_std.copy()
        self.delta_mean = source_dataset.delta_mean.copy()
        self.delta_std = source_dataset.delta_std.copy()
        self.position_scale = source_dataset.position_scale
        self.active_axes = source_dataset.active_axes
        self.operator_dim = source_dataset.operator_dim
        self.grid_bound_min = source_dataset.grid_bound_min.copy()
        self.grid_bound_max = source_dataset.grid_bound_max.copy()
        self.rot_invariant_radius = source_dataset.rot_invariant_radius
        self.num_node_types = source_dataset.num_node_types
        self.node_type_to_idx = (
            dict(source_dataset.node_type_to_idx)
            if source_dataset.node_type_to_idx is not None
            else None
        )
        self.sdf_mean = source_dataset.sdf_mean
        self.sdf_std = source_dataset.sdf_std

    def _create_subset(self, sample_ids: List[int], is_training: bool = False):
        subset = MeshGraphDataset.__new__(MeshGraphDataset)
        subset.h5_file = self.h5_file
        subset.config = self.config
        subset.input_dim = self.input_dim
        subset.output_dim = self.output_dim
        subset.num_pos_features = self.num_pos_features
        subset.use_node_types = self.use_node_types
        subset.dimension_tolerance = self.dimension_tolerance
        subset.grid_padding = self.grid_padding
        subset.configured_operator_dim = self.configured_operator_dim
        subset.sdf_source = self.sdf_source
        subset.sdf_sidecar = self.sdf_sidecar
        subset.has_sdf = self.has_sdf
        subset.sdf_mean = None
        subset.sdf_std = None
        subset.sample_ids = list(sample_ids)
        subset.num_features = self.num_features
        subset.num_timesteps = self.num_timesteps
        subset.num_node_types = None
        subset.node_type_to_idx = None
        subset.node_mean = None
        subset.node_std = None
        subset.delta_mean = None
        subset.delta_std = None
        subset.position_scale = None
        subset.active_axes = None
        subset.operator_dim = None
        subset.grid_bound_min = None
        subset.grid_bound_max = None
        subset.rot_invariant_radius = None
        subset._static_cache = {}
        subset._h5_handle = None
        subset.is_training = is_training
        subset.augment_geometry = bool(self.config.get('augment_geometry', False)) and is_training
        return subset

    def _resolve_split_ids(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
        """Always generate a deterministic seeded split (exact MGN semantics)."""
        rng = np.random.default_rng(seed)
        shuffled_ids = self.sample_ids.copy()
        rng.shuffle(shuffled_ids)
        n_samples = len(shuffled_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val:]
        print(f"Using seeded random split (seed={seed}).")
        return train_ids, val_ids, test_ids

    def _compute_zscore_and_coordinate_stats(self, use_parallel_stats: bool) -> None:
        print('Computing normalization + coordinate-domain statistics...')

        stats = compute_normalization_stats(
            self.h5_file, self.sample_ids, self.input_dim, self.output_dim,
            self.num_timesteps, self.num_pos_features,
            use_parallel=use_parallel_stats,
        )

        self.node_mean, self.node_std = finalize_moments(
            stats['node_sum'], stats['node_sumsq'], stats['node_count'])
        self.delta_mean, self.delta_std = finalize_moments(
            stats['delta_sum'], stats['delta_sumsq'], stats['delta_count'])

        self.position_scale = finalize_position_scale(stats['pos_sqnorm_sum'], stats['pos_count'])
        axis_min_norm, axis_max_norm = finalize_axis_bounds(
            stats['axis_min_raw'], stats['axis_max_raw'], self.position_scale)
        active_axes, extent = resolve_active_axes(axis_min_norm, axis_max_norm, self.dimension_tolerance)

        if self.configured_operator_dim != 'auto' and self.configured_operator_dim != len(active_axes):
            raise ValueError(
                f"operator_dim={self.configured_operator_dim} but training geometry resolves "
                f"{len(active_axes)} active axes {active_axes} (extents={extent.tolist()}). "
                "Set operator_dim auto or fix the mismatch."
            )
        self.active_axes = active_axes
        self.operator_dim = len(active_axes)

        pad = self.grid_padding * np.maximum(extent, 1e-8)
        grid_min = axis_min_norm - pad
        grid_max = axis_max_norm + pad

        self.rot_invariant_radius = finalize_rot_invariant_radius(
            stats['xy_radius_max_raw'], self.position_scale)
        if self.config.get('augment_geometry', False):
            R = self.rot_invariant_radius * (1.0 + self.grid_padding)
            if 0 in active_axes:
                grid_min[0], grid_max[0] = -R, R
            if 1 in active_axes:
                grid_min[1], grid_max[1] = -R, R
            print(f"  augment_geometry=True: x/y grid bounds forced to rotation-safe [-{R:.4f}, {R:.4f}]")

        self.grid_bound_min = grid_min.astype(np.float32)
        self.grid_bound_max = grid_max.astype(np.float32)

        print(f'  Node features - mean: {self.node_mean}, std: {self.node_std}')
        print(f'  Delta features - mean: {self.delta_mean}, std: {self.delta_std}')
        print(f'  position_scale: {self.position_scale:.6f}')
        print(f'  active_axes: {self.active_axes}  (operator_dim={self.operator_dim})')
        print(f'  grid_bound_min: {self.grid_bound_min}, grid_bound_max: {self.grid_bound_max}')
        print(f'  rot_invariant_radius: {self.rot_invariant_radius:.6f}')
        self._warn_on_degenerate_stats()

    def _warn_on_degenerate_stats(self) -> None:
        warnings = []
        if np.any(self.node_std < 1e-6):
            warnings.append("  CRITICAL: Near-zero node variance - feature is constant!")
        if np.any(self.delta_std < 1e-6):
            warnings.append("  CRITICAL: Near-zero delta variance - targets are constant!")
        for w in warnings:
            print(w)

    def _compute_node_type_info(self) -> None:
        print('Computing node type information...')
        with h5py.File(self.h5_file, 'r') as f:
            unique_types = set()
            for sid in self.sample_ids:
                node_types = f[f'data/{sid}/nodal_data'][-1, 0, :].astype(np.int32)
                unique_types.update(node_types.tolist())

            sorted_types = sorted(unique_types)
            self.node_type_to_idx = {t: i for i, t in enumerate(sorted_types)}
            self.num_node_types = len(unique_types)
            print(f'  Found {self.num_node_types} unique node types: {sorted_types}')

    def __len__(self) -> int:
        if self.num_timesteps > 1:
            return len(self.sample_ids) * (self.num_timesteps - 1)
        return len(self.sample_ids)

    def _get_h5_handle(self):
        if not hasattr(self, '_h5_handle') or self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_file, 'r', swmr=True)
        return self._h5_handle

    def _get_static_sample_data(self, sample_id: int, h5_handle, nodal_dset):
        """Cache per-sample topology and positional features inside each worker.

        Positional features and node types depend on reference geometry/topology
        only, which is constant across timesteps, so they are computed once per
        sample and cached (mirrors MeshGraphNets).
        """
        cache = self._static_cache
        cached = cache.get(sample_id)
        if cached is not None:
            return cached

        mesh_edge = h5_handle[f'data/{sample_id}/mesh_edge'][:]  # [2, M]
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        first_step = nodal_dset[:, 0, :]  # [features, N] timestep 0 only

        node_types = first_step[-1, :].astype(np.int32) if self.use_node_types else None

        x_pos = None
        if self.num_pos_features > 0:
            ref_pos_0 = first_step[:3, :].T
            x_pos = compute_positional_features(ref_pos_0, edge_index, self.num_pos_features)

        cached = (edge_index, x_pos, node_types)
        if len(cache) < 2000:
            cache[sample_id] = cached
        return cached

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5_handle'] = None
        state['_static_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        if getattr(self, '_h5_handle', None) is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            self._h5_handle = None

    def __getitem__(self, idx: int) -> Data:
        if self.num_timesteps > 1:
            sample_idx = idx // (self.num_timesteps - 1)
            time_idx = idx % (self.num_timesteps - 1)
        else:
            sample_idx = idx
            time_idx = 0

        sample_id = self.sample_ids[sample_idx]

        f = self._get_h5_handle()
        dset = f[f'data/{sample_id}/nodal_data']  # [F, T, N] on disk
        edge_index, x_pos, node_types = self._get_static_sample_data(sample_id, f, dset)
        part_ids = node_types

        if self.num_timesteps == 1:
            data_t = dset[:, 0, :].T  # [N, F]
            pos_raw = data_t[:, :3].copy()
            x_phys = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)
            target_delta = data_t[:, 3:3 + self.output_dim].copy()
        else:
            pair = dset[:, time_idx:time_idx + 2, :]  # [F, 2, N]
            data_t = pair[:, 0, :].T
            data_t1 = pair[:, 1, :].T
            pos_raw = data_t[:, :3].copy()
            x_phys = data_t[:, 3:3 + self.input_dim].copy()
            y_raw = data_t1[:, 3:3 + self.output_dim]
            target_delta = y_raw - x_phys[:, :self.output_dim]

        if getattr(self, 'augment_geometry', False):
            R = random_augmentation_matrix()
            pos_raw = pos_raw @ R.T
            if x_phys.shape[1] >= 3:
                x_phys[:, :3] = x_phys[:, :3] @ R.T
            if target_delta.shape[1] >= 3:
                target_delta[:, :3] = target_delta[:, :3] @ R.T

        if self.num_pos_features > 0:
            x_raw = np.concatenate([x_phys, x_pos], axis=1)
        else:
            x_raw = x_phys

        if self.node_mean is None or self.node_std is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: node statistics are missing.")
        x_norm = normalize_node_features(
            x_raw, self.node_mean, self.node_std,
            node_types=node_types if self.use_node_types else None,
            node_type_to_idx=self.node_type_to_idx, num_node_types=self.num_node_types,
        )
        target_norm = (target_delta - self.delta_mean) / self.delta_std
        pos_normalized = normalize_positions(pos_raw, self.position_scale)

        pos_t = torch.from_numpy(pos_raw.astype(np.float32))
        pos_norm_t = torch.from_numpy(pos_normalized.astype(np.float32))
        x_t = torch.from_numpy(x_norm.astype(np.float32))
        y_t = torch.from_numpy(target_norm.astype(np.float32))
        edge_index_t = torch.from_numpy(edge_index).long()
        part_ids_t = torch.from_numpy(part_ids).long() if part_ids is not None else None

        graph_data = Data(
            x=x_t,
            y=y_t,
            pos=pos_t,
            pos_normalized=pos_norm_t,
            edge_index=edge_index_t,
            sample_id=sample_id,
            time_idx=time_idx if self.num_timesteps > 1 else None,
            part_ids=part_ids_t,
        )

        if self.has_sdf:
            sdf_raw = load_sdf(self.h5_file, sample_id, self.sdf_source, self.sdf_sidecar,
                               time_idx=time_idx)
            sdf_norm = (sdf_raw - self.sdf_mean) / self.sdf_std
            graph_data.sdf = torch.from_numpy(sdf_norm.astype(np.float32)).unsqueeze(-1)

        return graph_data

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 0):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        train_ids, val_ids, test_ids = self._resolve_split_ids(train_ratio, val_ratio, test_ratio, seed)

        train_dataset = self._create_subset(train_ids, is_training=True)
        val_dataset = self._create_subset(val_ids, is_training=False)
        test_dataset = self._create_subset(test_ids, is_training=False)

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        print("Fitting preprocessing on train split only...")
        use_parallel_stats = self.config.get('use_parallel_stats', True)
        train_dataset.prepare_preprocessing(use_parallel_stats)
        val_dataset.inherit_preprocessing_from(train_dataset)
        test_dataset.inherit_preprocessing_from(train_dataset)

        return train_dataset, val_dataset, test_dataset
