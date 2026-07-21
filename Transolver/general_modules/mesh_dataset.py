import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from torch_geometric.data import Data

from general_modules.config_validation import validate_temporal_contract
from general_modules.dataset_stats import (
    compute_normalization_stats, finalize_moments, finalize_position_scale,
)
from general_modules.positional_features import compute_positional_features
from general_modules.time_integration import (
    AR_RT,
    resolve_rollout_window,
    resolve_time_integration,
)

POSITION_SCALE_EPS = 1e-8


def normalize_positions(pos_raw: np.ndarray, position_scale: float,
                         eps: float = POSITION_SCALE_EPS) -> np.ndarray:
    """Per-sample centered_isotropic geometry normalization (section 5.5).

    Shared by MeshGraphDataset.__getitem__ (training/direct inference) and
    inference_profiles' point-sample builders (temporal rollout, decoupled
    inference) so coordinate normalization cannot drift between call sites.
    """
    center = pos_raw.mean(axis=0)
    pos_centered = pos_raw - center
    return pos_centered / max(position_scale, eps)


def normalize_node_features(x_raw: np.ndarray, node_mean: np.ndarray, node_std: np.ndarray,
                             node_types: np.ndarray = None, node_type_to_idx: Dict = None,
                             num_node_types: int = None) -> np.ndarray:
    """Z-score node features, then append the one-hot node-type block after
    numeric normalization (section 5.5). Shared for the same reason as
    normalize_positions above."""
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


class MeshGraphDataset(Dataset):
    """HDF5-to-PyG loader for Transolver, matching the MeshGraphNets on-disk
    schema (IMPLEMENTATION_PLAN.md section 5) but with a Transolver-specific
    runtime contract: no edge_attr (baseline Physics-Attention does not consume
    edges), an added `pos_normalized` (centered_isotropic geometry input), and
    the corrected node-type row contract (last row, only when present).
    """

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        self.input_dim = config.get('input_var')
        self.output_dim = config.get('output_var')
        self.num_pos_features = int(config.get('positional_features', 0))
        self.use_node_types = config.get('use_node_types', False)
        self.num_node_types = None
        self.node_type_to_idx = None

        self.node_mean = None
        self.node_std = None
        self.delta_mean = None
        self.delta_std = None
        self.position_scale = None
        self._h5_handle = None
        self._static_cache = {}
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
                f"provides one (IMPLEMENTATION_PLAN.md section 5.1)."
            )

        config['num_timesteps'] = self.num_timesteps
        validate_temporal_contract(config)

        # Time integration scheme (see training_profiles/ar_rollout.py):
        # ar_ot trains on ground-truth pairs, ar_rt unrolls the model over a
        # window of timesteps and trains on its own predictions.
        self.time_integration = resolve_time_integration(config)
        self.rollout_window = resolve_rollout_window(config, self.num_timesteps)

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}, feature rows: {self.num_features}")
        print(f"  time_integration: {self.time_integration}"
              + (f" (rollout window: {self.rollout_window} steps)"
                 if self.time_integration == AR_RT else ""))

        self._validate_edge_indices()

    def _validate_edge_indices(self) -> None:
        """Fail fast on malformed topology instead of surfacing a later indexing error."""
        with h5py.File(self.h5_file, 'r') as f:
            for sid in self.sample_ids:
                grp = f[f'data/{sid}']
                n_nodes = grp['nodal_data'].shape[2]
                edge_data = grp['mesh_edge'][:]
                if edge_data.shape[1] == 0:
                    raise ValueError(f"sample {sid}: mesh_edge has 0 edges")
                emin, emax = int(edge_data.min()), int(edge_data.max())
                if emin < 0 or emax >= n_nodes:
                    raise ValueError(
                        f"sample {sid}: edge index out of range [{emin}, {emax}] "
                        f"for {n_nodes} nodes"
                    )

    def prepare_preprocessing(self) -> None:
        """Fit preprocessing statistics using this dataset's sample_ids only."""
        if self.use_node_types:
            self._compute_node_type_info()
        self._compute_zscore_stats()

    def inherit_preprocessing_from(self, source_dataset) -> None:
        """Reuse preprocessing fit on another dataset, typically the train split."""
        self.node_mean = source_dataset.node_mean.copy()
        self.node_std = source_dataset.node_std.copy()
        self.delta_mean = source_dataset.delta_mean.copy()
        self.delta_std = source_dataset.delta_std.copy()
        self.position_scale = source_dataset.position_scale
        self.num_node_types = source_dataset.num_node_types
        self.node_type_to_idx = (
            dict(source_dataset.node_type_to_idx)
            if source_dataset.node_type_to_idx is not None
            else None
        )

    def inherit_preprocessing_from_dict(self, normalization: Dict) -> None:
        """Inject checkpoint-loaded statistics (section 11): lets inference
        reuse this same class -- and therefore the exact same __getitem__
        normalization path as training -- for an unrelated HDF5 file whose
        own statistics were never fit."""
        self.node_mean = np.asarray(normalization['node_mean'])
        self.node_std = np.asarray(normalization['node_std'])
        self.delta_mean = np.asarray(normalization['delta_mean'])
        self.delta_std = np.asarray(normalization['delta_std'])
        self.position_scale = float(normalization['position_scale'])
        self.node_type_to_idx = normalization.get('node_type_to_idx')
        self.num_node_types = normalization.get('num_node_types')

    def write_preprocessing_to_hdf5(self, split_seed: int) -> None:
        """Persist train-derived preprocessing statistics to the HDF5 dataset.

        Opt-in (config `write_preprocessing`) and namespaced under
        metadata/normalization_params/transolver — never overwrite MeshGraphNets'
        own metadata/normalization_params group merely by training this
        architecture on the same shared file (section 5.5, section 9).
        """
        if any(value is None for value in (
            self.node_mean, self.node_std, self.delta_mean, self.delta_std,
            self.position_scale,
        )):
            raise RuntimeError("Cannot write preprocessing stats before prepare_preprocessing()")

        with h5py.File(self.h5_file, 'r+') as f:
            metadata = f.require_group('metadata')
            norm_root = metadata.require_group('normalization_params')
            norm_group = norm_root.require_group('transolver')

            def _write_array(name: str, value: np.ndarray) -> None:
                if name in norm_group:
                    del norm_group[name]
                norm_group.create_dataset(name, data=value.astype(np.float32))

            _write_array('node_mean', self.node_mean)
            _write_array('node_std', self.node_std)
            _write_array('delta_mean', self.delta_mean)
            _write_array('delta_std', self.delta_std)

            norm_group.attrs['position_scale'] = self.position_scale
            norm_group.attrs['coordinate_normalization'] = 'centered_isotropic'
            norm_group.attrs['normalization_source'] = 'train_split'
            norm_group.attrs['split_seed'] = int(split_seed)

    def _create_subset(self, sample_ids: List[int], is_training: bool = False):
        subset = MeshGraphDataset.__new__(MeshGraphDataset)
        subset.h5_file = self.h5_file
        subset.config = self.config
        subset.input_dim = self.input_dim
        subset.output_dim = self.output_dim
        subset.num_pos_features = self.num_pos_features
        subset.num_features = self.num_features
        subset.sample_ids = list(sample_ids)
        subset.num_timesteps = self.num_timesteps
        subset.time_integration = self.time_integration
        subset.rollout_window = self.rollout_window
        subset.use_node_types = self.use_node_types
        subset.num_node_types = None
        subset.node_type_to_idx = None
        subset.node_mean = None
        subset.node_std = None
        subset.delta_mean = None
        subset.delta_std = None
        subset.position_scale = None
        subset._static_cache = {}
        subset._h5_handle = None
        subset.is_training = is_training
        subset.augment_geometry = self.config.get('augment_geometry', False) and is_training
        return subset

    def _resolve_split_ids(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
        """Always generate a deterministic seeded split. Stored HDF5 splits are
        ignored (they are empty in the shared dataset and not authoritative)."""
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

    def _compute_zscore_stats(self) -> None:
        """Compute train-split z-score statistics for node/delta features, and
        the isotropic position_scale."""
        print('Computing z-score normalization statistics...')

        stats = compute_normalization_stats(
            self.h5_file, self.sample_ids, self.input_dim, self.output_dim,
            self.num_timesteps, self.num_pos_features,
            use_parallel=self.config.get('use_parallel_stats', True),
        )

        self.node_mean, self.node_std = finalize_moments(
            stats['node_sum'], stats['node_sumsq'], stats['node_count'])
        self.delta_mean, self.delta_std = finalize_moments(
            stats['delta_sum'], stats['delta_sumsq'], stats['delta_count'])
        self.position_scale = finalize_position_scale(
            stats['pos_sqnorm_sum'], stats['pos_count'])

        print(f'  Node features - mean: {self.node_mean}, std: {self.node_std}')
        print(f'  Delta features - mean: {self.delta_mean}, std: {self.delta_std}')
        print(f"  Delta features - min: {stats['delta_min'].astype(np.float32)}, "
              f"max: {stats['delta_max'].astype(np.float32)}")
        print(f'  position_scale (RMS radius, centered_isotropic): {self.position_scale:.6f}')

        self._warn_on_degenerate_stats(stats)

    def _warn_on_degenerate_stats(self, stats: Dict) -> None:
        """Print warnings for suspicious normalization statistics (section 5.2)."""
        print('\n  === Normalization Sanity Checks ===')
        warnings = []
        node_raw_std = np.sqrt(np.maximum(
            stats['node_sumsq'] / max(stats['node_count'], 1)
            - (stats['node_sum'] / max(stats['node_count'], 1)) ** 2, 0.0))
        delta_raw_std = np.sqrt(np.maximum(
            stats['delta_sumsq'] / max(stats['delta_count'], 1)
            - (stats['delta_sum'] / max(stats['delta_count'], 1)) ** 2, 0.0))
        floor = 1e-8
        if np.any(node_raw_std <= floor):
            idx = np.where(node_raw_std <= floor)[0]
            warnings.append(
                f"  WARNING: node feature(s) at index {idx.tolist()} have raw std <= {floor:.0e} "
                f"(floored) -- these channels carry no signal; normalized values will be near-zero "
                f"or dominated by floating-point noise."
            )
        if np.any(delta_raw_std <= floor):
            idx = np.where(delta_raw_std <= floor)[0]
            warnings.append(
                f"  WARNING: target delta channel(s) at index {idx.tolist()} have raw std <= "
                f"{floor:.0e} (floored) -- see IMPLEMENTATION_PLAN.md section 5.2 for the "
                f"accepted-tradeoff decision on ex1.h5's dz/stress channels."
            )
        if np.any(self.node_std > 100):
            warnings.append(f"  WARNING: Very large node std (> 100): {self.node_std[self.node_std > 100]}")
        if np.any(self.delta_std > 100):
            warnings.append(f"  WARNING: Very large delta std (> 100): {self.delta_std[self.delta_std > 100]}")
        for w in warnings:
            print(w)
        if not warnings:
            print('  All normalization statistics look reasonable')

    def _check_no_unseen_node_types(self) -> None:
        """Section 5.5: val/test must not contain a node type absent from the
        train-fitted mapping — the one-hot encoding has no slot for it."""
        if not self.use_node_types or self.node_type_to_idx is None:
            return
        known = set(self.node_type_to_idx.keys())
        unseen = set()
        with h5py.File(self.h5_file, 'r') as f:
            for sid in self.sample_ids:
                node_types = f[f'data/{sid}/nodal_data'][-1, 0, :].astype(np.int32)
                unseen.update(int(t) for t in node_types if int(t) not in known)
        if unseen:
            raise ValueError(
                f"Dataset split contains node type(s) {sorted(unseen)} not present in the "
                f"train split's node_type_to_idx mapping {known}. One-hot node-type encoding "
                f"has no slot for an unseen type (IMPLEMENTATION_PLAN.md section 5.5)."
            )

    def _compute_node_type_info(self) -> None:
        """Compute the number of unique node types from the dataset.

        Uses the last feature row, guarded by num_features > 7 (checked in
        __init__), matching MeshGraphNets' rollout contract rather than its
        (inconsistent) training-time indexing (section 5.1).
        """
        print('Computing node type information...')
        with h5py.File(self.h5_file, 'r') as f:
            unique_types = set()
            for sid in self.sample_ids:
                nodal_data = f[f'data/{sid}/nodal_data'][:]
                node_types = nodal_data[-1, 0, :].astype(np.int32)
                unique_types.update(node_types)

            sorted_types = sorted(unique_types)
            self.node_type_to_idx = {t: i for i, t in enumerate(sorted_types)}
            self.num_node_types = len(unique_types)
            print(f'  Found {self.num_node_types} unique node types: {sorted_types}')
            print(f'  Node type mapping: {self.node_type_to_idx}')

    def __len__(self) -> int:
        if self.num_timesteps > 1:
            return len(self.sample_ids) * self._windows_per_sample()
        else:
            return len(self.sample_ids)

    def _windows_per_sample(self) -> int:
        """Window start times per trajectory: (T-1) one-step pairs under AR-OT."""
        return max(1, self.num_timesteps - self.rollout_window)

    def _get_h5_handle(self):
        if not hasattr(self, '_h5_handle') or self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_file, 'r', swmr=True)
        return self._h5_handle

    def _get_static_sample_data(self, sample_id: int, h5_handle, nodal_dset):
        """Cache sample topology and positional features inside each worker.

        `nodal_dset` is the open h5py dataset (not a loaded array); only
        timestep 0 is read on a cache miss.
        """
        cache = getattr(self, '_static_cache', None)
        if cache is None:
            cache = {}
            self._static_cache = cache

        cached = cache.get(sample_id)
        if cached is not None:
            return cached

        mesh_edge = h5_handle[f'data/{sample_id}/mesh_edge'][:]  # [2, M]
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        first_step = nodal_dset[:, 0, :]  # [features, N] — timestep 0 only

        node_types = first_step[-1, :].astype(np.int32) if self.use_node_types else None

        x_pos = None
        if self.num_pos_features > 0:
            ref_pos_0 = first_step[:3, :].T
            x_pos = compute_positional_features(ref_pos_0, edge_index, self.num_pos_features)

        cached = (edge_index, x_pos, node_types)
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
        if hasattr(self, '_h5_handle') and self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            self._h5_handle = None

    def _random_augmentation_matrix(self) -> np.ndarray:
        """Random Z-axis rotation + optional x/y reflection matrix [3, 3]."""
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        if np.random.random() < 0.5:
            R[0, :] *= -1
        if np.random.random() < 0.5:
            R[1, :] *= -1
        return R

    def __getitem__(self, idx: int) -> Data:
        if self.num_timesteps > 1:
            windows = self._windows_per_sample()
            sample_idx = idx // windows
            time_idx = idx % windows
        else:
            sample_idx = idx
            time_idx = 0

        sample_id = self.sample_ids[sample_idx]

        f = self._get_h5_handle()
        dset = f[f'data/{sample_id}/nodal_data']
        edge_index, x_pos, node_types = self._get_static_sample_data(sample_id, f, dset)
        part_ids = node_types

        future_states = None  # [N, W, output_var] raw future states (AR-RT only)
        if self.num_timesteps == 1:
            data_t = dset[:, 0, :].T  # [N, F]
            pos = data_t[:, :3].copy()  # [N, 3]
            x_phys = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)
            y_raw = data_t[:, 3:3 + self.output_dim]
            target_delta = y_raw.copy()
        else:
            # One slice covers the whole rollout window; W=1 under AR-OT makes
            # this the same two-step read as before.
            window = self.rollout_window
            block = dset[:, time_idx:time_idx + window + 1, :]  # [F, W+1, N]
            data_t = block[:, 0, :].T
            data_t1 = block[:, 1, :].T
            pos = data_t[:, :3].copy()
            x_phys = data_t[:, 3:3 + self.input_dim].copy()
            y_raw = data_t1[:, 3:3 + self.output_dim]
            target_delta = y_raw - x_phys
            if self.time_integration == AR_RT:
                # Absolute ground-truth states: the rollout re-derives each
                # step's target as (state_gt - state_pred), not from a stored delta.
                future_states = np.ascontiguousarray(
                    block[3:3 + self.output_dim, 1:, :].transpose(2, 1, 0)
                ).astype(np.float32)

        if self.num_pos_features > 0:
            x_raw = np.concatenate([x_phys, x_pos], axis=1)
        else:
            x_raw = x_phys

        # Geometric augmentation FIRST, in raw coordinates (training only);
        # centering/scaling below always operates on the (possibly augmented)
        # positions (section 5.5).
        if getattr(self, 'augment_geometry', False):
            R = self._random_augmentation_matrix()
            pos = pos @ R.T
            x_raw[:, :3] = x_raw[:, :3] @ R.T
            target_delta[:, :3] = target_delta[:, :3] @ R.T
            if future_states is not None:
                # The rollout integrates in this rotated frame, so its
                # ground-truth targets must live there too.
                future_states[:, :, :3] = future_states[:, :, :3] @ R.T

        if self.node_mean is None or self.node_std is None or self.position_scale is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: statistics are missing.")
        if self.delta_mean is None or self.delta_std is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: delta statistics are missing.")

        pos_normalized = normalize_positions(pos, self.position_scale)

        if self.use_node_types and node_types is not None and self.node_type_to_idx is None:
            raise RuntimeError("use_node_types=True but node_type_to_idx was not fitted.")
        x_norm = normalize_node_features(
            x_raw, self.node_mean, self.node_std,
            node_types=(node_types if self.use_node_types else None),
            node_type_to_idx=self.node_type_to_idx, num_node_types=self.num_node_types,
        )

        target_norm = (target_delta - self.delta_mean) / self.delta_std

        pos_t = torch.from_numpy(pos.astype(np.float32))
        pos_normalized_t = torch.from_numpy(pos_normalized.astype(np.float32))
        x = torch.from_numpy(x_norm.astype(np.float32))
        y = torch.from_numpy(target_norm.astype(np.float32))
        edge_index_t = torch.from_numpy(edge_index).long()

        part_ids_tensor = torch.from_numpy(part_ids).long() if part_ids is not None else None

        graph_data = Data(
            x=x,
            y=y,
            pos=pos_t,
            pos_normalized=pos_normalized_t,
            edge_index=edge_index_t,
            sample_id=sample_id,
            time_idx=time_idx if self.num_timesteps > 1 else None,
            part_ids=part_ids_tensor,
        )

        if future_states is not None:
            # AR-RT payload. `y` (the step-0 normalized delta) is still set, so
            # a one-step rollout is numerically identical to the AR-OT path.
            graph_data.y_seq = torch.from_numpy(future_states)          # [N, W, output_var]
            graph_data.state0 = torch.from_numpy(
                np.ascontiguousarray(x_raw[:, :self.input_dim]).astype(np.float32)
            )                                                           # [N, input_var] physical units

        return graph_data

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 0):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        train_ids, val_ids, test_ids = self._resolve_split_ids(train_ratio, val_ratio, test_ratio, seed)
        if len(train_ids) == 0:
            raise ValueError(
                f"Train split is empty ({len(self.sample_ids)} total samples, "
                f"train_ratio={train_ratio}). Datasets with too few samples for an "
                f"80/10/10 split (e.g. one-sample files) must be used for inference only."
            )

        train_dataset = self._create_subset(train_ids, is_training=True)
        val_dataset = self._create_subset(val_ids, is_training=False)
        test_dataset = self._create_subset(test_ids, is_training=False)

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        print("Fitting preprocessing on train split only...")
        train_dataset.prepare_preprocessing()
        val_dataset.inherit_preprocessing_from(train_dataset)
        test_dataset.inherit_preprocessing_from(train_dataset)
        val_dataset._check_no_unseen_node_types()
        test_dataset._check_no_unseen_node_types()

        return train_dataset, val_dataset, test_dataset
