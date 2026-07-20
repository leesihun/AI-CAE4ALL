"""Isolated faithful Point-DeepONet paper benchmark.

This module deliberately does not import or modify the suite's shared loader or
Point-DeepONet implementation.  It reproduces the released executable topology,
fixed output clipping, and the release's all-selected-case scaling before the
800/200 split.  That last behavior is validation leakage, but retaining it is
necessary for a direct comparison to the paper's reported result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate


COMPONENTS = ("ux", "uy", "uz", "vm")
DIRECTIONS = ("ver", "hor", "dia")
PAPER_TARGET_AVERAGE_R2 = 0.897
PAPER_PARAMETER_COUNT = 251_936
PAPER_TRAIN_SHA256 = "b1153fad047e45bfe5bbdda15cb93ecb6e30983de4bdd7824a9111a044c88d33"
PAPER_VALID_SHA256 = "282cab5b4e3ec9dd45f8f59399a9bdda06bf99165c848cbf443fc66bd419b05a"

CLIPPING_RANGES: Mapping[str, tuple[tuple[float, float], ...]] = {
    "ver": ((-0.068, 0.473), (-0.093, 0.073), (-0.003, 0.824), (0.0, 232.19)),
    "hor": ((-0.421, 0.008), (-0.024, 0.029), (-0.388, 0.109), (0.0, 227.78)),
    "dia": ((-0.079, 0.016), (-0.057, 0.056), (-0.006, 0.214), (0.0, 172.19)),
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def direction_from_case(case_name: str) -> str:
    direction = case_name.split("_", 1)[0]
    if direction not in CLIPPING_RANGES:
        raise ValueError(f"unknown load direction in case {case_name!r}")
    return direction


def clip_targets(targets: np.ndarray, direction: str) -> np.ndarray:
    """Apply the released direction-specific physical output limits."""
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ValueError(f"targets must have shape [points, 4], got {targets.shape}")
    result = np.array(targets, dtype=np.float32, copy=True)
    for component, (low, high) in enumerate(CLIPPING_RANGES[direction]):
        np.clip(result[:, component], low, high, out=result[:, component])
    return result


@dataclass(frozen=True)
class FeatureMinMaxScaler:
    """Small serializable equivalent of MinMaxScaler(feature_range=(-1, 1))."""

    data_min: np.ndarray
    data_max: np.ndarray

    @classmethod
    def from_arrays(cls, arrays: Iterable[np.ndarray]) -> "FeatureMinMaxScaler":
        data_min: np.ndarray | None = None
        data_max: np.ndarray | None = None
        for array in arrays:
            values = np.asarray(array, dtype=np.float64)
            if values.ndim == 1:
                values = values[None, :]
            if values.ndim != 2 or values.shape[0] == 0:
                raise ValueError(f"scaler input must be a nonempty 2D array, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError("scaler input contains non-finite values")
            current_min = values.min(axis=0)
            current_max = values.max(axis=0)
            data_min = current_min if data_min is None else np.minimum(data_min, current_min)
            data_max = current_max if data_max is None else np.maximum(data_max, current_max)
        if data_min is None or data_max is None:
            raise ValueError("cannot fit a scaler without arrays")
        return cls(data_min=data_min, data_max=data_max)

    @property
    def n_features(self) -> int:
        return int(self.data_min.size)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {values64.shape[-1]}")
        # This matches sklearn's zero-range handling: a constant feature maps to -1.
        scale_range = self.data_max - self.data_min
        safe_range = np.where(scale_range == 0.0, 1.0, scale_range)
        result = 2.0 * (values64 - self.data_min) / safe_range - 1.0
        return result.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {values64.shape[-1]}")
        scale_range = self.data_max - self.data_min
        safe_range = np.where(scale_range == 0.0, 1.0, scale_range)
        result = (values64 + 1.0) * safe_range / 2.0 + self.data_min
        return result.astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"data_min": self.data_min.tolist(), "data_max": self.data_max.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Sequence[float]]) -> "FeatureMinMaxScaler":
        return cls(
            data_min=np.asarray(value["data_min"], dtype=np.float64),
            data_max=np.asarray(value["data_max"], dtype=np.float64),
        )


@dataclass(frozen=True)
class Preprocessing:
    branch_mlc: FeatureMinMaxScaler
    point_xyz: FeatureMinMaxScaler
    trunk_xyzd: FeatureMinMaxScaler
    output: FeatureMinMaxScaler
    fit_split: str = "train-only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_split": self.fit_split,
            "fixed_clipping_ranges": {k: [list(pair) for pair in v] for k, v in CLIPPING_RANGES.items()},
            "branch_mlc": self.branch_mlc.to_dict(),
            "point_xyz": self.point_xyz.to_dict(),
            "trunk_xyzd": self.trunk_xyzd.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Preprocessing":
        fit_split = str(value.get("fit_split"))
        if fit_split not in {"train-only", "released-all-selected-before-split"}:
            raise ValueError(f"unsupported checkpoint preprocessing scope {fit_split!r}")
        return cls(
            branch_mlc=FeatureMinMaxScaler.from_dict(value["branch_mlc"]),
            point_xyz=FeatureMinMaxScaler.from_dict(value["point_xyz"]),
            trunk_xyzd=FeatureMinMaxScaler.from_dict(value["trunk_xyzd"]),
            output=FeatureMinMaxScaler.from_dict(value["output"]),
            fit_split=fit_split,
        )


def read_manifest(prepared_dir: Path, split: str) -> list[str]:
    if split not in {"train", "valid"}:
        raise ValueError(f"split must be train or valid, got {split!r}")
    path = prepared_dir / "manifests" / f"{split}.txt"
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate case names in {path}")
    return names


def case_path(prepared_dir: Path, split: str, case_name: str) -> Path:
    return prepared_dir / "cases" / split / f"{case_name}.npz"


def _load_raw_case(path: Path, expected_name: str, expected_split: str, expected_points: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"xyzdmlc", "targets", "sample_indices", "case_name", "split"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        xyzdmlc = np.asarray(archive["xyzdmlc"])
        targets = np.asarray(archive["targets"])
        stored_name = str(archive["case_name"].item())
        stored_split = str(archive["split"].item())
        sample_indices = np.asarray(archive["sample_indices"])
    if stored_name != expected_name or stored_split != expected_split:
        raise ValueError(
            f"case metadata mismatch in {path}: {(stored_name, stored_split)} != {(expected_name, expected_split)}"
        )
    if xyzdmlc.shape != (expected_points, 9) or targets.shape != (expected_points, 4):
        raise ValueError(f"unexpected arrays in {path}: {xyzdmlc.shape}, {targets.shape}")
    if sample_indices.shape != (expected_points,):
        raise ValueError(f"unexpected sample_indices shape in {path}: {sample_indices.shape}")
    if xyzdmlc.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError(f"paper arrays must be float32 in {path}")
    if not np.isfinite(xyzdmlc).all() or not np.isfinite(targets).all():
        raise ValueError(f"non-finite values in {path}")
    # d (column 3) is the per-node signed-distance value.  Only mass, load,
    # and the three direction components are global case constants.
    constants = xyzdmlc[:, 4:9]
    if not np.array_equal(constants, np.broadcast_to(constants[0], constants.shape)):
        raise ValueError(f"m/l/cx/cy/cz must be constant within case {expected_name}")
    return xyzdmlc, targets


def fit_preprocessing(
    prepared_dir: Path,
    train_names: Sequence[str],
    n_points: int,
    valid_names: Sequence[str] = (),
) -> Preprocessing:
    """Fit released ranges on all selected cases, or train-only for smoke tests."""
    branch_arrays: list[np.ndarray] = []
    xyz_arrays: list[np.ndarray] = []
    trunk_arrays: list[np.ndarray] = []
    output_arrays: list[np.ndarray] = []
    for split, names in (("train", train_names), ("valid", valid_names)):
        for name in names:
            xyzdmlc, targets = _load_raw_case(case_path(prepared_dir, split, name), name, split, n_points)
            # The five m/l/cx/cy/cz values are checked above.  The released branch
            # consumes them; per-node signed distance d remains the fourth trunk feature.
            branch_arrays.append(xyzdmlc[0, 4:9][None, :])
            xyz_arrays.append(xyzdmlc[:, 0:3])
            trunk_arrays.append(xyzdmlc[:, 0:4])
            output_arrays.append(clip_targets(targets, direction_from_case(name)))
    return Preprocessing(
        branch_mlc=FeatureMinMaxScaler.from_arrays(branch_arrays),
        point_xyz=FeatureMinMaxScaler.from_arrays(xyz_arrays),
        trunk_xyzd=FeatureMinMaxScaler.from_arrays(trunk_arrays),
        output=FeatureMinMaxScaler.from_arrays(output_arrays),
        fit_split=("released-all-selected-before-split" if valid_names else "train-only"),
    )


class ReleasedCaseDataset(Dataset[dict[str, Any]]):
    """Lazy benchmark-only adapter for prepared per-case NPZ files."""

    def __init__(
        self,
        prepared_dir: Path,
        split: str,
        case_names: Sequence[str],
        preprocessing: Preprocessing,
        n_points: int,
        cache_in_memory: bool = False,
    ) -> None:
        self.prepared_dir = prepared_dir
        self.split = split
        self.case_names = list(case_names)
        self.preprocessing = preprocessing
        self.n_points = n_points
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, dict[str, Any]] = {}
        if not self.case_names:
            raise ValueError(f"empty {split} dataset")

    def __len__(self) -> int:
        return len(self.case_names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        name = self.case_names[index]
        xyzdmlc, raw_targets = _load_raw_case(
            case_path(self.prepared_dir, self.split, name), name, self.split, self.n_points
        )
        targets = clip_targets(raw_targets, direction_from_case(name))
        condition_mlc = xyzdmlc[0, 4:9].copy()
        item = {
            "case_name": name,
            "direction": direction_from_case(name),
            "condition_mlc_raw": torch.from_numpy(condition_mlc),
            "branch_mlc": torch.from_numpy(self.preprocessing.branch_mlc.transform(xyzdmlc[0, 4:9])),
            "point_xyz": torch.from_numpy(self.preprocessing.point_xyz.transform(xyzdmlc[:, 0:3])),
            "trunk_xyzd": torch.from_numpy(self.preprocessing.trunk_xyzd.transform(xyzdmlc[:, 0:4])),
            "target": torch.from_numpy(self.preprocessing.output.transform(targets)),
        }
        if self.cache_in_memory:
            self._cache[index] = item
        return item


class ReleasedBatchSampler:
    """Exact NumPy index sampler used by DeepXDE's released BatchSampler."""

    def __init__(self, num_samples: int, shuffle: bool = True) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.indices = np.arange(num_samples)
        self.epochs_completed = 0
        self.index_in_epoch = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def get_next(self, batch_size: int) -> np.ndarray:
        if batch_size > self.num_samples:
            raise ValueError(f"batch_size={batch_size} is larger than num_samples={self.num_samples}")
        start = self.index_in_epoch
        if start + batch_size <= self.num_samples:
            self.index_in_epoch += batch_size
            return self.indices[start:self.index_in_epoch]
        self.epochs_completed += 1
        rest_count = self.num_samples - start
        rest = np.copy(self.indices[start:self.num_samples])
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.index_in_epoch = batch_size - rest_count
        new = self.indices[:self.index_in_epoch]
        return np.hstack((rest, new))


class SineDenseLayer(nn.Module):
    """Released SIREN layer, including the authors' exact weight initialization."""

    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self) -> None:
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
        else:
            bound = math.sqrt(6 / self.in_features) / self.w0
            nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(values))


class ReleasedPointDeepONet(nn.Module):
    """The executable topology in the authors' released 5.Point_DeepONet/main.py."""

    def __init__(
        self,
        branch_hidden_dim: int = 100,
        trunk_hidden_dim: int = 100,
        trunk_encoding_hidden_dim: int = 100,
        fc_hidden_dim: int = 100,
        num_outputs: int = 4,
    ) -> None:
        super().__init__()
        activation = nn.SiLU()
        self.trunk_encoding = nn.Sequential(
            SineDenseLayer(3, trunk_encoding_hidden_dim, w0=10.0, is_first=True),
            SineDenseLayer(trunk_encoding_hidden_dim, trunk_encoding_hidden_dim * 2, w0=10.0),
            SineDenseLayer(trunk_encoding_hidden_dim * 2, trunk_encoding_hidden_dim, w0=10.0),
        )
        self.branch_net_global = nn.Sequential(
            nn.Linear(5, branch_hidden_dim), activation,
            nn.Linear(branch_hidden_dim, branch_hidden_dim * 2), activation,
            nn.Linear(branch_hidden_dim * 2, branch_hidden_dim), activation,
        )
        self.pointnet_branch = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.BatchNorm1d(32), activation,
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), activation,
            nn.Conv1d(64, 100, 1), nn.BatchNorm1d(100), activation,
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_encoding_hidden_dim + 1, trunk_hidden_dim), activation,
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2), activation,
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim * num_outputs), activation,
        )
        self.combined_branch = nn.Sequential(
            nn.Linear(branch_hidden_dim, fc_hidden_dim), activation,
            nn.Linear(fc_hidden_dim, fc_hidden_dim * 2), activation,
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim), activation,
        )
        self.b = nn.Parameter(torch.zeros(num_outputs))
        self.num_outputs = num_outputs

    def forward(
        self,
        branch_mlc: torch.Tensor,
        point_xyz: torch.Tensor,
        trunk_xyzd: torch.Tensor,
    ) -> torch.Tensor:
        branch_output = self.branch_net_global(branch_mlc)
        pointnet_output = self.pointnet_branch(point_xyz.transpose(2, 1)).max(dim=2)[0]
        combined_output = branch_output + pointnet_output

        trunk_xyz = trunk_xyzd[:, :, :3]
        trunk_other = trunk_xyzd[:, :, 3:]
        batch_size, num_points, _ = trunk_xyz.shape
        trunk_xyz_encoded = self.trunk_encoding(trunk_xyz.reshape(-1, 3)).reshape(batch_size, num_points, -1)
        trunk_combined = torch.cat([trunk_xyz_encoded, trunk_other], dim=2)

        mix = combined_output.unsqueeze(1) * trunk_xyz_encoded
        x_trunk = self.trunk_net(trunk_combined.reshape(batch_size * num_points, -1))
        x_trunk = x_trunk.reshape(batch_size, num_points, mix.shape[-1], self.num_outputs)
        combined_output = self.combined_branch(mix.mean(dim=1))
        output = torch.einsum("bh,bnhc->bnc", combined_output, x_trunk) + self.b
        return torch.tanh(output)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("prepared_dir", "output_dir"):
        value = Path(config[key])
        config[key] = str((config_path.parent / value).resolve()) if not value.is_absolute() else str(value)
    config["config_path"] = str(config_path)
    return config


def _require_equal(config: Mapping[str, Any], dotted_key: str, expected: Any) -> None:
    value: Any = config
    for part in dotted_key.split("."):
        value = value[part]
    if value != expected:
        raise ValueError(f"paper guard: {dotted_key} must be {expected!r}, got {value!r}")


def validate_config(config: Mapping[str, Any]) -> None:
    profile = config.get("profile")
    if profile not in {"paper_n1000_p5000", "cpu_smoke"}:
        raise ValueError(f"unsupported benchmark profile {profile!r}")
    common = {
        "model.branch_components": "mlc",
        "model.pointnet_components": "xyz",
        "model.trunk_components": "xyzd",
        "model.output_components": "xyzs",
        "model.branch_hidden_dim": 100,
        "model.trunk_hidden_dim": 100,
        "model.trunk_encoding_hidden_dim": 100,
        "model.fc_hidden_dim": 100,
        "model.pointnet_channels": [32, 64, 100],
        "model.siren_w0": 10.0,
        "train.loss": "mse",
        "train.optimizer": "AdamW",
        "train.learning_rate": 0.001,
        "train.weight_decay": 0.00001,
        "train.learning_rate_decay.type": "inverse_time",
        "train.learning_rate_decay.decay_steps": 1,
        "train.learning_rate_decay.decay_rate": 0.0001,
        "preprocessing.output_clipping": "released_fixed_by_direction",
    }
    for key, expected in common.items():
        _require_equal(config, key, expected)
    if profile == "paper_n1000_p5000":
        exact = {
            "dataset.n_samples": 1000,
            "dataset.n_train": 800,
            "dataset.n_valid": 200,
            "dataset.n_points": 5000,
            "dataset.train_manifest_sha256": PAPER_TRAIN_SHA256,
            "dataset.valid_manifest_sha256": PAPER_VALID_SHA256,
            "train.iterations": 40000,
            "train.batch_size": 16,
            "train.seed": 2024,
            "train.sampler": "deepxde_numpy_batch_sampler",
            "preprocessing.learned_scaler_fit_split": "released_all_selected_before_split",
            "evaluation.paper_target_average_r2": PAPER_TARGET_AVERAGE_R2,
            "evaluation.aggregate": "mean_of_12_direction_component_pooled_r2",
        }
        for key, expected in exact.items():
            _require_equal(config, key, expected)
        if config["dataset"].get("train_cases") or config["dataset"].get("valid_cases"):
            raise ValueError("paper guard: case overrides are forbidden")
    else:
        _require_equal(config, "preprocessing.learned_scaler_fit_split", "train")
        if config["train"]["device"] != "cpu":
            raise ValueError("smoke guard: device must be cpu")
        if int(config["train"]["iterations"]) > 5 or int(config["train"]["batch_size"]) > 2:
            raise ValueError("smoke guard: at most 5 iterations and batch size 2")
        if not config["dataset"].get("train_cases") or not config["dataset"].get("valid_cases"):
            raise ValueError("smoke guard: explicit train and valid cases are required")

    model = ReleasedPointDeepONet()
    if count_parameters(model) != PAPER_PARAMETER_COUNT:
        raise AssertionError(
            f"released topology changed: expected {PAPER_PARAMETER_COUNT:,} parameters, got {count_parameters(model):,}"
        )


def validate_prepared_data(config: Mapping[str, Any], deep: bool = False) -> dict[str, Any]:
    validate_config(config)
    prepared_dir = Path(config["prepared_dir"])
    train_names = read_manifest(prepared_dir, "train")
    valid_names = read_manifest(prepared_dir, "valid")
    dataset_config = config["dataset"]
    if config["profile"] == "paper_n1000_p5000":
        if len(train_names) != 800 or len(valid_names) != 200:
            raise ValueError(f"paper guard: expected 800/200 manifests, got {len(train_names)}/{len(valid_names)}")
        train_hash = _sha256(prepared_dir / "manifests" / "train.txt")
        valid_hash = _sha256(prepared_dir / "manifests" / "valid.txt")
        if train_hash != PAPER_TRAIN_SHA256 or valid_hash != PAPER_VALID_SHA256:
            raise ValueError(f"paper guard: manifest digest mismatch: {train_hash}, {valid_hash}")
        selected_train = train_names
        selected_valid = valid_names
    else:
        selected_train = list(dataset_config["train_cases"])
        selected_valid = list(dataset_config["valid_cases"])

    missing = [
        str(case_path(prepared_dir, split, name))
        for split, names in (("train", selected_train), ("valid", selected_valid))
        for name in names
        if not case_path(prepared_dir, split, name).is_file()
    ]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} prepared cases are missing; first paths:\n{preview}")
    if deep:
        n_points = int(dataset_config["n_points"])
        for split, names in (("train", selected_train), ("valid", selected_valid)):
            for name in names:
                _load_raw_case(case_path(prepared_dir, split, name), name, split, n_points)
    return {
        "profile": config["profile"],
        "train_cases": len(selected_train),
        "valid_cases": len(selected_valid),
        "deep_array_validation": deep,
        "parameter_count": PAPER_PARAMETER_COUNT,
    }


def selected_case_names(config: Mapping[str, Any], split: str) -> list[str]:
    configured = config["dataset"].get(f"{split}_cases")
    return list(configured) if configured else read_manifest(Path(config["prepared_dir"]), split)


def make_model(config: Mapping[str, Any]) -> ReleasedPointDeepONet:
    model_config = config["model"]
    model = ReleasedPointDeepONet(
        branch_hidden_dim=int(model_config["branch_hidden_dim"]),
        trunk_hidden_dim=int(model_config["trunk_hidden_dim"]),
        trunk_encoding_hidden_dim=int(model_config["trunk_encoding_hidden_dim"]),
        fc_hidden_dim=int(model_config["fc_hidden_dim"]),
    )
    if count_parameters(model) != PAPER_PARAMETER_COUNT:
        raise ValueError("paper topology parameter-count guard failed")
    return model


def make_dataset(
    config: Mapping[str, Any],
    split: str,
    preprocessing: Preprocessing,
) -> ReleasedCaseDataset:
    return ReleasedCaseDataset(
        prepared_dir=Path(config["prepared_dir"]),
        split=split,
        case_names=selected_case_names(config, split),
        preprocessing=preprocessing,
        n_points=int(config["dataset"]["n_points"]),
        cache_in_memory=bool(config["dataset"].get("cache_in_memory", False)),
    )


def make_loader(
    config: Mapping[str, Any],
    split: str,
    preprocessing: Preprocessing,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    dataset = make_dataset(config, split, preprocessing)
    batch_size = int(config["train"]["batch_size"] if split == "train" else config["evaluation"]["batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=bool(config["train"].get("pin_memory", False)),
        drop_last=False,
    )


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["train"]["device"])
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("paper config requests CUDA, but CUDA is unavailable")
    return torch.device(requested)


def _forward_batch(model: ReleasedPointDeepONet, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    return model(
        batch["branch_mlc"].to(device),
        batch["point_xyz"].to(device),
        batch["trunk_xyzd"].to(device),
    )


class _R2Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sse = 0.0
        self.sae = 0.0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        true64 = np.asarray(truth, dtype=np.float64).reshape(-1)
        pred64 = np.asarray(prediction, dtype=np.float64).reshape(-1)
        residual = true64 - pred64
        self.n += true64.size
        self.sum_y += float(true64.sum())
        self.sum_y2 += float(np.square(true64).sum())
        self.sse += float(np.square(residual).sum())
        self.sae += float(np.abs(residual).sum())

    def result(self) -> dict[str, float | int | None]:
        denominator = self.sum_y2 - self.sum_y * self.sum_y / self.n if self.n else 0.0
        r2 = None if denominator <= 0.0 else 1.0 - self.sse / denominator
        return {
            "n_values": self.n,
            "mae": self.sae / self.n if self.n else None,
            "rmse": math.sqrt(self.sse / self.n) if self.n else None,
            "r2": r2,
        }


@torch.no_grad()
def evaluate_model(
    model: ReleasedPointDeepONet,
    loader: DataLoader[dict[str, Any]],
    preprocessing: Preprocessing,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accumulators = {
        direction: {component: _R2Accumulator() for component in COMPONENTS}
        for direction in DIRECTIONS
    }
    case_count = 0
    for batch in loader:
        prediction_scaled = _forward_batch(model, batch, device).cpu().numpy()
        target_scaled = batch["target"].numpy()
        prediction = preprocessing.output.inverse_transform(prediction_scaled)
        target = preprocessing.output.inverse_transform(target_scaled)
        for batch_index, direction in enumerate(batch["direction"]):
            case_count += 1
            for component_index, component in enumerate(COMPONENTS):
                accumulators[direction][component].update(
                    target[batch_index, :, component_index], prediction[batch_index, :, component_index]
                )

    by_direction_component: dict[str, dict[str, Any]] = {}
    r2_values: list[float] = []
    for direction in DIRECTIONS:
        by_direction_component[direction] = {}
        for component in COMPONENTS:
            metric = accumulators[direction][component].result()
            by_direction_component[direction][component] = metric
            if metric["r2"] is not None and math.isfinite(float(metric["r2"])):
                r2_values.append(float(metric["r2"]))
    if not r2_values:
        raise ValueError("evaluation produced no finite R2 values")
    average_r2 = float(np.mean(r2_values))
    paper_comparable = len(r2_values) == len(DIRECTIONS) * len(COMPONENTS)
    return {
        "case_count": case_count,
        "r2_terms_available": len(r2_values),
        "aggregate_definition": (
            "mean_of_12_direction_component_pooled_r2"
            if paper_comparable
            else "mean_of_available_direction_component_pooled_r2_smoke_only"
        ),
        "by_direction_component": by_direction_component,
        "average_r2": average_r2,
        "paper_target_average_r2": PAPER_TARGET_AVERAGE_R2,
        "paper_comparable": paper_comparable,
        "difference_from_paper": average_r2 - PAPER_TARGET_AVERAGE_R2 if paper_comparable else None,
        "meets_or_exceeds_paper": average_r2 >= PAPER_TARGET_AVERAGE_R2 if paper_comparable else None,
    }


def _checkpoint_payload(
    model: ReleasedPointDeepONet,
    optimizer: torch.optim.Optimizer,
    preprocessing: Preprocessing,
    config: Mapping[str, Any],
    iteration: int,
    best_average_r2: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": config["profile"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "preprocessing": preprocessing.to_dict(),
        "iteration": iteration,
        "best_average_r2": best_average_r2,
        "parameter_count": count_parameters(model),
    }


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_prepared_data(config, deep=False)
    seed = int(config["train"]["seed"])
    set_random_seed(seed)
    device = _device_from_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = selected_case_names(config, "train")
    valid_names = selected_case_names(config, "valid")
    preprocessing = fit_preprocessing(
        Path(config["prepared_dir"]),
        train_names,
        int(config["dataset"]["n_points"]),
        valid_names if config["profile"] == "paper_n1000_p5000" else (),
    )
    _json_dump(output_dir / "preprocessing.json", preprocessing.to_dict())
    _json_dump(output_dir / "resolved_config.json", dict(config))

    train_dataset = make_dataset(config, "train", preprocessing)
    batch_sampler = ReleasedBatchSampler(len(train_dataset), shuffle=True)
    valid_loader = make_loader(config, "valid", preprocessing, shuffle=False)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    decay = config["train"]["learning_rate_decay"]
    decay_steps = float(decay["decay_steps"])
    decay_rate = float(decay["decay_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (1.0 + decay_rate * (step / decay_steps))
    )
    loss_function = nn.MSELoss()
    iterations = int(config["train"]["iterations"])
    validate_every = int(config["evaluation"]["validate_every"])
    history: list[dict[str, Any]] = []
    best_average_r2 = -math.inf
    started = time.perf_counter()

    for iteration in range(1, iterations + 1):
        indices = batch_sampler.get_next(int(config["train"]["batch_size"]))
        batch = default_collate([train_dataset[int(index)] for index in indices])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward_batch(model, batch, device)
        target = batch["target"].to(device)
        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        record: dict[str, Any] = {"iteration": iteration, "train_mse": float(loss.detach().cpu())}
        should_validate = iteration == iterations or iteration % validate_every == 0
        if should_validate:
            metrics = evaluate_model(model, valid_loader, preprocessing, device)
            if config["profile"] == "paper_n1000_p5000" and not metrics["paper_comparable"]:
                raise ValueError("paper evaluation requires all 12 direction/component R2 terms")
            record["validation"] = metrics
            average_r2 = float(metrics["average_r2"])
            if average_r2 > best_average_r2:
                best_average_r2 = average_r2
                torch.save(
                    _checkpoint_payload(model, optimizer, preprocessing, config, iteration, best_average_r2),
                    output_dir / "checkpoint_best.pt",
                )
        history.append(record)

    elapsed = time.perf_counter() - started
    torch.save(
        _checkpoint_payload(model, optimizer, preprocessing, config, iterations, best_average_r2),
        output_dir / "checkpoint_last.pt",
    )
    result = {
        "profile": config["profile"],
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "best_average_r2": best_average_r2,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "paper_target_average_r2": PAPER_TARGET_AVERAGE_R2,
        "parameter_count": count_parameters(model),
        "training_sampler": "deepxde_numpy_batch_sampler",
        "preprocessing_fit_split": preprocessing.fit_split,
        "history": history,
    }
    _json_dump(output_dir / "training_metrics.json", result)
    return result


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    validate_prepared_data(config, deep=False)
    device = _device_from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("parameter_count") != PAPER_PARAMETER_COUNT:
        raise ValueError("checkpoint schema or paper parameter-count guard failed")
    if checkpoint.get("profile") != config["profile"]:
        raise ValueError("checkpoint/config profile mismatch")
    preprocessing = Preprocessing.from_dict(checkpoint["preprocessing"])
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader = make_loader(config, "valid", preprocessing, shuffle=False)
    metrics = evaluate_model(model, loader, preprocessing, device)
    if config["profile"] == "paper_n1000_p5000" and not metrics["paper_comparable"]:
        raise ValueError("paper evaluation requires all 12 direction/component R2 terms")
    metrics["checkpoint"] = str(checkpoint_path.resolve())
    _json_dump(Path(config["output_dir"]) / "evaluation_metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train", "evaluate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "validate":
            subparser.add_argument("--deep", action="store_true")
        if command == "evaluate":
            subparser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        result = validate_prepared_data(config, deep=args.deep)
    elif args.command == "train":
        result = train(config)
    else:
        result = evaluate_checkpoint(config, args.checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
