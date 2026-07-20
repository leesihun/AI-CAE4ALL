#!/usr/bin/env python3
"""Strict post-inference evaluation for the suite Plasticity benchmark.

This module reads only de-normalized rollout HDF5 files.  It deliberately lives
outside every training and inference path so metric collection cannot change
model behavior or throughput.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


MODELS = (
    "meshgraphnets",
    "hi_meshgraphnets",
    "deeponet",
    "point_deeponet",
    "fno",
    "gino",
    "transolver",
)
COMPONENTS = ("u_x_mm", "u_y_mm", "u_z_mm")
ROLLOUT_RE = re.compile(r"^rollout_sample(?P<sample>\d+)_steps(?P<steps>\d+)\.h5$")
SCHEMA_VERSION = "plasticity_rollout_evaluation_v1"
PINNED_TRUTH_SHA256 = "5970cdcd362e94f5a54e0f7d18893b11c51f5e1ab345712bddfbbe8d130ad8be"
PRODUCTION_TEST_IDS = (
    1, 10, 15, 34, 41, 42, 49, 57, 63, 65, 69, 76, 102, 113, 117, 137,
    141, 167, 177, 182, 195, 212, 272, 275, 286, 287, 290, 296, 300, 303,
    313, 314, 317, 333, 366, 372, 410, 432, 437, 451, 490, 491, 500, 502,
    503, 518, 526, 528, 529, 534, 535, 543, 547, 548, 553, 555, 583, 586,
    609, 611, 617, 621, 622, 628, 634, 637, 641, 648, 651, 669, 672, 675,
    687, 689, 702, 706, 720, 735, 736, 749, 752, 779, 791, 809, 821, 828,
    832, 837, 908, 910, 915, 927, 932, 936, 943, 955, 961, 969, 973, 984,
)


def _suite_seed42_test_ids() -> tuple[int, ...]:
    """Return the pinned tuple after independently reproducing the split."""
    sample_ids = np.arange(987, dtype=np.int64)
    np.random.default_rng(42).shuffle(sample_ids)
    generated = tuple(sorted(int(value) for value in sample_ids[789 + 98 :]))
    if generated != PRODUCTION_TEST_IDS:
        raise RuntimeError("Pinned Plasticity seed-42 test IDs no longer match NumPy split")
    return PRODUCTION_TEST_IDS


@dataclass(frozen=True)
class EvaluationContract:
    expected_cases: int = 100
    num_features: int = 8
    num_timesteps: int = 20
    num_nodes: int = 3131
    num_edges: int = 6130
    split_seed: int = 42
    expected_truth_ids: tuple[int, ...] = field(default_factory=_suite_seed42_test_ids)
    geometry_tolerance: float = 1.0e-5
    # All writers copy the four seed-state channels directly from the truth
    # float32 array before rollout, so 1e-6 permits no meaningful case swap.
    seed_state_tolerance: float = 1.0e-6
    pinned_ground_truth_sha256: str | None = PINNED_TRUTH_SHA256

    @property
    def rollout_steps(self) -> int:
        return self.num_timesteps - 1

    @property
    def nodal_shape(self) -> tuple[int, int, int]:
        return self.num_features, self.num_timesteps, self.num_nodes


PRODUCTION_CONTRACT = EvaluationContract()


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def _normalized_path_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _provenance_matches_model(value: str, model: str) -> bool:
    normalized = _normalized_path_token(value)
    token = _normalized_path_token(model)
    if model in {"meshgraphnets", "hi_meshgraphnets"}:
        identifies_hi_mgn = "himeshgraphnets" in normalized
        if identifies_hi_mgn != (model == "hi_meshgraphnets"):
            return False
    if token not in normalized:
        return False
    # Canonical DeepONet must not accept Point-DeepONet artifacts merely
    # because its shorter token is a substring.
    if model == "deeponet" and "pointdeeponet" in normalized:
        return False
    remaining = normalized.replace(token, "")
    for other_model in MODELS:
        if other_model == model:
            continue
        other_token = _normalized_path_token(other_model)
        if other_token in remaining:
            return False
    return True


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metric CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _discover_rollouts(
    prediction_dir: Path, contract: EvaluationContract
) -> dict[int, Path]:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    h5_files = sorted(prediction_dir.glob("*.h5"))
    malformed = [path.name for path in h5_files if ROLLOUT_RE.fullmatch(path.name) is None]
    if malformed:
        raise ValueError(f"Unexpected HDF5 files in prediction directory: {malformed[:10]}")

    rollouts: dict[int, Path] = {}
    for path in h5_files:
        match = ROLLOUT_RE.fullmatch(path.name)
        assert match is not None
        sample_id = int(match.group("sample"))
        steps = int(match.group("steps"))
        if steps != contract.rollout_steps:
            raise ValueError(
                f"{path.name}: steps={steps}, expected {contract.rollout_steps}"
            )
        if sample_id in rollouts:
            raise ValueError(f"Duplicate rollout for sample {sample_id}")
        rollouts[sample_id] = path

    expected = set(contract.expected_truth_ids)
    actual = set(rollouts)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if len(rollouts) != contract.expected_cases or missing or extra:
        raise ValueError(
            "Rollout ID contract failed: "
            f"files={len(rollouts)} expected={contract.expected_cases}; "
            f"missing={missing[:10]}; extra={extra[:10]}"
        )
    return rollouts


def _validate_truth(
    handle: h5py.File, contract: EvaluationContract
) -> tuple[list[int], np.ndarray]:
    if "data" not in handle:
        raise ValueError("Ground truth is missing /data")
    truth_ids = sorted(int(key) for key in handle["data"].keys())
    expected_ids = list(contract.expected_truth_ids)
    if len(truth_ids) != contract.expected_cases or truth_ids != expected_ids:
        raise ValueError(
            "Ground-truth IDs are not the exact suite held-out split: "
            f"count={len(truth_ids)} expected={contract.expected_cases}"
        )
    if int(handle.attrs.get("num_samples", -1)) != contract.expected_cases:
        raise ValueError("Ground-truth num_samples attribute is incorrect")
    if int(handle.attrs.get("suite_split_seed", -1)) != contract.split_seed:
        raise ValueError("Ground-truth suite_split_seed is incorrect")
    if _decode(handle.attrs.get("suite_split_role", "")) != "test":
        raise ValueError("Ground-truth suite_split_role must be 'test'")
    if "splits/test" not in handle:
        raise ValueError("Ground truth is missing /splits/test")
    stored_test = sorted(int(value) for value in handle["splits/test"][:])
    if stored_test != expected_ids:
        raise ValueError("Ground-truth /splits/test does not match /data IDs")
    if "metadata/time_normalized" not in handle:
        raise ValueError("Ground truth is missing /metadata/time_normalized")
    time_values = np.asarray(handle["metadata/time_normalized"][:], dtype=np.float64)
    if time_values.shape != (contract.num_timesteps,):
        raise ValueError(f"Unexpected time coordinate shape: {time_values.shape}")
    _require_finite("Ground-truth time coordinates", time_values)
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("Ground-truth time coordinates must be strictly increasing")
    return truth_ids, time_values


def _read_timing_and_provenance(
    sample_group: h5py.Group,
    *,
    sample_id: int,
    model: str,
) -> tuple[float | None, str | None, str | None]:
    metadata = sample_group.get("metadata")
    if metadata is None:
        raise ValueError(f"Sample {sample_id}: rollout metadata group is missing")
    metadata_sample_id = int(metadata.attrs.get("sample_id", -1))
    if metadata_sample_id != sample_id:
        raise ValueError(
            f"Sample {sample_id}: metadata sample_id={metadata_sample_id} is inconsistent"
        )
    raw_timing = metadata.attrs.get("total_rollout_time_s")
    timing = None if raw_timing is None else float(raw_timing)
    if timing is not None and (not math.isfinite(timing) or timing < 0):
        raise ValueError("total_rollout_time_s must be finite and non-negative")
    raw_model = metadata.attrs.get("model_path")
    raw_config = metadata.attrs.get("config_file")
    if raw_model is None or not _decode(raw_model).strip():
        raise ValueError(f"Sample {sample_id}: model_path provenance is missing")
    if raw_config is None or not _decode(raw_config).strip():
        raise ValueError(f"Sample {sample_id}: config_file provenance is missing")
    model_path = _decode(raw_model).strip()
    config_file = _decode(raw_config).strip()
    if not _provenance_matches_model(model_path, model):
        raise ValueError(
            f"Sample {sample_id}: model_path does not identify requested model {model}"
        )
    if not _provenance_matches_model(config_file, model):
        raise ValueError(
            f"Sample {sample_id}: config_file does not identify requested model {model}"
        )
    return (
        timing,
        model_path,
        config_file,
    )


def evaluate_rollouts(
    *,
    model: str,
    ground_truth: Path,
    predictions: Path,
    output_json: Path,
    output_csv: Path,
    epsilon: float = 1.0e-12,
    contract: EvaluationContract = PRODUCTION_CONTRACT,
) -> dict[str, object]:
    """Validate and score one model's complete held-out rollout directory."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    ground_truth = ground_truth.resolve()
    predictions = predictions.resolve()
    output_json = output_json.resolve()
    output_csv = output_csv.resolve()
    if not ground_truth.is_file():
        raise FileNotFoundError(f"Ground truth not found: {ground_truth}")
    if _paths_alias(output_json, output_csv):
        raise ValueError("output JSON and CSV paths must be distinct")
    if _paths_alias(output_json, ground_truth) or _paths_alias(output_csv, ground_truth):
        raise ValueError("Output artifacts must not alias the ground-truth HDF5")

    truth_stat_before = ground_truth.stat()
    truth_sha256 = _sha256_file(ground_truth)
    truth_stat_after_hash = ground_truth.stat()
    if (
        truth_stat_before.st_size != truth_stat_after_hash.st_size
        or truth_stat_before.st_mtime_ns != truth_stat_after_hash.st_mtime_ns
    ):
        raise RuntimeError(f"Ground truth changed while being hashed: {ground_truth}")
    if contract.pinned_ground_truth_sha256 is not None:
        expected_hash = contract.pinned_ground_truth_sha256.lower()
        if not _is_sha256(expected_hash):
            raise ValueError("Pinned ground-truth SHA256 is malformed")
        if truth_sha256 != expected_hash:
            raise ValueError(
                "Ground-truth SHA256 mismatch: "
                f"got {truth_sha256}, expected {expected_hash}"
            )
    rollouts = _discover_rollouts(predictions, contract)
    for rollout_path in rollouts.values():
        if _paths_alias(output_json, rollout_path) or _paths_alias(output_csv, rollout_path):
            raise ValueError("Output artifacts must not alias an input rollout")

    rows: list[dict[str, object]] = []
    per_case_full: list[float] = []
    per_case_time_average: list[float] = []
    per_case_final: list[float] = []
    timing_values: list[float] = []
    missing_timing_ids: list[int] = []
    checkpoint_paths: set[str] = set()
    config_files: set[str] = set()
    prediction_manifest: list[dict[str, object]] = []
    component_sse = np.zeros(3, dtype=np.float64)
    component_sae = np.zeros(3, dtype=np.float64)
    component_count = 0
    total_sse = 0.0
    total_target_sse = 0.0
    max_geometry_error = 0.0
    max_initial_state_error = 0.0
    rollouts_with_mesh_edges = 0
    zero_time_denominators = 0

    with h5py.File(ground_truth, "r") as truth_file:
        truth_ids, time_values = _validate_truth(truth_file, contract)

        for sample_id in truth_ids:
            truth_group = truth_file[f"data/{sample_id}"]
            truth_nodal = np.asarray(truth_group["nodal_data"], dtype=np.float64)
            if truth_nodal.shape != contract.nodal_shape:
                raise ValueError(
                    f"Truth sample {sample_id}: nodal shape {truth_nodal.shape}, "
                    f"expected {contract.nodal_shape}"
                )
            _require_finite(f"Truth sample {sample_id}", truth_nodal)
            source_index = int(truth_group["metadata"].attrs.get("source_index", -1))
            if source_index != sample_id:
                raise ValueError(
                    f"Truth sample {sample_id}: source_index={source_index} is inconsistent"
                )
            truth_edges = np.asarray(truth_group["mesh_edge"])
            if truth_edges.shape != (2, contract.num_edges):
                raise ValueError(
                    f"Truth sample {sample_id}: mesh_edge shape {truth_edges.shape}"
                )

            rollout_path = rollouts[sample_id]
            rollout_stat_before = rollout_path.stat()
            with h5py.File(rollout_path, "r") as rollout_file:
                if int(rollout_file.attrs.get("num_samples", -1)) != 1:
                    raise ValueError(f"{rollout_path.name}: num_samples must be 1")
                if int(rollout_file.attrs.get("num_features", -1)) != contract.num_features:
                    raise ValueError(f"{rollout_path.name}: num_features mismatch")
                if int(rollout_file.attrs.get("num_timesteps", -1)) != contract.num_timesteps:
                    raise ValueError(f"{rollout_path.name}: num_timesteps mismatch")
                if "data" not in rollout_file or set(rollout_file["data"].keys()) != {str(sample_id)}:
                    raise ValueError(
                        f"{rollout_path.name}: /data must contain only sample {sample_id}"
                    )
                prediction_group = rollout_file[f"data/{sample_id}"]
                prediction_nodal = np.asarray(
                    prediction_group["nodal_data"], dtype=np.float64
                )
                if prediction_nodal.shape != contract.nodal_shape:
                    raise ValueError(
                        f"{rollout_path.name}: nodal shape {prediction_nodal.shape}, "
                        f"expected {contract.nodal_shape}"
                    )
                _require_finite(rollout_path.name, prediction_nodal)

                geometry_error = float(
                    np.max(np.abs(prediction_nodal[0:3] - truth_nodal[0:3]))
                )
                max_geometry_error = max(max_geometry_error, geometry_error)
                if geometry_error > contract.geometry_tolerance:
                    raise ValueError(
                        f"{rollout_path.name}: geometry mismatch {geometry_error:.3e} "
                        f"> {contract.geometry_tolerance:.3e}"
                    )
                seed_state_error = float(
                    np.max(np.abs(prediction_nodal[3:7, 0] - truth_nodal[3:7, 0]))
                )
                max_initial_state_error = max(max_initial_state_error, seed_state_error)
                if seed_state_error > contract.seed_state_tolerance:
                    raise ValueError(
                        f"{rollout_path.name}: seed state channels 3:7 mismatch "
                        f"{seed_state_error:.3e} > {contract.seed_state_tolerance:.3e}"
                    )

                if "mesh_edge" in prediction_group:
                    prediction_edges = np.asarray(prediction_group["mesh_edge"])
                    if prediction_edges.shape != truth_edges.shape or not np.array_equal(
                        prediction_edges, truth_edges
                    ):
                        raise ValueError(f"{rollout_path.name}: mesh_edge mismatch")
                    rollouts_with_mesh_edges += 1
                elif model != "transolver":
                    raise ValueError(
                        f"{rollout_path.name}: mesh_edge is required for model {model}"
                    )

                timing, model_path, config_file = _read_timing_and_provenance(
                    prediction_group, sample_id=sample_id, model=model
                )
                if timing is None:
                    missing_timing_ids.append(sample_id)
                else:
                    timing_values.append(timing)
                if model_path:
                    checkpoint_paths.add(model_path)
                if config_file:
                    config_files.add(config_file)

            rollout_stat_after = rollout_path.stat()
            if (
                rollout_stat_before.st_size != rollout_stat_after.st_size
                or rollout_stat_before.st_mtime_ns != rollout_stat_after.st_mtime_ns
            ):
                raise RuntimeError(f"Rollout changed while being evaluated: {rollout_path}")
            rollout_sha256 = _sha256_file(rollout_path)
            rollout_stat_after_hash = rollout_path.stat()
            if (
                rollout_stat_after.st_size != rollout_stat_after_hash.st_size
                or rollout_stat_after.st_mtime_ns != rollout_stat_after_hash.st_mtime_ns
            ):
                raise RuntimeError(f"Rollout changed while being hashed: {rollout_path}")
            prediction_manifest.append(
                {
                    "sample_id": sample_id,
                    "filename": rollout_path.name,
                    "bytes": rollout_stat_after_hash.st_size,
                    "sha256": rollout_sha256,
                }
            )

            # Saved channels 3:6 are physical ux/uy/uz.  Channel 6 is the
            # static die profile and is intentionally excluded.
            target = truth_nodal[3:6, 1 : contract.num_timesteps, :]
            prediction = prediction_nodal[3:6, 1 : contract.num_timesteps, :]
            difference = prediction - target
            difference_squared = difference * difference
            target_squared = target * target

            case_sse = float(difference_squared.sum(dtype=np.float64))
            case_target_sse = float(target_squared.sum(dtype=np.float64))
            case_relative_l2 = math.sqrt(case_sse) / max(
                math.sqrt(case_target_sse), epsilon
            )
            time_sse = difference_squared.sum(axis=(0, 2), dtype=np.float64)
            time_target_sse = target_squared.sum(axis=(0, 2), dtype=np.float64)
            time_relative_l2 = np.sqrt(time_sse) / np.maximum(
                np.sqrt(time_target_sse), epsilon
            )
            zero_time_denominators += int(np.count_nonzero(time_target_sse == 0.0))
            case_time_average = float(time_relative_l2.mean())
            case_final = float(time_relative_l2[-1])

            per_case_full.append(case_relative_l2)
            per_case_time_average.append(case_time_average)
            per_case_final.append(case_final)
            total_sse += case_sse
            total_target_sse += case_target_sse
            component_sse += difference_squared.sum(axis=(1, 2), dtype=np.float64)
            component_sae += np.abs(difference).sum(axis=(1, 2), dtype=np.float64)
            component_count += difference.shape[1] * difference.shape[2]

            for local_time, time_index in enumerate(range(1, contract.num_timesteps)):
                time_difference = difference[:, local_time, :]
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id,
                        "source_index": source_index,
                        "time_index": time_index,
                        "time_normalized": float(time_values[time_index]),
                        "relative_l2": float(time_relative_l2[local_time]),
                        "error_l2": float(math.sqrt(time_sse[local_time])),
                        "target_l2": float(math.sqrt(time_target_sse[local_time])),
                        "case_full_trajectory_relative_l2": case_relative_l2,
                        "case_time_averaged_relative_l2": case_time_average,
                        "rmse_u_x_mm": float(np.sqrt(np.mean(time_difference[0] ** 2))),
                        "rmse_u_y_mm": float(np.sqrt(np.mean(time_difference[1] ** 2))),
                        "rmse_u_z_mm": float(np.sqrt(np.mean(time_difference[2] ** 2))),
                        "mae_u_x_mm": float(np.mean(np.abs(time_difference[0]))),
                        "mae_u_y_mm": float(np.mean(np.abs(time_difference[1]))),
                        "mae_u_z_mm": float(np.mean(np.abs(time_difference[2]))),
                        "rollout_time_seconds": "" if timing is None else timing,
                        "prediction_file": rollout_path.name,
                        "prediction_sha256": rollout_sha256,
                    }
                )

    truth_stat_after = ground_truth.stat()
    if (
        truth_stat_before.st_size != truth_stat_after.st_size
        or truth_stat_before.st_mtime_ns != truth_stat_after.st_mtime_ns
    ):
        raise RuntimeError(f"Ground truth changed while being evaluated: {ground_truth}")
    if len(checkpoint_paths) != 1:
        raise ValueError(
            f"Expected one consistent checkpoint path across all rollouts, got {checkpoint_paths}"
        )
    if len(config_files) != 1:
        raise ValueError(
            f"Expected one consistent config path across all rollouts, got {config_files}"
        )

    component_metrics = {
        name: {
            "rmse": float(math.sqrt(component_sse[index] / component_count)),
            "mae": float(component_sae[index] / component_count),
        }
        for index, name in enumerate(COMPONENTS)
    }
    primary_value = float(np.mean(per_case_full))
    manifest_text = json.dumps(prediction_manifest, sort_keys=True, separators=(",", ":"))
    truth_ids_text = ",".join(str(value) for value in contract.expected_truth_ids)
    timing_summary: dict[str, object] = {
        "attribute": "data/{sample_id}/metadata.attrs.total_rollout_time_s",
        "available_cases": len(timing_values),
        "missing_cases": len(missing_timing_ids),
        "missing_sample_ids": missing_timing_ids,
    }
    if timing_values:
        timing_summary.update(
            {
                "total_seconds": float(sum(timing_values)),
                "per_case_seconds": _summary(timing_values),
            }
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "complete": True,
        "primary_metric": {
            "name": "mean_per_case_full_trajectory_relative_l2",
            "value": primary_value,
            "lower_is_better": True,
            "definition": (
                "arithmetic mean over cases of ||prediction-target||_2 / "
                "||target||_2 on de-normalized ux,uy,uz over timesteps 1..19"
            ),
        },
        "dataset": {
            "ground_truth": str(ground_truth),
            "ground_truth_bytes": truth_stat_after.st_size,
            "ground_truth_sha256": truth_sha256,
            "split_seed": contract.split_seed,
            "split_role": "test",
            "expected_cases": contract.expected_cases,
            "truth_ids_sha256": _sha256_text(truth_ids_text),
            "truth_sample_ids": list(contract.expected_truth_ids),
        },
        "predictions": {
            "directory": str(predictions),
            "files": len(prediction_manifest),
            "manifest_sha256": _sha256_text(manifest_text),
            "manifest": prediction_manifest,
        },
        "evaluation_contract": {
            "saved_displacement_feature_indices": [3, 4, 5],
            "component_names": list(COMPONENTS),
            "excluded_saved_feature_index": 6,
            "excluded_feature": "die_profile_mm",
            "evaluated_time_indices": list(range(1, contract.num_timesteps)),
            "excluded_seed_time_index": 0,
            "expected_nodal_shape": list(contract.nodal_shape),
            "expected_rollout_steps": contract.rollout_steps,
            "epsilon": epsilon,
        },
        "validation": {
            "complete": True,
            "evaluated_cases": len(per_case_full),
            "csv_rows": len(rows),
            "max_abs_geometry_error": max_geometry_error,
            "geometry_tolerance": contract.geometry_tolerance,
            "max_abs_seed_state_channels_3_to_6_error": max_initial_state_error,
            "seed_state_tolerance": contract.seed_state_tolerance,
            "rollouts_with_mesh_edges": rollouts_with_mesh_edges,
            "rollouts_without_mesh_edges": contract.expected_cases - rollouts_with_mesh_edges,
            "zero_per_timestep_target_denominators": zero_time_denominators,
            "all_values_finite": True,
        },
        "metrics": {
            "mean_per_case_full_trajectory_relative_l2": primary_value,
            "global_full_trajectory_relative_l2": float(
                math.sqrt(total_sse) / max(math.sqrt(total_target_sse), epsilon)
            ),
            "mean_per_case_time_averaged_per_timestep_relative_l2": float(
                np.mean(per_case_time_average)
            ),
            "final_time_mean_per_case_relative_l2": float(np.mean(per_case_final)),
            "per_component": component_metrics,
        },
        "distributions": {
            "per_case_full_trajectory_relative_l2": _summary(per_case_full),
            "per_case_time_averaged_per_timestep_relative_l2": _summary(
                per_case_time_average
            ),
            "per_case_final_time_relative_l2": _summary(per_case_final),
        },
        "timing": timing_summary,
        "provenance": {
            "checkpoint_paths": sorted(checkpoint_paths),
            "config_files": sorted(config_files),
        },
        "artifacts": {},
    }

    _write_csv(output_csv, rows)
    csv_stat_before_hash = output_csv.stat()
    csv_sha256 = _sha256_file(output_csv)
    csv_stat_after_hash = output_csv.stat()
    if (
        csv_stat_before_hash.st_size != csv_stat_after_hash.st_size
        or csv_stat_before_hash.st_mtime_ns != csv_stat_after_hash.st_mtime_ns
    ):
        raise RuntimeError(f"Metric CSV changed while being hashed: {output_csv}")
    result["artifacts"] = {
        "summary_json": {"path": str(output_json)},
        "per_case_time_csv": {
            "path": str(output_csv),
            "bytes": csv_stat_after_hash.st_size,
            "sha256": csv_sha256,
            "rows": len(rows),
        },
    }
    _atomic_write_text(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--ground-truth", type=Path, default=here / "plasticity_seed42_test.h5"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_dir = args.predictions.resolve()
    output_json = (
        args.output_json or prediction_dir / "plasticity_metrics.json"
    ).resolve()
    output_csv = (
        args.output_csv or prediction_dir / "plasticity_per_case_time.csv"
    ).resolve()
    result = evaluate_rollouts(
        model=args.model,
        ground_truth=args.ground_truth,
        predictions=prediction_dir,
        output_json=output_json,
        output_csv=output_csv,
        epsilon=args.epsilon,
    )
    print(json.dumps(result["primary_metric"], indent=2, sort_keys=True))
    print(f"Summary JSON: {output_json}")
    print(f"Per-case/time CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
