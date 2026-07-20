#!/usr/bin/env python3
"""Combine all seven strict Plasticity rollout evaluations into one ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from evaluate_rollouts import (
    COMPONENTS,
    MODELS,
    PINNED_TRUTH_SHA256,
    PRODUCTION_TEST_IDS,
    SCHEMA_VERSION,
    _is_sha256,
    _provenance_matches_model,
    _sha256_file,
    _sha256_text,
)


COMPARISON_SCHEMA_VERSION = "plasticity_seven_model_comparison_v1"
PRIMARY_METRIC = "mean_per_case_full_trajectory_relative_l2"


class IncompleteComparisonError(RuntimeError):
    """Raised after writing an explicit incomplete comparison report."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _finite_metric(payload: dict[str, object], path: str) -> float:
    current: object = payload
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing metric field {path}")
        current = current[key]
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"Metric {path} is non-finite")
    return value


def _nonnegative_metric(payload: dict[str, object], path: str) -> float:
    value = _finite_metric(payload, path)
    if value < 0:
        raise ValueError(f"Metric {path} is negative")
    return value


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12)


def _validate_manifest(payload: dict[str, object]) -> dict[int, dict[str, object]]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict) or int(predictions.get("files", -1)) != 100:
        raise ValueError("Prediction file count must be 100")
    manifest = predictions.get("manifest")
    if not isinstance(manifest, list) or len(manifest) != 100:
        raise ValueError("Prediction manifest must contain 100 entries")
    by_id: dict[int, dict[str, object]] = {}
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("Prediction manifest entry is not an object")
        sample_id = int(entry.get("sample_id", -1))
        if sample_id in by_id:
            raise ValueError(f"Duplicate prediction manifest ID {sample_id}")
        if entry.get("filename") != f"rollout_sample{sample_id}_steps19.h5":
            raise ValueError(f"Prediction manifest filename mismatch for {sample_id}")
        if int(entry.get("bytes", 0)) <= 0:
            raise ValueError(f"Prediction manifest byte count is invalid for {sample_id}")
        if not _is_sha256(entry.get("sha256")):
            raise ValueError(f"Prediction manifest SHA256 is invalid for {sample_id}")
        by_id[sample_id] = entry
    if tuple(sorted(by_id)) != PRODUCTION_TEST_IDS:
        raise ValueError("Prediction manifest IDs are not the exact production test IDs")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    if predictions.get("manifest_sha256") != _sha256_text(canonical):
        raise ValueError("Prediction manifest digest is inconsistent")
    return by_id


def _validate_csv(
    *,
    csv_path: Path,
    artifact: dict[str, object],
    model: str,
    manifest: dict[int, dict[str, object]],
) -> dict[str, float]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metric CSV does not exist: {csv_path}")
    stat_before = csv_path.stat()
    digest = _sha256_file(csv_path)
    stat_after = csv_path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError(f"Metric CSV changed while being hashed: {csv_path}")
    if int(artifact.get("bytes", -1)) != stat_after.st_size:
        raise ValueError("Metric CSV byte count disagrees with its artifact record")
    if artifact.get("sha256") != digest or not _is_sha256(digest):
        raise ValueError("Metric CSV SHA256 disagrees with its artifact record")
    if int(artifact.get("rows", -1)) != 1900:
        raise ValueError("Metric CSV artifact row count must be 1900")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1900:
        raise ValueError(f"Metric CSV has {len(rows)} rows, expected 1900")

    relative_values: list[float] = []
    final_values: list[float] = []
    error_squared_sum = 0.0
    target_squared_sum = 0.0
    component_rmse_squared = {component: [] for component in COMPONENTS}
    component_mae = {component: [] for component in COMPONENTS}
    case_full: dict[int, float] = {}
    case_time_average: dict[int, float] = {}
    case_error_squared: dict[int, float] = {}
    case_target_squared: dict[int, float] = {}
    times_by_case: dict[int, set[int]] = {}
    timing_by_case: dict[int, float | None] = {}

    for row in rows:
        if row.get("model") != model:
            raise ValueError("Metric CSV contains a row for the wrong model")
        sample_id = int(row["sample_id"])
        if sample_id not in manifest or int(row["source_index"]) != sample_id:
            raise ValueError("Metric CSV sample/source ID contract failed")
        time_index = int(row["time_index"])
        if time_index not in range(1, 20):
            raise ValueError("Metric CSV time_index is outside 1..19")
        expected_time = time_index / 19.0
        expected_time_float32 = float(np.float32(expected_time))
        if not (
            _same_float(float(row["time_normalized"]), expected_time)
            or _same_float(float(row["time_normalized"]), expected_time_float32)
        ):
            raise ValueError("Metric CSV normalized time coordinate is inconsistent")
        if row.get("prediction_file") != manifest[sample_id]["filename"]:
            raise ValueError("Metric CSV prediction filename disagrees with manifest")
        if row.get("prediction_sha256") != manifest[sample_id]["sha256"]:
            raise ValueError("Metric CSV prediction SHA256 disagrees with manifest")

        numeric_names = (
            "relative_l2",
            "error_l2",
            "target_l2",
            "case_full_trajectory_relative_l2",
            "case_time_averaged_relative_l2",
            "rmse_u_x_mm",
            "rmse_u_y_mm",
            "rmse_u_z_mm",
            "mae_u_x_mm",
            "mae_u_y_mm",
            "mae_u_z_mm",
        )
        values = {name: float(row[name]) for name in numeric_names}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Metric CSV contains a negative or non-finite metric")
        relative_values.append(values["relative_l2"])
        if time_index == 19:
            final_values.append(values["relative_l2"])
        error_squared_sum += values["error_l2"] ** 2
        target_squared_sum += values["target_l2"] ** 2
        case_error_squared[sample_id] = (
            case_error_squared.get(sample_id, 0.0) + values["error_l2"] ** 2
        )
        case_target_squared[sample_id] = (
            case_target_squared.get(sample_id, 0.0) + values["target_l2"] ** 2
        )
        times_by_case.setdefault(sample_id, set()).add(time_index)

        for target_dict, key in (
            (case_full, "case_full_trajectory_relative_l2"),
            (case_time_average, "case_time_averaged_relative_l2"),
        ):
            if sample_id in target_dict and not _same_float(target_dict[sample_id], values[key]):
                raise ValueError(f"Metric CSV case aggregate varies within sample {sample_id}")
            target_dict[sample_id] = values[key]

        for component in COMPONENTS:
            component_rmse_squared[component].append(values[f"rmse_{component}"] ** 2)
            component_mae[component].append(values[f"mae_{component}"])

        timing_text = row.get("rollout_time_seconds", "")
        timing = None if timing_text == "" else float(timing_text)
        if timing is not None and (not math.isfinite(timing) or timing < 0):
            raise ValueError("Metric CSV rollout timing is negative or non-finite")
        if sample_id in timing_by_case and timing_by_case[sample_id] != timing:
            raise ValueError(f"Metric CSV rollout timing varies within sample {sample_id}")
        timing_by_case[sample_id] = timing

    if tuple(sorted(times_by_case)) != PRODUCTION_TEST_IDS:
        raise ValueError("Metric CSV IDs are not the exact production test IDs")
    if any(times != set(range(1, 20)) for times in times_by_case.values()):
        raise ValueError("Metric CSV does not contain exactly 19 rows per case")
    for sample_id in PRODUCTION_TEST_IDS:
        sample_relative = [
            float(row["relative_l2"])
            for row in rows
            if int(row["sample_id"]) == sample_id
        ]
        if not _same_float(case_time_average[sample_id], sum(sample_relative) / 19.0):
            raise ValueError("Metric CSV case time-average is inconsistent")
        reconstructed_full = math.sqrt(case_error_squared[sample_id]) / max(
            math.sqrt(case_target_squared[sample_id]), 1.0e-12
        )
        if not _same_float(case_full[sample_id], reconstructed_full):
            raise ValueError("Metric CSV case full-trajectory relative L2 is inconsistent")

    return {
        "mean_per_case_full_trajectory_relative_l2": float(
            sum(case_full.values()) / 100.0
        ),
        "global_full_trajectory_relative_l2": float(
            math.sqrt(error_squared_sum) / max(math.sqrt(target_squared_sum), 1.0e-12)
        ),
        "mean_per_case_time_averaged_per_timestep_relative_l2": float(
            sum(relative_values) / 1900.0
        ),
        "final_time_mean_per_case_relative_l2": float(sum(final_values) / 100.0),
        **{
            f"rmse_{component}": float(
                math.sqrt(sum(component_rmse_squared[component]) / 1900.0)
            )
            for component in COMPONENTS
        },
        **{
            f"mae_{component}": float(sum(component_mae[component]) / 1900.0)
            for component in COMPONENTS
        },
        "timing_available_cases": float(
            sum(value is not None for value in timing_by_case.values())
        ),
        "timing_total_seconds": float(
            sum(value for value in timing_by_case.values() if value is not None)
        ),
    }


def _validate_result(path: Path, model: str) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema_version in {path}")
    if payload.get("model") != model:
        raise ValueError(f"{path}: model={payload.get('model')!r}, expected {model!r}")
    if payload.get("complete") is not True:
        raise ValueError(f"{path}: evaluator did not mark the result complete")
    validation = payload.get("validation")
    dataset = payload.get("dataset")
    primary = payload.get("primary_metric")
    contract = payload.get("evaluation_contract")
    if not all(isinstance(value, dict) for value in (validation, dataset, primary, contract)):
        raise ValueError(f"{path}: missing validation/dataset/metric contract")
    assert isinstance(validation, dict)
    assert isinstance(dataset, dict)
    assert isinstance(primary, dict)
    assert isinstance(contract, dict)
    if validation.get("complete") is not True:
        raise ValueError(f"{path}: validation is incomplete")
    if int(validation.get("evaluated_cases", -1)) != 100:
        raise ValueError(f"{path}: evaluated_cases must be 100")
    if int(validation.get("csv_rows", -1)) != 1900:
        raise ValueError(f"{path}: csv_rows must be 1900")
    if validation.get("all_values_finite") is not True:
        raise ValueError(f"{path}: finite-value validation did not pass")
    if int(dataset.get("expected_cases", -1)) != 100:
        raise ValueError(f"{path}: expected_cases must be 100")
    if int(dataset.get("split_seed", -1)) != 42 or dataset.get("split_role") != "test":
        raise ValueError(f"{path}: production split identity is wrong")
    if tuple(dataset.get("truth_sample_ids", ())) != PRODUCTION_TEST_IDS:
        raise ValueError(f"{path}: production truth ID tuple is wrong")
    ids_text = ",".join(str(value) for value in PRODUCTION_TEST_IDS)
    if dataset.get("truth_ids_sha256") != _sha256_text(ids_text):
        raise ValueError(f"{path}: production truth ID digest is wrong")
    if dataset.get("ground_truth_sha256") != PINNED_TRUTH_SHA256:
        raise ValueError(f"{path}: ground-truth SHA256 is not the pinned artifact")
    if not _is_sha256(dataset.get("ground_truth_sha256")):
        raise ValueError(f"{path}: malformed ground-truth SHA256")
    if primary.get("name") != PRIMARY_METRIC or primary.get("lower_is_better") is not True:
        raise ValueError(f"{path}: wrong primary metric contract")
    if contract.get("saved_displacement_feature_indices") != [3, 4, 5]:
        raise ValueError(f"{path}: displacement channel contract differs")
    if contract.get("evaluated_time_indices") != list(range(1, 20)):
        raise ValueError(f"{path}: time-index contract differs")
    if contract.get("expected_nodal_shape") != [8, 20, 3131]:
        raise ValueError(f"{path}: nodal shape contract differs")
    if int(contract.get("expected_rollout_steps", -1)) != 19:
        raise ValueError(f"{path}: rollout-step contract differs")
    if int(contract.get("excluded_saved_feature_index", -1)) != 6:
        raise ValueError(f"{path}: excluded die-profile channel differs")
    if contract.get("excluded_feature") != "die_profile_mm":
        raise ValueError(f"{path}: excluded feature name differs")
    if int(contract.get("excluded_seed_time_index", -1)) != 0:
        raise ValueError(f"{path}: seed-time exclusion differs")
    if float(contract.get("epsilon", float("nan"))) != 1.0e-12:
        raise ValueError(f"{path}: epsilon contract differs")
    required_edges = 0 if model == "transolver" else 100
    if int(validation.get("rollouts_with_mesh_edges", -1)) < required_edges:
        raise ValueError(f"{path}: required rollout mesh edges are missing")
    if (
        int(validation.get("rollouts_with_mesh_edges", -1))
        + int(validation.get("rollouts_without_mesh_edges", -1))
        != 100
    ):
        raise ValueError(f"{path}: rollout mesh-edge accounting is inconsistent")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}: provenance object is missing")
    checkpoints = provenance.get("checkpoint_paths")
    configs = provenance.get("config_files")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise ValueError(f"{path}: checkpoint provenance must contain one path")
    if not isinstance(configs, list) or len(configs) != 1:
        raise ValueError(f"{path}: config provenance must contain one path")
    if not _provenance_matches_model(str(checkpoints[0]), model):
        raise ValueError(f"{path}: checkpoint provenance identifies the wrong model")
    if not _provenance_matches_model(str(configs[0]), model):
        raise ValueError(f"{path}: config provenance identifies the wrong model")

    manifest = _validate_manifest(payload)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{path}: artifact records are missing")
    csv_artifact = artifacts.get("per_case_time_csv")
    summary_artifact = artifacts.get("summary_json")
    if (
        not isinstance(summary_artifact, dict)
        or Path(str(summary_artifact.get("path", ""))).resolve() != path.resolve()
    ):
        raise ValueError(f"{path}: summary JSON artifact path is inconsistent")
    if not isinstance(csv_artifact, dict) or not str(csv_artifact.get("path", "")):
        raise ValueError(f"{path}: CSV artifact record is missing")
    csv_path = Path(str(csv_artifact["path"])).resolve()
    if csv_path == path.resolve() or (csv_path.exists() and os.path.samefile(csv_path, path)):
        raise ValueError(f"{path}: CSV aliases the result JSON")
    csv_metrics = _validate_csv(
        csv_path=csv_path,
        artifact=csv_artifact,
        model=model,
        manifest=manifest,
    )

    primary_value = _nonnegative_metric(
        payload, "metrics.mean_per_case_full_trajectory_relative_l2"
    )
    declared_primary_value = float(primary.get("value", float("nan")))
    if not math.isfinite(declared_primary_value) or declared_primary_value != primary_value:
        raise ValueError(f"{path}: declared primary value disagrees with metrics")

    row: dict[str, object] = {
        "rank": "",
        "model": model,
        "status": "ok",
        "mean_per_case_full_trajectory_relative_l2": primary_value,
        "global_full_trajectory_relative_l2": _nonnegative_metric(
            payload, "metrics.global_full_trajectory_relative_l2"
        ),
        "mean_per_case_time_averaged_per_timestep_relative_l2": _nonnegative_metric(
            payload,
            "metrics.mean_per_case_time_averaged_per_timestep_relative_l2",
        ),
        "final_time_mean_per_case_relative_l2": _nonnegative_metric(
            payload, "metrics.final_time_mean_per_case_relative_l2"
        ),
        "rmse_u_x_mm": _nonnegative_metric(payload, "metrics.per_component.u_x_mm.rmse"),
        "rmse_u_y_mm": _nonnegative_metric(payload, "metrics.per_component.u_y_mm.rmse"),
        "rmse_u_z_mm": _nonnegative_metric(payload, "metrics.per_component.u_z_mm.rmse"),
        "mae_u_x_mm": _nonnegative_metric(payload, "metrics.per_component.u_x_mm.mae"),
        "mae_u_y_mm": _nonnegative_metric(payload, "metrics.per_component.u_y_mm.mae"),
        "mae_u_z_mm": _nonnegative_metric(payload, "metrics.per_component.u_z_mm.mae"),
        "timing_available_cases": int(payload.get("timing", {}).get("available_cases", 0)),
        "timing_total_seconds": payload.get("timing", {}).get("total_seconds", ""),
        "result_json": str(path.resolve()),
        "error": "",
    }
    json_to_csv_fields = {
        "mean_per_case_full_trajectory_relative_l2": "mean_per_case_full_trajectory_relative_l2",
        "global_full_trajectory_relative_l2": "global_full_trajectory_relative_l2",
        "mean_per_case_time_averaged_per_timestep_relative_l2": "mean_per_case_time_averaged_per_timestep_relative_l2",
        "final_time_mean_per_case_relative_l2": "final_time_mean_per_case_relative_l2",
        "rmse_u_x_mm": "rmse_u_x_mm",
        "rmse_u_y_mm": "rmse_u_y_mm",
        "rmse_u_z_mm": "rmse_u_z_mm",
        "mae_u_x_mm": "mae_u_x_mm",
        "mae_u_y_mm": "mae_u_y_mm",
        "mae_u_z_mm": "mae_u_z_mm",
    }
    for row_field, csv_field in json_to_csv_fields.items():
        if not _same_float(float(row[row_field]), csv_metrics[csv_field]):
            raise ValueError(f"{path}: JSON metric {row_field} disagrees with CSV")
    timing = payload.get("timing", {})
    available_cases = int(timing.get("available_cases", 0))
    missing_cases = int(timing.get("missing_cases", -1))
    if available_cases < 0 or missing_cases < 0 or available_cases + missing_cases != 100:
        raise ValueError(f"{path}: timing case accounting is inconsistent")
    if int(csv_metrics["timing_available_cases"]) != available_cases:
        raise ValueError(f"{path}: timing availability disagrees with CSV")
    if available_cases:
        timing_total = float(timing.get("total_seconds", float("nan")))
        if not math.isfinite(timing_total) or timing_total < 0:
            raise ValueError(f"{path}: timing total is negative or non-finite")
        if not _same_float(timing_total, csv_metrics["timing_total_seconds"]):
            raise ValueError(f"{path}: timing total disagrees with CSV")
    identity = {
        "ground_truth_sha256": dataset.get("ground_truth_sha256"),
        "truth_ids_sha256": dataset.get("truth_ids_sha256"),
        "split_seed": dataset.get("split_seed"),
        "evaluation_contract": contract,
        "geometry_tolerance": validation.get("geometry_tolerance"),
        "seed_state_tolerance": validation.get("seed_state_tolerance"),
    }
    if not identity["ground_truth_sha256"] or not identity["truth_ids_sha256"]:
        raise ValueError(f"{path}: missing ground-truth identity hashes")
    return row, identity


def _markdown(result: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Plasticity seven-model rollout comparison",
        "",
        f"- Complete: **{str(result['complete']).lower()}**",
        "- Primary metric: arithmetic mean of per-case full-trajectory relative L2",
        "- Evaluation: de-normalized `u_x`, `u_y`, `u_z`; timesteps 1 through 19",
        "- Static die-profile channel and seed timestep 0 are excluded",
        "",
    ]
    missing = result["models_missing"]
    invalid = result["models_invalid"]
    if missing:
        lines.append(f"- Missing models: {', '.join(missing)}")
    if invalid:
        lines.append(f"- Invalid models: {', '.join(invalid)}")
    if missing or invalid:
        lines.extend(
            [
                "",
                "**This is not a complete seven-model result and must not be reported as one.**",
                "",
            ]
        )

    lines.extend(
        [
            "| Rank | Model | Status | Mean case trajectory relL2 | Global relL2 | Mean time-averaged relL2 | Final-time relL2 |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        def display(key: str) -> str:
            value = row[key]
            return "" if value == "" else f"{float(value):.8g}"

        lines.append(
            f"| {row['rank']} | {row['model']} | {row['status']} | "
            f"{display('mean_per_case_full_trajectory_relative_l2')} | "
            f"{display('global_full_trajectory_relative_l2')} | "
            f"{display('mean_per_case_time_averaged_per_timestep_relative_l2')} | "
            f"{display('final_time_mean_per_case_relative_l2')} |"
        )
    return "\n".join(lines) + "\n"


def combine_results(
    *,
    results_root: Path,
    output_dir: Path,
    raise_on_incomplete: bool = True,
) -> dict[str, object]:
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    rows_by_model: dict[str, dict[str, object]] = {}
    identities: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    invalid: list[str] = []

    empty_metrics = {
        "rank": "",
        "mean_per_case_full_trajectory_relative_l2": "",
        "global_full_trajectory_relative_l2": "",
        "mean_per_case_time_averaged_per_timestep_relative_l2": "",
        "final_time_mean_per_case_relative_l2": "",
        "rmse_u_x_mm": "",
        "rmse_u_y_mm": "",
        "rmse_u_z_mm": "",
        "mae_u_x_mm": "",
        "mae_u_y_mm": "",
        "mae_u_z_mm": "",
        "timing_available_cases": "",
        "timing_total_seconds": "",
    }

    for model in MODELS:
        path = results_root / model / "inference" / "plasticity_metrics.json"
        if not path.is_file():
            missing.append(model)
            rows_by_model[model] = {
                **empty_metrics,
                "model": model,
                "status": "missing",
                "result_json": str(path),
                "error": "result JSON not found",
            }
            continue
        try:
            row, identity = _validate_result(path, model)
            rows_by_model[model] = row
            identities[model] = identity
        except Exception as exc:
            invalid.append(model)
            rows_by_model[model] = {
                **empty_metrics,
                "model": model,
                "status": "invalid",
                "result_json": str(path.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }

    if identities:
        identity_keys = {
            model: json.dumps(identity, sort_keys=True, separators=(",", ":"))
            for model, identity in identities.items()
        }
        counts = {
            key: sum(candidate == key for candidate in identity_keys.values())
            for key in set(identity_keys.values())
        }
        reference_key = max(
            counts,
            key=lambda key: (
                counts[key],
                -min(MODELS.index(model) for model, value in identity_keys.items() if value == key),
            ),
        )
        reference_model = next(
            model for model in MODELS if identity_keys.get(model) == reference_key
        )
        reference_identity = identities[reference_model]
        for model in MODELS:
            if model in identities and identities[model] != reference_identity:
                invalid.append(model)
                identities.pop(model)
                previous = rows_by_model[model]
                rows_by_model[model] = {
                    **empty_metrics,
                    "model": model,
                    "status": "invalid",
                    "result_json": previous["result_json"],
                    "error": "dataset or evaluation contract differs from other models",
                }

    valid_rows = [rows_by_model[model] for model in MODELS if model in identities]
    valid_rows.sort(
        key=lambda row: (
            float(row["mean_per_case_full_trajectory_relative_l2"]),
            str(row["model"]),
        )
    )
    for rank, row in enumerate(valid_rows, start=1):
        row["rank"] = rank
    rows = valid_rows + [rows_by_model[model] for model in MODELS if model not in identities]
    missing = sorted(set(missing), key=MODELS.index)
    invalid = sorted(set(invalid), key=MODELS.index)
    complete = not missing and not invalid and len(valid_rows) == len(MODELS)

    result: dict[str, object] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "complete": complete,
        "qualification": (
            "complete seven-model comparison"
            if complete
            else "INCOMPLETE: missing or invalid models must not be treated as success"
        ),
        "primary_metric": PRIMARY_METRIC,
        "lower_is_better": True,
        "models_expected": list(MODELS),
        "models_present_and_valid": [str(row["model"]) for row in valid_rows],
        "models_missing": missing,
        "models_invalid": invalid,
        "shared_identity": next(iter(identities.values())) if identities else None,
        "results": rows,
        "artifacts": {
            "json": str(output_dir / "comparison.json"),
            "csv": str(output_dir / "comparison.csv"),
            "markdown": str(output_dir / "comparison.md"),
        },
    }
    _write_csv(output_dir / "comparison.csv", rows)
    _atomic_write_text(
        output_dir / "comparison.json", json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write_text(output_dir / "comparison.md", _markdown(result, rows))

    if not complete and raise_on_incomplete:
        raise IncompleteComparisonError(
            f"Incomplete comparison: missing={missing}, invalid={invalid}. "
            f"Diagnostic artifacts were written to {output_dir}."
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("output/benchmarks/plasticity")
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (args.output_dir or args.results_root).resolve()
    try:
        result = combine_results(
            results_root=args.results_root,
            output_dir=output_dir,
            raise_on_incomplete=True,
        )
    except IncompleteComparisonError as exc:
        print(exc)
        return 2
    print(json.dumps({"complete": result["complete"], "ranking": result["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
