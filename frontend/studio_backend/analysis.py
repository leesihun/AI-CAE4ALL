"""Real numeric analysis over existing repository files: evaluation, comparison,
optimization, and export. Every function reads actual HDF5/CSV rows and writes
its report under frontend/runtime/; nothing here invents a metric or score.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from studio_backend.paths import FRONTEND_ROOT, RUNTIME_ROOT, SUITE_ROOT, relative, safe_repo_path, slug, utc_now


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_constraint(text: str) -> tuple[str, str, float]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*(<=|>=|<|>)\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*", text)
    if not match:
        raise ValueError(f"Invalid constraint {text!r}; use forms such as stress <= 250.")
    return match.group(1), match.group(2), float(match.group(3))


def run_field_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise ValueError("Field evaluation requires h5py and numpy.") from exc

    allowed = (SUITE_ROOT / "dataset", SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT)
    prediction_path = safe_repo_path(str(payload.get("prediction_path", "")), allowed)
    truth_path = safe_repo_path(str(payload.get("truth_path", "")), allowed)
    if not prediction_path.exists():
        raise ValueError("Prediction path does not exist.")
    if not truth_path.is_file() or truth_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Ground truth must be an existing HDF5 file.")

    prediction_start = max(0, int(payload.get("prediction_start", 3)))
    truth_start = max(0, int(payload.get("truth_start", 3)))
    num_fields = max(1, int(payload.get("num_fields", 1)))
    prediction_files = (
        sorted(path for path in prediction_path.rglob("*") if path.suffix.lower() in {".h5", ".hdf5"})
        if prediction_path.is_dir()
        else [prediction_path]
    )
    if not prediction_files:
        raise ValueError("Prediction path contains no HDF5 files.")

    with h5py.File(truth_path, "r") as truth_handle:
        if "data" not in truth_handle:
            raise ValueError("Ground-truth HDF5 is missing the /data group.")
        truth_ids = {str(sample_id) for sample_id in truth_handle["data"].keys()}

    pairs: list[tuple[Path, str, str]] = []
    for file_path in prediction_files:
        try:
            with h5py.File(file_path, "r") as handle:
                if "data" not in handle:
                    continue
                for sample_id in handle["data"].keys():
                    truth_id = str(sample_id)
                    if truth_id not in truth_ids:
                        match = re.search(r"sample(\d+)", file_path.stem, re.IGNORECASE)
                        if match and match.group(1) in truth_ids:
                            truth_id = match.group(1)
                        elif len(truth_ids) == 1:
                            truth_id = next(iter(truth_ids))
                        else:
                            continue
                    pairs.append((file_path, str(sample_id), truth_id))
        except OSError:
            continue
    if not pairs:
        raise ValueError("No prediction samples could be matched to ground-truth sample IDs.")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for file_path, prediction_id, truth_id in pairs:
        with h5py.File(file_path, "r") as prediction_handle, h5py.File(truth_path, "r") as truth_handle:
            prediction_key = f"data/{prediction_id}/nodal_data"
            truth_key = f"data/{truth_id}/nodal_data"
            if prediction_key not in prediction_handle or truth_key not in truth_handle:
                skipped.append(f"{relative(file_path)}:{prediction_id} missing nodal_data")
                continue
            prediction_dataset = prediction_handle[prediction_key]
            truth_dataset = truth_handle[truth_key]
            if prediction_dataset.ndim != 3 or truth_dataset.ndim != 3:
                skipped.append(f"{relative(file_path)}:{prediction_id} requires rank-3 nodal_data")
                continue
            if prediction_start + num_fields > prediction_dataset.shape[0] or truth_start + num_fields > truth_dataset.shape[0]:
                skipped.append(f"{relative(file_path)}:{prediction_id} field range is outside nodal_data")
                continue
            if prediction_dataset.shape[2] != truth_dataset.shape[2]:
                skipped.append(
                    f"{relative(file_path)}:{prediction_id} node mismatch "
                    f"{prediction_dataset.shape[2]} != {truth_dataset.shape[2]}"
                )
                continue
            timesteps = min(prediction_dataset.shape[1], truth_dataset.shape[1])
            prediction = np.asarray(
                prediction_dataset[prediction_start:prediction_start + num_fields, :timesteps, :],
                dtype=np.float64,
            )
            truth = np.asarray(
                truth_dataset[truth_start:truth_start + num_fields, :timesteps, :],
                dtype=np.float64,
            )
        finite = np.isfinite(prediction) & np.isfinite(truth)
        if not finite.any():
            skipped.append(f"{relative(file_path)}:{prediction_id} contains no finite paired values")
            continue
        prediction_values = prediction[finite]
        truth_values = truth[finite]
        difference = prediction_values - truth_values
        denominator = max(float(np.linalg.norm(truth_values)), np.finfo(np.float64).eps)
        truth_centered = truth_values - truth_values.mean()
        total_variance = float(np.dot(truth_centered, truth_centered))
        squared_error = float(np.dot(difference, difference))
        rows.append(
            {
                "prediction_file": relative(file_path),
                "prediction_sample": prediction_id,
                "truth_sample": truth_id,
                "fields": num_fields,
                "timesteps": timesteps,
                "nodes": int(prediction.shape[2]),
                "values": int(prediction_values.size),
                "relative_l2": float(np.linalg.norm(difference) / denominator),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_absolute_error": float(np.max(np.abs(difference))),
                "r2": float(1.0 - squared_error / total_variance) if total_variance > 0 else None,
            }
        )
    if not rows:
        raise ValueError("Matched files were found, but none had compatible field arrays. Check the reported field rows and node count.")

    metric_names = ("relative_l2", "mae", "rmse", "max_absolute_error", "r2")
    aggregate: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=np.float64)
        if not values.size:
            continue
        aggregate[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    report_id = uuid.uuid4().hex[:12]
    report_dir = RUNTIME_ROOT / "evaluation" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "id": report_id,
        "created_at": utc_now(),
        "prediction_path": relative(prediction_path),
        "truth_path": relative(truth_path),
        "prediction_start": prediction_start,
        "truth_start": truth_start,
        "num_fields": num_fields,
        "evaluated_samples": len(rows),
        "skipped": skipped,
        "aggregate": aggregate,
        "per_sample_csv": relative(csv_path),
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = relative(report_path)
    return report


def comparison_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect selected comparison CSVs so the UI never guesses columns."""

    raw_paths = [str(item).strip() for item in payload.get("csv_paths", []) if str(item).strip()]
    if not raw_paths:
        return {"sources": [], "common_columns": [], "numeric_columns": [], "group_columns": []}
    if len(raw_paths) > 12:
        raise ValueError("Inspect at most 12 runs at once.")
    sources: list[dict[str, Any]] = []
    common_columns: list[str] | None = None
    numeric_by_source: list[set[str]] = []
    for raw_path in raw_paths:
        csv_path = safe_repo_path(raw_path, (SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT))
        if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
            raise ValueError(f"{raw_path!r} is not an existing CSV file.")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)[:500]
        if common_columns is None:
            common_columns = columns
        else:
            common_columns = [column for column in common_columns if column in columns]
        numeric = {
            column
            for column in columns
            if any(_finite_float(row.get(column)) is not None for row in rows)
        }
        numeric_by_source.append(numeric)
        sources.append({"path": relative(csv_path), "columns": columns, "rows_sampled": len(rows)})
    common = common_columns or []
    numeric_columns = [column for column in common if all(column in numeric for numeric in numeric_by_source)]
    return {
        "sources": sources,
        "common_columns": common,
        "numeric_columns": numeric_columns,
        "group_columns": [column for column in common if column not in numeric_columns],
    }


def optimization_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect one candidate CSV and return only columns backed by real values."""

    raw_path = str(payload.get("csv_path", "")).strip()
    if not raw_path:
        return {
            "path": "",
            "columns": [],
            "numeric_columns": [],
            "objective_columns": [],
            "identifier_columns": [],
            "rows_sampled": 0,
        }
    csv_path = safe_repo_path(
        raw_path,
        (SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT),
    )
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise ValueError("Optimization schema requires an existing output CSV file.")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)[:500]
    if not columns:
        raise ValueError("Optimization CSV has no header columns.")
    numeric_counts = {
        column: sum(_finite_float(row.get(column)) is not None for row in rows)
        for column in columns
    }
    numeric_columns = [column for column in columns if numeric_counts[column] > 0]
    identifier_names = {
        "id", "index", "row", "sample", "sample_id", "candidate", "candidate_id",
        "case", "case_id", "model", "run", "step", "epoch", "iteration", "time",
        "timestamp", "seed", "fold",
        "fields", "timesteps", "nodes", "points", "elements", "values", "count",
        "rows", "columns", "batch", "batch_size",
    }
    identifier_columns = [
        column for column in columns
        if column.casefold() in identifier_names
        or column.casefold().endswith(("_id", "_index", "_sample", "_count"))
    ]
    objective_columns = [
        column for column in numeric_columns if column not in identifier_columns
    ]
    return {
        "path": relative(csv_path),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "objective_columns": objective_columns,
        "identifier_columns": identifier_columns,
        "numeric_counts": numeric_counts,
        "rows_sampled": len(rows),
    }


def run_model_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    raw_paths = payload.get("csv_paths")
    if not raw_paths:
        single = str(payload.get("csv_path", "")).strip()
        raw_paths = [single] if single else []
    raw_paths = [str(item).strip() for item in raw_paths if str(item).strip()]
    if not raw_paths:
        raise ValueError("Model comparison requires at least one existing CSV file.")
    if len(raw_paths) > 12:
        raise ValueError("Compare at most 12 runs at once.")

    metric = str(payload.get("metric", "")).strip()
    group_column = str(payload.get("group_column", "model")).strip() or "model"
    direction = str(payload.get("direction", "min")).strip().lower()
    if not metric:
        raise ValueError("Select a numeric metric column.")
    if direction not in {"min", "max"}:
        raise ValueError("Comparison direction must be min or max.")

    multi_run = len(raw_paths) > 1
    sources: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    total_rows = 0
    for raw_path in raw_paths:
        csv_path = safe_repo_path(raw_path, (SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT))
        if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
            raise ValueError(f"{raw_path!r} is not an existing CSV file.")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        if not raw_rows:
            raise ValueError(f"{relative(csv_path)} has no rows.")
        if metric not in raw_rows[0]:
            raise ValueError(f"{relative(csv_path)} must contain a {metric!r} column.")
        has_group_column = group_column in raw_rows[0]
        # A run's own CSV may have no explicit group/model column (e.g. a
        # per-sample evaluation output) - fall back to naming every one of its
        # rows after the run itself so multiple runs stay distinguishable.
        run_label = csv_path.stem
        numeric_in_source = 0
        for index, row in enumerate(raw_rows):
            value = _finite_float(row.get(metric))
            if value is None:
                continue
            numeric_in_source += 1
            group_value = row[group_column] if has_group_column and row.get(group_column) else str(index)
            name = f"{run_label} · {group_value}" if multi_run else group_value
            ranked.append({
                "index": index,
                "name": name,
                "value": value,
                "row": row,
                "source": relative(csv_path),
                "run": run_label,
            })
        sources.append({
            "path": relative(csv_path),
            "run": run_label,
            "rows": len(raw_rows),
            "numeric_rows": numeric_in_source,
            "has_group_column": has_group_column,
        })
        total_rows += len(raw_rows)
    if not ranked:
        raise ValueError(f"Column {metric!r} has no numeric values in any selected CSV.")
    ranked.sort(key=lambda item: item["value"], reverse=direction == "max")
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    report_id = uuid.uuid4().hex[:12]
    report = {
        "id": report_id,
        "created_at": utc_now(),
        "csv_path": sources[0]["path"],
        "sources": sources,
        "runs": len(sources),
        "group_column": group_column,
        "metric": metric,
        "direction": direction,
        "rows": total_rows,
        "numeric_rows": len(ranked),
        "best": ranked[0],
        "ranked": ranked[:200],
    }
    report_dir = RUNTIME_ROOT / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = relative(report_path)
    return report


def export_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    source = safe_repo_path(str(payload.get("path", "")), (SUITE_ROOT,))
    if not source.exists():
        raise ValueError("Selected export source does not exist.")
    export_root = RUNTIME_ROOT / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and (source == export_root or source in export_root.parents):
        raise ValueError("Cannot export a directory that contains the Studio export destination.")
    label = slug(str(payload.get("label", source.stem)), source.stem)
    token = uuid.uuid4().hex[:8]
    if source.is_dir():
        destination_base = export_root / f"{label}-{token}"
        destination = Path(shutil.make_archive(str(destination_base), "zip", root_dir=source))
    else:
        destination = export_root / f"{label}-{token}{source.suffix}"
        shutil.copy2(source, destination)
    browser_path = "/" + destination.resolve().relative_to(FRONTEND_ROOT).as_posix()
    return {
        "ok": True,
        "source": relative(source),
        "path": relative(destination),
        "browser_path": browser_path,
        "size": destination.stat().st_size,
        "created_at": utc_now(),
    }


def run_optimization(payload: dict[str, Any]) -> dict[str, Any]:
    csv_path = safe_repo_path(
        str(payload.get("csv_path", "")),
        (SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT),
    )
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise ValueError("Optimization requires an existing output CSV file.")
    objectives = [item.strip() for item in str(payload.get("objectives", "")).split(",") if item.strip()]
    if not objectives:
        raise ValueError("Provide at least one objective column.")
    directions_raw = [item.strip().lower() for item in str(payload.get("directions", "")).split(",") if item.strip()]
    directions = directions_raw or ["min"] * len(objectives)
    if len(directions) == 1 and len(objectives) > 1:
        directions *= len(objectives)
    if len(directions) != len(objectives) or any(item not in {"min", "max"} for item in directions):
        raise ValueError("Directions must contain one min/max entry per objective.")
    constraints = [
        parse_constraint(item)
        for item in str(payload.get("constraints", "")).split(";")
        if item.strip()
    ]
    top_k = max(1, min(int(payload.get("top_k", 10)), 200))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Optimization CSV has no data rows.")
    missing = [name for name in objectives if name not in rows[0]]
    missing.extend(name for name, _, _ in constraints if name not in rows[0])
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(set(missing)))}")

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        objective_values = [_finite_float(row.get(name)) for name in objectives]
        constraint_values = {name: _finite_float(row.get(name)) for name, _, _ in constraints}
        if any(value is None for value in objective_values) or any(value is None for value in constraint_values.values()):
            continue
        feasible = True
        violations = []
        for name, operator, threshold in constraints:
            value = constraint_values[name]
            ok = (
                value <= threshold if operator == "<=" else
                value >= threshold if operator == ">=" else
                value < threshold if operator == "<" else
                value > threshold
            )
            if not ok:
                feasible = False
                violations.append({"column": name, "value": value, "operator": operator, "threshold": threshold})
        candidates.append(
            {
                "index": index,
                "id": row.get("id") or row.get("sample") or row.get("candidate") or row.get("model") or str(index),
                "objectives": dict(zip(objectives, objective_values)),
                "values": objective_values,
                "feasible": feasible,
                "violations": violations,
                "row": row,
            }
        )
    feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        comparisons = []
        for index, direction in enumerate(directions):
            l_value, r_value = left["values"][index], right["values"][index]
            comparisons.append((l_value <= r_value, l_value < r_value) if direction == "min" else (l_value >= r_value, l_value > r_value))
        return all(item[0] for item in comparisons) and any(item[1] for item in comparisons)

    pareto = [
        candidate
        for candidate in feasible_candidates
        if not any(dominates(other, candidate) for other in feasible_candidates if other is not candidate)
    ]
    for candidate in pareto:
        candidate["crowding"] = 0.0
    if len(pareto) > 2:
        for objective_index in range(len(objectives)):
            ordered = sorted(pareto, key=lambda candidate: candidate["values"][objective_index])
            ordered[0]["crowding"] = ordered[-1]["crowding"] = math.inf
            low = ordered[0]["values"][objective_index]
            high = ordered[-1]["values"][objective_index]
            if high == low:
                continue
            for position in range(1, len(ordered) - 1):
                ordered[position]["crowding"] += (
                    ordered[position + 1]["values"][objective_index]
                    - ordered[position - 1]["values"][objective_index]
                ) / (high - low)
    selected = sorted(pareto, key=lambda candidate: candidate.get("crowding", 0.0), reverse=True)[:top_k]
    report_id = uuid.uuid4().hex[:12]
    report = {
        "id": report_id,
        "created_at": utc_now(),
        "csv_path": relative(csv_path),
        "objectives": dict(zip(objectives, directions)),
        "constraints": [
            {"column": name, "operator": operator, "threshold": threshold}
            for name, operator, threshold in constraints
        ],
        "rows": len(rows),
        "numeric_candidates": len(candidates),
        "feasible": len(feasible_candidates),
        "pareto": len(pareto),
        "selected": [
            {
                "id": candidate["id"],
                "index": candidate["index"],
                "objectives": candidate["objectives"],
                "crowding": None if math.isinf(candidate.get("crowding", 0.0)) else candidate.get("crowding", 0.0),
                "row": candidate["row"],
            }
            for candidate in selected
        ],
    }
    report_dir = RUNTIME_ROOT / "optimization"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = relative(report_path)
    return report
