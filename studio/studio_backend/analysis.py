"""Real numeric analysis over existing repository files: evaluation, comparison,
optimization, and export. Every function reads actual HDF5/CSV rows and writes
its report under studio/runtime/; nothing here invents a metric or score.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from studio_backend.paths import FRONTEND_ROOT, RUNTIME_ROOT, SUITE_ROOT, relative, safe_repo_path, slug, utc_now
from studio_backend.paths import result_roots


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


HDF5_SUFFIXES = {".h5", ".hdf5"}
EVALUATION_FILE_LIMIT = 512
_PREDICTION_ARRAYS = (
    "Y_pred", "predictions", "prediction", "arrays/predictions",
    "Y", "targets", "arrays/targets",
)
_TRUTH_ARRAYS = (
    "Y", "Y_true", "targets", "arrays/targets",
    "predictions", "Y_pred", "arrays/predictions",
)


def _evaluation_imports():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise ValueError("Field evaluation requires h5py and numpy.") from exc
    return h5py, np


def _decode_hdf5(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _hdf5_names(handle: Any, key: str) -> list[str]:
    """Read a small string name vector without materializing numeric arrays."""

    sources = [handle]
    metadata = handle.get("metadata") if hasattr(handle, "get") else None
    if metadata is not None:
        sources.append(metadata)
    for source in sources:
        try:
            if key in source.attrs:
                raw = source.attrs[key]
            elif key in source:
                raw = source[key][()]
            else:
                continue
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if isinstance(raw, (bytes, str)):
            return [_decode_hdf5(raw)]
        try:
            return [_decode_hdf5(item) for item in raw]
        except TypeError:
            return [_decode_hdf5(raw)]
    return []


def _field_names(names: list[str], count: int) -> list[str]:
    clean = [str(name).strip() for name in names[:count]]
    clean += [f"field {index}" for index in range(len(clean), count)]
    return clean


def _hdf5_source(raw: Any, label: str) -> tuple[Path, list[Path], bool]:
    path = safe_repo_path(str(raw or ""), result_roots())
    if not path.exists():
        raise ValueError(f"{label} path does not exist.")
    if path.is_file():
        if path.suffix.lower() not in HDF5_SUFFIXES:
            raise ValueError(f"{label} must be an HDF5 file or a directory containing HDF5 files.")
        return path, [path], False
    candidates = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in HDF5_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"{label} directory contains no HDF5 files.")
    return path, candidates[:EVALUATION_FILE_LIMIT], len(candidates) > EVALUATION_FILE_LIMIT


def _sample_aliases(path: Path, sample_id: str, handle: Any) -> list[str]:
    aliases = [str(sample_id), path.stem]
    declared = handle.attrs.get("sample_id")
    if declared is not None:
        aliases.append(_decode_hdf5(declared))
    match = re.search(r"sample[_-]?(\d+)", path.stem, re.IGNORECASE)
    if match:
        aliases.append(match.group(1))
    return list(dict.fromkeys(alias for alias in aliases if alias))


# The shared mesh HDF5 contract (docs/reference/DATASET_FORMAT.md): rows 0:3 of
# nodal_data are the reference coordinates, rows 3:3+input_var the state the
# model predicts, and any cond_var rows follow.
MESH_COORDINATE_ROWS = 3


# The trailing row every mesh rollout writer appends: a per-node integer part /
# node-type label, copied through unchanged exactly like the coordinates. It is
# categorical, it is never predicted, and pairing it against a displacement field
# is meaningless -- but with no declared output_var it looked like just another
# state row and was offered for scoring.
_NODE_TYPE_NAMES = frozenset({"part no.", "part_no", "part", "node_type", "nodetype", "node type", "part id", "part_id"})


def _is_node_type_name(name: str) -> bool:
    return str(name).strip().lower() in _NODE_TYPE_NAMES


def _mesh_target_indices(
    handle: Any,
    field_count: int,
    row_source: str = "nodal_data",
    names: list[str] | None = None,
) -> tuple[list[int], str]:
    """Rows a mesh prediction may be scored on, per the suite's HDF5 contract.

    The state rows are what a model predicts, so they are the targets; the
    reference coordinates never are -- every rollout copies them in verbatim,
    so scoring them yields zero error and R^2 = 1 for any model whatsoever.

    The previous derivation assumed an [inputs | conditions | outputs] layout,
    which is not the suite's. On ex9 (builder_input_var 4, builder_output_var 4,
    seven rows) it computed rows 4:8, overflowed, returned nothing, and left
    every row -- coordinates included -- as a candidate. That is how a pipeline
    came to report every metric exactly 0.0 on 87 held-out samples.
    """
    # `nodal_field` is SimulGen-VAE's reconstruct output: num_var physical rows,
    # no coordinates, whatever num_var happens to be (ex3 uses 4). Only
    # `nodal_data` follows the coordinates-first contract, so the array name --
    # not the row count -- is what decides. A row-count guess silently dropped
    # three of four channels on any 4-row coordinate-less file.
    if row_source == "nodal_field" or field_count <= MESH_COORDINATE_ROWS:
        return list(range(field_count)), "every row (no reference-coordinate rows)"
    try:
        output_count = int(handle.attrs.get("builder_output_var", handle.attrs.get("output_var", 0)) or 0)
        input_count = int(handle.attrs.get("builder_input_var", handle.attrs.get("input_var", 0)) or 0)
    except (TypeError, ValueError):
        output_count = input_count = 0
    state_count = output_count or input_count
    if state_count > 0 and MESH_COORDINATE_ROWS + state_count <= field_count:
        return (
            list(range(MESH_COORDINATE_ROWS, MESH_COORDINATE_ROWS + state_count)),
            "state rows after the reference coordinates (builder output_var)",
        )
    tail = list(range(MESH_COORDINATE_ROWS, field_count))
    if names and tail and _is_node_type_name(names[tail[-1]]):
        return tail[:-1], "state rows after the reference coordinates, minus the node-type row"
    return tail, "all rows after the reference coordinates"


def _carries_coordinate_rows(source: dict[str, Any]) -> bool:
    """Whether this mesh source's rows begin with the reference coordinates.

    Not every mesh-shaped file does. SimulGen-VAE's `reconstruct` writes only the
    `num_var` physical channels to `data/{id}/nodal_field` -- two rows on ex9, no
    coordinates at all -- so a blanket "rows 0:3 are coordinates" rule would
    discard the only channels it produces and report "no safe field mapping" for
    a perfectly scoreable reconstruction.

    Declared names settle it when present. Otherwise the array name does: only
    `nodal_data` follows the coordinates-first contract, while `nodal_field`
    carries physical rows alone. Falling back to a row count instead was wrong
    for any coordinate-less file with more than three rows -- SimulGen-VAE's ex3
    reconstruct writes four.
    """
    names = [str(name).strip().lower() for name in (source.get("field_names") or [])]
    declared = bool(source.get("declared_field_names")) and len(names) >= MESH_COORDINATE_ROWS
    if declared:
        if all(_is_coordinate_name(name) for name in names[:MESH_COORDINATE_ROWS]):
            return True
        # Declared names that are NOT recognisable coordinates do not prove the
        # rows are physical: a writer free to name a row "x" is equally free to
        # name it "disp_1". Fall through to the array-name contract rather than
        # concluding there are no coordinate rows -- concluding wrongly here is
        # what puts a row scoring R2 = 1 into the report.
        if source.get("row_source") == "nodal_field":
            return False
        return int(source.get("field_count") or 0) > MESH_COORDINATE_ROWS
    if source.get("row_source") == "nodal_field":
        return False
    return int(source.get("field_count") or 0) > MESH_COORDINATE_ROWS


# Reference-coordinate row names seen across the suite's writers. The check used
# to be `name.endswith("_coord")` alone, so a file naming its rows x/y/z -- the
# obvious spelling, and what several exporters emit -- disabled both the filter
# and the coordinate-only guard, and the coordinates were scored.
_COORDINATE_NAMES = frozenset({
    "x", "y", "z",
    "x_coord", "y_coord", "z_coord",
    "coord_x", "coord_y", "coord_z",
    "node_x", "node_y", "node_z",
    "pos_x", "pos_y", "pos_z",
    "mesh_x", "mesh_y", "mesh_z",
})


def _is_coordinate_name(name: str) -> bool:
    clean = str(name).strip().lower()
    return clean.endswith("_coord") or clean in _COORDINATE_NAMES


def _is_coordinate_row(source: dict[str, Any], index: int) -> bool:
    """True for a mesh source's reference-coordinate rows, by name or position."""
    if source.get("kind") != "mesh":
        return False
    names = source.get("field_names") or []
    if index < len(names) and _is_coordinate_name(names[index]):
        return True
    return index < MESH_COORDINATE_ROWS and _carries_coordinate_rows(source)


def _inspect_hdf5_file(path: Path, role: str) -> dict[str, Any]:
    h5py, np = _evaluation_imports()
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if isinstance(data, h5py.Group):
            records: list[dict[str, Any]] = []
            counts: list[int] = []
            for sample_id in sorted(data.keys(), key=lambda value: (not str(value).isdigit(), str(value))):
                group = data[sample_id]
                name = "nodal_data" if "nodal_data" in group else "nodal_field" if "nodal_field" in group else ""
                if not name:
                    continue
                dataset = group[name]
                shape = [int(value) for value in dataset.shape]
                if dataset.ndim == 3:
                    counts.append(shape[0])
                records.append({
                    "id": str(sample_id),
                    "aliases": _sample_aliases(path, str(sample_id), handle),
                    "file": relative(path),
                    "dataset": f"data/{sample_id}/{name}",
                    "shape": shape,
                    "rank_ok": dataset.ndim == 3,
                    "explicit_id": True,
                })
            count = counts[0] if counts else 0
            # Every record in one file uses the same array name; `name` still
            # holds the last one seen and they cannot differ within a contract.
            row_source = name
            names = _field_names(_hdf5_names(handle, "feature_names"), count)
            target_indices, target_basis = _mesh_target_indices(handle, count, row_source, names)
            return {
                "contract": "mesh_state",
                "kind": "mesh",
                "records": records,
                "field_count": count,
                "field_names": names,
                "declared_field_names": bool(_hdf5_names(handle, "feature_names")),
                "row_source": row_source,
                "target_indices": target_indices,
                "target_basis": target_basis,
                "arrays": [{"path": "data/{sample}/nodal_data", "shape": records[0]["shape"] if records else []}],
                "embedded_truth": False,
            }

        nodes = handle.get("nodes")
        predicted_name = next(
            (name for name in ("predicted_denorm", "predicted_norm") if isinstance(nodes, h5py.Group) and name in nodes),
            None,
        )
        if isinstance(nodes, h5py.Group) and "pos" in nodes and predicted_name:
            target_name = next((name for name in ("target_denorm", "target_norm") if name in nodes), None)
            predicted = nodes[predicted_name]
            count = int(predicted.shape[1]) if predicted.ndim == 2 else 1
            sample_id = _decode_hdf5(handle.attrs.get("sample_id", path.stem))
            record = {
                "id": sample_id,
                "aliases": _sample_aliases(path, sample_id, handle),
                "file": relative(path),
                "dataset": f"nodes/{predicted_name}",
                "truth_dataset": f"nodes/{target_name}" if target_name else "",
                "shape": [int(value) for value in predicted.shape],
                "truth_shape": [int(value) for value in nodes[target_name].shape] if target_name else [],
                "rank_ok": predicted.ndim in {1, 2},
                "explicit_id": True,
            }
            names = _field_names(_hdf5_names(handle, "output_names"), count)
            return {
                "contract": "native_inference_result",
                "kind": "mesh",
                "records": [record],
                "field_count": count,
                "field_names": names,
                "declared_field_names": bool(_hdf5_names(handle, "output_names")),
                "target_indices": list(range(count)),
                "target_basis": "paired inference output channels",
                "arrays": [
                    {"path": record["dataset"], "shape": record["shape"]},
                    *([{"path": record["truth_dataset"], "shape": record["truth_shape"]}] if target_name else []),
                ],
                "embedded_truth": bool(target_name),
            }

        candidates = _PREDICTION_ARRAYS if role == "prediction" else _TRUTH_ARRAYS
        array_name = next(
            (
                name for name in candidates
                if name in handle and isinstance(handle[name], h5py.Dataset)
                and np.issubdtype(handle[name].dtype, np.number)
            ),
            None,
        )
        if array_name is None:
            return {
                "contract": "unsupported",
                "kind": "unsupported",
                "records": [],
                "field_count": 0,
                "field_names": [],
                "declared_field_names": False,
                "target_indices": [],
                "target_basis": "",
                "arrays": [],
                "embedded_truth": False,
            }
        dataset = handle[array_name]
        row_count = int(dataset.shape[0]) if dataset.ndim else 1
        field_count = int(dataset.shape[1]) if dataset.ndim >= 2 else 1
        id_values = _hdf5_names(handle, "sample_ids")
        explicit_ids = len(id_values) == row_count
        records = [
            {
                "id": id_values[index] if explicit_ids else str(index),
                "aliases": [id_values[index] if explicit_ids else str(index)],
                "file": relative(path),
                "dataset": array_name,
                "row": index,
                "shape": [int(value) for value in dataset.shape],
                "rank_ok": dataset.ndim >= 1,
                "explicit_id": explicit_ids,
            }
            for index in range(row_count)
        ]
        declared_names = _hdf5_names(handle, "output_names") or _hdf5_names(handle, "feature_names")
        embedded_name = next(
            (
                name for name in ("Y_true", "targets", "arrays/targets")
                if role == "prediction" and name in handle and isinstance(handle[name], h5py.Dataset)
                and list(handle[name].shape) == list(dataset.shape)
            ),
            None,
        )
        if embedded_name:
            for record in records:
                record["truth_dataset"] = embedded_name
                record["truth_shape"] = record["shape"]
        contract = "operator_grid" if dataset.ndim >= 3 and "common/query_xy" in handle else "table"
        return {
            "contract": contract,
            "kind": "table",
            "records": records,
            "field_count": field_count,
            "field_names": _field_names(declared_names, field_count),
            "declared_field_names": bool(declared_names),
            "target_indices": list(range(field_count)),
            "target_basis": "output array columns",
            "arrays": [
                {"path": array_name, "shape": [int(value) for value in dataset.shape]},
                *([{"path": embedded_name, "shape": [int(value) for value in handle[embedded_name].shape]}] if embedded_name else []),
            ],
            "embedded_truth": bool(embedded_name),
        }


def _inspect_hdf5_source(path: Path, files: list[Path], truncated: bool, role: str) -> dict[str, Any]:
    inspections: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for file_path in files:
        try:
            inspections.append(_inspect_hdf5_file(file_path, role))
        except (OSError, ValueError) as exc:
            unreadable.append(f"{relative(file_path)}: {exc}")
    contracts = {item["contract"] for item in inspections}
    primary = inspections[0] if inspections else {
        "contract": "unsupported", "kind": "unsupported", "field_count": 0,
        "field_names": [], "declared_field_names": False, "row_source": "", "target_indices": [],
        "target_basis": "", "embedded_truth": False,
    }
    records = [record for item in inspections for record in item["records"]]
    arrays: list[dict[str, Any]] = []
    for item in inspections:
        for array in item["arrays"]:
            if array not in arrays:
                arrays.append(array)
    field_counts = {int(item["field_count"]) for item in inspections if item["field_count"]}
    mixed = len(contracts) > 1 or len(field_counts) > 1
    return {
        "path": relative(path),
        "kind": "mixed" if mixed else primary["kind"],
        "contract": "mixed" if mixed else primary["contract"],
        "files": len(files),
        "samples": len(records),
        "field_count": int(primary["field_count"]),
        "field_names": list(primary["field_names"]),
        "declared_field_names": bool(primary["declared_field_names"]),
        # Carried through so the coordinate rules see which array the rows came
        # from; without it they fell back to the row-count guess again.
        "row_source": primary.get("row_source", ""),
        "target_indices": list(primary["target_indices"]),
        "target_basis": primary["target_basis"],
        "arrays": arrays,
        "sample_records": records,
        "embedded_truth": bool(inspections) and all(item["embedded_truth"] for item in inspections),
        "truncated": truncated,
        "unreadable": unreadable[:20],
    }


def _match_sample_records(prediction: dict[str, Any], truth: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    predicted = prediction["sample_records"]
    actual = truth["sample_records"]
    if not predicted or not actual:
        return [], "none"
    if prediction["kind"] == "table" and truth["kind"] == "table":
        explicit = all(item["explicit_id"] for item in predicted) and all(item["explicit_id"] for item in actual)
        if not explicit:
            if len(predicted) != len(actual):
                return [], "position-count-mismatch"
            return list(zip(predicted, actual)), "position"
    available = list(actual)
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left in predicted:
        left_aliases = set(left["aliases"])
        index = next(
            (index for index, right in enumerate(available) if left["id"] == right["id"]),
            None,
        )
        if index is None:
            index = next(
                (index for index, right in enumerate(available) if left_aliases.intersection(right["aliases"])),
                None,
            )
        if index is not None:
            matched.append((left, available.pop(index)))
    if not matched and len(predicted) == len(actual) == 1:
        return [(predicted[0], actual[0])], "single"
    return matched, "id"


def _recommended_field_pairs(prediction: dict[str, Any], truth: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[str]]:
    warnings: list[str] = []
    prediction_indices = prediction["target_indices"] or list(range(prediction["field_count"]))
    truth_indices = truth["target_indices"] or list(range(truth["field_count"]))
    # Coordinates are never a prediction. Without this, a rollout whose physical
    # channels were named differently from the truth file's (the native writers
    # used to hardcode NASA-CRM names) matched *only* x/y/z_coord by name and
    # scored a perfect model.
    prediction_indices = [index for index in prediction_indices if not _is_coordinate_row(prediction, index)]
    truth_indices = [index for index in truth_indices if not _is_coordinate_row(truth, index)]
    prediction_names = prediction["field_names"]
    truth_names = truth["field_names"]
    if prediction["declared_field_names"] and truth["declared_field_names"]:
        truth_by_name = {
            truth_names[index]: index for index in truth_indices
            if truth_names.count(truth_names[index]) == 1
        }
        pairs = [
            {
                "name": prediction_names[index],
                "prediction_index": index,
                "truth_index": truth_by_name[prediction_names[index]],
                "prediction_name": prediction_names[index],
                "truth_name": prediction_names[index],
            }
            for index in prediction_indices
            if prediction_names[index] in truth_by_name
            and prediction_names.count(prediction_names[index]) == 1
        ]
        if pairs:
            return pairs, "declared field names", warnings
    if len(prediction_indices) == len(truth_indices) and prediction_indices:
        warnings.append(
            "Field names were absent or did not match; equal-sized output fields are mapped by position. Confirm this mapping before scoring."
        )
        pairs = []
        for prediction_index, truth_index in zip(prediction_indices, truth_indices):
            name = truth_names[truth_index] if truth["declared_field_names"] else prediction_names[prediction_index]
            pairs.append({
                "name": name,
                "prediction_index": prediction_index,
                "truth_index": truth_index,
                "prediction_name": prediction_names[prediction_index],
                "truth_name": truth_names[truth_index],
            })
        return pairs, "equal output count by position", warnings
    return [], "", warnings


def _schema_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    prediction_path, prediction_files, prediction_truncated = _hdf5_source(payload.get("prediction_path"), "Prediction")
    truth_path, truth_files, truth_truncated = _hdf5_source(payload.get("truth_path"), "Ground truth")
    prediction = _inspect_hdf5_source(prediction_path, prediction_files, prediction_truncated, "prediction")
    truth = _inspect_hdf5_source(truth_path, truth_files, truth_truncated, "truth")
    errors: list[str] = []
    warnings: list[str] = []
    if prediction["contract"] in {"unsupported", "mixed"}:
        errors.append(
            "Prediction HDF5 contract is unsupported or mixed. Use mesh data/{sample}/nodal_data, MLP Y_pred/predictions, operator arrays/predictions, or native nodes/predicted_* arrays."
        )
    if truth["contract"] in {"unsupported", "mixed"}:
        errors.append(
            "Ground-truth HDF5 contract is unsupported or mixed. Use mesh data/{sample}/nodal_data or table Y/targets arrays."
        )
    if prediction["unreadable"]:
        warnings.append(f"{len(prediction['unreadable'])} prediction HDF5 file(s) could not be read.")
    if truth["unreadable"]:
        warnings.append(f"{len(truth['unreadable'])} truth HDF5 file(s) could not be read.")
    if prediction["truncated"] or truth["truncated"]:
        warnings.append(f"Contract inspection is limited to {EVALUATION_FILE_LIMIT} files per source.")

    embedded = prediction["embedded_truth"]
    matched, strategy = _match_sample_records(prediction, truth)
    if not matched and not embedded:
        if strategy == "position-count-mismatch":
            errors.append(
                "Table row counts differ and neither file declares sample_ids; add matching sample_ids or select arrays with equal rows."
            )
        else:
            errors.append("No prediction samples match ground truth by sample ID.")

    if not embedded and prediction["kind"] != truth["kind"]:
        errors.append(
            f"Prediction contract {prediction['contract']} cannot be compared directly with truth contract {truth['contract']}."
        )

    if embedded:
        field_pairs = [
            {
                "name": prediction["field_names"][index],
                "prediction_index": index,
                "truth_index": index,
                "prediction_name": prediction["field_names"][index],
                "truth_name": prediction["field_names"][index],
            }
            for index in (prediction["target_indices"] or range(prediction["field_count"]))
        ]
        mapping_basis = "paired truth array stored beside each prediction"
        mode = "embedded"
        warnings.append(
            "The prediction files contain shape-aligned truth arrays; scoring will use those exact paired arrays. The selected truth file is retained as provenance."
        )
    else:
        field_pairs, mapping_basis, field_warnings = _recommended_field_pairs(prediction, truth)
        warnings.extend(field_warnings)
        mode = prediction["kind"]
        if not field_pairs and prediction["kind"] not in {"unsupported", "mixed"}:
            errors.append(
                "No safe field mapping was found. Select named prediction/truth fields explicitly; do not assume unequal channel layouts align."
            )

    incompatible_shapes = 0
    compatible_shapes = 0
    for left, right in matched:
        left_shape = left["shape"]
        right_shape = right["shape"]
        if prediction["kind"] == truth["kind"] == "mesh":
            ok = (
                left.get("rank_ok") and right.get("rank_ok")
                and len(left_shape) == len(right_shape) == 3
                and left_shape[1] == right_shape[1]
                and left_shape[2] == right_shape[2]
            )
        else:
            left_tail = left_shape[2:] if len(left_shape) >= 2 else []
            right_tail = right_shape[2:] if len(right_shape) >= 2 else []
            ok = left.get("rank_ok") and right.get("rank_ok") and left_tail == right_tail
        compatible_shapes += int(bool(ok))
        incompatible_shapes += int(not ok)
    if matched and not compatible_shapes and not embedded:
        errors.append(
            "Matched samples have incompatible value shapes (mesh timestep/node counts or table trailing dimensions differ)."
        )
    elif incompatible_shapes:
        warnings.append(f"{incompatible_shapes} matched sample(s) have incompatible shapes and will be skipped.")

    recommended = {
        "mode": mode,
        "prediction_array": prediction["arrays"][0]["path"] if prediction["arrays"] else "",
        "truth_array": (
            prediction["arrays"][1]["path"]
            if embedded and len(prediction["arrays"]) > 1
            else truth["arrays"][0]["path"] if truth["arrays"] else ""
        ),
        "field_pairs": field_pairs,
        "sample_strategy": "embedded" if embedded else strategy,
        "basis": mapping_basis,
        "confidence": "exact" if embedded or mapping_basis == "declared field names" else "confirm",
    }
    prediction_indices = [item["prediction_index"] for item in field_pairs]
    truth_indices = [item["truth_index"] for item in field_pairs]
    if prediction_indices and prediction_indices == list(range(min(prediction_indices), max(prediction_indices) + 1)):
        recommended["prediction_start"] = min(prediction_indices)
    if truth_indices and truth_indices == list(range(min(truth_indices), max(truth_indices) + 1)):
        recommended["truth_start"] = min(truth_indices)
    if prediction_indices and truth_indices:
        recommended["num_fields"] = len(field_pairs)

    schema = {
        "compatible": not errors,
        "errors": errors,
        "warnings": warnings,
        "prediction": prediction,
        "truth": truth,
        "sample_matching": {
            "strategy": "embedded" if embedded else strategy,
            "prediction_count": prediction["samples"],
            "truth_count": truth["samples"],
            "overlap_count": len(matched),
            "compatible_shape_count": compatible_shapes,
            "incompatible_shape_count": incompatible_shapes,
            "matched_ids": [left["id"] for left, _ in matched[:50]],
            "truncated": len(matched) > 50,
        },
        "recommended_mapping": recommended,
    }
    return schema, prediction_path, truth_path


def evaluation_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect two HDF5 sources and return an evidence-backed scoring contract."""

    schema, _, _ = _schema_payload(payload)
    return schema


def _resolved_field_pairs(payload: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    requested = payload.get("field_pairs")
    if not isinstance(requested, list) or not requested:
        return list(schema["recommended_mapping"]["field_pairs"])
    prediction = schema["prediction"]
    truth = schema["truth"]
    pairs: list[dict[str, Any]] = []
    for raw in requested:
        if not isinstance(raw, dict):
            raise ValueError("Each field_pairs entry must be an object.")
        prediction_name = str(raw.get("prediction_name", raw.get("name", ""))).strip()
        truth_name = str(raw.get("truth_name", raw.get("name", ""))).strip()
        try:
            prediction_index = (
                int(raw["prediction_index"])
                if "prediction_index" in raw
                else prediction["field_names"].index(prediction_name)
            )
            truth_index = (
                int(raw["truth_index"])
                if "truth_index" in raw
                else truth["field_names"].index(truth_name)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not resolve requested field pair {raw!r}.") from exc
        if not 0 <= prediction_index < prediction["field_count"]:
            raise ValueError(f"Prediction field index {prediction_index} is outside the available fields.")
        truth_limit = prediction["field_count"] if schema["recommended_mapping"]["mode"] == "embedded" else truth["field_count"]
        if not 0 <= truth_index < truth_limit:
            raise ValueError(f"Truth field index {truth_index} is outside the available fields.")
        pairs.append({
            "name": str(raw.get("name") or truth_name or prediction_name or f"field {len(pairs)}"),
            "prediction_index": prediction_index,
            "truth_index": truth_index,
            "prediction_name": prediction["field_names"][prediction_index],
            "truth_name": (
                prediction["field_names"][truth_index]
                if schema["recommended_mapping"]["mode"] == "embedded"
                else truth["field_names"][truth_index]
            ),
        })
    if len({item["prediction_index"] for item in pairs}) != len(pairs) or len({item["truth_index"] for item in pairs}) != len(pairs):
        raise ValueError("Field mapping cannot reuse a prediction or truth field index.")
    return pairs


def _metric_row(np: Any, prediction: Any, truth: Any, metadata: dict[str, Any]) -> dict[str, Any] | None:
    if prediction.shape != truth.shape:
        return None
    finite = np.isfinite(prediction) & np.isfinite(truth)
    if not finite.any():
        return None
    prediction_values = prediction[finite]
    truth_values = truth[finite]
    difference = prediction_values - truth_values
    denominator = max(float(np.linalg.norm(truth_values)), np.finfo(np.float64).eps)
    truth_centered = truth_values - truth_values.mean()
    total_variance = float(np.dot(truth_centered, truth_centered))
    squared_error = float(np.dot(difference, difference))
    return {
        **metadata,
        "values": int(prediction_values.size),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "r2": float(1.0 - squared_error / total_variance) if total_variance > 0 else None,
    }


def _record_path(record: dict[str, Any]) -> Path:
    return safe_repo_path(record["file"], result_roots())


def _evaluate_mesh(
    schema: dict[str, Any], field_pairs: list[dict[str, Any]], prediction_start: int,
    truth_start: int, num_fields: int, use_legacy_rows: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    h5py, np = _evaluation_imports()
    matched, _ = _match_sample_records(schema["prediction"], schema["truth"])
    if use_legacy_rows:
        field_pairs = [
            {
                "name": f"field {offset}",
                "prediction_index": prediction_start + offset,
                "truth_index": truth_start + offset,
            }
            for offset in range(num_fields)
        ]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for left, right in matched:
        with h5py.File(_record_path(left), "r") as prediction_handle, h5py.File(_record_path(right), "r") as truth_handle:
            prediction_dataset = prediction_handle[left["dataset"]]
            truth_dataset = truth_handle[right["dataset"]]
            prediction_indices = [item["prediction_index"] for item in field_pairs]
            truth_indices = [item["truth_index"] for item in field_pairs]
            if (
                not prediction_indices or max(prediction_indices) >= prediction_dataset.shape[0]
                or max(truth_indices) >= truth_dataset.shape[0]
            ):
                skipped.append(f"{left['file']}:{left['id']} field mapping is outside nodal_data")
                continue
            if (
                prediction_dataset.ndim != 3
                or truth_dataset.ndim != 3
                or prediction_dataset.shape[1:] != truth_dataset.shape[1:]
            ):
                skipped.append(
                    f"{left['file']}:{left['id']} has incompatible timestep/node shape "
                    f"{tuple(prediction_dataset.shape[1:])} vs {tuple(truth_dataset.shape[1:])}"
                )
                continue
            timesteps = int(prediction_dataset.shape[1])
            prediction = np.asarray(prediction_dataset[prediction_indices, :timesteps, :], dtype=np.float64)
            truth = np.asarray(truth_dataset[truth_indices, :timesteps, :], dtype=np.float64)
        row = _metric_row(np, prediction, truth, {
            "contract": "mesh_state",
            "prediction_file": left["file"],
            "prediction_sample": left["id"],
            "truth_sample": right["id"],
            "fields": len(field_pairs),
            "timesteps": timesteps,
            "nodes": int(prediction.shape[2]),
        })
        if row is None:
            skipped.append(f"{left['file']}:{left['id']} contains no finite, shape-aligned values")
        else:
            rows.append(row)
    return rows, skipped


def _evaluate_table(schema: dict[str, Any], field_pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    h5py, np = _evaluation_imports()
    matched, _ = _match_sample_records(schema["prediction"], schema["truth"])
    prediction_indices = [item["prediction_index"] for item in field_pairs]
    truth_indices = [item["truth_index"] for item in field_pairs]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for left, right in matched:
        with h5py.File(_record_path(left), "r") as prediction_handle, h5py.File(_record_path(right), "r") as truth_handle:
            prediction_dataset = prediction_handle[left["dataset"]]
            truth_dataset = truth_handle[right["dataset"]]
            if max(prediction_indices, default=-1) >= (prediction_dataset.shape[1] if prediction_dataset.ndim >= 2 else 1):
                skipped.append(f"{left['file']}:{left['id']} prediction field mapping is outside {left['dataset']}")
                continue
            if max(truth_indices, default=-1) >= (truth_dataset.shape[1] if truth_dataset.ndim >= 2 else 1):
                skipped.append(f"{right['file']}:{right['id']} truth field mapping is outside {right['dataset']}")
                continue
            if prediction_dataset.ndim == 1:
                prediction = np.asarray([prediction_dataset[left["row"]]], dtype=np.float64)
            else:
                prediction = np.asarray(prediction_dataset[left["row"], prediction_indices, ...], dtype=np.float64)
            if truth_dataset.ndim == 1:
                truth = np.asarray([truth_dataset[right["row"]]], dtype=np.float64)
            else:
                truth = np.asarray(truth_dataset[right["row"], truth_indices, ...], dtype=np.float64)
        row = _metric_row(np, prediction, truth, {
            "contract": schema["prediction"]["contract"],
            "prediction_file": left["file"],
            "prediction_sample": left["id"],
            "truth_sample": right["id"],
            "fields": len(field_pairs),
            "timesteps": 1,
            "nodes": int(prediction.size // max(1, len(field_pairs))),
        })
        if row is None:
            skipped.append(f"{left['file']}:{left['id']} contains no finite, shape-aligned values")
        else:
            rows.append(row)
    return rows, skipped


def _evaluate_embedded(schema: dict[str, Any], field_pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    h5py, np = _evaluation_imports()
    prediction_indices = [item["prediction_index"] for item in field_pairs]
    truth_indices = [item["truth_index"] for item in field_pairs]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in schema["prediction"]["sample_records"]:
        with h5py.File(_record_path(record), "r") as handle:
            prediction_dataset = handle[record["dataset"]]
            truth_dataset = handle[record["truth_dataset"]]
            if prediction_dataset.ndim == 1:
                prediction = np.asarray([prediction_dataset[()]], dtype=np.float64)
                truth = np.asarray([truth_dataset[()]], dtype=np.float64)
            elif schema["prediction"]["contract"] == "native_inference_result":
                prediction = np.asarray(prediction_dataset[:, prediction_indices].T, dtype=np.float64)
                truth = np.asarray(truth_dataset[:, truth_indices].T, dtype=np.float64)
            else:
                row_index = int(record.get("row", 0))
                prediction = np.asarray(prediction_dataset[row_index, prediction_indices, ...], dtype=np.float64)
                truth = np.asarray(truth_dataset[row_index, truth_indices, ...], dtype=np.float64)
        row = _metric_row(np, prediction, truth, {
            "contract": schema["prediction"]["contract"],
            "prediction_file": record["file"],
            "prediction_sample": record["id"],
            "truth_sample": f"embedded:{record['id']}",
            "fields": len(field_pairs),
            "timesteps": 1,
            "nodes": int(prediction.size // max(1, len(field_pairs))),
        })
        if row is None:
            skipped.append(f"{record['file']}:{record['id']} embedded truth is not shape-aligned or finite")
        else:
            rows.append(row)
    return rows, skipped


def run_field_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    """Score mesh, table, operator-grid, or paired native inference arrays."""

    _, np = _evaluation_imports()
    schema, prediction_path, truth_path = _schema_payload(payload)
    mode = schema["recommended_mapping"]["mode"]
    explicit_pairs = isinstance(payload.get("field_pairs"), list) and bool(payload["field_pairs"])
    use_legacy_rows = (
        mode == "mesh"
        and not explicit_pairs
        and any(key in payload for key in ("prediction_start", "truth_start", "num_fields"))
    )
    structural_errors = [
        message for message in schema["errors"]
        if not message.startswith("No safe field mapping was found.")
    ]
    if structural_errors:
        raise ValueError("Evaluation contract is incompatible: " + " ".join(structural_errors))
    mapping = schema["recommended_mapping"]
    if (
        not explicit_pairs
        and not use_legacy_rows
        and mapping.get("confidence") == "confirm"
        and not bool(payload.get("confirm_mapping"))
    ):
        raise ValueError(
            "The available fields can only be mapped by position. Inspect and explicitly confirm the field mapping before scoring."
        )
    field_pairs = _resolved_field_pairs(payload, schema)
    if not field_pairs and not use_legacy_rows:
        raise ValueError("Select at least one explicit prediction/truth field pair before scoring.")
    prediction_start = max(0, int(payload.get("prediction_start", 3)))
    truth_start = max(0, int(payload.get("truth_start", 3)))
    num_fields = max(1, int(payload.get("num_fields", 1)))
    # Last line of defence, independent of how the pairs were chosen (recommended,
    # explicit, or legacy rows): a score computed only on reference-coordinate
    # rows is not a score of the model. Refuse rather than report R^2 = 1.
    if mode == "mesh":
        only_coordinates = (
            (prediction_start + num_fields <= MESH_COORDINATE_ROWS and truth_start + num_fields <= MESH_COORDINATE_ROWS)
            if use_legacy_rows
            else bool(field_pairs) and all(
                _is_coordinate_row(schema["prediction"], int(pair["prediction_index"]))
                and _is_coordinate_row(schema["truth"], int(pair["truth_index"]))
                for pair in field_pairs
            )
        )
        if only_coordinates:
            raise ValueError(
                "Every paired field is a reference-coordinate row (nodal_data rows 0:3). Coordinates are copied "
                "into every rollout unchanged, so scoring them reports zero error for any model. Pair the physical "
                "state rows (3 onwards) instead -- if the prediction and truth files name them differently, "
                "select the pairs explicitly in the Evaluation workspace."
            )
    if mode == "embedded":
        rows, skipped = _evaluate_embedded(schema, field_pairs)
        truth_source = "embedded"
    elif mode == "table":
        rows, skipped = _evaluate_table(schema, field_pairs)
        truth_source = "selected"
    elif mode == "mesh":
        if schema["prediction"]["contract"] != "mesh_state" or schema["truth"]["contract"] != "mesh_state":
            raise ValueError("External mesh evaluation requires mesh-state HDF5 on both sides.")
        rows, skipped = _evaluate_mesh(
            schema, field_pairs, prediction_start, truth_start, num_fields, use_legacy_rows
        )
        if use_legacy_rows:
            field_pairs = [
                {"name": f"field {offset}", "prediction_index": prediction_start + offset, "truth_index": truth_start + offset}
                for offset in range(num_fields)
            ]
        truth_source = "selected"
    else:
        raise ValueError("Evaluation contract is unsupported.")
    if not rows:
        detail = f" First issue: {skipped[0]}" if skipped else ""
        raise ValueError("Matched samples were found, but none had compatible finite arrays." + detail)

    aggregate: dict[str, Any] = {}
    for metric in ("relative_l2", "mae", "rmse", "max_absolute_error", "r2"):
        values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=np.float64)
        if values.size:
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
        "contract": schema["prediction"]["contract"],
        "truth_source": truth_source,
        "field_pairs": field_pairs,
        "prediction_start": prediction_start if use_legacy_rows else None,
        "truth_start": truth_start if use_legacy_rows else None,
        "num_fields": len(field_pairs),
        "evaluated_samples": len(rows),
        "skipped": skipped,
        "aggregate": aggregate,
        "per_sample_csv": relative(csv_path),
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = relative(report_path)
    return report


# Numeric columns that describe the shape of a row rather than a model's
# quality. Shared by the comparison and optimization schemas.
NON_METRIC_COLUMNS = frozenset({
    "fields", "timesteps", "nodes", "points", "elements", "values", "count",
    "rows", "columns", "batch", "batch_size", "index", "row", "step", "epoch",
    "iteration", "seed", "fold", "prediction_sample", "truth_sample",
    "sample", "sample_id", "candidate_id", "case_id",
})


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
        csv_path = safe_repo_path(raw_path, result_roots())
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
        sources.append({
            "path": relative(csv_path),
            "columns": columns,
            "rows_sampled": len(rows),
            # Both are written by run_field_evaluation into every per-sample CSV,
            # and together they are the only evidence available here about WHAT
            # was scored: the array contract, and which held-out samples.
            "contract": next(
                (str(row.get("contract")) for row in rows if str(row.get("contract", "")).strip()),
                "",
            ),
            "samples": sorted({
                str(row.get("truth_sample") or row.get("prediction_sample") or "").strip()
                for row in rows
                if str(row.get("truth_sample") or row.get("prediction_sample") or "").strip()
            }),
        })
    common = common_columns or []
    # Studio writes the row's shape bookkeeping (fields/timesteps/nodes/values)
    # into every per_sample_metrics.csv. They are numeric, so they were offered
    # as rankable metrics -- ranking two models by "nodes" is meaningless and
    # the run button enabled it happily.
    numeric_columns = [
        column for column in common
        if all(column in numeric for numeric in numeric_by_source)
        and column.casefold() not in NON_METRIC_COLUMNS
    ]
    warnings = _comparison_qualification_warnings(sources)
    # The per-source sample sets exist only to answer "is this the same held-out
    # set". Shipping a 900-entry list per run to the browser on every selection
    # change is not worth the answer.
    for source in sources:
        source["sample_count"] = len(source.pop("samples", ()))
    return {
        "sources": sources,
        "common_columns": common,
        "numeric_columns": numeric_columns,
        "group_columns": [column for column in common if column not in numeric_columns],
        "warnings": warnings,
    }


def _comparison_qualification_warnings(sources: list[dict[str, Any]]) -> list[str]:
    """Check the comparison's own stated precondition instead of asserting it.

    The block shipped a `qualification: same held-out set` row and the workspace
    told the user to "select outputs from the same held-out dataset" -- then
    ranked whatever was selected, across different array contracts and disjoint
    sample sets, without a word. Ranking a mean over two different test sets is
    not a comparison, so say so when the evidence in the CSVs disagrees.
    """
    warnings: list[str] = []
    if len(sources) < 2:
        return warnings
    contracts = {source["contract"] for source in sources if source.get("contract")}
    if len(contracts) > 1:
        warnings.append(
            "These runs were scored under different array contracts "
            f"({', '.join(sorted(contracts))}); their metrics are not directly comparable."
        )
    sample_sets = [set(source.get("samples") or ()) for source in sources]
    known = [item for item in sample_sets if item]
    if len(known) >= 2:
        shared = set.intersection(*known)
        union = set.union(*known)
        if not shared:
            warnings.append(
                "The selected runs share no sample IDs, so they were evaluated on "
                "different held-out sets. A mean-vs-mean ranking is not meaningful here."
            )
        elif len(shared) < len(union):
            warnings.append(
                f"Only {len(shared)} of {len(union)} sample IDs are common to every "
                "selected run; the means below are taken over different sample sets."
            )
    return warnings


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
    raw_group_column = payload.get("group_column")
    group_column = "model" if raw_group_column is None else str(raw_group_column).strip()
    direction = str(payload.get("direction", "min")).strip().lower()
    if not metric:
        raise ValueError("Select a numeric metric column.")
    if direction not in {"min", "max"}:
        raise ValueError("Comparison direction must be min or max.")

    csv_paths: list[Path] = []
    for raw_path in raw_paths:
        csv_path = safe_repo_path(raw_path, result_roots())
        if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
            raise ValueError(f"{raw_path!r} is not an existing CSV file.")
        csv_paths.append(csv_path)

    multi_run = len(csv_paths) > 1
    stem_counts = {
        path.stem: sum(candidate.stem == path.stem for candidate in csv_paths)
        for path in csv_paths
    }
    sources: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    total_rows = 0
    total_numeric = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        if not raw_rows:
            raise ValueError(f"{relative(csv_path)} has no rows.")
        if metric not in raw_rows[0]:
            raise ValueError(f"{relative(csv_path)} must contain a {metric!r} column.")
        has_group_column = bool(group_column) and group_column in raw_rows[0]
        run_label = csv_path.stem
        if stem_counts[run_label] > 1:
            run_label = f"{csv_path.parent.name}/{run_label}"
        numeric_in_source = 0
        grouped: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(raw_rows):
            value = _finite_float(row.get(metric))
            if value is None:
                continue
            numeric_in_source += 1
            group_value = (
                str(row.get(group_column) or "").strip() or "(missing group)"
                if has_group_column
                else run_label
            )
            bucket = grouped.setdefault(group_value, {"values": [], "first_index": index})
            bucket["values"].append(value)
        for group_value, bucket in grouped.items():
            values = bucket["values"]
            mean_value = sum(values) / len(values)
            name = (
                f"{run_label} · {group_value}"
                if multi_run and has_group_column
                else group_value if has_group_column else run_label
            )
            ranked.append({
                "index": bucket["first_index"],
                "name": name,
                "group": group_value if has_group_column else None,
                "value": mean_value,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "source": relative(csv_path),
                "run": run_label,
            })
        sources.append({
            "path": relative(csv_path),
            "run": run_label,
            "rows": len(raw_rows),
            "numeric_rows": numeric_in_source,
            "has_group_column": has_group_column,
            "groups": len(grouped),
            # Same two fields the schema inspects, so the saved report records
            # the comparability caveat rather than only the live panel showing it.
            "contract": next(
                (str(row.get("contract")) for row in raw_rows if str(row.get("contract", "")).strip()),
                "",
            ),
            "samples": sorted({
                str(row.get("truth_sample") or row.get("prediction_sample") or "").strip()
                for row in raw_rows
                if str(row.get("truth_sample") or row.get("prediction_sample") or "").strip()
            }),
        })
        total_rows += len(raw_rows)
        total_numeric += numeric_in_source
    if not ranked:
        raise ValueError(f"Column {metric!r} has no numeric values in any selected CSV.")
    ranked.sort(key=lambda item: item["value"], reverse=direction == "max")
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    qualification_warnings = _comparison_qualification_warnings(sources)
    for source in sources:
        source["sample_count"] = len(source.pop("samples", ()))
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
        "aggregation": "arithmetic mean of finite values within each source/group",
        "rows": total_rows,
        "numeric_rows": total_numeric,
        "ranked_groups": len(ranked),
        "best": ranked[0],
        "ranked": ranked[:200],
        "warnings": qualification_warnings,
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


CANDIDATE_TABLE_NAME = "candidates.csv"
OPTIMIZE_TABLE_NAME = "optimize_summary.csv"


def write_optimize_summary_table(output_dir: Path) -> dict[str, Any] | None:
    """Turn SDFFlow `mode optimize`'s `summary.json` into a small CSV.

    That mode produces one winning design plus two comparison references, not
    a population of i.i.d. candidates, so it does not fit `write_candidate_table`
    (built around `sample_*_meta.json`, which optimize mode never writes). This
    reuses the same {path, rows} shape so the CAD Generator block's results
    panel renders it exactly like the sample-mode gallery does.

    Returns None when no `summary.json` is present, so callers can try this
    before falling back to the sample-mode table without special-casing which
    mode actually ran.
    """
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    verified = summary.get("verified")
    if not verified:
        return None

    backend = summary.get("analysis_backend", "fea")
    mesh_key = "num_nodes" if backend == "surrogate" else "num_tets"
    rows: list[dict[str, Any]] = []
    for tag in ("baseline", "typical", "optimized"):
        entry = verified.get(tag)
        if not entry:
            continue
        stl_path = output_dir / f"{tag}.stl"
        rows.append({
            "id": tag,
            "mass_kg": round(entry.get("mass_kg", 0.0), 4),
            "peak_von_mises_mpa": round(entry.get("peak_von_mises_MPa", 0.0), 2),
            "max_displacement_mm": round(entry.get("max_displacement_mm", 0.0), 4),
            mesh_key: entry.get(mesh_key, ""),
            "analysis_backend": backend,
            "path": stl_path.name if stl_path.is_file() else "",
        })
    if not rows:
        return None
    table = output_dir / OPTIMIZE_TABLE_NAME
    fieldnames = list(rows[0])
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {"path": relative(table), "rows": len(rows)}


def write_candidate_table(output_dir: Path) -> dict[str, Any] | None:
    """Turn a generator run's `sample_*_meta.json` into a candidate CSV.

    The Optimization block needs a CSV with numeric columns; SDFFlow's sample
    mode writes STLs plus a metadata JSON that already holds exactly the right
    numbers per shape (volume, area, extents, watertight, faces). Nothing joined
    the two, so the `generator -> optimization` edge the shipped template draws
    carried nothing usable and the only way to optimize generated geometry was
    to build the table by hand outside the GUI.

    Returns None when the directory holds no generator metadata, so callers can
    treat "not a generation step" and "generation produced nothing" the same.
    """
    metas = sorted(output_dir.glob("sample_*_meta.json"))
    if not metas:
        return None
    rows: list[dict[str, Any]] = []
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seed = meta.get("seed")
        cond_names = meta.get("cond_names") or []
        cond_values = meta.get("cond_values")
        # A batch's requested design point is what makes a sweep readable later;
        # unconditional batches simply leave the columns blank.
        requested = dict(zip(cond_names, cond_values)) if cond_values else {}
        for entry in meta.get("results") or []:
            if not entry.get("valid"):
                continue
            extents = list(entry.get("extents") or [])
            while len(extents) < 3:
                extents.append("")
            volume = entry.get("volume")
            area = entry.get("area")
            row = {
                "id": f"seed{seed}_{entry.get('index', len(rows)):03d}",
                "seed": seed,
                "volume": volume,
                "area": area,
                "bbox_x": extents[0],
                "bbox_y": extents[1],
                "bbox_z": extents[2],
                "watertight": int(bool(entry.get("watertight"))),
                "faces": entry.get("faces"),
                "vertices": entry.get("vertices"),
                "compactness": (volume / area) if isinstance(volume, (int, float)) and area else "",
                "path": os.path.basename(str(entry.get("path", ""))),
            }
            for name in cond_names:
                row[f"requested_{name}"] = requested.get(name, "")
            rows.append(row)
    if not rows:
        return None
    table = output_dir / CANDIDATE_TABLE_NAME
    fieldnames = list(rows[0])
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {"path": relative(table), "rows": len(rows)}


# Column names that identify a candidate row, most specific first. The old
# fallback chain (id / sample / candidate / model) never looked at `*_id`
# columns, so a CSV whose identity lives in `candidate_id` was labelled by its
# `model` column instead -- four distinct designs reported under two names,
# while the workspace card promised "source-row provenance".
_CANDIDATE_ID_COLUMNS = ("candidate_id", "sample_id", "case_id", "design_id", "id", "candidate", "sample", "case", "design", "name", "model")


def _candidate_id_column(columns: list[str], requested: str = "") -> str:
    """Which column names a candidate. Explicit request wins when it exists."""
    lookup = {str(column).casefold(): str(column) for column in columns}
    wanted = str(requested or "").strip()
    if wanted and wanted.casefold() in lookup:
        return lookup[wanted.casefold()]
    for candidate in _CANDIDATE_ID_COLUMNS:
        if candidate in lookup:
            return lookup[candidate]
    for column in columns:
        if str(column).casefold().endswith("_id"):
            return str(column)
    return ""


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
    # top_k arrives as null from a cleared number input; int(None) is a
    # TypeError that surfaced as a raw HTTP 500 instead of a usable message.
    raw_top_k = payload.get("top_k", 10)
    try:
        top_k = max(1, min(int(raw_top_k if raw_top_k not in (None, "") else 10), 200))
    except (TypeError, ValueError):
        raise ValueError("top_k must be a whole number between 1 and 200.") from None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError("Optimization CSV has no data rows.")
    id_column = _candidate_id_column(columns, str(payload.get("id_column", "")))
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
                "id": (str(row.get(id_column, "")).strip() if id_column else "") or str(index),
                "objectives": dict(zip(objectives, objective_values)),
                "values": objective_values,
                "feasible": feasible,
                "violations": violations,
                "row": row,
            }
        )
    feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]

    # Why nothing survived. The per-candidate violations were computed and then
    # dropped, so an all-infeasible run rendered "0 feasible / 0 Pareto" with an
    # empty list and no way to tell a typo'd threshold from a genuinely
    # infeasible batch. Aggregate per constraint: how many rows it rejected and
    # the closest value observed, which is what tells you where to move it.
    rejections: list[dict[str, Any]] = []
    for name, operator, threshold in constraints:
        blocked = [
            violation["value"]
            for candidate in candidates
            for violation in candidate["violations"]
            if violation["column"] == name and violation["operator"] == operator
        ]
        if not blocked:
            continue
        closest = min(blocked, key=lambda value: abs(value - threshold))
        rejections.append({
            "column": name,
            "operator": operator,
            "threshold": threshold,
            "rows": len(blocked),
            "closest_value": closest,
        })
    rejections.sort(key=lambda item: item["rows"], reverse=True)

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
    if pareto:
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
        # rows - numeric_candidates was visible as two different numbers with no
        # explanation; a row is skipped when any objective or constraint cell is
        # blank or non-numeric.
        "skipped_rows": len(rows) - len(candidates),
        "feasible": len(feasible_candidates),
        "rejections": rejections,
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

    # The block's "selected" port is typed `candidates`, and Export downstream of
    # it produced a JSON report -- not a candidate table. Write the selected rows
    # back out in the source CSV's own columns (plus rank and crowding) so the
    # port delivers what its type promises and the result is openable in a
    # spreadsheet.
    selected_path = report_dir / f"{report_id}_selected.csv"
    # The two added columns must not collide with the source CSV's own headers:
    # a candidate table that already has a "rank" column would otherwise emit a
    # duplicate header, and the spread below would silently overwrite the Pareto
    # rank with the source value.
    def _free_column(base: str) -> str:
        taken = {column.casefold() for column in columns}
        if base.casefold() not in taken:
            return base
        for suffix in range(2, 100):
            candidate_name = f"{base}_{suffix}"
            if candidate_name.casefold() not in taken:
                return candidate_name
        return f"{base}_pareto"

    rank_column = _free_column("pareto_rank")
    crowding_column = _free_column("pareto_crowding")
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[rank_column, crowding_column, *columns])
        writer.writeheader()
        for rank, candidate in enumerate(selected, start=1):
            crowding = candidate.get("crowding", 0.0)
            writer.writerow({
                **{column: candidate["row"].get(column, "") for column in columns},
                rank_column: rank,
                crowding_column: "boundary" if math.isinf(crowding) else f"{crowding:.6g}",
            })
    report["selected_csv"] = relative(selected_path)
    report["path"] = report["selected_csv"]
    return report
