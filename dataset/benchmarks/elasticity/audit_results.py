#!/usr/bin/env python3
"""Mechanically audit completion of the full Elasticity validation run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
OUTPUT_ROOT = SUITE_ROOT / "output" / "benchmarks" / "elasticity"
CONFIG_ROOT = SUITE_ROOT / "configs" / "benchmarks" / "elasticity"
MODELS = ("point_deeponet", "deeponet", "fno", "gino", "transolver")
TRAIN_CONFIGS = {
    "point_deeponet": "config_train_point_deeponet.txt",
    "deeponet": "config_train_deeponet.txt",
    "fno": "config_train_fno.txt",
    "gino": "config_train_gino.txt",
    "transolver": "config_train_transolver_paper.txt",
}
ROLLOUT_RE = re.compile(r"^rollout_sample(?P<sample>\d+)_steps(?P<steps>\d+)\.h5$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_value(path: Path, wanted: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "%", "'")):
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2 and parts[0] == wanted:
            return parts[1]
    raise KeyError(f"{wanted} is absent from {path}")


def check_dataset(
    path: Path,
    expected_sources: list[int],
    xy: np.ndarray,
    sigma: np.ndarray,
) -> None:
    with h5py.File(path, "r") as handle:
        ids = sorted(int(key) for key in handle["data"].keys())
        if ids != list(range(len(expected_sources))):
            raise ValueError(f"{path.name}: sample IDs are not contiguous")
        for sample_id, source_index in enumerate(expected_sources):
            group = handle[f"data/{sample_id}"]
            if int(group.attrs["source_index"]) != source_index:
                raise ValueError(f"{path.name}: sample {sample_id} source index mismatch")
            nodal = np.asarray(group["nodal_data"])
            edge = np.asarray(group["mesh_edge"])
            if nodal.shape != (4, 1, 972) or edge.shape != (2, 971):
                raise ValueError(
                    f"{path.name}: sample {sample_id} shapes {nodal.shape}, {edge.shape}"
                )
            expected_xy = np.asarray(xy[:, :, source_index].T, dtype=np.float32)
            expected_sigma = np.asarray(sigma[:, source_index], dtype=np.float32)
            if not np.array_equal(nodal[0:2, 0, :], expected_xy):
                raise ValueError(f"{path.name}: sample {sample_id} coordinates differ from source")
            if np.any(nodal[2, 0, :] != 0):
                raise ValueError(f"{path.name}: sample {sample_id} z coordinate is not zero")
            if not np.array_equal(nodal[3, 0, :], expected_sigma):
                raise ValueError(f"{path.name}: sample {sample_id} stress differs from source")
            if int(edge.min()) < 0 or int(edge.max()) >= 972:
                raise ValueError(f"{path.name}: sample {sample_id} edge index is invalid")


def check_checkpoint(model: str) -> dict[str, float | int]:
    checkpoint_path = OUTPUT_ROOT / model / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_epochs = int(config_value(CONFIG_ROOT / TRAIN_CONFIGS[model], "training_epochs"))
    epoch = int(checkpoint["epoch"])
    valid_loss = float(checkpoint["valid_loss"])
    if epoch != expected_epochs - 1:
        raise ValueError(f"{model}: checkpoint epoch {epoch}, expected {expected_epochs - 1}")
    if not math.isfinite(valid_loss):
        raise ValueError(f"{model}: non-finite checkpoint validation loss")
    if "normalization" not in checkpoint:
        raise ValueError(f"{model}: checkpoint lacks training normalization")
    return {"epoch": epoch, "valid_loss": valid_loss}


def check_predictions(model: str) -> dict[str, float | int]:
    prediction_dir = OUTPUT_ROOT / model / "inference"
    files: dict[int, Path] = {}
    for path in prediction_dir.glob("rollout_sample*_steps*.h5"):
        match = ROLLOUT_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        if sample_id in files:
            raise ValueError(f"{model}: duplicate rollout for sample {sample_id}")
        files[sample_id] = path
    if sorted(files) != list(range(200)):
        missing = sorted(set(range(200)) - set(files))
        extra = sorted(set(files) - set(range(200)))
        raise ValueError(f"{model}: rollout IDs incomplete; missing={missing[:10]}, extra={extra[:10]}")

    for sample_id, path in files.items():
        with h5py.File(path, "r") as handle:
            nodal = np.asarray(handle[f"data/{sample_id}/nodal_data"])
        if nodal.ndim != 3 or nodal.shape[0] < 4 or nodal.shape[2] != 972:
            raise ValueError(f"{model}: sample {sample_id} output shape {nodal.shape}")
        if not np.all(np.isfinite(nodal[0:4])):
            raise ValueError(f"{model}: sample {sample_id} output contains NaN or Inf")

    summary_path = prediction_dir / "relative_l2.json"
    csv_path = prediction_dir / "relative_l2.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evaluated_samples") != 200 or summary.get("expected_samples") != 200:
        raise ValueError(f"{model}: metric summary does not cover 200 cases")
    if summary.get("complete") is not True:
        raise ValueError(f"{model}: metric summary is marked incomplete")
    mean = float(summary["relative_l2"]["mean"])
    if not math.isfinite(mean):
        raise ValueError(f"{model}: non-finite mean relative L2")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200:
        raise ValueError(f"{model}: per-sample metric CSV has {len(rows)} rows")
    if [int(row["sample_id"]) for row in rows] != list(range(200)):
        raise ValueError(f"{model}: per-sample metric CSV IDs are incomplete")
    if [int(row["source_index"]) for row in rows] != list(range(1800, 2000)):
        raise ValueError(f"{model}: per-sample metric CSV source indices are wrong")
    return {"rollouts": len(files), "mean_relative_l2": mean}


def main() -> None:
    errors: list[str] = []
    details: dict[str, object] = {}
    provenance = json.loads((HERE / "elasticity.provenance.json").read_text(encoding="utf-8"))
    source_dir = HERE / "source" / "original_data"

    try:
        for name, expected in provenance["source_sha256"].items():
            actual = sha256(source_dir / name)
            if actual != expected:
                raise ValueError(f"{name}: SHA256 {actual}, expected {expected}")
        xy = np.load(source_dir / "Random_UnitCell_XY_10.npy", mmap_mode="r")
        sigma = np.load(source_dir / "Random_UnitCell_sigma_10.npy", mmap_mode="r")
        check_dataset(HERE / "elasticity_train.h5", list(range(1250)), xy, sigma)
        check_dataset(HERE / "elasticity_test.h5", list(range(1800, 2000)), xy, sigma)
        details["dataset"] = {
            "source_hashes": "verified",
            "training_cases": 1250,
            "published_test_cases": 200,
            "source_mapping": "verified for every converted case",
        }
    except Exception as exc:
        errors.append(f"dataset: {type(exc).__name__}: {exc}")

    model_details: dict[str, object] = {}
    for model in MODELS:
        try:
            model_details[model] = {
                "checkpoint": check_checkpoint(model),
                "evaluation": check_predictions(model),
            }
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    details["models"] = model_details

    try:
        comparison = json.loads((OUTPUT_ROOT / "comparison.json").read_text(encoding="utf-8"))
        rows = comparison["results"]
        if [row["model"] for row in rows] != list(MODELS):
            raise ValueError("comparison model order or coverage is wrong")
        for name in ("comparison.md", "comparison.csv"):
            if not (OUTPUT_ROOT / name).is_file():
                raise FileNotFoundError(name)
        details["comparison"] = "verified JSON, CSV, and Markdown for all five models"
    except Exception as exc:
        errors.append(f"comparison: {type(exc).__name__}: {exc}")

    audit = {"status": "PASS" if not errors else "FAIL", "errors": errors, "details": details}
    output_path = OUTPUT_ROOT / "audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
