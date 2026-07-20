#!/usr/bin/env python3
"""Evaluate saved suite rollouts on the published Elasticity test cases.

This is deliberately a post-inference tool.  It does not import or modify any
training or inference hot path, and it evaluates de-normalized values already
written by those paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


ROLLOUT_RE = re.compile(r"^rollout_sample(?P<sample>\d+)_steps(?P<steps>\d+)\.h5$")
PAPER_REFERENCES = {
    "transolver": {
        "reported_mean_relative_l2": 0.0064,
        "reported_display": "0.0064 mean relative L2",
        "reported_model": "Transolver",
        "comparison": (
            "same benchmark and architecture-aligned, but not a strict reproduction: "
            "the suite uses a shuffled training composition, normalized-MSE training, "
            "and its native scheduler/runtime"
        ),
        "paper": "https://arxiv.org/abs/2402.02366",
    },
    "fno": {
        "reported_mean_relative_l2": 0.0229,
        "reported_display": "0.0229 mean relative L2",
        "reported_model": "Geo-FNO",
        "comparison": "context only: suite FNO is not the paper's Geo-FNO implementation",
        "paper": "https://arxiv.org/abs/2402.02366",
    },
    "deeponet": {
        "reported_mean_relative_l2": None,
        "reported_display": "different operator tasks; no matching Elasticity value",
        "reported_model": "DeepONet",
        "comparison": "not comparable: the original DeepONet paper does not report this Elasticity benchmark",
        "paper": "https://arxiv.org/abs/1910.03193",
    },
    "point_deeponet": {
        "reported_mean_relative_l2": None,
        "reported_display": "R-squared 0.923 for von Mises stress",
        "reported_model": "Point-DeepONet",
        "comparison": "not comparable: the Point-DeepONet paper uses DeepJEB inputs and reports R-squared",
        "paper": "https://arxiv.org/abs/2412.18362",
    },
    "gino": {
        "reported_mean_relative_l2": None,
        "reported_display": "different 3D vehicle-pressure task; no matching scalar",
        "reported_model": "GINO",
        "comparison": "not comparable: the GINO paper uses a 3D vehicle-flow benchmark",
        "paper": "https://arxiv.org/abs/2309.00583",
    },
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(PAPER_REFERENCES))
    parser.add_argument("--ground-truth", type=Path, default=here / "elasticity_test.h5")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def discover_rollouts(directory: Path) -> dict[int, Path]:
    rollouts: dict[int, Path] = {}
    for path in sorted(directory.glob("rollout_sample*_steps*.h5")):
        match = ROLLOUT_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        if sample_id in rollouts:
            raise ValueError(f"Multiple rollout files found for sample {sample_id}")
        rollouts[sample_id] = path
    return rollouts


def read_prediction(path: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        nodal = np.asarray(handle[f"data/{sample_id}/nodal_data"])
    if nodal.ndim != 3 or nodal.shape[0] < 4:
        raise ValueError(f"Unexpected prediction shape in {path}: {nodal.shape}")
    positions = np.asarray(nodal[0:3, -1, :], dtype=np.float64)
    prediction = np.asarray(nodal[3, -1, :], dtype=np.float64)
    return positions, prediction


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    truth_path = args.ground_truth.resolve()
    prediction_dir = args.predictions.resolve()
    output_path = (args.output or prediction_dir / "relative_l2.json").resolve()
    csv_path = output_path.with_suffix(".csv")

    if not truth_path.is_file():
        raise FileNotFoundError(f"Ground-truth HDF5 not found: {truth_path}")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    rollouts = discover_rollouts(prediction_dir)
    rows: list[dict[str, float | int | str]] = []
    with h5py.File(truth_path, "r") as truth_file:
        truth_ids = sorted(int(key) for key in truth_file["data"].keys())
        missing = sorted(set(truth_ids) - set(rollouts))
        extra = sorted(set(rollouts) - set(truth_ids))
        if missing and not args.allow_partial:
            raise ValueError(
                f"Missing {len(missing)} of {len(truth_ids)} predictions; "
                f"first missing IDs: {missing[:10]}"
            )
        if extra:
            raise ValueError(f"Prediction IDs are absent from ground truth: {extra[:10]}")

        eval_ids = sorted(set(truth_ids) & set(rollouts))
        if not eval_ids:
            raise ValueError("No prediction files match the ground-truth sample IDs")

        for sample_id in eval_ids:
            truth_group = truth_file[f"data/{sample_id}"]
            truth_nodal = np.asarray(truth_group["nodal_data"])
            truth_positions = np.asarray(truth_nodal[0:3, 0, :], dtype=np.float64)
            target = np.asarray(truth_nodal[3, 0, :], dtype=np.float64)
            prediction_positions, prediction = read_prediction(rollouts[sample_id], sample_id)

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Sample {sample_id} prediction shape {prediction.shape} != target {target.shape}"
                )
            position_error = float(np.max(np.abs(prediction_positions - truth_positions)))
            if position_error > 1e-5:
                raise ValueError(
                    f"Sample {sample_id} geometry mismatch (max abs error {position_error:.3e})"
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError(f"Sample {sample_id} prediction contains NaN or Inf")

            numerator = float(np.linalg.norm(prediction - target))
            denominator = float(np.linalg.norm(target))
            relative_l2 = numerator / max(denominator, args.epsilon)
            rows.append({
                "sample_id": sample_id,
                "source_index": int(truth_group.attrs["source_index"]),
                "relative_l2": relative_l2,
                "error_l2": numerator,
                "target_l2": denominator,
                "prediction_file": rollouts[sample_id].name,
            })

    values = np.asarray([float(row["relative_l2"]) for row in rows], dtype=np.float64)
    reference = PAPER_REFERENCES[args.model]
    reported = reference["reported_mean_relative_l2"]
    result = {
        "model": args.model,
        "metric": "mean per-sample ||prediction-target||_2 / ||target||_2",
        "evaluation_space": "de-normalized stress from saved inference HDF5",
        "ground_truth": str(truth_path),
        "predictions": str(prediction_dir),
        "evaluated_samples": len(rows),
        "expected_samples": len(truth_ids),
        "complete": len(rows) == len(truth_ids),
        "relative_l2": summarize(values),
        "paper_reference": reference,
        "mean_minus_reported": None if reported is None else float(np.mean(values) - reported),
        "mean_over_reported": None if reported is None else float(np.mean(values) / reported),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Per-sample CSV: {csv_path}")
    print(f"Summary JSON: {output_path}")


if __name__ == "__main__":
    main()
