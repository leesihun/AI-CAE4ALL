#!/usr/bin/env python3
"""Evaluate de-normalized FNO Darcy rollouts against the paper test set."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


PAPER_RESULT = 0.0108
PAPER = "https://arxiv.org/abs/2010.08895"
ROLLOUT_RE = re.compile(r"^rollout_sample(?P<sample>\d+)_steps(?P<steps>\d+)\.h5$")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=here / "darcy_test.h5")
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="Directory containing rollout_sample*_steps1.h5 files",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def discover(directory: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(directory.glob("rollout_sample*_steps*.h5")):
        match = ROLLOUT_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        if int(match.group("steps")) != 1:
            raise ValueError(f"Expected one-step Darcy rollout, got {path.name}")
        if sample_id in found:
            raise ValueError(f"Duplicate rollout for sample {sample_id}")
        found[sample_id] = path
    return found


def main() -> None:
    args = parse_args()
    truth_path = args.ground_truth.resolve()
    prediction_dir = args.predictions.resolve()
    output_path = (args.output or prediction_dir / "relative_l2.json").resolve()
    csv_path = output_path.with_suffix(".csv")
    rollouts = discover(prediction_dir)
    rows: list[dict[str, float | int | str]] = []

    with h5py.File(truth_path, "r") as truth_h5:
        benchmark_protocol = str(
            truth_h5.attrs.get("benchmark_protocol", "suite_residual_v1")
        )
        paper_direct = benchmark_protocol == "paper_direct_solution_v1"
        truth_ids = sorted(int(key) for key in truth_h5["data"])
        missing = sorted(set(truth_ids) - set(rollouts))
        extra = sorted(set(rollouts) - set(truth_ids))
        if missing and not args.allow_partial:
            raise ValueError(f"Missing {len(missing)} predictions; first IDs: {missing[:10]}")
        if extra:
            raise ValueError(f"Unexpected prediction IDs: {extra[:10]}")

        for sample_id in sorted(set(truth_ids) & set(rollouts)):
            truth_group = truth_h5[f"data/{sample_id}"]
            truth_nodal = np.asarray(truth_group["nodal_data"])
            coefficient = np.asarray(truth_nodal[3, 0, :], dtype=np.float64)
            target_t1 = np.asarray(truth_nodal[3, 1, :], dtype=np.float64)
            target = target_t1 - coefficient if paper_direct else target_t1
            truth_position = np.asarray(truth_nodal[0:3, 1, :], dtype=np.float64)
            with h5py.File(rollouts[sample_id], "r") as prediction_h5:
                prediction_nodal = np.asarray(
                    prediction_h5[f"data/{sample_id}/nodal_data"]
                )
            prediction_t1 = np.asarray(prediction_nodal[3, -1, :], dtype=np.float64)
            prediction = prediction_t1 - coefficient if paper_direct else prediction_t1
            prediction_position = np.asarray(prediction_nodal[0:3, -1, :], dtype=np.float64)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Sample {sample_id}: prediction {prediction.shape} != target {target.shape}"
                )
            position_error = float(np.max(np.abs(prediction_position - truth_position)))
            if position_error > 1e-6:
                raise ValueError(
                    f"Sample {sample_id}: geometry mismatch, max error {position_error:.3e}"
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError(f"Sample {sample_id}: prediction contains NaN or Inf")
            numerator = float(np.linalg.norm(prediction - target))
            denominator = float(np.linalg.norm(target))
            rows.append({
                "sample_id": sample_id,
                "source_file": str(truth_group.attrs["source_file"]),
                "source_index": int(truth_group.attrs["source_index"]),
                "relative_l2": numerator / max(denominator, args.epsilon),
                "error_l2": numerator,
                "target_l2": denominator,
                "prediction_file": rollouts[sample_id].name,
            })

    if not rows:
        raise ValueError("No matching predictions")
    values = np.asarray([row["relative_l2"] for row in rows], dtype=np.float64)
    mean = float(np.mean(values))
    result = {
        "model": "fno",
        "benchmark": "Darcy flow, 85x85",
        "benchmark_protocol": benchmark_protocol,
        "metric": "mean per-sample ||prediction-target||_2 / ||target||_2",
        "evaluation_space": (
            "de-normalized direct solution recovered as saved timestep 1 minus coefficient"
            if paper_direct else "de-normalized solution at saved rollout timestep 1"
        ),
        "evaluated_samples": len(rows),
        "expected_samples": len(truth_ids),
        "complete": len(rows) == len(truth_ids),
        "relative_l2": {
            "mean": mean,
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "paper_reference": {
            "paper": PAPER,
            "reported_mean_relative_l2": PAPER_RESULT,
            "resolution": 85,
            "reported_model": "FNO",
        },
        "mean_minus_reported": mean - PAPER_RESULT,
        "mean_over_reported": mean / PAPER_RESULT,
        "qualification": (
            "Opt-in paper protocol: exact optimization/test composition, direct-solution "
            "target, paper FNO core, decoded relative-L2 loss, Adam, and StepLR; the exact "
            "regular grid still passes through the suite splat/sample adapter."
            if paper_direct else
            "Same paper task, resolution, test size, and metric; not a strict reproduction "
            "because the unchanged suite uses a shuffled training pool, normalized residual "
            "MSE, and its native architecture/runtime details."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
