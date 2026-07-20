#!/usr/bin/env python3
"""Evaluate the exact 111 CarCFD cases in de-normalized pressure units."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
NEURAL_OPERATOR_ROOT = SUITE_ROOT / "Neural_Operator"
sys.path.insert(0, str(NEURAL_OPERATOR_ROOT))
sys.path.insert(0, str(HERE))

from carcfd_dataset import CarCFDPaperDataset  # noqa: E402
from general_modules.data_spec import DataSpec  # noqa: E402
from model.gino_carcfd import CarCFDGINODecoder  # noqa: E402
from train_carcfd import load_benchmark_config, resolve_suite_path  # noqa: E402


PAPER_RESULT = 0.0712
PAPER_URL = "https://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--allow-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config.resolve())
    dataset_path = resolve_suite_path(config["dataset_path"])
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else resolve_suite_path(config["checkpoint_path"])
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else resolve_suite_path(config.get("evaluation_output", checkpoint_path.with_suffix(".relative_l2.json")))
    )
    predictions_path = (
        args.predictions.resolve()
        if args.predictions is not None
        else output_path.with_suffix(".predictions.h5")
    )

    dataset = CarCFDPaperDataset(dataset_path, "test")
    if not args.allow_diagnostic:
        if dataset.protocol != "gino_carcfd_hybrid_decoder_v1" or dataset.resolution != 64:
            raise ValueError(
                f"Full hybrid evaluation requires the 64^3 Open3D dataset, got {dataset.protocol}, "
                f"resolution {dataset.resolution}."
            )
        if len(dataset) != 111:
            raise ValueError(f"Paper evaluation requires exactly 111 test cases, got {len(dataset)}.")

    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_data_spec = DataSpec.from_dict(checkpoint["data_config"])
    model_config = dict(checkpoint["model_config"])
    # Runtime-only choices may be safely overridden without changing weights.
    model_config["gino_query_chunk_size"] = int(
        config.get("gino_query_chunk_size", model_config.get("gino_query_chunk_size", 1024))
    )
    model_config["gino_use_torch_cluster"] = bool(
        config.get("gino_use_torch_cluster", model_config.get("gino_use_torch_cluster", True))
    )
    model = CarCFDGINODecoder(model_config, checkpoint_data_spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    use_amp = bool(config.get("use_amp", False))
    rows: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with h5py.File(predictions_path, "w") as prediction_h5:
        prediction_h5.attrs["checkpoint"] = str(checkpoint_path)
        prediction_h5.attrs["dataset"] = str(dataset_path)
        prediction_h5.attrs["pressure_mean"] = dataset.pressure_mean
        prediction_h5.attrs["pressure_std"] = dataset.pressure_std
        prediction_h5.attrs["pressure_eps"] = dataset.pressure_eps
        prediction_h5.attrs["evaluation_space"] = "de-normalized pressure"
        prediction_group = prediction_h5.create_group("predictions")
        with torch.no_grad():
            for graph in loader:
                graph = graph.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp and device.type == "cuda",
                ):
                    prediction_normalized = model(graph)
                target_normalized = graph.y
                prediction = dataset.de_normalize_pressure(prediction_normalized.float())
                target = dataset.de_normalize_pressure(target_normalized.float())

                for graph_index in range(graph.ptr.numel() - 1):
                    start, end = int(graph.ptr[graph_index]), int(graph.ptr[graph_index + 1])
                    case_id = dataset.case_ids[cursor]
                    cursor += 1
                    pred_case = prediction[start:end].detach().cpu().numpy().reshape(-1)
                    target_case = target[start:end].detach().cpu().numpy().reshape(-1)
                    pred_norm_case = prediction_normalized[start:end].float().detach().cpu().numpy().reshape(-1)
                    target_norm_case = target_normalized[start:end].float().detach().cpu().numpy().reshape(-1)
                    numerator = float(np.linalg.norm(pred_case - target_case))
                    denominator = float(np.linalg.norm(target_case))
                    normalized_numerator = float(np.linalg.norm(pred_norm_case - target_norm_case))
                    normalized_denominator = float(np.linalg.norm(target_norm_case))
                    rows.append(
                        {
                            "case_id": case_id,
                            "relative_l2": numerator / max(denominator, 1.0e-12),
                            "error_l2": numerator,
                            "target_l2": denominator,
                            "normalized_relative_l2": normalized_numerator / max(normalized_denominator, 1.0e-12),
                        }
                    )
                    case_group = prediction_group.create_group(case_id)
                    case_group.create_dataset("pressure", data=pred_case, compression="gzip")
                    case_group.create_dataset("target", data=target_case, compression="gzip")

    if cursor != len(dataset):
        raise AssertionError(f"Evaluated {cursor} cases but dataset contains {len(dataset)}.")
    values = np.asarray([row["relative_l2"] for row in rows], dtype=np.float64)
    normalized_values = np.asarray(
        [row["normalized_relative_l2"] for row in rows], dtype=np.float64
    )
    mean = float(np.mean(values))
    report = {
        "model": "GINO decoder-only",
        "benchmark": "ShapeNet Car surface pressure",
        "benchmark_protocol": dataset.protocol,
        "metric": "mean per-case ||prediction-target||_2 / ||target||_2",
        "evaluation_space": "de-normalized physical pressure",
        "evaluated_samples": len(rows),
        "expected_samples": 111,
        "complete": len(rows) == 111,
        "relative_l2": {
            "mean": mean,
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "diagnostic_normalized_relative_l2_mean": float(np.mean(normalized_values)),
        "paper_reference": {
            "paper": PAPER_URL,
            "reported_test_mean_relative_l2": PAPER_RESULT,
            "reported_train_normalized_relative_l2": 0.0637,
            "reported_model": "GINO decoder-only",
            "grid_resolution": 64,
            "epochs": 100,
        },
        "mean_minus_reported": mean - PAPER_RESULT,
        "mean_over_reported": mean / PAPER_RESULT,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
    }
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
