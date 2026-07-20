#!/usr/bin/env python3
"""One-case real-resolution CUDA memory gate for isolated CarCFD GINO.

This does not train a model or write a checkpoint.  It executes one forward,
backward, and Adam update so the reported CUDA peak includes optimizer-state
initialization as well as the 64-cubed operator activations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
NEURAL_OPERATOR_ROOT = SUITE_ROOT / "Neural_Operator"
sys.path.insert(0, str(NEURAL_OPERATOR_ROOT))
sys.path.insert(0, str(HERE))

from carcfd_dataset import CarCFDPaperDataset  # noqa: E402
from model.gino_carcfd import CarCFDGINODecoder  # noqa: E402
from train_carcfd import (  # noqa: E402
    data_spec,
    load_benchmark_config,
    relative_l2,
    resolve_suite_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()

    config = load_benchmark_config(args.config.resolve())
    dataset_path = (
        args.dataset.resolve()
        if args.dataset is not None
        else resolve_suite_path(config["dataset_path"])
    )
    dataset = CarCFDPaperDataset(dataset_path, "train")
    if dataset.resolution != 64:
        raise ValueError(f"The CUDA memory gate requires a 64-cubed artifact, got {dataset.resolution}.")
    if dataset.diagnostic_only and not args.allow_diagnostic:
        raise ValueError("Diagnostic artifact requires explicit --allow-diagnostic.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-resolution memory gate.")

    device = torch.device(str(config.get("device", "cuda:0")))
    model_config = dict(config)
    model_config["gino_variant"] = "paper_decoder"
    model = CarCFDGINODecoder(model_config, data_spec()).to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 2.5e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    graph = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    graph = graph.to(device)
    coverage = model.coverage_preflight(graph)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(graph)
    loss = relative_l2(prediction, graph.y, graph.ptr)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)

    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "case_id": dataset.case_ids[0],
                "diagnostic_only": dataset.diagnostic_only,
                "resolution": dataset.resolution,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "loss": float(loss.detach().cpu()),
                "coverage": coverage,
                "cuda": {
                    "device": str(device),
                    "total_MiB": total / 2**20,
                    "free_before_MiB": free_before / 2**20,
                    "free_after_MiB": free_after / 2**20,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                    "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
