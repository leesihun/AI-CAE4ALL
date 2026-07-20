#!/usr/bin/env python3
"""Standalone trainer for the opt-in GINO CarCFD paper decoder.

It is deliberately benchmark-local so the suite's default MSE loop and model
factory remain unchanged. The full config is a documented hybrid reconstruction
targeting the paper task/metric; it is not a byte-identical released experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

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


def parse_scalar(value: str):
    value = value.strip()
    if "," in value:
        return [parse_scalar(item) for item in value.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if any(mark in value.lower() for mark in (".", "e")) else int(value)
    except ValueError:
        return value


def load_benchmark_config(path: Path) -> dict:
    config: dict = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected 'key value'.")
        config[parts[0].lower()] = parse_scalar(parts[1])
    return config


def resolve_suite_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (SUITE_ROOT / path).resolve()


def data_spec() -> DataSpec:
    return DataSpec(
        input_var=1,
        output_var=1,
        positional_dim=0,
        node_type_dim=0,
        global_condition_dim=0,
        operator_dim=3,
        active_axes=(0, 1, 2),
        has_sdf=False,
        has_integration_weights=False,
        num_timesteps=1,
    )


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    losses = []
    for graph_index in range(ptr.numel() - 1):
        start, end = int(ptr[graph_index]), int(ptr[graph_index + 1])
        numerator = torch.linalg.vector_norm(prediction[start:end] - target[start:end])
        denominator = torch.linalg.vector_norm(target[start:end]).clamp_min(1.0e-12)
        losses.append(numerator / denominator)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, dataset, device, use_amp: bool) -> dict[str, float]:
    model.eval()
    values: list[float] = []
    for graph in loader:
        graph = graph.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            prediction = model(graph)
        prediction = dataset.de_normalize_pressure(prediction.float())
        target = dataset.de_normalize_pressure(graph.y.float())
        for graph_index in range(graph.ptr.numel() - 1):
            start, end = int(graph.ptr[graph_index]), int(graph.ptr[graph_index + 1])
            value = torch.linalg.vector_norm(prediction[start:end] - target[start:end])
            value = value / torch.linalg.vector_norm(target[start:end]).clamp_min(1.0e-12)
            values.append(float(value.cpu()))
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.export_model_config(),
            "data_config": data_spec().to_dict(),
            "benchmark_config": config,
            "benchmark_protocol": "gino_carcfd_hybrid_decoder_v1",
        },
        temporary,
    )
    temporary.replace(path)


def build_paper_scheduler(optimizer, config: dict):
    """Paper-hybrid schedule: halve the Adam learning rate after epoch 50."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.get("lr_step_size", 50)),
        gamma=float(config.get("lr_gamma", 0.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config.resolve())

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_path = resolve_suite_path(config["dataset_path"])
    train_dataset = CarCFDPaperDataset(dataset_path, "train")
    test_dataset = CarCFDPaperDataset(dataset_path, "test")
    require_full_hybrid = bool(config.get("require_full_hybrid_protocol", True))
    if require_full_hybrid:
        if train_dataset.protocol != "gino_carcfd_hybrid_decoder_v1":
            raise ValueError(
                f"Full hybrid run requires its Open3D HDF5, got {train_dataset.protocol!r}."
            )
        if train_dataset.resolution != 64 or len(train_dataset) != 500 or len(test_dataset) != 111:
            raise ValueError(
                "Full GINO hybrid run requires resolution 64 and exact 500/111 manifests; "
                f"got {train_dataset.resolution} and {len(train_dataset)}/{len(test_dataset)}."
            )

    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(requested_device)
    batch_size = int(config.get("batch_size", 1))
    workers = int(config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model_config = dict(config)
    model_config["gino_variant"] = "paper_decoder"
    model = CarCFDGINODecoder(model_config, data_spec()).to(device)
    learning_rate = float(config.get("learning_rate", 2.5e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = build_paper_scheduler(optimizer, config)
    epochs = int(config.get("training_epochs", 100))
    use_amp = bool(config.get("use_amp", False))
    clip_norm = float(config.get("max_grad_norm", 0.0))
    checkpoint_path = resolve_suite_path(config["checkpoint_path"])
    log_path = resolve_suite_path(config.get("log_path", checkpoint_path.with_suffix(".jsonl")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = int(config.get("checkpoint_interval", 10))
    test_interval = int(config.get("test_interval", 0))

    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "protocol": train_dataset.protocol,
                "resolution": train_dataset.resolution,
                "manifest_counts": [len(train_dataset), len(test_dataset)],
                "device": str(device),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "paper_target_mean_relative_l2": train_dataset.paper_target,
            },
            indent=2,
        )
    )

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        running = 0.0
        cases = 0
        for graph in train_loader:
            graph = graph.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp and device.type == "cuda",
            ):
                prediction = model(graph)
                loss = relative_l2(prediction, graph.y, graph.ptr)
            loss.backward()
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            graph_count = graph.ptr.numel() - 1
            running += float(loss.detach().cpu()) * graph_count
            cases += graph_count
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_normalized_relative_l2": running / max(1, cases),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        if test_interval > 0 and epoch % test_interval == 0:
            record["test_denormalized_relative_l2"] = evaluate(
                model, test_loader, test_dataset, device, use_amp
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, config)

    final = evaluate(model, test_loader, test_dataset, device, use_amp)
    final_record = {
        "event": "final_test",
        "metric": "mean per-case de-normalized relative L2",
        "result": final,
        "paper_reported": 0.0712,
        "mean_minus_paper": final["mean"] - 0.0712,
        "mean_over_paper": final["mean"] / 0.0712,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(final_record) + "\n")
    print(json.dumps(final_record, indent=2))


if __name__ == "__main__":
    main()
