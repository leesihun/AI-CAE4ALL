#!/usr/bin/env python3
"""Exact-protocol Transolver-v1 Elasticity training/evaluation entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
TRANSOLVER_ROOT = SUITE_ROOT / "transolver"
if str(TRANSOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSOLVER_ROOT))

from model.paper_elasticity import PaperElasticityTransolver  # noqa: E402


SOURCE_HASHES = {
    "Random_UnitCell_XY_10.npy":
        "29c615b7c8b0ef94252e4def4cd9999653b8759104619eb728b91a1cac5b665f",
    "Random_UnitCell_sigma_10.npy":
        "eb8102b580001bab80ee99bfc33289f727491e1d8edab2a4a94abc37a348fa1a",
}
OFFICIAL_COMMIT = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"
PAPER_MEAN_RELATIVE_L2 = 0.0064


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_value(value: str) -> object:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected KEY VALUE")
        key, value = fields
        config[key.lower()] = parse_value(value.strip())
    return config


REQUIRED = {
    "source_dir", "output_dir", "device", "seed", "epochs", "batch_size",
    "learning_rate", "weight_decay", "max_grad_norm", "hidden_dim",
    "num_layers", "num_heads", "slice_num", "mlp_ratio", "dropout",
    "eval_interval_epochs", "checkpoint_interval_epochs", "resume",
}


def validate_config(config: dict[str, object]) -> None:
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    unknown = config.keys() - REQUIRED
    if unknown:
        raise ValueError(f"Unknown keys: {sorted(unknown)}")
    expected = {
        "epochs": 500,
        "batch_size": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "max_grad_norm": 0.1,
        "hidden_dim": 128,
        "num_layers": 8,
        "num_heads": 8,
        "slice_num": 64,
        "mlp_ratio": 1,
        "dropout": 0.0,
    }
    mismatches = [
        f"{key}={config[key]!r} (expected {wanted!r})"
        for key, wanted in expected.items() if config[key] != wanted
    ]
    if mismatches:
        raise ValueError("Paper Elasticity config requires: " + "; ".join(mismatches))


def suite_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else SUITE_ROOT / path


def load_paper_data(source_dir: Path) -> tuple[torch.Tensor, ...]:
    paths = {name: source_dir / name for name in SOURCE_HASHES}
    for name, expected in SOURCE_HASHES.items():
        actual = file_sha256(paths[name])
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
    xy_np = np.load(paths["Random_UnitCell_XY_10.npy"], mmap_mode="r")
    sigma_np = np.load(paths["Random_UnitCell_sigma_10.npy"], mmap_mode="r")
    if xy_np.shape != (972, 2, 2000) or sigma_np.shape != (972, 2000):
        raise ValueError(f"Unexpected source shapes: XY={xy_np.shape}, sigma={sigma_np.shape}")
    xy = torch.from_numpy(np.array(xy_np, dtype=np.float32, copy=True)).permute(2, 0, 1)
    sigma = torch.from_numpy(np.array(sigma_np, dtype=np.float32, copy=True)).permute(1, 0)
    return xy[:1000], sigma[:1000], xy[-200:], sigma[-200:]


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(prediction - target, dim=1)
    # Match the released TestLoss.rel implementation exactly.  The verified
    # Elasticity targets all have non-zero norms, so no safety clamp is needed
    # on this isolated paper-reproduction path.
    denominator = torch.linalg.vector_norm(target, dim=1)
    return numerator / denominator


@torch.no_grad()
def evaluate(
    model: PaperElasticityTransolver,
    test_xy: torch.Tensor,
    test_sigma: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    values = []
    for index in range(test_xy.shape[0]):
        xy = test_xy[index:index + 1].to(device)
        target = test_sigma[index:index + 1].to(device)
        prediction = model(xy) * target_std + target_mean
        values.append(relative_l2(prediction, target).cpu())
    errors = torch.cat(values).double()
    return {
        "cases": int(errors.numel()),
        "mean_relative_l2": float(errors.mean().item()),
        "std_relative_l2": float(errors.std(unbiased=False).item()),
        "median_relative_l2": float(errors.median().item()),
        "min_relative_l2": float(errors.min().item()),
        "max_relative_l2": float(errors.max().item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: PaperElasticityTransolver,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    shuffle_generator: torch.Generator,
    identity: dict[str, object],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "shuffle_generator_state": shuffle_generator.get_state(),
        "identity": identity,
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "model_config": model.export_model_config(),
        "official_commit": OFFICIAL_COMMIT,
    }, temporary)
    os.replace(temporary, path)


def run(config_path: Path, eval_only: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    validate_config(config)
    source_dir = suite_path(config["source_dir"])
    output_dir = suite_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pth"
    log_path = output_dir / "train.jsonl"
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_xy, train_sigma, test_xy, test_sigma = load_paper_data(source_dir)
    target_mean_cpu = train_sigma.mean().reshape(1, 1)
    target_std_cpu = (train_sigma.std() + 1.0e-8).reshape(1, 1)
    # The release stores encoded training targets in its TensorDataset, then
    # decodes each batch immediately before computing relative L2.  Preserve
    # that float32 round trip for exact training-path parity.
    train_sigma_encoded = (train_sigma - target_mean_cpu) / target_std_cpu
    target_mean = target_mean_cpu.to(device)
    target_std = target_std_cpu.to(device)
    identity = {
        "source_hashes": SOURCE_HASHES,
        "official_train": [0, 999],
        "official_test": [1800, 1999],
        "seed": seed,
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "official_commit": OFFICIAL_COMMIT,
    }

    model = PaperElasticityTransolver(
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        slice_num=int(config["slice_num"]),
        mlp_ratio=int(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(seed)
    start_epoch = 0

    if (bool(config["resume"]) or eval_only) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError("Checkpoint identity does not match source/config")
        model.load_state_dict(checkpoint["model"])
        if not eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
    elif eval_only:
        raise FileNotFoundError(checkpoint_path)

    if eval_only:
        metrics = evaluate(model, test_xy, test_sigma, target_mean, target_std, device)
        result = {
            "mode": "paper_evaluation",
            "checkpoint": str(checkpoint_path.resolve()),
            "metrics": metrics,
            "paper_mean_relative_l2": PAPER_MEAN_RELATIVE_L2,
            "ratio_to_paper": metrics["mean_relative_l2"] / PAPER_MEAN_RELATIVE_L2,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    start_time = time.perf_counter()
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    for epoch in range(start_epoch, epochs):
        model.train()
        permutation = torch.randperm(1000, generator=shuffle_generator)
        train_sum = 0.0
        for offset in range(0, 1000, batch_size):
            indices = permutation[offset:offset + batch_size]
            xy = train_xy[indices].to(device)
            target_encoded = train_sigma_encoded[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xy) * target_std + target_mean
            target = target_encoded * target_std + target_mean
            loss = relative_l2(prediction, target).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["max_grad_norm"])
            )
            optimizer.step()
            train_sum += float(loss.detach().item())
        scheduler.step()
        train_mean = train_sum / 1000.0
        record: dict[str, object] = {
            "epoch": epoch,
            "train_mean_relative_l2": train_mean,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if epoch % int(config["eval_interval_epochs"]) == 0 or epoch + 1 == epochs:
            record["paper_test"] = evaluate(
                model, test_xy, test_sigma, target_mean, target_std, device
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if (
            (epoch + 1) % int(config["checkpoint_interval_epochs"]) == 0
            or epoch + 1 == epochs
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                shuffle_generator=shuffle_generator,
                identity=identity,
                target_mean=target_mean,
                target_std=target_std,
            )

    metrics = evaluate(model, test_xy, test_sigma, target_mean, target_std, device)
    result = {
        "mode": "paper_training_complete",
        "checkpoint": str(checkpoint_path.resolve()),
        "model_config": model.export_model_config(),
        "official_commit": OFFICIAL_COMMIT,
        "official_train_source_indices": [0, 999],
        "official_test_source_indices": [1800, 1999],
        "target_mean": float(target_mean_cpu.item()),
        "target_std_unbiased_plus_1e-8": float(target_std_cpu.item()),
        "metrics": metrics,
        "paper_mean_relative_l2": PAPER_MEAN_RELATIVE_L2,
        "ratio_to_paper": metrics["mean_relative_l2"] / PAPER_MEAN_RELATIVE_L2,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.eval_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
