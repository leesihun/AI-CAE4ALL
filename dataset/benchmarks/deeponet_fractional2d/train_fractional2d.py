#!/usr/bin/env python3
"""Train/evaluate the isolated 2D fractional-Laplacian DeepONet profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
NEURAL_OPERATOR_ROOT = SUITE_ROOT / "Neural_Operator"
if str(NEURAL_OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_OPERATOR_ROOT))

from model.deeponet_fractional2d import FractionalLaplacianDeepONet  # noqa: E402


def parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith("%"):
                continue
            fields = line.split(None, 1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected KEY VALUE")
            key, value = fields
            key = key.lower()
            if key in config:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            config[key] = parse_scalar(value.strip())
    return config


REQUIRED_KEYS = {
    "benchmark_profile",
    "dataset_dir",
    "output_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "hidden_width",
    "eval_interval_epochs",
    "checkpoint_interval_epochs",
}
ALLOWED_KEYS = REQUIRED_KEYS | {
    "device",
    "function_eval_chunk",
    "resume",
    "paper_plot_normalized_mse",
    "steps_per_epoch_limit",
}


def validate_config(config: dict[str, object]) -> None:
    missing = REQUIRED_KEYS - config.keys()
    unknown = config.keys() - ALLOWED_KEYS
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if int(config["epochs"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if int(config["eval_interval_epochs"]) <= 0:
        raise ValueError("eval_interval_epochs must be positive")
    if int(config["checkpoint_interval_epochs"]) <= 0:
        raise ValueError("checkpoint_interval_epochs must be positive")
    if int(config.get("steps_per_epoch_limit", 1)) <= 0:
        raise ValueError("steps_per_epoch_limit must be positive when provided")
    if int(config["hidden_width"]) != 60:
        raise ValueError("The paper profile requires hidden_width=60")
    profile = str(config["benchmark_profile"]).lower()
    if profile not in {"paper", "smoke"}:
        raise ValueError("benchmark_profile must be paper or smoke")
    if profile == "paper":
        expected = {
            "epochs": 5000,
            "batch_size": 100000,
            "learning_rate": 0.001,
            "hidden_width": 60,
            "eval_interval_epochs": 100,
        }
        mismatches = [
            f"{key}={config[key]!r} (expected {value!r})"
            for key, value in expected.items() if config[key] != value
        ]
        if "steps_per_epoch_limit" in config:
            mismatches.append("steps_per_epoch_limit is diagnostic-only")
        if mismatches:
            raise ValueError("paper profile requires: " + "; ".join(mismatches))


def resolve_suite_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else SUITE_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_queries(query_xy: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    num_query = query_xy.shape[0]
    xy = query_xy.repeat(alpha.shape[0], 1)
    repeated_alpha = alpha.repeat_interleave(num_query).unsqueeze(1)
    return torch.cat((xy, repeated_alpha), dim=1)


def decode_expanded_indices(
    flat: torch.Tensor, num_functions: int, num_query: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the released alpha -> function -> query expanded row order."""
    query_index = flat.remainder(num_query)
    quotient = torch.div(flat, num_query, rounding_mode="floor")
    function_index = quotient.remainder(num_functions)
    alpha_index = torch.div(quotient, num_functions, rounding_mode="floor")
    combined_query_index = alpha_index * num_query + query_index
    return function_index, alpha_index, query_index, combined_query_index


@torch.no_grad()
def evaluate(
    model: FractionalLaplacianDeepONet,
    branch: torch.Tensor,
    queries: torch.Tensor,
    target: torch.Tensor,
    function_chunk: int,
) -> dict[str, float]:
    model.eval()
    trunk_code = model.encode_trunk(queries)
    squared_error = torch.zeros((), dtype=torch.float64, device=branch.device)
    squared_target = torch.zeros((), dtype=torch.float64, device=branch.device)
    per_function_relative_l2: list[torch.Tensor] = []

    flat_target = target.reshape(target.shape[0], -1)
    for start in range(0, branch.shape[0], function_chunk):
        end = min(start + function_chunk, branch.shape[0])
        branch_code = model.encode_branch(branch[start:end])
        prediction = branch_code @ trunk_code.T + model.bias
        truth = flat_target[start:end]
        error = prediction - truth
        squared_error += torch.sum(error.double() ** 2)
        squared_target += torch.sum(truth.double() ** 2)
        numerator = torch.linalg.vector_norm(error.double(), dim=1)
        denominator = torch.linalg.vector_norm(truth.double(), dim=1).clamp_min(1.0e-30)
        per_function_relative_l2.append((numerator / denominator).cpu())

    normalized_mse = float((squared_error / squared_target).item())
    per_function = torch.cat(per_function_relative_l2)
    return {
        "global_normalized_mse": normalized_mse,
        "global_relative_l2": float(np.sqrt(normalized_mse)),
        "mean_per_function_relative_l2": float(per_function.mean().item()),
        "std_per_function_relative_l2": float(per_function.std(unbiased=False).item()),
        "median_per_function_relative_l2": float(per_function.median().item()),
    }


def save_checkpoint(
    path: Path,
    model: FractionalLaplacianDeepONet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    optimizer_steps: int,
    best_training_batch_loss: float,
    config: dict[str, object],
    sampler_rng_state: tuple,
    validation_identity: dict[str, object],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "best_training_batch_loss": best_training_batch_loss,
        "config": config,
        "model_config": model.export_model_config(),
        "sampler_rng_state": sampler_rng_state,
        "validation_identity": validation_identity,
    }, temporary)
    os.replace(temporary, path)


def run(config_path: Path, eval_only: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    validate_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    requested_device = str(config.get("device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Config requests {requested_device}, but CUDA is unavailable")
    device = torch.device(requested_device)

    dataset_path = resolve_suite_path(config["dataset_dir"])
    dataset_sha256 = file_sha256(dataset_path)
    output_dir = resolve_suite_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pth"
    best_path = output_dir / "best_train_batch.pth"
    log_path = output_dir / "train.jsonl"

    with h5py.File(dataset_path, "r") as handle:
        if handle.attrs.get("format") != "deeponet_fractional_laplacian_2d_compact_v1":
            raise ValueError(f"Unexpected benchmark format in {dataset_path}")
        relation = str(handle.attrs["released_train_test_relation"])
        if relation != "identical_hard_links":
            raise ValueError(f"Direct paper mode requires released duplicate test data, got {relation}")
        if handle["train/branch_values"].id != handle["test/branch_values"].id:
            raise ValueError("Released train/test branch datasets are not HDF5 hard links")
        if handle["train/targets"].id != handle["test/targets"].id:
            raise ValueError("Released train/test target datasets are not HDF5 hard links")
        if str(config["benchmark_profile"]).lower() == "paper":
            expected_direction_hash = (
                "864a98b3af71806c1922feed53b9f77da29189f67a52bba0a88f7503d332e949"
            )
            actual_direction_hash = str(
                handle.attrs.get("sobol_direction_file_sha256", "")
            )
            if actual_direction_hash != expected_direction_hash:
                raise ValueError(
                    "paper profile requires the MATLAB Joe-Kuo 2003 Sobol table; "
                    f"got hash {actual_direction_hash!r}"
                )
            if "common/sobol_logical_index" not in handle:
                raise ValueError("paper profile requires common/sobol_logical_index")
            if int(handle["common/sobol_logical_index"][0]) != 28:
                raise ValueError("paper profile Sobol first retained logical index must be 28")
            if "train/coefficients" not in handle:
                raise ValueError("paper profile requires retained Sobol coefficients")
            if float(handle["train/coefficients"][0, 0]) != -1.125:
                raise ValueError("paper profile first retained coefficient must be -1.125")
        branch_np = handle["train/branch_values"][:]
        target_np = handle["train/targets"][:]
        query_np = handle["common/query_xy"][:]
        alpha_np = handle["common/alpha"][:]
        dataset_metadata = {key: handle.attrs[key] for key in handle.attrs.keys()}

    branch = torch.as_tensor(branch_np, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    query_xy = torch.as_tensor(query_np, dtype=torch.float32, device=device)
    alpha = torch.as_tensor(alpha_np, dtype=torch.float32, device=device)
    queries = build_queries(query_xy, alpha)
    num_functions, num_alpha, num_query = target.shape
    if branch.shape != (num_functions, 225):
        raise ValueError(f"Paper branch shape must be [N,225], got {tuple(branch.shape)}")
    if queries.shape != (num_alpha * num_query, 3):
        raise ValueError(f"Paper trunk shape mismatch: {tuple(queries.shape)}")
    if str(config["benchmark_profile"]).lower() == "paper":
        paper_shape = (5000, 10, 225)
        if tuple(target.shape) != paper_shape:
            raise ValueError(
                f"paper profile requires target shape {paper_shape}, got {tuple(target.shape)}"
            )

    model = FractionalLaplacianDeepONet(
        branch_dim=225, query_dim=3, width=int(config["hidden_width"]), seed=seed
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    start_epoch = 0
    optimizer_steps = 0
    best_training_batch_loss = float("inf")
    rng = np.random.RandomState(seed)
    validation_identity = {
        "benchmark_profile": str(config["benchmark_profile"]).lower(),
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "hidden_width": int(config["hidden_width"]),
    }

    resume = bool(config.get("resume", True))
    if (resume or eval_only) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("validation_identity") != validation_identity:
            raise ValueError(
                "Checkpoint validation identity does not match this config/dataset: "
                f"saved={checkpoint.get('validation_identity')}, current={validation_identity}"
            )
        model.load_state_dict(checkpoint["model"])
        if not eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_steps = int(checkpoint["optimizer_steps"])
        best_training_batch_loss = float(checkpoint["best_training_batch_loss"])
        if not eval_only:
            rng.set_state(checkpoint["sampler_rng_state"])
    elif eval_only:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}")

    function_eval_chunk = int(config.get("function_eval_chunk", 256))
    if eval_only:
        final_metrics = evaluate(model, branch, queries, target, function_eval_chunk)
        official_best_metrics = None
        if best_path.exists():
            best_checkpoint = torch.load(
                best_path, map_location=device, weights_only=False
            )
            if best_checkpoint.get("validation_identity") != validation_identity:
                raise ValueError("Official-best checkpoint identity does not match dataset/config")
            model.load_state_dict(best_checkpoint["model"])
            official_best_metrics = evaluate(
                model, branch, queries, target, function_eval_chunk
            )
        profile = str(config["benchmark_profile"]).lower()
        result = {
            "mode": "paper_evaluation" if profile == "paper" else "smoke_evaluation",
            "benchmark_profile": profile,
            "final_checkpoint": str(checkpoint_path.resolve()),
            "monitor_checkpoint": str(best_path.resolve()) if best_path.exists() else None,
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha256,
            "released_test_duplicates_train": True,
            "final_released_test_metrics": final_metrics,
            "monitor_released_test_metrics": official_best_metrics,
        }
        if profile == "paper":
            result.update({
                "official_best_checkpoint": (
                    str(best_path.resolve()) if best_path.exists() else None
                ),
                "official_style_best_released_test_metrics": official_best_metrics,
                "primary_paper_comparison": "official_style_best_released_test_metrics",
            })
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    total_triples = int(target.numel())
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    eval_interval = int(config["eval_interval_epochs"])
    checkpoint_interval = int(config["checkpoint_interval_epochs"])
    full_steps_per_epoch = total_triples // batch_size
    if full_steps_per_epoch < 1:
        raise ValueError(
            f"batch_size={batch_size} exceeds expanded triples={total_triples}"
        )
    steps_per_epoch = min(
        full_steps_per_epoch,
        int(config.get("steps_per_epoch_limit", full_steps_per_epoch)),
    )
    if str(config["benchmark_profile"]).lower() == "paper":
        if full_steps_per_epoch != 112 or steps_per_epoch != 112:
            raise ValueError(
                "paper profile requires exactly 112 full batches per epoch; "
                f"got full={full_steps_per_epoch}, selected={steps_per_epoch}"
            )
    start_time = time.perf_counter()

    # Official expanded row order is alpha -> function -> query.  We shuffle
    # those row IDs, but compute each branch/trunk network only once per unique
    # input. With no batch-dependent layers this is mathematically identical
    # to materializing repeated 225-wide branch rows (about 9.6 GiB/split).
    for epoch in range(start_epoch, epochs):
        permutation = rng.permutation(total_triples)
        loss_value = float("nan")
        for batch_index in range(steps_per_epoch):
            offset = batch_index * batch_size
            flat_np = permutation[offset:offset + batch_size]
            flat = torch.as_tensor(flat_np, dtype=torch.long, device=device)
            function_index, alpha_index, query_index, combined_query_index = (
                decode_expanded_indices(flat, num_functions, num_query)
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            branch_code = model.encode_branch(branch)
            trunk_code = model.encode_trunk(queries)
            prediction = model.decode_encoded(
                branch_code[function_index], trunk_code[combined_query_index]
            ).squeeze(1)
            truth = target[function_index, alpha_index, query_index]
            loss = torch.mean((prediction - truth) ** 2) / torch.mean(truth**2)
            loss_value = float(loss.detach().item())
            official_best_metrics = None
            is_official_monitor = (
                batch_index == steps_per_epoch - 1
                and (epoch % eval_interval == 0 or epoch + 1 == epochs)
            )
            if is_official_monitor and loss_value < best_training_batch_loss:
                # Released code measures/saves before applying this final
                # batch update, then still executes the Adam step below.
                best_training_batch_loss = loss_value
                official_best_metrics = evaluate(
                    model, branch, queries, target, function_eval_chunk
                )
                model.train()
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    optimizer_steps,
                    best_training_batch_loss,
                    config,
                    rng.get_state(),
                    validation_identity,
                )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        record: dict[str, object] = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "last_training_batch_normalized_mse": loss_value,
            "best_training_batch_normalized_mse": best_training_batch_loss,
            "steps_per_epoch": steps_per_epoch,
            "full_steps_per_epoch": full_steps_per_epoch,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if official_best_metrics is not None:
            record["official_style_best_released_test"] = official_best_metrics
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if is_official_monitor:
            print(json.dumps(record, sort_keys=True), flush=True)

        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                optimizer_steps,
                best_training_batch_loss,
                config,
                rng.get_state(),
                validation_identity,
            )

    final_metrics = evaluate(model, branch, queries, target, function_eval_chunk)
    if not best_path.exists():
        raise RuntimeError("Official-style best checkpoint was not created")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    official_best_metrics = evaluate(
        model, branch, queries, target, function_eval_chunk
    )
    paper_approx = float(config.get("paper_plot_normalized_mse", 1.2e-3))
    profile = str(config["benchmark_profile"]).lower()
    result = {
        "mode": "paper_training_complete" if profile == "paper" else "smoke",
        "benchmark_profile": profile,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "final_checkpoint": str(checkpoint_path.resolve()),
        "monitor_checkpoint": str(best_path.resolve()),
        "model_config": model.export_model_config(),
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "full_steps_per_epoch": full_steps_per_epoch,
        "batch_size": batch_size,
        "expanded_triples_per_released_split": total_triples,
        "released_test_duplicates_train": True,
        "final_released_test_metrics": final_metrics,
        "monitor_released_test_metrics": official_best_metrics,
        "dataset_metadata": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in dataset_metadata.items()
        },
    }
    if profile == "paper":
        result.update({
            "official_best_checkpoint": str(best_path.resolve()),
            "official_style_best_released_test_metrics": official_best_metrics,
            "primary_paper_comparison": "official_style_best_released_test_metrics",
            "paper_plot_normalized_mse_approximate": paper_approx,
            "diagnostic_ratio_to_plot_approximation": (
                official_best_metrics["global_normalized_mse"] / paper_approx
            ),
            "paper_comparator_warning": (
                "Figure 2e is plot-derived; 0.0012 is approximate and the ratio is "
                "diagnostic, not an exact pass/fail threshold."
            ),
        })
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
