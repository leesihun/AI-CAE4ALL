#!/usr/bin/env python3
"""Run the Elasticity benchmark without touching production runtime code."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
CONFIG_ROOT = SUITE_ROOT / "configs" / "benchmarks" / "elasticity"
MODELS = ("point_deeponet", "deeponet", "fno", "gino", "transolver")
TRAIN_CONFIGS = {
    "point_deeponet": "config_train_point_deeponet.txt",
    "deeponet": "config_train_deeponet.txt",
    "fno": "config_train_fno.txt",
    "gino": "config_train_gino.txt",
    "transolver": "config_train_transolver_paper.txt",
}
INFER_CONFIGS = {model: f"config_infer_{model}.txt" for model in MODELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential preflight, training, inference, and relative-L2 evaluation"
    )
    parser.add_argument(
        "phase", choices=("preflight", "train", "infer", "evaluate", "all"),
        nargs="?", default="all",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use 20 train-pool and 2 published-test cases, one epoch, and isolated outputs.",
    )
    parser.add_argument("--run-name", help="Smoke output label (default: timestamp).")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=SUITE_ROOT, check=True)


def copy_subset(source: Path, target: Path, count: int) -> None:
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["benchmark_role"] = f"smoke_subset_first_{count}"
            data = dst.create_group("data")
            for sample_id in range(count):
                src.copy(src[f"data/{sample_id}"], data, name=str(sample_id))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_config(path: Path) -> list[tuple[str | None, str]]:
    rows: list[tuple[str | None, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "%", "'")):
            rows.append((None, raw))
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse config line in {path}: {raw!r}")
        rows.append((parts[0], parts[1]))
    return rows


def write_runtime_config(source: Path, target: Path, overrides: dict[str, str]) -> None:
    rows = parse_config(source)
    seen: set[str] = set()
    output: list[str] = []
    for key, raw_or_value in rows:
        if key is None:
            output.append(raw_or_value)
            continue
        value = overrides.get(key, raw_or_value)
        output.append(f"{key}\t{value}")
        seen.add(key)
    missing = sorted(set(overrides) - seen)
    if missing:
        raise ValueError(f"Overrides are absent from {source}: {missing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")


def smoke_configs(models: list[str], run_name: str) -> tuple[dict[str, Path], dict[str, Path]]:
    smoke_train = HERE / "elasticity_smoke_train.h5"
    smoke_test = HERE / "elasticity_smoke_test.h5"
    copy_subset(HERE / "elasticity_train.h5", smoke_train, 20)
    copy_subset(HERE / "elasticity_test.h5", smoke_test, 2)

    runtime_root = SUITE_ROOT / "output" / "benchmarks" / "elasticity" / "smoke" / run_name
    config_dir = runtime_root / "runtime_configs"
    train_configs: dict[str, Path] = {}
    infer_configs: dict[str, Path] = {}
    for model in models:
        model_root = runtime_root / model
        common = {
            "dataset_dir": "../dataset/benchmarks/elasticity/elasticity_smoke_train.h5",
            "infer_dataset": "../dataset/benchmarks/elasticity/elasticity_smoke_test.h5",
            "modelpath": f"../output/benchmarks/elasticity/smoke/{run_name}/{model}/model.pth",
            "inference_output_dir": f"../output/benchmarks/elasticity/smoke/{run_name}/{model}/inference",
        }
        train_target = config_dir / f"config_train_{model}.txt"
        infer_target = config_dir / f"config_infer_{model}.txt"
        train_overrides = common | {
            "log_file_dir": f"../output/benchmarks/elasticity/smoke/{run_name}/{model}/train.log",
            "training_epochs": "1",
            "warmup_epochs": "1",
            "test_interval": "1",
            "test_max_batches": "2",
        }
        if model != "transolver":
            train_overrides["checkpoint_interval"] = "1"
        write_runtime_config(
            CONFIG_ROOT / TRAIN_CONFIGS[model], train_target, train_overrides,
        )
        write_runtime_config(
            CONFIG_ROOT / INFER_CONFIGS[model], infer_target,
            common | {
                "log_file_dir": f"../output/benchmarks/elasticity/smoke/{run_name}/{model}/infer.log",
            },
        )
        train_configs[model] = train_target
        infer_configs[model] = infer_target
    return train_configs, infer_configs


def write_comparison(result_root: Path, prediction_dirs: dict[str, Path], smoke: bool) -> None:
    import torch

    rows: list[dict[str, object]] = []
    for model, prediction_dir in prediction_dirs.items():
        result = json.loads((prediction_dir / "relative_l2.json").read_text(encoding="utf-8"))
        reference = result["paper_reference"]
        checkpoint_path = prediction_dir.parent / "model.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_config_path = (
            result_root / "runtime_configs" / f"config_train_{model}.txt"
            if smoke else CONFIG_ROOT / TRAIN_CONFIGS[model]
        )
        train_config = {
            key: value for key, value in parse_config(train_config_path) if key is not None
        }
        rows.append({
            "model": model,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "internal_validation_normalized_mse": float(checkpoint["valid_loss"]),
            "training_epochs": int(train_config["training_epochs"]),
            "batch_size": int(train_config["batch_size"]),
            "evaluated_samples": result["evaluated_samples"],
            "mean_relative_l2": result["relative_l2"]["mean"],
            "std_relative_l2": result["relative_l2"]["std"],
            "median_relative_l2": result["relative_l2"]["median"],
            "reported_model": reference["reported_model"],
            "reported_mean_relative_l2": reference["reported_mean_relative_l2"],
            "reported_display": reference.get(
                "reported_display",
                ("n/a" if reference["reported_mean_relative_l2"] is None
                 else f"{float(reference['reported_mean_relative_l2']):.6f} mean relative L2"),
            ),
            "paper": reference["paper"],
            "comparison": reference["comparison"],
            "mean_minus_reported": result["mean_minus_reported"],
            "mean_over_reported": result["mean_over_reported"],
        })

    provenance = json.loads((HERE / "elasticity.provenance.json").read_text(encoding="utf-8"))
    protocol = {
        "dataset": provenance["benchmark"],
        "source_sha256": provenance["source_sha256"],
        "mesh_points_per_sample": provenance["mesh_points_per_sample"],
        "training_pool_source_indices": (
            [0, 19] if smoke else provenance["suite_training_pool_source_indices"]
        ),
        "training_pool_size": 20 if smoke else provenance["suite_training_pool_size"],
        "split": (
            "unchanged seeded 80/10/10 split (16/2/2)"
            if smoke else provenance["suite_split"]
        ),
        "published_test_source_indices": (
            [1800, 1801] if smoke else provenance["published_test_source_indices"]
        ),
        "published_test_size": 2 if smoke else 200,
        "training_objective": "normalized mean squared error (unchanged production runtime)",
        "reported_metric": "mean per-sample relative L2 on de-normalized saved inference fields",
        "comparability_note": provenance["comparability_note"],
    }
    result_root.mkdir(parents=True, exist_ok=True)
    csv_path = result_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (result_root / "comparison.json").write_text(
        json.dumps({"smoke": smoke, "protocol": protocol, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Elasticity validation comparison",
        "",
        ("Smoke run only; these one-epoch values are not accuracy results."
         if smoke else
         "Full benchmark on the isolated published 200-case test partition."),
        "",
        "## Protocol",
        "",
        f"- Public dataset: {protocol['dataset']} ({protocol['mesh_points_per_sample']} points per case).",
        ("- Training pool: source cases 0-19; unchanged seeded split gives 16/2/2 train/validation/internal-test cases."
         if smoke else
         "- Training pool: source cases 0-1249; unchanged seeded split gives 1,000/125/125 train/validation/internal-test cases."),
        (f"- Final test: published source cases 1800-1801 ({protocol['published_test_size']} smoke cases)."
         if smoke else
         f"- Final test: exact published source cases 1800-1999 ({protocol['published_test_size']} cases), isolated in a separate HDF5 file."),
        f"- Training objective: {protocol['training_objective']}.",
        f"- Reported metric: {protocol['reported_metric']}.",
        ("- Qualification: smoke results are functional checks only and are not accuracy comparisons."
         if smoke else
         "- Qualification: the official test partition is exact, but this is not a strict paper reproduction. The 1,000 optimization cases are the suite's seeded selection from source cases 0-1249 rather than exactly source cases 0-999, and training retains normalized MSE plus the suite scheduler/runtime rather than the authors' decoded relative-L2 recipe."),
        "",
        "## Results",
        "",
        "| Suite model | Epoch | Batch | Internal val MSE | Samples | Mean relative L2 | Median | Paper reference | Reported paper result | Mean / reported rel-L2 | Comparability |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        reported = row["reported_mean_relative_l2"]
        reported_text = str(row["reported_display"])
        ratio = row["mean_over_reported"]
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}x"
        paper_model = row["reported_model"] or "n/a"
        if row["paper"]:
            paper_model = f"[{paper_model}]({row['paper']})"
        comparison = str(row["comparison"]).replace("|", "\\|")
        lines.append(
            f"| {row['model']} | {row['checkpoint_epoch']} | {row['batch_size']} | "
            f"{float(row['internal_validation_normalized_mse']):.6e} | "
            f"{row['evaluated_samples']} | {float(row['mean_relative_l2']):.6f} | "
            f"{float(row['median_relative_l2']):.6f} | {paper_model} | "
            f"{reported_text} | {ratio_text} | {comparison} |"
        )
    lines.extend([
        "",
        "Only the Transolver row uses the same benchmark and an architecture-aligned configuration. It is not a strict reproduction because of the training-composition, objective, scheduler, and suite-runtime qualifications above. Geo-FNO is contextual for the suite's FNO adapter; Point-DeepONet and GINO were published on different 3D datasets and metrics; no like-for-like paper number exists for this DeepONet adapter on Elasticity.",
        "",
        "## Source integrity",
        "",
        f"- `Random_UnitCell_XY_10.npy`: `{provenance['source_sha256']['Random_UnitCell_XY_10.npy']}`",
        f"- `Random_UnitCell_sigma_10.npy`: `{provenance['source_sha256']['Random_UnitCell_sigma_10.npy']}`",
    ])
    (result_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison: {result_root / 'comparison.md'}")


def main() -> None:
    args = parse_args()
    models = list(args.models)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.smoke and args.run_name:
        raise ValueError("--run-name is only used with --smoke")

    if args.smoke:
        train_configs, infer_configs = smoke_configs(models, run_name)
        truth = HERE / "elasticity_smoke_test.h5"
    else:
        train_configs = {model: CONFIG_ROOT / TRAIN_CONFIGS[model] for model in models}
        infer_configs = {model: CONFIG_ROOT / INFER_CONFIGS[model] for model in models}
        truth = HERE / "elasticity_test.h5"

    launcher = str(SUITE_ROOT / "AI_CAE4ALL_main.py")
    phases = ("preflight", "train", "infer", "evaluate") if args.phase == "all" else (args.phase,)

    if "preflight" in phases:
        for model in models:
            run([sys.executable, launcher, "--config", str(train_configs[model]), "--check"])

    if "train" in phases:
        for model in models:
            run([sys.executable, launcher, "--config", str(train_configs[model])])

    if "infer" in phases:
        for model in models:
            run([sys.executable, launcher, "--config", str(infer_configs[model]), "--check"])
            if model == "transolver":
                run([
                    sys.executable, str(HERE / "infer_transolver_compat.py"),
                    "--config", str(infer_configs[model]),
                ])
            else:
                run([sys.executable, launcher, "--config", str(infer_configs[model])])

    if "evaluate" in phases:
        prediction_dirs: dict[str, Path] = {}
        for model in models:
            if args.smoke:
                predictions = (
                    SUITE_ROOT / "output" / "benchmarks" / "elasticity" / "smoke" /
                    run_name / model / "inference"
                )
            else:
                predictions = SUITE_ROOT / "output" / "benchmarks" / "elasticity" / model / "inference"
            prediction_dirs[model] = predictions
            run([
                sys.executable, str(HERE / "evaluate_relative_l2.py"),
                "--model", model,
                "--ground-truth", str(truth),
                "--predictions", str(predictions),
            ])
        comparison_root = (
            SUITE_ROOT / "output" / "benchmarks" / "elasticity" / "smoke" / run_name
            if args.smoke else SUITE_ROOT / "output" / "benchmarks" / "elasticity"
        )
        write_comparison(comparison_root, prediction_dirs, args.smoke)


if __name__ == "__main__":
    main()
