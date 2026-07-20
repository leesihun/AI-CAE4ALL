from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest
import h5py
import numpy as np
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from run_campaign import (  # noqa: E402
    CAMPAIGN_MODELS,
    EVALUATION_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    GPU_PEAK_LIMIT_MIB,
    MIN_PAIR_THROUGHPUT_GAIN,
    RESOURCE_INDEX_SCHEMA_VERSION,
    RESOURCE_PROBE_SCHEMA_VERSION,
    TRAIN_PROFILES,
    CampaignLock,
    CampaignError,
    CommandResult,
    PlasticityCampaign,
    SubprocessRunner,
    _parse_models,
    paper_completion_gate_or_user_waiver,
)


def _write_config(path: Path, model: str, mode: str) -> None:
    native_model = "MeshGraphNets" if model in {"meshgraphnets", "hi_meshgraphnets"} else model
    dataset = {
        "meshgraphnets": "plasticity_meshgraphnets_runtime.h5",
        "hi_meshgraphnets": "plasticity_hi_meshgraphnets_runtime.h5",
    }.get(model, "plasticity.h5")
    batch_size, accumulation = {
        "point_deeponet": (2, 2),
        "gino": (1, 4),
        "transolver": (1, 4),
    }.get(model, (4, 1))
    path.write_text(
        "\n".join(
            [
                f"model\t{native_model}",
                f"mode\t{mode}",
                "gpu_ids\t0",
                "parallel_mode\tddp",
                f"dataset_dir\t../dataset/benchmarks/plasticity/{dataset}",
                "infer_dataset\t../dataset/benchmarks/plasticity/plasticity_seed42_test.h5",
                f"modelpath\t../output/benchmarks/plasticity/{model}/model.pth",
                f"inference_output_dir\t../output/benchmarks/plasticity/{model}/inference",
                "infer_timesteps\t19",
                "split_seed\t42",
                "input_var\t4",
                "output_var\t4",
                f"batch_size\t{batch_size}",
                f"grad_accum_steps\t{accumulation}",
                "use_amp\tTrue",
                "num_workers\t2",
                "write_preprocessing\tFalse",
                f"training_epochs\t{'500' if mode == 'train' else '1'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _make_suite(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    benchmark = root / "dataset" / "benchmarks" / "plasticity"
    configs = root / "configs" / "benchmarks" / "plasticity"
    benchmark.mkdir(parents=True)
    configs.mkdir(parents=True)
    for name in ("CAE_ML_Suite_main.py",):
        (root / name).write_text("# fixture\n", encoding="utf-8")
    for name in ("evaluate_rollouts.py", "compare_results.py"):
        (benchmark / name).write_text("# fixture\n", encoding="utf-8")

    source = benchmark / "plasticity.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["benchmark"] = "plasticity"
        metadata = handle.create_group("metadata")
        normalization = metadata.create_group("normalization_params")
        normalization.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        normalization.create_dataset("std", data=np.ones(8, dtype=np.float32))
        data = handle.create_group("data")
        data.create_dataset("payload", data=np.arange(12, dtype=np.float32))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(source, benchmark / "plasticity_meshgraphnets_runtime.h5")
    shutil.copy2(source, benchmark / "plasticity_hi_meshgraphnets_runtime.h5")
    (benchmark / "plasticity_seed42_test.h5").write_bytes(b"truth")
    (benchmark / "plasticity.provenance.json").write_text(
        json.dumps({"converted_hdf5_sha256": source_hash}), encoding="utf-8"
    )
    for model in CAMPAIGN_MODELS:
        _write_config(configs / f"config_train_{model}.txt", model, "train")
        _write_config(configs / f"config_infer_{model}.txt", model, "inference")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_paper_gate(root: Path) -> Path:
    output = root / "output" / "benchmarks"
    artifacts = output / "paper_validation_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report = artifacts / "report.md"
    report.write_text("# validated\n", encoding="utf-8")
    validations: dict[str, object] = {}
    for index, model in enumerate(("fno", "transolver", "deeponet", "point_deeponet", "gino")):
        artifact = artifacts / f"{model}.json"
        artifact.write_text(json.dumps({"model": model}), encoding="utf-8")
        validations[model] = {
            "complete": True,
            "status": "passed",
            "benchmark": f"fixture-{model}",
            "metric": "relative_l2",
            "paper_value": 0.1 + index * 0.01,
            "measured_value": 0.1 + index * 0.01,
            "primary_artifact": {
                "path": artifact.relative_to(root).as_posix(),
                "sha256": _sha256(artifact),
            },
        }
    gate = output / "paper_validation_completion_gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": GATE_SCHEMA_VERSION,
                "complete": True,
                "status": "passed",
                "completed_at": "2026-07-19T00:00:00+00:00",
                "report": {
                    "path": report.relative_to(root).as_posix(),
                    "sha256": _sha256(report),
                },
                "validations": validations,
            }
        ),
        encoding="utf-8",
    )
    return gate


def _write_probe_evidence(
    root: Path,
    *,
    profiles: Mapping[str, str],
    device: str,
    cpu_eligible: bool = False,
    pair_certified: bool = False,
    improvement: float = 0.20,
    peak_mib: int = 5000,
) -> str:
    models = sorted(profiles)
    mode = "pair" if len(models) == 2 else "single"
    model_identity: list[dict[str, object]] = []
    model_records: dict[str, object] = {}
    for model in models:
        resolved = {
            name: (batch, accumulation)
            for name, batch, accumulation in TRAIN_PROFILES[model]
        }
        batch, accumulation = resolved[profiles[model]]
        identity_item = {
            "model": model,
            "profile": profiles[model],
            "batch_size": batch,
            "grad_accum_steps": accumulation,
        }
        model_identity.append(identity_item)
        model_records[model] = {**identity_item, "complete": True}
    source = root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
    identity = {
        "mode": mode,
        "device": device,
        "models": model_identity,
        "models_key": ",".join(models),
        "profile_key": ",".join(f"{model}={profiles[model]}" for model in models),
        "source_sha256": _sha256(source),
        "selected_ids_sha256": hashlib.sha256(b"fixture-selected-ids").hexdigest(),
        "cases": 32,
        "seed": 42,
    }
    identity_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    completed_at = "2026-07-19T01:00:00+00:00"
    run_id = f"fixture-{identity_key[:12]}"
    record: dict[str, object] = {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "device": device,
        "state": "complete",
        "complete": True,
        "completed_at": completed_at,
        "gate": {
            "schema_version": GATE_SCHEMA_VERSION,
            "status": "passed",
            "complete": True,
        },
        "dataset": {
            "source_sha256": identity["source_sha256"],
            "selected_ids_sha256": identity["selected_ids_sha256"],
            "cases": 32,
            "seed": 42,
        },
        "models": model_records,
        "baselines": ([{"model": model} for model in models] if mode == "pair" else []),
        "identity": identity,
        "identity_key": identity_key,
    }
    if device == "gpu":
        record["gpu"] = {"complete": True, "peak_total_used_mib": peak_mib}
    if mode == "single":
        record["projection"] = {
            "conservative": True,
            "projected_wall_seconds_500_epochs": 3600.0,
        }
        record["placement"] = {
            "cpu_eligible": cpu_eligible,
            "model": models[0],
            "device": device,
            "profile": profiles[models[0]],
        }
        if device == "gpu":
            record["resource_fit"] = {
                "peak_limit_mib": GPU_PEAK_LIMIT_MIB,
                "observed_peak_total_used_mib": peak_mib,
                "peak_pass": peak_mib <= GPU_PEAK_LIMIT_MIB,
            }
    else:
        record["certification"] = {
            "certified": pair_certified,
            "peak_limit_mib": GPU_PEAK_LIMIT_MIB,
            "observed_peak_total_used_mib": peak_mib,
            "peak_pass": peak_mib <= GPU_PEAK_LIMIT_MIB,
            "required_throughput_improvement": MIN_PAIR_THROUGHPUT_GAIN,
            "throughput_improvement_fraction": improvement,
            "throughput_pass": improvement >= MIN_PAIR_THROUGHPUT_GAIN,
        }
    result_path = (
        root
        / "output"
        / "benchmarks"
        / "plasticity"
        / "resource_probe"
        / "runs"
        / run_id
        / "resource_probe_result.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record), encoding="utf-8")
    reference = {
        "identity_key": identity_key,
        "identity": identity,
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "completed_at": completed_at,
        "run_id": run_id,
    }
    index_path = result_path.parents[2] / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {
            "schema_version": RESOURCE_INDEX_SCHEMA_VERSION,
            "updated_at": completed_at,
            "latest_completed_single": {},
            "latest_cpu_eligible": {},
            "latest_certified_pair": {},
        }
    if mode == "single":
        index["latest_completed_single"][identity_key] = reference
        if cpu_eligible:
            index["latest_cpu_eligible"][identity_key] = reference
    elif pair_certified:
        index["latest_certified_pair"][identity_key] = reference
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return identity_key


def _fixture_config_pins(root: Path) -> dict[str, dict[str, str]]:
    config_dir = root / "configs" / "benchmarks" / "plasticity"
    return {
        model: {
            phase: hashlib.sha256(
                (config_dir / f"config_{phase}_{model}.txt").read_bytes()
            ).hexdigest()
            for phase in ("train", "infer")
        }
        for model in CAMPAIGN_MODELS
    }


def _fake_result_validator(path: Path, model: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model or payload.get("complete") is not True:
        raise ValueError("fake strict validator rejected result")
    return (
        {"mean_per_case_full_trajectory_relative_l2": payload["primary_metric"]["value"]},
        {"fixture_identity": "strict-validator-boundary"},
    )


def _campaign(root: Path, runner) -> PlasticityCampaign:
    _write_paper_gate(root)
    truth = root / "dataset" / "benchmarks" / "plasticity" / "plasticity_seed42_test.h5"
    return PlasticityCampaign(
        root,
        python_executable="python",
        runner=runner,
        pinned_config_sha256=_fixture_config_pins(root),
        pinned_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
        result_validator=_fake_result_validator,
    )


def _write_metrics(inference: Path, model: str) -> None:
    inference.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model": model,
        "complete": True,
        "validation": {"evaluated_cases": 100},
        "predictions": {"files": 100, "manifest_sha256": f"manifest-{model}"},
        "primary_metric": {
            "name": "mean_per_case_full_trajectory_relative_l2",
            "value": 0.01,
        },
    }
    (inference / "plasticity_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with (inference / "plasticity_per_case_time.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "time_index"])
        for index in range(1900):
            writer.writerow([index // 19, index % 19 + 1])


def _write_rollouts(inference: Path) -> None:
    inference.mkdir(parents=True, exist_ok=True)
    for sample_id in range(100):
        (inference / f"rollout_sample{sample_id}_steps19.h5").write_bytes(b"rollout")


def _write_checkpoint(
    root: Path,
    model: str,
    *,
    epoch: int = 499,
    config_path: Path | None = None,
) -> None:
    output = root / "output" / "benchmarks" / "plasticity" / model / "model.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    normalization = {
        "node_mean": torch.zeros(8),
        "node_std": torch.ones(8),
        "delta_mean": torch.zeros(4),
        "delta_std": torch.ones(4),
    }
    common = {
        "epoch": epoch,
        "model_state_dict": {"weight": torch.ones(1)},
        "ema_state_dict": {"module.weight": torch.ones(1)},
        "optimizer_state_dict": {"state": {}},
        "scheduler_state_dict": {"last_epoch": epoch},
        "normalization": normalization,
    }
    if model in {"meshgraphnets", "hi_meshgraphnets"}:
        normalization.update({"edge_mean": torch.zeros(8), "edge_std": torch.ones(8)})
        if model == "hi_meshgraphnets":
            normalization.update(
                {
                    "coarse_edge_means": [torch.zeros(8), torch.zeros(8)],
                    "coarse_edge_stds": [torch.ones(8), torch.ones(8)],
                }
            )
        checkpoint = {
            **common,
            "model_config": {
                "input_var": 4,
                "output_var": 4,
                "edge_var": 8,
                "latent_dim": 128,
                "message_passing_num": 15,
                "positional_features": 4,
                "use_node_types": False,
                "use_world_edges": False,
                "use_multiscale": model == "hi_meshgraphnets",
                **(
                    {
                        "multiscale_levels": 2,
                        "mp_per_level": [4, 6, 8, 6, 4],
                        "coarsening_type": "voronoi_seedmean",
                        "voronoi_clusters": [500, 100],
                    }
                    if model == "hi_meshgraphnets"
                    else {}
                ),
            },
        }
    elif model == "transolver":
        checkpoint = {
            **common,
            "checkpoint_version": 1,
            "model_config": {
                "model": "transolver",
                "input_var": 4,
                "output_var": 4,
                "positional_features": 4,
                "use_node_types": False,
                "latent_dim": 128,
                "num_layers": 8,
                "num_heads": 8,
                "slice_num": 64,
                "attention_kernel": "slice_space",
                "mlp_ratio": 1,
                "dropout": 0.0,
                "temperature_init": 0.5,
                "temperature_min": 0.1,
                "temperature_max": 5.0,
                "small_output_init": True,
                "use_checkpointing": True,
                "num_timesteps": 20,
            },
            "data_config": {
                "split_seed": 42,
                "coordinate_normalization": "centered_isotropic",
                "num_timesteps": 20,
                "chunk_size": 1024,
                "infer_mode": "direct",
                "infer_chunk_size": 0,
                "feature_loss_weights": [1.0, 1.0, 1.0, 1.0],
                "std_noise": 0.0,
                "noise_gamma": 1,
            },
        }
    else:
        source = root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
        architectures = {
            "point_deeponet": {
                "point_variant": "mesh_state",
                "point_sensor_count": 2048,
                "point_hidden_channels": 128,
                "point_feature_dim": 128,
                "pointnet_depth": 3,
                "point_condition_depth": 2,
                "point_trunk_depth": 3,
                "point_refiner_depth": 2,
                "point_siren_omega0": 30.0,
                "point_output_activation": "identity",
                "query_dim": 6,
            },
            "deeponet": {
                "deeponet_branch_source": "fixed_sensors",
                "deeponet_sensor_resolution": [32, 16],
                "deeponet_hidden_channels": 256,
                "deeponet_branch_depth": 3,
                "deeponet_trunk_depth": 3,
                "deeponet_basis_dim": 128,
                "deeponet_activation": "silu",
                "deeponet_multi_output": "split_both",
                "branch_in_dim": 5120,
                "query_dim": 6,
            },
            "fno": {
                "fno_grid_resolution": [64, 32],
                "fno_modes": [16, 12],
                "fno_hidden_channels": 64,
                "fno_layers": 4,
                "fno_use_channel_mlp": True,
                "fno_norm": "none",
                "in_channels": 12,
            },
            "gino": {
                "gino_variant": "mesh_state",
                "gino_grid_resolution": [64, 32],
                "gino_fno_modes": [16, 12],
                "gino_fno_hidden_channels": 64,
                "gino_fno_layers": 4,
                "gino_in_radius": 0.08,
                "gino_out_radius": 0.08,
                "gino_kernel_hidden": 64,
                "gino_use_torch_cluster": False,
                "source_feat_dim": 8,
            },
        }
        checkpoint = {
            **common,
            "schema_version": "deeponet_repo_v1",
            "selected_model": model,
            "model_config": {"model_name": model, **architectures[model]},
            "data_config": {
                "input_var": 4,
                "output_var": 4,
                "positional_dim": 4,
                "node_type_dim": 0,
                "global_condition_dim": 0,
                "operator_dim": 2,
                "active_axes": [0, 1],
                "has_sdf": False,
                "num_timesteps": 20,
            },
            "source_reference": {
                "config_file": str(
                    (
                        config_path
                        or root
                        / "configs"
                        / "benchmarks"
                        / "plasticity"
                        / f"config_train_{model}.txt"
                    ).resolve()
                ),
                "dataset": {
                    "path": str(source.resolve()),
                    "size": source.stat().st_size,
                    "head_sha1": hashlib.sha1(source.read_bytes()[: 1024 * 1024]).hexdigest(),
                },
            },
        }
    torch.save(checkpoint, output)


def _write_complete_model(root: Path, model: str) -> None:
    model_root = root / "output" / "benchmarks" / "plasticity" / model
    model_root.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(root, model)
    inference = model_root / "inference"
    _write_rollouts(inference)
    _write_metrics(inference, model)


class FakeRunner:
    def __init__(
        self,
        root: Path,
        *,
        fail_model: str | None = None,
        fail_phase: str | None = None,
        omit_checkpoint: bool = False,
        mutate_checkpoint_on_evaluate: str | None = None,
        unexpected_file_on_evaluate: str | None = None,
        phase_delay: float = 0.0,
    ) -> None:
        self.root = root
        self.fail_model = fail_model
        self.fail_phase = fail_phase
        self.omit_checkpoint = omit_checkpoint
        self.mutate_checkpoint_on_evaluate = mutate_checkpoint_on_evaluate
        self.unexpected_file_on_evaluate = unexpected_file_on_evaluate
        self.phase_delay = phase_delay
        self.calls: list[list[str]] = []
        self.call_records: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._active_gpu = 0
        self._active_cpu = 0
        self.max_active_gpu = 0
        self.max_active_total = 0

    @staticmethod
    def _option(command: Sequence[str], name: str) -> str | None:
        try:
            return str(command[list(command).index(name) + 1])
        except ValueError:
            return None

    @staticmethod
    def _model_from_config(command: Sequence[str]) -> tuple[str, str]:
        config = Path(FakeRunner._option(command, "--config") or "")
        stem = config.stem
        if stem.startswith("config_train_"):
            return stem.removeprefix("config_train_"), "train"
        return stem.removeprefix("config_infer_"), "infer"

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path | None,
        stderr_path: Path | None,
        job_id: str | None = None,
        env: Mapping[str, str] | None = None,
        on_start: Callable[[int], None] | None = None,
    ) -> CommandResult:
        command = list(command)
        with self._lock:
            self.calls.append(command)
            self.call_records.append(
                {"command": command, "job_id": job_id, "env": dict(env or {})}
            )
        if on_start is not None:
            on_start(10000 + len(self.calls))
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text("fake stdout\n", encoding="utf-8")
        if stderr_path is not None:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text("", encoding="utf-8")

        target = Path(command[1]).name
        if target == "CAE_ML_Suite_main.py":
            model, phase = self._model_from_config(command)
            if "--check" in command:
                return CommandResult(0, stdout="preflight passed")
            lane = "cpu" if (env or {}).get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
            with self._lock:
                if lane == "cpu":
                    self._active_cpu += 1
                else:
                    self._active_gpu += 1
                self.max_active_gpu = max(self.max_active_gpu, self._active_gpu)
                self.max_active_total = max(
                    self.max_active_total, self._active_gpu + self._active_cpu
                )
            try:
                if self.phase_delay:
                    time.sleep(self.phase_delay)
                if model == self.fail_model and phase == self.fail_phase:
                    return CommandResult(9, stderr="intentional failure")
                model_root = self.root / "output" / "benchmarks" / "plasticity" / model
                config_path = Path(self._option(command, "--config") or "")
                if phase == "train" and not self.omit_checkpoint:
                    _write_checkpoint(self.root, model, config_path=config_path)
                    if model == "transolver":
                        working = (
                            self.root
                            / "dataset"
                            / "benchmarks"
                            / "plasticity"
                            / "plasticity_transolver_runtime.h5"
                        )
                        with h5py.File(working, "r+") as handle:
                            root_group = handle["metadata/normalization_params"]
                            group = root_group.create_group("transolver")
                            for name, values in {
                                "node_mean": np.zeros(8),
                                "node_std": np.ones(8),
                                "delta_mean": np.zeros(4),
                                "delta_std": np.ones(4),
                            }.items():
                                group.create_dataset(name, data=np.asarray(values, dtype=np.float32))
                            group.attrs["position_scale"] = 1.0
                            group.attrs["coordinate_normalization"] = "centered_isotropic"
                            group.attrs["normalization_source"] = "train_split"
                            group.attrs["split_seed"] = 42
                if phase == "infer":
                    _write_rollouts(model_root / "inference")
                return CommandResult(0)
            finally:
                with self._lock:
                    if lane == "cpu":
                        self._active_cpu -= 1
                    else:
                        self._active_gpu -= 1

        if target == "evaluate_rollouts.py":
            model = self._option(command, "--model")
            assert model is not None
            prediction = Path(self._option(command, "--predictions") or "")
            output_json_raw = self._option(command, "--output-json")
            output_csv_raw = self._option(command, "--output-csv")
            if output_json_raw is None and output_csv_raw is None:
                _write_metrics(prediction, model)
            else:
                output_json = Path(output_json_raw or "")
                output_csv = Path(output_csv_raw or "")
                output_json.parent.mkdir(parents=True, exist_ok=True)
                _write_metrics(output_json.parent, model)
                generated_json = output_json.parent / "plasticity_metrics.json"
                generated_csv = output_json.parent / "plasticity_per_case_time.csv"
                generated_json.replace(output_json)
                generated_csv.replace(output_csv)
            if model == self.mutate_checkpoint_on_evaluate:
                checkpoint = prediction.parent / "model.pth"
                checkpoint.write_bytes(checkpoint.read_bytes() + b"mutated")
            if model == self.unexpected_file_on_evaluate:
                (prediction / "unexpected.tmp").write_text("unexpected", encoding="utf-8")
            return CommandResult(0)

        if target == "compare_results.py":
            output_root = Path(self._option(command, "--output-dir") or "")
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "comparison.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "models_present_and_valid": list(CAMPAIGN_MODELS),
                    }
                ),
                encoding="utf-8",
            )
            (output_root / "comparison.csv").write_text("model\n", encoding="utf-8")
            (output_root / "comparison.md").write_text("# complete\n", encoding="utf-8")
            return CommandResult(0)
        raise AssertionError(f"Unexpected command: {command}")


def _noncheck_suite_phases(calls: list[list[str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for command in calls:
        if Path(command[1]).name != "CAE_ML_Suite_main.py" or "--check" in command:
            continue
        result.append(FakeRunner._model_from_config(command))
    return result


def test_full_campaign_uses_safe_single_gpu_fallback_and_completes_atomically(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)

    assert campaign.run() == 0

    assert all("--check" in command for command in runner.calls[:7])
    assert runner.max_active_gpu == 1
    assert _noncheck_suite_phases(runner.calls) == [
        phase for model in CAMPAIGN_MODELS for phase in ((model, "train"), (model, "infer"))
    ]
    assert Path(runner.calls[-1][1]).name == "compare_results.py"
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert status["complete"] is True
    assert status["state"] == "complete"
    assert status["comparison"]["state"] == "complete"
    assert all(status["models"][model]["state"] == "complete" for model in CAMPAIGN_MODELS)
    for model in CAMPAIGN_MODELS:
        model_status = status["models"][model]
        assert len(model_status["checkpoint_sha256"]) == 64
        assert model_status["checkpoint_identity"]["epoch"] == 499
        assert model_status["checkpoint_identity"]["sha256"] == model_status["checkpoint_sha256"]
    assert not list(campaign.status_path.parent.glob(".campaign_status.json.tmp.*"))
    assert (campaign.log_root / "meshgraphnets" / "train.attempt01.stdout.log").is_file()
    assert (
        campaign.log_root / "campaign" / "compare_results.attempt01.stderr.log"
    ).is_file()


def test_safe_recovery_skips_only_strictly_revalidated_complete_model(tmp_path):
    root = _make_suite(tmp_path)
    _write_complete_model(root, "meshgraphnets")
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)

    assert campaign.run(models=("meshgraphnets", "point_deeponet")) == 0

    phases = _noncheck_suite_phases(runner.calls)
    assert ("meshgraphnets", "train") not in phases
    assert ("meshgraphnets", "infer") not in phases
    assert phases == [("point_deeponet", "train"), ("point_deeponet", "infer")]
    assert any(
        Path(command[1]).name == "evaluate_rollouts.py"
        and FakeRunner._option(command, "--model") == "meshgraphnets"
        for command in runner.calls
    )
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["state"] == "incomplete"
    assert status["models"]["meshgraphnets"]["assessment"]["recovery_validated"] is True
    assert status["comparison"]["state"] == "not_run"


def test_ambiguous_preexisting_output_is_refused_before_launch(tmp_path):
    root = _make_suite(tmp_path)
    checkpoint = root / "output" / "benchmarks" / "plasticity" / "fno" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-only")
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="ambiguous preexisting"):
        campaign.run(models=("fno",))

    assert runner.calls == []
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["state"] == "failed"
    assert status["models"]["fno"]["state"] == "ambiguous"


def test_nonzero_training_fails_fast_and_never_compares(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root, fail_model="meshgraphnets", fail_phase="train")
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="meshgraphnets train failed"):
        campaign.run(models=("meshgraphnets", "point_deeponet"))

    assert _noncheck_suite_phases(runner.calls) == [("meshgraphnets", "train")]
    assert all(Path(command[1]).name != "compare_results.py" for command in runner.calls)
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["state"] == "failed"


def test_success_without_checkpoint_fails_before_inference(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root, omit_checkpoint=True)
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="checkpoint is missing or empty"):
        campaign.run(models=("gino",))

    assert _noncheck_suite_phases(runner.calls) == [("gino", "train")]


def test_dry_run_preflights_but_writes_no_campaign_artifacts(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)

    assert campaign.run(models=("deeponet", "fno"), dry_run=True) == 0

    assert len(runner.calls) == 7
    assert all("--check" in command for command in runner.calls)
    assert not campaign.status_path.exists()
    assert not campaign.log_root.exists()
    assert not campaign.lock_path.exists()
    assert _parse_models(["fno,deeponet"]) == ("deeponet", "fno")


def test_explicit_user_waiver_does_not_claim_paper_validation_passed(tmp_path):
    root = _make_suite(tmp_path)
    record = paper_completion_gate_or_user_waiver(
        root / "output" / "benchmarks" / "missing.json",
        root,
        allow_incomplete=True,
    )
    assert record["complete"] is False
    assert record["status"] == "waived_by_user"
    assert record["validations"] == {}
    assert record["authorization"]["type"] == "explicit_cli_override"
    assert record["authorization"]["scope"] == "plasticity_campaign"


def test_preliminary_epoch_budget_is_runtime_only_and_checkpoint_bound(tmp_path):
    root = _make_suite(tmp_path)
    truth = root / "dataset" / "benchmarks" / "plasticity" / "plasticity_seed42_test.h5"
    canonical = root / "configs" / "benchmarks" / "plasticity" / "config_train_fno.txt"
    canonical_before = canonical.read_bytes()
    campaign = PlasticityCampaign(
        root,
        runner=FakeRunner(root),
        pinned_config_sha256=_fixture_config_pins(root),
        pinned_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
        result_validator=_fake_result_validator,
        epoch_budget=5,
    )
    campaign.execution_plan = campaign._build_execution_plan(("fno",))
    materialized = campaign._materialize_runtime_configs(("fno",))
    runtime = campaign.effective_train_configs["fno"]
    runtime_text = runtime.read_text(encoding="utf-8")

    assert canonical.read_bytes() == canonical_before
    assert "training_epochs\t5" in runtime_text
    assert "warmup_epochs\t1" in runtime_text
    assert "val_interval\t1" in runtime_text
    assert materialized["configs"]["fno"]["train_updates"]["training_epochs"] == "5"

    _write_checkpoint(root, "fno", epoch=4, config_path=runtime)
    identity = campaign._validate_checkpoint("fno")
    assert identity["epoch"] == 4
    assert identity["epoch_budget"] == 5


def test_kernel_lock_rejects_second_contender_despite_stale_pid_metadata(tmp_path):
    lock_path = tmp_path / ".campaign.lock"
    lock_path.write_text(json.dumps({"pid": 99999999, "stale": True}), encoding="utf-8")
    first = CampaignLock(lock_path).acquire()
    try:
        with pytest.raises(CampaignError, match="Another Plasticity campaign owns"):
            CampaignLock(lock_path).acquire()
    finally:
        first.release()

    # Stale text is not authority: after kernel ownership is released a new
    # contender can acquire the same persistent lock file safely.
    third = CampaignLock(lock_path).acquire()
    third.release()


def test_exact_config_hash_drift_is_refused(tmp_path):
    root = _make_suite(tmp_path)
    pins = _fixture_config_pins(root)
    config = root / "configs" / "benchmarks" / "plasticity" / "config_train_fno.txt"
    config.write_text(config.read_text(encoding="utf-8") + "% drift\n", encoding="utf-8")
    truth = root / "dataset" / "benchmarks" / "plasticity" / "plasticity_seed42_test.h5"
    campaign = PlasticityCampaign(
        root,
        runner=FakeRunner(root),
        pinned_config_sha256=pins,
        pinned_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
        result_validator=_fake_result_validator,
    )
    with pytest.raises(CampaignError, match="Pinned config drift"):
        campaign.run(models=("fno",), dry_run=True)


def test_semantic_config_drift_is_refused_even_with_matching_hash_pin(tmp_path):
    root = _make_suite(tmp_path)
    config = root / "configs" / "benchmarks" / "plasticity" / "config_infer_gino.txt"
    config.write_text(
        config.read_text(encoding="utf-8").replace("split_seed\t42", "split_seed\t7"),
        encoding="utf-8",
    )
    truth = root / "dataset" / "benchmarks" / "plasticity" / "plasticity_seed42_test.h5"
    campaign = PlasticityCampaign(
        root,
        runner=FakeRunner(root),
        pinned_config_sha256=_fixture_config_pins(root),
        pinned_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
        result_validator=_fake_result_validator,
    )
    with pytest.raises(CampaignError, match="Pinned semantic drift"):
        campaign.run(models=("gino",), dry_run=True)


def test_intermediate_epoch_checkpoint_is_not_recoverable(tmp_path):
    root = _make_suite(tmp_path)
    _write_complete_model(root, "deeponet")
    _write_checkpoint(root, "deeponet", epoch=450)
    campaign = _campaign(root, FakeRunner(root))

    with pytest.raises(CampaignError, match="ambiguous preexisting"):
        campaign.run(models=("deeponet",))
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert "expected final epoch 499" in status["models"]["deeponet"]["assessment"]["reason"]


def test_checkpoint_identity_mismatch_is_not_recoverable(tmp_path):
    root = _make_suite(tmp_path)
    _write_complete_model(root, "point_deeponet")
    checkpoint_path = (
        root / "output" / "benchmarks" / "plasticity" / "point_deeponet" / "model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["selected_model"] = "deeponet"
    torch.save(checkpoint, checkpoint_path)
    campaign = _campaign(root, FakeRunner(root))

    with pytest.raises(CampaignError, match="ambiguous preexisting"):
        campaign.run(models=("point_deeponet",))
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert "selected_model" in status["models"]["point_deeponet"]["assessment"]["reason"]


def test_checkpoint_hash_must_remain_stable_through_evaluation(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root, mutate_checkpoint_on_evaluate="fno")
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="checkpoint changed"):
        campaign.run(models=("fno",))
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["models"]["fno"]["state"] == "failed"


def test_post_evaluation_allowed_file_audit_rejects_surprise_artifact(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root, unexpected_file_on_evaluate="gino")
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="allowed-file audit failed"):
        campaign.run(models=("gino",))


def test_fresh_mgn_requires_bit_identical_working_copy(tmp_path):
    root = _make_suite(tmp_path)
    working = (
        root
        / "dataset"
        / "benchmarks"
        / "plasticity"
        / "plasticity_meshgraphnets_runtime.h5"
    )
    working.write_bytes(working.read_bytes() + b"drift")
    campaign = _campaign(root, FakeRunner(root))

    with pytest.raises(CampaignError, match="bit-identical working copy"):
        campaign.run(models=("meshgraphnets",), dry_run=True)


def _replace_source_with_minimal_hdf5(root: Path) -> tuple[Path, Path]:
    benchmark = root / "dataset" / "benchmarks" / "plasticity"
    source = benchmark / "plasticity.h5"
    working = benchmark / "plasticity_meshgraphnets_runtime.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["benchmark"] = "plasticity"
        metadata = handle.create_group("metadata")
        normalization = metadata.create_group("normalization_params")
        normalization.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        normalization.create_dataset("std", data=np.ones(8, dtype=np.float32))
        data = handle.create_group("data")
        data.create_dataset("payload", data=np.arange(12, dtype=np.float32))
    shutil.copy2(source, working)
    shutil.copy2(source, benchmark / "plasticity_hi_meshgraphnets_runtime.h5")
    provenance = benchmark / "plasticity.provenance.json"
    provenance.write_text(
        json.dumps({"converted_hdf5_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    return source, working


def test_completed_mgn_recovery_allows_only_normalization_metadata_drift(tmp_path):
    root = _make_suite(tmp_path)
    _, working = _replace_source_with_minimal_hdf5(root)
    _write_complete_model(root, "meshgraphnets")
    with h5py.File(working, "r+") as handle:
        normalization = handle["metadata/normalization_params"]
        for name, values in {
            "node_mean": np.zeros(8),
            "node_std": np.ones(8),
            "edge_mean": np.zeros(8),
            "edge_std": np.ones(8),
            "delta_mean": np.zeros(4),
            "delta_std": np.ones(4),
        }.items():
            normalization.create_dataset(name, data=np.asarray(values, dtype=np.float32))
        normalization.attrs["edge_feature_layout"] = (
            "deformed_dx,deformed_dy,deformed_dz,deformed_dist,"
            "ref_dx,ref_dy,ref_dz,ref_dist"
        )
        normalization.attrs["edge_var"] = 8
        normalization.attrs["normalization_source"] = "train_split"
        normalization.attrs["split_seed"] = 42
    campaign = _campaign(root, FakeRunner(root))

    assert campaign.run(models=("meshgraphnets",)) == 0
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    validation = status["models"]["meshgraphnets"]["assessment"][
        "mgn_working_copy_validation"
    ]
    assert validation["mode"] == "normalization_only_drift"


def test_truth_hash_drift_is_refused_before_preflight(tmp_path):
    root = _make_suite(tmp_path)
    truth = root / "dataset" / "benchmarks" / "plasticity" / "plasticity_seed42_test.h5"
    pinned = hashlib.sha256(truth.read_bytes()).hexdigest()
    truth.write_bytes(truth.read_bytes() + b"drift")
    campaign = PlasticityCampaign(
        root,
        runner=FakeRunner(root),
        pinned_config_sha256=_fixture_config_pins(root),
        pinned_truth_sha256=pinned,
        result_validator=_fake_result_validator,
    )
    with pytest.raises(CampaignError, match="not the pinned campaign truth"):
        campaign.run(models=("fno",), dry_run=True)


def test_real_launch_fails_closed_on_missing_paper_gate_before_runner_call(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)
    campaign.paper_gate_path.unlink()

    with pytest.raises(CampaignError, match="completion gate exists"):
        campaign.run(models=("fno",))

    assert runner.calls == []
    assert not campaign.status_path.exists()
    assert not campaign.runtime_config_root.exists()


def test_dry_run_remains_allowed_without_paper_gate_and_materializes_nothing(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root)
    campaign = _campaign(root, runner)
    campaign.paper_gate_path.unlink()

    assert campaign.run(models=("fno",), dry_run=True) == 0
    assert len(runner.calls) == len(CAMPAIGN_MODELS)
    assert not campaign.runtime_config_root.exists()
    assert not campaign.status_path.exists()
    assert not campaign.lock_path.exists()


def test_strict_cpu_probe_runs_only_deeponet_cpu_lane_concurrently_with_gpu(tmp_path):
    root = _make_suite(tmp_path)
    _write_probe_evidence(
        root,
        profiles={"deeponet": "fallback_1x4"},
        device="cpu",
        cpu_eligible=True,
    )
    runner = FakeRunner(root, phase_delay=0.03)
    campaign = _campaign(root, runner)

    assert campaign.run(models=("deeponet", "fno")) == 0

    assert runner.max_active_gpu == 1
    assert runner.max_active_total >= 2
    deep_records = [
        value
        for value in runner.call_records
        if value["job_id"] and str(value["job_id"]).startswith("deeponet.train")
    ]
    assert len(deep_records) == 1
    environment = deep_records[0]["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["OMP_NUM_THREADS"] == "8"
    runtime = campaign.effective_train_configs["deeponet"].read_text(encoding="utf-8")
    assert "gpu_ids\t-1" in runtime
    assert "use_amp\tFalse" in runtime
    assert "batch_size\t1" in runtime
    assert "grad_accum_steps\t4" in runtime


def test_only_exact_certified_pair_can_overlap_on_gpu(tmp_path):
    root = _make_suite(tmp_path)
    identity_key = _write_probe_evidence(
        root,
        profiles={"fno": "fallback_1x4", "gino": "canonical"},
        device="gpu",
        pair_certified=True,
    )
    runner = FakeRunner(root, phase_delay=0.03)
    campaign = _campaign(root, runner)

    assert campaign.run(models=("fno", "gino")) == 0

    assert runner.max_active_gpu == 2
    assert campaign.execution_plan["gpu_groups"] == [["fno", "gino"]]
    assert campaign.execution_plan["models"]["fno"]["evidence"]["identity_key"] == identity_key


def test_uncertified_pair_falls_back_to_one_gpu_job(tmp_path):
    root = _make_suite(tmp_path)
    _write_probe_evidence(
        root,
        profiles={"fno": "fallback_1x4", "gino": "canonical"},
        device="gpu",
        pair_certified=True,
        improvement=0.05,
    )
    runner = FakeRunner(root, phase_delay=0.02)
    campaign = _campaign(root, runner)

    assert campaign.run(models=("fno", "gino")) == 0

    assert runner.max_active_gpu == 1
    assert campaign.execution_plan["gpu_groups"] == [["fno"], ["gino"]]


def test_training_failure_has_no_automatic_profile_retry_into_same_output(tmp_path):
    root = _make_suite(tmp_path)
    runner = FakeRunner(root, fail_model="fno", fail_phase="train")
    campaign = _campaign(root, runner)

    with pytest.raises(CampaignError, match="fno train failed"):
        campaign.run(models=("fno",))

    train_calls = [
        value
        for value in runner.call_records
        if value["job_id"] and str(value["job_id"]).startswith("fno.train")
    ]
    assert len(train_calls) == 1
    runtime = campaign.effective_train_configs["fno"].read_text(encoding="utf-8")
    assert "batch_size\t1" in runtime
    assert "grad_accum_steps\t4" in runtime
    assert not campaign.checkpoint("fno").exists()


def test_hi_checkpoint_and_runtime_copy_are_distinct_from_baseline(tmp_path):
    root = _make_suite(tmp_path)
    campaign = _campaign(root, FakeRunner(root))

    assert campaign.run(models=("hi_meshgraphnets",)) == 0

    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    identity = status["models"]["hi_meshgraphnets"]["checkpoint_identity"]
    assert identity["checkpoint_schema"] == "hi_meshgraphnets_final_v1"
    validation = status["models"]["hi_meshgraphnets"]["runtime_dataset_validation"]
    assert validation["mode"] == "bit_identical"
    assert campaign.hi_mgn_working_copy != campaign.mgn_working_copy


def test_transolver_3_uses_isolated_runtime_hdf5_and_expected_mutation_only(tmp_path):
    root = _make_suite(tmp_path)
    source = root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
    source_hash = _sha256(source)
    campaign = _campaign(root, FakeRunner(root))

    assert campaign.run(models=("transolver",)) == 0

    assert _sha256(source) == source_hash
    assert campaign.transolver_working_copy.is_file()
    train_runtime = campaign.effective_train_configs["transolver"].read_text(
        encoding="utf-8"
    )
    assert "plasticity_transolver_runtime.h5" in train_runtime
    assert "write_preprocessing\tTrue" in train_runtime
    status = json.loads(campaign.status_path.read_text(encoding="utf-8"))
    assert (
        status["models"]["transolver"]["runtime_dataset_validation"]["mode"]
        == "normalization_only_drift"
    )
    assert (
        status["models"]["transolver"]["checkpoint_identity"]["checkpoint_schema"]
        == "transolver_checkpoint_v1"
    )


def _windows_pid_alive(pid: int) -> bool:
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(process)


def test_subprocess_runner_owns_and_cancels_two_concurrent_jobs(tmp_path):
    runner = SubprocessRunner()
    outcomes: list[CommandResult] = []
    started: list[int] = []
    started_lock = threading.Lock()

    def record_started(pid: int) -> None:
        with started_lock:
            started.append(pid)

    def invoke(index: int) -> None:
        outcomes.append(
            runner.run(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                cwd=tmp_path,
                stdout_path=tmp_path / f"job{index}.stdout.log",
                stderr_path=tmp_path / f"job{index}.stderr.log",
                job_id=f"fixture.job{index}",
                on_start=record_started,
            )
        )

    threads = [threading.Thread(target=invoke, args=(index,), daemon=True) for index in (1, 2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with started_lock:
            if len(started) == 2:
                break
        time.sleep(0.02)
    with started_lock:
        assert len(started) == 2
        assert len(set(started)) == 2

    runner.cancel_active()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert len(outcomes) == 2
    assert all(result.returncode != 0 for result in outcomes)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_windows_owned_job_cancels_parent_and_grandchild_without_orphan(tmp_path):
    pid_file = tmp_path / "pids.json"
    parent_code = (
        "import json,os,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
        f"open({str(pid_file)!r},'w').write(json.dumps({{'parent':os.getpid(),'child':child.pid}}));"
        "time.sleep(120)"
    )
    runner = SubprocessRunner()
    outcome: list[CommandResult] = []

    def invoke() -> None:
        outcome.append(
            runner.run(
                [sys.executable, "-c", parent_code],
                cwd=tmp_path,
                stdout_path=tmp_path / "parent.stdout.log",
                stderr_path=tmp_path / "parent.stderr.log",
            )
        )

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.is_file(), "dummy parent did not report its grandchild"
    pids = json.loads(pid_file.read_text(encoding="utf-8"))
    assert _windows_pid_alive(int(pids["parent"]))
    assert _windows_pid_alive(int(pids["child"]))

    runner.cancel_active()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert outcome and outcome[0].returncode != 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(
        _windows_pid_alive(int(pids[name])) for name in ("parent", "child")
    ):
        time.sleep(0.05)
    assert not _windows_pid_alive(int(pids["parent"]))
    assert not _windows_pid_alive(int(pids["child"]))
