from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from resource_probe import (  # noqa: E402
    GATE_SCHEMA_VERSION,
    REQUIRED_PAPER_VALIDATIONS,
    SCHEMA_VERSION,
    ProbeError,
    _flat_config,
    _index_identity,
    _sha256_file,
    _single_projection,
    _validate_isolated_output_root,
    build_dry_run_plan,
    certify_pair,
    completion_gate_or_user_waiver,
    deterministic_probe_ids,
    main,
    materialize_runtime_config,
    parse_completed_batches,
    prepare_model_plans,
    prepare_probe_hdf5,
    resolve_profile,
    update_probe_index,
    validate_completion_gate,
)


def _write_gate(path: Path, *, complete: bool = True) -> None:
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "complete": complete,
        "status": "passed" if complete else "running",
        "completed_at": "2026-07-19T12:00:00+00:00" if complete else None,
        "validations": {
            name: {"complete": complete, "status": "passed" if complete else "running"}
            for name in REQUIRED_PAPER_VALIDATIONS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_hdf5(path: Path, *, samples: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "cae_ml_suite_mesh_hdf5_v1"
        handle.attrs["num_samples"] = samples
        handle.attrs["num_timesteps"] = 3
        handle.attrs["num_features"] = 8
        topology = handle.create_group("topology")
        topology.create_dataset("mesh_edge_structured_quad", data=np.asarray([[0, 1], [1, 2]]))
        topology.create_dataset("quad_cells", data=np.asarray([[0, 1, 2, 3]]))
        metadata = handle.create_group("metadata")
        metadata.create_dataset("feature_names", data=np.asarray([b"x", b"y"]))
        metadata.create_dataset("time_normalized", data=np.asarray([0.0, 0.5, 1.0]))
        metadata.create_group("normalization_params").create_dataset("mean", data=np.zeros(8))
        metadata.create_dataset("provenance_json", data=np.bytes_("{}"))
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.arange(0, max(samples - 3, 0), dtype=np.int64))
        splits.create_dataset(
            "unused", data=np.arange(max(samples - 3, 0), max(samples - 2, 0), dtype=np.int64)
        )
        splits.create_dataset(
            "test", data=np.arange(max(samples - 2, 0), samples, dtype=np.int64)
        )
        splits.create_dataset("val", data=np.empty(0, dtype=np.int64))
        metadata["splits"] = splits
        data = handle.create_group("data")
        for sample_id in range(samples):
            group = data.create_group(str(sample_id))
            group.create_dataset(
                "nodal_data",
                data=np.full((8, 3, 4), float(sample_id), dtype=np.float32),
            )
            group["mesh_edge"] = topology["mesh_edge_structured_quad"]
            group.create_dataset("die_profile", data=np.full((4,), sample_id, dtype=np.float32))
            sample_metadata = group.create_group("metadata")
            sample_metadata.attrs["source_index"] = sample_id


def _write_config(path: Path, model: str) -> None:
    native = "MeshGraphNets" if model in {"meshgraphnets", "hi_meshgraphnets"} else model
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"model\t{native}",
                "mode\ttrain",
                "gpu_ids\t0",
                "parallel_mode\tddp",
                "log_file_dir\told/train.log",
                "modelpath\told/model.pth",
                "dataset_dir\told/data.h5",
                "infer_dataset\told/data.h5",
                "inference_output_dir\told/inference",
                "training_epochs\t500",
                "batch_size\t4",
                "grad_accum_steps\t1",
                "num_workers\t2",
                "use_amp\tTrue",
                "checkpoint_interval\t50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_completion_gate_fails_closed_and_requires_every_validation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ProbeError, match="locked"):
        validate_completion_gate(missing)

    gate = tmp_path / "gate.json"
    _write_gate(gate, complete=False)
    with pytest.raises(ProbeError, match="not explicitly complete"):
        validate_completion_gate(gate)

    _write_gate(gate)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["validations"]["gino"]["status"] = "running"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProbeError, match="gino"):
        validate_completion_gate(gate)

    _write_gate(gate)
    record = validate_completion_gate(gate)
    assert record["complete"] is True
    assert record["sha256"] == _sha256_file(gate)


def test_explicit_user_waiver_does_not_claim_paper_validation_passed(
    tmp_path: Path,
) -> None:
    record = completion_gate_or_user_waiver(
        tmp_path / "missing.json", allow_incomplete=True
    )
    assert record["complete"] is False
    assert record["status"] == "waived_by_user"
    assert record["validations"] == []
    assert record["authorization"]["type"] == "explicit_cli_override"
    assert record["authorization"]["scope"] == "plasticity_resource_probe"


def test_probe_hdf5_is_deterministic_schema_preserving_and_source_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    output = tmp_path / "run" / "probe.h5"
    _write_source_hdf5(source)
    before_hash = _sha256_file(source)
    expected_ids = deterministic_probe_ids(tuple(range(12)), 4, 42)

    record = prepare_probe_hdf5(
        source,
        output,
        cases=4,
        seed=42,
        expected_source_sha256=before_hash,
    )

    assert record["selected_ids"] == list(expected_ids)
    assert record["cases"] == 4
    assert _sha256_file(source) == before_hash
    with h5py.File(output, "r") as handle:
        assert set(handle) == {"data", "metadata", "splits", "topology"}
        assert int(handle.attrs["num_samples"]) == 4
        assert bool(handle.attrs["resource_probe"])
        assert handle["metadata/splits"].id == handle["splits"].id
        for sample_id in expected_ids:
            assert handle[f"data/{sample_id}/nodal_data"].shape == (8, 3, 4)


def test_cpu_runtime_config_is_isolated_and_disables_cuda_amp(tmp_path: Path) -> None:
    canonical = (
        tmp_path
        / "suite"
        / "configs"
        / "benchmarks"
        / "plasticity"
        / "config_train_deeponet.txt"
    )
    _write_config(canonical, "deeponet")
    canonical_before = canonical.read_bytes()
    model_root = tmp_path / "isolated" / "models" / "deeponet"
    runtime = tmp_path / "isolated" / "configs" / "deeponet_cpu.txt"
    dataset = tmp_path / "isolated" / "datasets" / "probe.h5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"synthetic")

    record = materialize_runtime_config(
        canonical,
        runtime,
        model="deeponet",
        profile="canonical",
        batch_size=4,
        grad_accum_steps=1,
        device="cpu",
        gpu_id=0,
        dataset=dataset,
        model_root=model_root,
    )
    values = _flat_config(runtime)

    assert canonical.read_bytes() == canonical_before
    assert values["gpu_ids"] == "-1"
    assert values["use_amp"] == "False"
    assert values["num_workers"] == "0"
    assert values["training_epochs"] == "1"
    assert record["effective_batch_size"] == 4
    assert str(model_root.resolve()) in values["modelpath"]
    assert ":" not in values["log_file_dir"]
    log_base = tmp_path / "suite" / "Neural_Operator" / "outputs"
    assert (log_base / values["log_file_dir"]).resolve() == (model_root / "train.log").resolve()


def test_writable_models_receive_three_distinct_hash_equal_probe_copies(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    configs = suite / "configs" / "benchmarks" / "plasticity"
    for model in ("meshgraphnets", "hi_meshgraphnets", "transolver"):
        _write_config(configs / f"config_train_{model}.txt", model)
    common = tmp_path / "run" / "datasets" / "common.h5"
    _write_source_hdf5(common)
    dataset_record = {
        "cases": 12,
        "num_timesteps": 3,
        "selected_ids_sha256": "a" * 64,
        "source_sha256": "b" * 64,
    }
    profiles = {
        "meshgraphnets": "fallback_1x4",
        "hi_meshgraphnets": "fallback_1x4",
        "transolver": "canonical",
    }

    plans, copies = prepare_model_plans(
        suite_root=suite,
        run_root=tmp_path / "run",
        models=tuple(profiles),
        profiles=profiles,
        device="gpu",
        gpu_id=0,
        cpu_threads=8,
        python_executable=sys.executable,
        common_dataset=common,
        dataset_record=dataset_record,
    )

    paths = {plan.dataset.resolve() for plan in plans.values()}
    assert len(paths) == 3
    assert common.resolve() not in paths
    assert {_sha256_file(path) for path in paths} == {_sha256_file(common)}
    assert set(model for model in profiles if model in copies) == set(profiles)


def test_progress_parser_and_pair_certification_thresholds() -> None:
    text = " 50%|x| 5/10\r100%|x| 10/10\nValidation: 100%|x| 2/2"
    assert parse_completed_batches(text, 10) == [10]
    records = {
        "deeponet": {"complete": True, "train_pairs_processed": 100},
        "fno": {"complete": True, "train_pairs_processed": 100},
    }
    baselines = [
        {"model": "deeponet", "wall_seconds": 10.0, "train_pairs_processed": 100},
        {"model": "fno", "wall_seconds": 10.0, "train_pairs_processed": 100},
    ]
    gpu = {"complete": True, "peak_total_used_mib": 6500}
    passed = certify_pair(
        model_records=records,
        baselines=baselines,
        pair_wall_seconds=18.0,
        gpu_record=gpu,
    )
    assert passed["certified"] is True
    assert passed["throughput_improvement_fraction"] == pytest.approx(1 / 9)

    failed = certify_pair(
        model_records=records,
        baselines=baselines,
        pair_wall_seconds=18.0,
        gpu_record={"complete": True, "peak_total_used_mib": 6657},
    )
    assert failed["certified"] is False
    assert failed["peak_pass"] is False


def test_dry_run_is_read_only_and_hi_defaults_to_one_by_four(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    source = suite / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not opened by dry run")
    canonical = suite / "configs" / "benchmarks" / "plasticity" / "config_train_hi_meshgraphnets.txt"
    _write_config(canonical, "hi_meshgraphnets")
    output_root = suite / "output" / "benchmarks" / "plasticity" / "resource_probe"

    plan = build_dry_run_plan(
        suite_root=suite,
        output_root=output_root,
        run_id="DRY_RUN",
        models=("hi_meshgraphnets",),
        profiles={"hi_meshgraphnets": "fallback_1x4"},
        device="gpu",
        gpu_id=0,
        cpu_threads=8,
        cases=32,
        seed=42,
        python_executable=sys.executable,
        gate_path=suite / "output" / "benchmarks" / "paper_validation_completion_gate.json",
    )

    assert plan["models"]["hi_meshgraphnets"]["batch_size"] == 1
    assert plan["models"]["hi_meshgraphnets"]["grad_accum_steps"] == 4
    with pytest.raises(ProbeError, match="not available"):
        resolve_profile("point_deeponet", "fallback_1x4")
    with pytest.raises(ProbeError, match="dedicated resource_probe subtree"):
        _validate_isolated_output_root(
            suite,
            suite / "output" / "benchmarks" / "plasticity" / "fno",
        )
    assert not output_root.exists()


def test_execute_without_gate_refuses_before_creating_outputs(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    output_root = suite / "probe-output"
    exit_code = main(
        [
            "--execute",
            "--model",
            "deeponet",
            "--device",
            "cpu",
            "--suite-root",
            str(suite),
            "--output-root",
            str(output_root),
        ]
    )
    assert exit_code == 2
    assert not output_root.exists()


def test_cpu_projection_and_atomic_index_discovery(tmp_path: Path) -> None:
    model_record = {
        "profile": "canonical",
        "batch_size": 4,
        "grad_accum_steps": 1,
        "train_pairs_processed": 475,
        "wall_seconds": 10.0,
        "throughput_pairs_per_second": 47.5,
        "complete": True,
    }
    projection = _single_projection(model_record)
    assert projection["projected_full_train_pairs"] == 7_495_500
    assert projection["projected_wall_hours_500_epochs"] > 0

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "cpu-deeponet",
        "mode": "single",
        "device": "cpu",
        "state": "complete",
        "complete": True,
        "completed_at": "2026-07-19T12:30:00+00:00",
        "dataset": {
            "source_sha256": "a" * 64,
            "selected_ids_sha256": "b" * 64,
            "cases": 32,
            "seed": 42,
        },
        "models": {"deeponet": model_record},
        "projection": projection,
        "placement": {"cpu_eligible": True},
    }
    result_path = tmp_path / "runs" / "cpu-deeponet" / "resource_probe_result.json"
    result_path.parent.mkdir(parents=True)
    identity_key, identity = _index_identity(result)
    result["identity_key"] = identity_key
    result["identity"] = identity
    result_path.write_text(json.dumps(result), encoding="utf-8")
    index_path = tmp_path / "index.json"

    update_probe_index(index_path, result_path, result)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == "plasticity_resource_probe_index_v1"
    assert len(index["latest_completed_single"]) == 1
    assert len(index["latest_cpu_eligible"]) == 1
    entry = next(iter(index["latest_cpu_eligible"].values()))
    assert entry["identity"]["models_key"] == "deeponet"
    assert entry["identity"]["profile_key"] == "deeponet=canonical"
    assert entry["result_sha256"] == _sha256_file(result_path)
