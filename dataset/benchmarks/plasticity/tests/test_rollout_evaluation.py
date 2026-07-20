from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from compare_results import (  # noqa: E402
    COMPARISON_SCHEMA_VERSION,
    IncompleteComparisonError,
    combine_results,
)
from evaluate_rollouts import (  # noqa: E402
    MODELS,
    PINNED_TRUTH_SHA256,
    PRODUCTION_TEST_IDS,
    EvaluationContract,
    _provenance_matches_model,
    _sha256_file,
    _sha256_text,
    _suite_seed42_test_ids,
    evaluate_rollouts,
)


@pytest.fixture
def contract() -> EvaluationContract:
    return EvaluationContract(
        expected_cases=2,
        num_features=8,
        num_timesteps=4,
        num_nodes=5,
        num_edges=4,
        split_seed=42,
        expected_truth_ids=(1, 3),
        geometry_tolerance=1.0e-6,
        seed_state_tolerance=1.0e-6,
        pinned_ground_truth_sha256=None,
    )


def _nodal(sample_id: int, contract: EvaluationContract) -> np.ndarray:
    values = np.zeros(contract.nodal_shape, dtype=np.float32)
    nodes = np.arange(1, contract.num_nodes + 1, dtype=np.float32)
    values[0] = nodes[None, :]
    values[1] = float(sample_id)
    for component in range(3):
        for time_index in range(contract.num_timesteps):
            values[3 + component, time_index] = (
                0.01
                * (sample_id + 1)
                * (component + 1)
                * (time_index + 1)
                * nodes
            )
    values[6] = 7.0 + sample_id
    return values


def _edges(contract: EvaluationContract) -> np.ndarray:
    return np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)[
        :, : contract.num_edges
    ]


def _write_truth(path: Path, contract: EvaluationContract) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["num_samples"] = contract.expected_cases
        handle.attrs["suite_split_seed"] = contract.split_seed
        handle.attrs["suite_split_role"] = "test"
        splits = handle.create_group("splits")
        splits.create_dataset("test", data=np.asarray(contract.expected_truth_ids))
        metadata = handle.create_group("metadata")
        metadata.create_dataset(
            "time_normalized",
            data=np.linspace(0.0, 1.0, contract.num_timesteps),
        )
        data = handle.create_group("data")
        for sample_id in contract.expected_truth_ids:
            group = data.create_group(str(sample_id))
            group.create_dataset("nodal_data", data=_nodal(sample_id, contract))
            group.create_dataset("mesh_edge", data=_edges(contract))
            sample_metadata = group.create_group("metadata")
            sample_metadata.attrs["source_index"] = sample_id


def _write_rollouts(
    directory: Path,
    contract: EvaluationContract,
    *,
    relative_scale: float = 1.1,
    model: str = "fno",
    include_mesh_edges: bool | None = None,
) -> None:
    if include_mesh_edges is None:
        include_mesh_edges = model != "transolver"
    directory.mkdir(parents=True)
    for ordinal, sample_id in enumerate(contract.expected_truth_ids):
        nodal = _nodal(sample_id, contract)
        nodal[3:6, 1:] *= relative_scale
        # The excluded static condition can be arbitrarily wrong without
        # affecting a displacement-only score.
        nodal[6, 1:] = 999.0
        path = directory / f"rollout_sample{sample_id}_steps{contract.rollout_steps}.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["num_samples"] = 1
            handle.attrs["num_features"] = contract.num_features
            handle.attrs["num_timesteps"] = contract.num_timesteps
            group = handle.create_group("data").create_group(str(sample_id))
            group.create_dataset("nodal_data", data=nodal)
            if include_mesh_edges:
                group.create_dataset("mesh_edge", data=_edges(contract))
            metadata = group.create_group("metadata")
            metadata.attrs["sample_id"] = sample_id
            metadata.attrs["total_rollout_time_s"] = float(ordinal + 1)
            metadata.attrs["model_path"] = f"output/plasticity/{model}/model.pth"
            metadata.attrs["config_file"] = f"config_infer_{model}.txt"


def _evaluate_fixture(tmp_path: Path, contract: EvaluationContract):
    tmp_path.mkdir(parents=True, exist_ok=True)
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    output_json = predictions / "plasticity_metrics.json"
    output_csv = predictions / "plasticity_per_case_time.csv"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    result = evaluate_rollouts(
        model="fno",
        ground_truth=truth,
        predictions=predictions,
        output_json=output_json,
        output_csv=output_csv,
        contract=contract,
    )
    return truth, predictions, output_json, output_csv, result


def test_seed42_contract_has_exactly_100_unique_ids():
    ids = _suite_seed42_test_ids()
    assert len(ids) == 100
    assert len(set(ids)) == 100
    assert ids == tuple(sorted(ids))
    assert ids == PRODUCTION_TEST_IDS


def test_real_ground_truth_hash_is_pinned():
    truth = BENCHMARK_DIR / "plasticity_seed42_test.h5"
    assert _sha256_file(truth) == PINNED_TRUTH_SHA256


def test_evaluator_scores_only_displacement_times_after_seed(tmp_path, contract):
    _, _, output_json, output_csv, result = _evaluate_fixture(tmp_path, contract)

    assert result["complete"] is True
    assert result["primary_metric"]["value"] == pytest.approx(0.1, abs=2.0e-7)
    assert result["metrics"]["global_full_trajectory_relative_l2"] == pytest.approx(
        0.1, abs=2.0e-7
    )
    assert result["metrics"][
        "mean_per_case_time_averaged_per_timestep_relative_l2"
    ] == pytest.approx(0.1, abs=2.0e-7)
    assert result["metrics"]["final_time_mean_per_case_relative_l2"] == pytest.approx(
        0.1, abs=2.0e-7
    )
    assert result["validation"]["rollouts_with_mesh_edges"] == 2
    assert result["validation"]["rollouts_without_mesh_edges"] == 0
    assert result["timing"]["total_seconds"] == 3.0
    assert output_json.is_file()
    assert json.loads(output_json.read_text(encoding="utf-8"))["complete"] is True
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == contract.expected_cases * contract.rollout_steps
    assert {int(row["time_index"]) for row in rows} == {1, 2, 3}
    assert all(float(row["relative_l2"]) == pytest.approx(0.1, abs=2.0e-7) for row in rows)


def test_evaluator_rejects_missing_rollout(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    next(predictions.glob("rollout_sample3_*.h5")).unlink()
    with pytest.raises(ValueError, match="Rollout ID contract failed"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


def test_evaluator_rejects_geometry_mismatch(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    path = next(predictions.glob("rollout_sample1_*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["data/1/nodal_data"][0, 2, 2] += 0.1
    with pytest.raises(ValueError, match="geometry mismatch"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


def test_evaluator_rejects_nonfinite_prediction(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    path = next(predictions.glob("rollout_sample1_*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["data/1/nodal_data"][3, 2, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


def test_evaluator_rejects_swapped_seed_die_channel(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    path = next(predictions.glob("rollout_sample1_*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["data/1/nodal_data"][6, 0, :] = 10.0  # sample 3's die seed
    with pytest.raises(ValueError, match="seed state channels 3:7 mismatch"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("sample_id", 3, "metadata sample_id"),
        ("model_path", "output/plasticity/gino/model.pth", "model_path does not identify"),
        ("config_file", "config_infer_gino.txt", "config_file does not identify"),
    ],
)
def test_evaluator_rejects_wrong_metadata_or_provenance(
    tmp_path, contract, attribute, value, message
):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    path = next(predictions.glob("rollout_sample1_*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["data/1/metadata"].attrs[attribute] = value
    with pytest.raises(ValueError, match=message):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


def test_evaluator_rejects_inconsistent_valid_provenance_across_cases(
    tmp_path, contract
):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    path = next(predictions.glob("rollout_sample3_*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["data/3/metadata"].attrs["config_file"] = (
            "runtime_config_infer_fno.txt"
        )
    with pytest.raises(ValueError, match="one consistent config path"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=contract,
        )


def test_meshgraphnets_provenance_disambiguates_baseline_and_hi_mgn():
    assert _provenance_matches_model(
        "output/plasticity/meshgraphnets/model.pth", "meshgraphnets"
    )
    assert not _provenance_matches_model(
        "output/plasticity/hi_meshgraphnets/model.pth", "meshgraphnets"
    )
    assert _provenance_matches_model(
        "config_infer_hi_meshgraphnets.txt", "hi_meshgraphnets"
    )
    assert not _provenance_matches_model(
        "config_infer_meshgraphnets.txt", "hi_meshgraphnets"
    )


def test_missing_mesh_edges_rejected_except_for_transolver(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    _write_truth(truth, contract)

    fno_predictions = tmp_path / "fno"
    _write_rollouts(
        fno_predictions, contract, model="fno", include_mesh_edges=False
    )
    with pytest.raises(ValueError, match="mesh_edge is required"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=fno_predictions,
            output_json=fno_predictions / "metrics.json",
            output_csv=fno_predictions / "metrics.csv",
            contract=contract,
        )

    transolver_predictions = tmp_path / "transolver"
    _write_rollouts(
        transolver_predictions,
        contract,
        model="transolver",
        include_mesh_edges=False,
    )
    result = evaluate_rollouts(
        model="transolver",
        ground_truth=truth,
        predictions=transolver_predictions,
        output_json=transolver_predictions / "metrics.json",
        output_csv=transolver_predictions / "metrics.csv",
        contract=contract,
    )
    assert result["validation"]["rollouts_without_mesh_edges"] == 2


@pytest.mark.parametrize("model", MODELS)
def test_representative_writer_schemas_are_accepted(tmp_path, contract, model):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / model
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract, model=model)
    result = evaluate_rollouts(
        model=model,
        ground_truth=truth,
        predictions=predictions,
        output_json=predictions / "metrics.json",
        output_csv=predictions / "metrics.csv",
        contract=contract,
    )
    expected_edges = 0 if model == "transolver" else 2
    assert result["validation"]["rollouts_with_mesh_edges"] == expected_edges


def test_custom_contract_can_explicitly_disable_or_change_truth_pin(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    wrong_pin = EvaluationContract(
        **{
            **contract.__dict__,
            "pinned_ground_truth_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="Ground-truth SHA256 mismatch"):
        evaluate_rollouts(
            model="fno",
            ground_truth=truth,
            predictions=predictions,
            output_json=predictions / "metrics.json",
            output_csv=predictions / "metrics.csv",
            contract=wrong_pin,
        )


def test_evaluator_rejects_output_aliases(tmp_path, contract):
    truth = tmp_path / "truth.h5"
    predictions = tmp_path / "predictions"
    _write_truth(truth, contract)
    _write_rollouts(predictions, contract)
    rollout = next(predictions.glob("*.h5"))
    cases = (
        (predictions / "same.json", predictions / "same.json"),
        (truth, predictions / "metrics.csv"),
        (predictions / "metrics.json", rollout),
    )
    for output_json, output_csv in cases:
        with pytest.raises(ValueError, match="alias|distinct"):
            evaluate_rollouts(
                model="fno",
                ground_truth=truth,
                predictions=predictions,
                output_json=output_json,
                output_csv=output_csv,
                contract=contract,
            )


def _write_seven_results(results_root: Path, template: dict[str, object]) -> None:
    truth_ids_text = ",".join(str(value) for value in PRODUCTION_TEST_IDS)
    for index, model in enumerate(MODELS):
        payload = deepcopy(template)
        payload["model"] = model
        value = 0.01 * (index + 1)
        payload["primary_metric"]["value"] = value
        payload["metrics"]["mean_per_case_full_trajectory_relative_l2"] = value
        payload["metrics"]["global_full_trajectory_relative_l2"] = value
        payload["metrics"][
            "mean_per_case_time_averaged_per_timestep_relative_l2"
        ] = value
        payload["metrics"]["final_time_mean_per_case_relative_l2"] = value
        for component in ("u_x_mm", "u_y_mm", "u_z_mm"):
            payload["metrics"]["per_component"][component] = {
                "rmse": value,
                "mae": value,
            }
        payload["dataset"].update(
            {
                "ground_truth_sha256": PINNED_TRUTH_SHA256,
                "split_seed": 42,
                "split_role": "test",
                "expected_cases": 100,
                "truth_ids_sha256": _sha256_text(truth_ids_text),
                "truth_sample_ids": list(PRODUCTION_TEST_IDS),
            }
        )
        manifest = [
            {
                "sample_id": sample_id,
                "filename": f"rollout_sample{sample_id}_steps19.h5",
                "bytes": 1,
                "sha256": _sha256_text(f"{model}:{sample_id}"),
            }
            for sample_id in PRODUCTION_TEST_IDS
        ]
        payload["predictions"] = {
            "directory": str(results_root / model / "inference"),
            "files": 100,
            "manifest_sha256": _sha256_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            ),
            "manifest": manifest,
        }
        payload["evaluation_contract"].update(
            {
                "saved_displacement_feature_indices": [3, 4, 5],
                "component_names": ["u_x_mm", "u_y_mm", "u_z_mm"],
                "excluded_saved_feature_index": 6,
                "excluded_feature": "die_profile_mm",
                "evaluated_time_indices": list(range(1, 20)),
                "excluded_seed_time_index": 0,
                "expected_nodal_shape": [8, 20, 3131],
                "expected_rollout_steps": 19,
                "epsilon": 1.0e-12,
            }
        )
        payload["validation"].update(
            {
                "complete": True,
                "evaluated_cases": 100,
                "csv_rows": 1900,
                "all_values_finite": True,
                "rollouts_with_mesh_edges": 0 if model == "transolver" else 100,
                "rollouts_without_mesh_edges": 100 if model == "transolver" else 0,
                "geometry_tolerance": 1.0e-5,
                "seed_state_tolerance": 1.0e-6,
            }
        )
        payload["timing"] = {
            "available_cases": 100,
            "missing_cases": 0,
            "missing_sample_ids": [],
            "total_seconds": 100.0,
        }
        payload["provenance"] = {
            "checkpoint_paths": [f"output/plasticity/{model}/model.pth"],
            "config_files": [f"config_infer_{model}.txt"],
        }

        inference = results_root / model / "inference"
        inference.mkdir(parents=True)
        csv_path = inference / "plasticity_per_case_time.csv"
        rows = []
        manifest_by_id = {entry["sample_id"]: entry for entry in manifest}
        for sample_id in PRODUCTION_TEST_IDS:
            for time_index in range(1, 20):
                entry = manifest_by_id[sample_id]
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_id,
                        "source_index": sample_id,
                        "time_index": time_index,
                        # Production HDF5 stores this coordinate as float32.
                        "time_normalized": float(np.float32(time_index / 19.0)),
                        "relative_l2": value,
                        "error_l2": value,
                        "target_l2": 1.0,
                        "case_full_trajectory_relative_l2": value,
                        "case_time_averaged_relative_l2": value,
                        "rmse_u_x_mm": value,
                        "rmse_u_y_mm": value,
                        "rmse_u_z_mm": value,
                        "mae_u_x_mm": value,
                        "mae_u_y_mm": value,
                        "mae_u_z_mm": value,
                        "rollout_time_seconds": 1.0,
                        "prediction_file": entry["filename"],
                        "prediction_sha256": entry["sha256"],
                    }
                )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload["artifacts"] = {
            "summary_json": {"path": str(inference / "plasticity_metrics.json")},
            "per_case_time_csv": {
                "path": str(csv_path.resolve()),
                "bytes": csv_path.stat().st_size,
                "sha256": _sha256_file(csv_path),
                "rows": 1900,
            },
        }
        path = inference / "plasticity_metrics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_comparison_ranks_all_seven_complete_results(tmp_path, contract):
    _, _, _, _, template = _evaluate_fixture(tmp_path / "fixture", contract)
    results_root = tmp_path / "results"
    _write_seven_results(results_root, template)

    result = combine_results(results_root=results_root, output_dir=results_root)
    assert result["complete"] is True
    assert result["schema_version"] == "plasticity_seven_model_comparison_v1"
    assert result["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert result["models_missing"] == []
    assert [row["model"] for row in result["results"]] == list(MODELS)
    assert [row["rank"] for row in result["results"]] == list(range(1, 8))
    assert (results_root / "comparison.json").is_file()
    assert (results_root / "comparison.csv").is_file()
    assert (results_root / "comparison.md").is_file()
    assert "seven-model" in (results_root / "comparison.md").read_text(encoding="utf-8")


def test_comparison_writes_incomplete_status_and_exits_failure(tmp_path, contract):
    _, _, _, _, template = _evaluate_fixture(tmp_path / "fixture", contract)
    results_root = tmp_path / "results"
    _write_seven_results(results_root, template)
    (results_root / "hi_meshgraphnets" / "inference" / "plasticity_metrics.json").unlink()

    with pytest.raises(IncompleteComparisonError, match=r"missing=\['hi_meshgraphnets'\]"):
        combine_results(results_root=results_root, output_dir=results_root)
    result = json.loads((results_root / "comparison.json").read_text(encoding="utf-8"))
    assert result["complete"] is False
    assert result["models_missing"] == ["hi_meshgraphnets"]
    row = next(row for row in result["results"] if row["model"] == "hi_meshgraphnets")
    assert row["status"] == "missing"
    assert row["rank"] == ""


def test_comparison_rejects_negative_forged_metric(tmp_path, contract):
    _, _, _, _, template = _evaluate_fixture(tmp_path / "fixture", contract)
    results_root = tmp_path / "results"
    _write_seven_results(results_root, template)
    path = results_root / "hi_meshgraphnets" / "inference" / "plasticity_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["primary_metric"]["value"] = -1.0
    payload["metrics"]["mean_per_case_full_trajectory_relative_l2"] = -1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IncompleteComparisonError, match=r"invalid=\['hi_meshgraphnets'\]"):
        combine_results(results_root=results_root, output_dir=results_root)
    comparison = json.loads((results_root / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["complete"] is False
    assert comparison["models_invalid"] == ["hi_meshgraphnets"]


def test_comparison_rejects_csv_hash_or_content_forgery(tmp_path, contract):
    _, _, _, _, template = _evaluate_fixture(tmp_path / "fixture", contract)
    results_root = tmp_path / "results"
    _write_seven_results(results_root, template)
    csv_path = results_root / "fno" / "inference" / "plasticity_per_case_time.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "forged\n", encoding="utf-8")

    with pytest.raises(IncompleteComparisonError, match=r"invalid=\['fno'\]"):
        combine_results(results_root=results_root, output_dir=results_root)
    comparison = json.loads((results_root / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["models_invalid"] == ["fno"]
