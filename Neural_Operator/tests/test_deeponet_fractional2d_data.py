import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch


SUITE_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative_path):
    path = SUITE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matlab_standard_sobol_uses_old_direction_table_and_skip_leap():
    prepare = _load(
        "prepare_fractional2d",
        "dataset/benchmarks/deeponet_fractional2d/prepare_fractional2d.py",
    )
    coefficients, logical = prepare.released_sobol_coefficients(4)
    assert logical.tolist() == [28, 53, 78, 103]
    assert coefficients.shape == (4, 15)
    assert coefficients[0, 0] == -1.125
    # Dimension 3 distinguishes the Joe-Kuo 2003 MATLAB table from SciPy's
    # newer default direction table at this retained point.
    assert coefficients[0, 2] == -1.125
    assert np.all(coefficients >= -2.0) and np.all(coefficients < 2.0)


def test_matlab_meshgrid_column_major_order():
    prepare = _load(
        "prepare_fractional2d_mesh",
        "dataset/benchmarks/deeponet_fractional2d/prepare_fractional2d.py",
    )
    polar, xy = prepare.matlab_polar_grid(3)
    expected = np.asarray([
        [0.0, 0.0], [0.0, np.pi], [0.0, 2.0 * np.pi],
        [0.475, 0.0], [0.475, np.pi], [0.475, 2.0 * np.pi],
    ])
    assert np.allclose(polar[:6], expected)
    assert np.allclose(xy[:3], 0.0, atol=1.0e-15)


def test_expanded_row_order_is_alpha_function_query():
    trainer = _load(
        "train_fractional2d_indices",
        "dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py",
    )
    flat = torch.tensor([0, 224, 225, 5000 * 225, 5000 * 225 + 7])
    function, alpha, query, combined = trainer.decode_expanded_indices(flat, 5000, 225)
    assert function.tolist() == [0, 0, 1, 0, 0]
    assert alpha.tolist() == [0, 0, 0, 1, 1]
    assert query.tolist() == [0, 224, 0, 0, 7]
    assert combined.tolist() == [0, 224, 0, 225, 232]


def test_paper_config_means_560000_optimizer_updates():
    trainer = _load(
        "train_fractional2d_config",
        "dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py",
    )
    config = trainer.load_config(
        SUITE_ROOT / "configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt"
    )
    trainer.validate_config(config)
    assert config["epochs"] == 5000
    assert config["batch_size"] == 100000
    triples = 5000 * 10 * 225
    assert triples // config["batch_size"] == 112
    assert config["epochs"] * (triples // config["batch_size"]) == 560000


def test_compact_pairing_matches_materialized_predictions_and_gradients():
    from model.deeponet_fractional2d import FractionalLaplacianDeepONet

    torch.manual_seed(7)
    compact_model = FractionalLaplacianDeepONet(seed=12345)
    materialized_model = FractionalLaplacianDeepONet(seed=12345)
    branch = torch.randn(3, 225)
    queries = torch.randn(4, 3)
    function_index = torch.tensor([0, 2, 1, 2, 0])
    query_index = torch.tensor([3, 0, 2, 1, 1])
    truth = torch.randn(5)

    branch_code = compact_model.encode_branch(branch)
    trunk_code = compact_model.encode_trunk(queries)
    compact_prediction = compact_model.decode_encoded(
        branch_code[function_index], trunk_code[query_index]
    ).squeeze(1)
    compact_loss = torch.mean((compact_prediction - truth) ** 2) / torch.mean(truth**2)
    compact_loss.backward()

    materialized_prediction = materialized_model(
        branch[function_index], queries[query_index]
    ).squeeze(1)
    materialized_loss = (
        torch.mean((materialized_prediction - truth) ** 2) / torch.mean(truth**2)
    )
    materialized_loss.backward()

    assert torch.allclose(compact_prediction, materialized_prediction, atol=1.0e-7)
    assert torch.allclose(compact_loss, materialized_loss, atol=1.0e-7)
    for compact, materialized in zip(
        compact_model.parameters(), materialized_model.parameters()
    ):
        assert torch.allclose(compact.grad, materialized.grad, atol=1.0e-6)


def test_paper_profile_rejects_diagnostic_step_limit():
    trainer = _load(
        "train_fractional2d_guard",
        "dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py",
    )
    config = trainer.load_config(
        SUITE_ROOT / "configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt"
    )
    config["steps_per_epoch_limit"] = 2
    try:
        trainer.validate_config(config)
    except ValueError as error:
        assert "diagnostic-only" in str(error)
    else:
        raise AssertionError("paper profile accepted diagnostic step limit")


def _write_smoke_h5(path):
    rng = np.random.RandomState(19)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "deeponet_fractional_laplacian_2d_compact_v1"
        handle.attrs["released_train_test_relation"] = "identical_hard_links"
        common = handle.create_group("common")
        common.create_dataset("query_xy", data=rng.randn(5, 2).astype(np.float32))
        common.create_dataset("alpha", data=np.asarray([0.2, 0.8], dtype=np.float32))
        arrays = handle.create_group("arrays")
        arrays.create_dataset("branch_values", data=rng.randn(8, 225).astype(np.float32))
        arrays.create_dataset("targets", data=rng.randn(8, 2, 5).astype(np.float32))
        train = handle.create_group("train")
        test = handle.create_group("test")
        train["branch_values"] = arrays["branch_values"]
        test["branch_values"] = arrays["branch_values"]
        train["targets"] = arrays["targets"]
        test["targets"] = arrays["targets"]


def _write_smoke_config(path, dataset, output, epochs):
    path.write_text(
        "\n".join([
            "benchmark_profile smoke",
            f"dataset_dir {dataset}",
            f"output_dir {output}",
            "device cpu",
            "seed 12345",
            f"epochs {epochs}",
            "batch_size 4",
            "learning_rate 0.001",
            "hidden_width 60",
            "eval_interval_epochs 1",
            "checkpoint_interval_epochs 1",
            "function_eval_chunk 4",
            "steps_per_epoch_limit 2",
            "resume True",
            "paper_plot_normalized_mse 0.0012",
        ]) + "\n",
        encoding="utf-8",
    )


def test_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    trainer = _load(
        "train_fractional2d_resume",
        "dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py",
    )
    dataset = tmp_path / "smoke.h5"
    _write_smoke_h5(dataset)

    uninterrupted_config = tmp_path / "uninterrupted.txt"
    uninterrupted_output = tmp_path / "uninterrupted"
    _write_smoke_config(uninterrupted_config, dataset, uninterrupted_output, 2)
    trainer.run(uninterrupted_config)

    resumed_config = tmp_path / "resumed.txt"
    resumed_output = tmp_path / "resumed"
    _write_smoke_config(resumed_config, dataset, resumed_output, 1)
    trainer.run(resumed_config)
    _write_smoke_config(resumed_config, dataset, resumed_output, 2)
    trainer.run(resumed_config)

    uninterrupted = torch.load(
        uninterrupted_output / "last.pth", map_location="cpu", weights_only=False
    )
    resumed = torch.load(
        resumed_output / "last.pth", map_location="cpu", weights_only=False
    )
    assert uninterrupted["optimizer_steps"] == resumed["optimizer_steps"] == 4
    for key, value in uninterrupted["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
