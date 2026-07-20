import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import paper_benchmark as benchmark  # noqa: E402


def _write_case(root: Path, split: str, name: str, n_points: int, condition_offset: float) -> None:
    direction = name.split("_", 1)[0]
    index = np.arange(n_points, dtype=np.float32)
    xyz = np.stack((index / n_points, index / (2 * n_points), -index / n_points), axis=1)
    sdf = np.full((n_points, 1), 0.2 + condition_offset, dtype=np.float32)
    mlc = np.array(
        [1.0 + condition_offset, 40.0 + condition_offset, 0.1, 0.2, 0.3], dtype=np.float32
    )
    conditions = np.broadcast_to(mlc, (n_points, 5)).copy()
    xyzdmlc = np.concatenate((xyz, sdf, conditions), axis=1).astype(np.float32)
    phase = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    targets = np.stack(
        (
            -0.02 + 0.01 * phase,
            -0.01 + 0.005 * phase,
            -0.03 + 0.02 * phase,
            10.0 + 4.0 * phase,
        ),
        axis=1,
    ).astype(np.float32)
    path = root / "cases" / split / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyzdmlc=xyzdmlc,
        targets=targets,
        sample_indices=np.arange(n_points, dtype=np.int32),
        case_name=np.array(name),
        split=np.array(split),
    )
    assert direction in benchmark.DIRECTIONS


def _smoke_config(root: Path, output: Path, n_points: int) -> dict:
    return {
        "schema_version": 1,
        "profile": "cpu_smoke",
        "prepared_dir": str(root),
        "output_dir": str(output),
        "dataset": {
            "n_samples": 2,
            "n_train": 1,
            "n_valid": 1,
            "n_points": n_points,
            "train_cases": ["hor_train"],
            "valid_cases": ["hor_valid"],
        },
        "model": {
            "branch_components": "mlc",
            "pointnet_components": "xyz",
            "trunk_components": "xyzd",
            "output_components": "xyzs",
            "branch_hidden_dim": 100,
            "trunk_hidden_dim": 100,
            "trunk_encoding_hidden_dim": 100,
            "fc_hidden_dim": 100,
            "pointnet_channels": [32, 64, 100],
            "siren_w0": 10.0,
        },
        "preprocessing": {
            "learned_scaler_fit_split": "train",
            "output_clipping": "released_fixed_by_direction",
        },
        "train": {
            "iterations": 1,
            "batch_size": 1,
            "loss": "mse",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "learning_rate_decay": {
                "type": "inverse_time",
                "decay_steps": 1,
                "decay_rate": 0.0001,
            },
            "seed": 2024,
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
        },
        "evaluation": {
            "batch_size": 1,
            "validate_every": 1,
            "aggregate": "available_direction_component_pooled_r2_smoke_only",
            "paper_target_average_r2": 0.897,
        },
    }


class PaperBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "prepared"
        self.output = Path(self.temp.name) / "output"
        manifests = self.root / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "train.txt").write_text("hor_train\n", encoding="utf-8")
        (manifests / "valid.txt").write_text("hor_valid\n", encoding="utf-8")
        _write_case(self.root, "train", "hor_train", 16, condition_offset=0.0)
        _write_case(self.root, "valid", "hor_valid", 16, condition_offset=100.0)
        self.config = _smoke_config(self.root, self.output, n_points=16)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_released_topology_shape_range_and_parameter_count(self) -> None:
        model = benchmark.ReleasedPointDeepONet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 5), torch.zeros(2, 17, 3), torch.zeros(2, 17, 4))
        self.assertEqual(tuple(output.shape), (2, 17, 4))
        self.assertTrue(torch.all(output <= 1.0))
        self.assertTrue(torch.all(output >= -1.0))
        self.assertEqual(benchmark.count_parameters(model), benchmark.PAPER_PARAMETER_COUNT)

    def test_train_only_scaling_and_released_feature_routing(self) -> None:
        preprocessing = benchmark.fit_preprocessing(self.root, ["hor_train"], 16)
        dataset = benchmark.ReleasedCaseDataset(
            self.root, "valid", ["hor_valid"], preprocessing, n_points=16
        )
        item = dataset[0]
        self.assertEqual(tuple(item["condition_mlc_raw"].shape), (5,))
        self.assertEqual(tuple(item["branch_mlc"].shape), (5,))
        self.assertEqual(tuple(item["point_xyz"].shape), (16, 3))
        self.assertEqual(tuple(item["trunk_xyzd"].shape), (16, 4))
        # A validation-only extreme must remain outside [-1, 1], proving the
        # learned range did not inspect validation data.
        self.assertGreater(float(item["branch_mlc"][0]), 1.0)
        self.assertGreater(float(item["trunk_xyzd"][0, 3]), 1.0)

    def test_paper_scaling_includes_all_selected_cases_before_split(self) -> None:
        preprocessing = benchmark.fit_preprocessing(
            self.root, ["hor_train"], 16, valid_names=["hor_valid"]
        )
        item = benchmark.ReleasedCaseDataset(
            self.root, "valid", ["hor_valid"], preprocessing, n_points=16
        )[0]
        self.assertEqual(preprocessing.fit_split, "released-all-selected-before-split")
        self.assertLessEqual(float(item["branch_mlc"].max()), 1.0)
        self.assertLessEqual(float(item["trunk_xyzd"].max()), 1.0)

    def test_released_numpy_sampler_matches_deepxde_epoch_boundary(self) -> None:
        np.random.seed(2024)
        expected_first_epoch = np.arange(8)
        np.random.shuffle(expected_first_epoch)
        expected_second_epoch = expected_first_epoch.copy()
        np.random.shuffle(expected_second_epoch)

        np.random.seed(2024)
        sampler = benchmark.ReleasedBatchSampler(8, shuffle=True)
        np.testing.assert_array_equal(sampler.get_next(3), expected_first_epoch[:3])
        np.testing.assert_array_equal(sampler.get_next(3), expected_first_epoch[3:6])
        np.testing.assert_array_equal(
            sampler.get_next(3), np.hstack((expected_first_epoch[6:8], expected_second_epoch[:1]))
        )
        self.assertEqual(sampler.epochs_completed, 1)

    def test_model_matches_executable_official_source(self) -> None:
        source_path = BENCHMARK_DIR / "source" / "point_deeponet_official" / "5.Point_DeepONet" / "main.py"
        if not source_path.is_file():
            self.skipTest("official Point-DeepONet source clone is unavailable")
        source = source_path.read_text(encoding="utf-8")
        sine_source = source[source.index("class SineDenseLayer"):source.index("\ndef parse_arguments")]
        model_source = source[
            source.index("class DeepONetCartesianProd"):source.index("\ndef define_model")
        ]
        model_source = model_source.replace(
            "class DeepONetCartesianProd(dde.maps.NN, nn.Module):",
            "class OfficialPointDeepONet(nn.Module):",
        )
        model_source = model_source.replace(
            "super(DeepONetCartesianProd, self).__init__()", "super().__init__()"
        )
        model_source = model_source.replace("dde.maps.NN.__init__(self)", "")
        namespace = {"np": np, "torch": torch, "nn": torch.nn}
        exec(sine_source + "\n" + model_source, namespace)
        official_class = namespace["OfficialPointDeepONet"]

        torch.manual_seed(123)
        official = official_class(5, 3, 17, 4, "Glorot normal", 100, 100, 100, 100, None, 4).eval()
        torch.manual_seed(123)
        isolated = benchmark.ReleasedPointDeepONet().eval()
        self.assertEqual(list(official.state_dict()), list(isolated.state_dict()))
        for name, value in official.state_dict().items():
            torch.testing.assert_close(value, isolated.state_dict()[name], rtol=0, atol=0)
        inputs = (torch.randn(2, 5), torch.randn(2, 17, 3), torch.randn(2, 17, 4))
        with torch.no_grad():
            torch.testing.assert_close(official(inputs), isolated(*inputs), rtol=0, atol=0)

    def test_clipping_is_direction_specific(self) -> None:
        raw = np.array([[-9.0, 9.0, -9.0, 999.0]], dtype=np.float32)
        clipped = benchmark.clip_targets(raw, "hor")
        np.testing.assert_allclose(clipped[0], [-0.421, 0.029, -0.388, 227.78], rtol=0, atol=2e-6)

    def test_paper_guard_rejects_hyperparameter_drift(self) -> None:
        paper = benchmark.load_config(BENCHMARK_DIR / "configs" / "paper_n1000_p5000.json")
        benchmark.validate_config(paper)
        changed = copy.deepcopy(paper)
        changed["train"]["iterations"] = 39999
        with self.assertRaisesRegex(ValueError, "iterations"):
            benchmark.validate_config(changed)

    def test_cpu_train_and_checkpoint_evaluate(self) -> None:
        result = benchmark.train(self.config)
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["parameter_count"], benchmark.PAPER_PARAMETER_COUNT)
        checkpoint = self.output / "checkpoint_last.pt"
        self.assertTrue(checkpoint.is_file())
        metrics = benchmark.evaluate_checkpoint(self.config, checkpoint)
        self.assertEqual(metrics["case_count"], 1)
        self.assertEqual(metrics["r2_terms_available"], 4)
        self.assertTrue((self.output / "evaluation_metrics.json").is_file())
        parsed = json.loads((self.output / "evaluation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["aggregate_definition"], "mean_of_available_direction_component_pooled_r2_smoke_only")
        self.assertFalse(parsed["paper_comparable"])
        self.assertIsNone(parsed["difference_from_paper"])

    def test_real_prepared_case_contract_when_present(self) -> None:
        prepared = BENCHMARK_DIR / "prepared" / "n1000_p5000"
        real_path = prepared / "cases" / "train" / "hor_20_506.npz"
        if not real_path.is_file():
            self.skipTest("selective real-case download is not present")
        xyzdmlc, targets = benchmark._load_raw_case(real_path, "hor_20_506", "train", 5000)
        self.assertEqual(xyzdmlc.shape, (5000, 9))
        self.assertEqual(targets.shape, (5000, 4))
        self.assertGreater(float(np.ptp(xyzdmlc[:, 3])), 0.0)
        self.assertTrue(np.array_equal(xyzdmlc[:, 4:9], np.broadcast_to(xyzdmlc[0, 4:9], (5000, 5))))


if __name__ == "__main__":
    unittest.main()
