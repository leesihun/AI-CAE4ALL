from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from studio_backend.analysis import evaluation_schema, run_field_evaluation
from studio_backend.paths import RUNTIME_ROOT


class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import h5py  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional feature dependency
            raise unittest.SkipTest(str(exc)) from exc
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def test_mesh_schema_uses_named_builder_output_window_and_runs_legacy_rows(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-mesh-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "rollout_sample7.h5"
            truth = root / "truth.h5"
            values = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4)
            names = ["x", "y", "z", "mach", "pressure", "velocity", "node_type"]
            self._mesh(prediction, {"7": values + 0.25}, names, input_count=3, condition_count=1, output_count=2)
            self._mesh(truth, {"7": values}, names, input_count=3, condition_count=1, output_count=2)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertTrue(schema["compatible"], schema["errors"])
            self.assertEqual(schema["prediction"]["contract"], "mesh_state")
            self.assertEqual(schema["sample_matching"]["matched_ids"], ["7"])
            mapping = schema["recommended_mapping"]
            self.assertEqual([item["name"] for item in mapping["field_pairs"]], ["pressure", "velocity"])
            self.assertEqual(mapping["prediction_start"], 4)
            self.assertEqual(mapping["truth_start"], 4)
            self.assertEqual(mapping["num_fields"], 2)

            report = run_field_evaluation({
                "prediction_path": str(prediction),
                "truth_path": str(truth),
                "prediction_start": 4,
                "truth_start": 4,
                "num_fields": 2,
            })
            self.assertEqual(report["contract"], "mesh_state")
            self.assertEqual(report["evaluated_samples"], 1)
            self.assertAlmostEqual(report["aggregate"]["mae"]["mean"], 0.25)

    def test_mesh_schema_reports_node_mismatch_before_scoring(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-node-mismatch-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "prediction.h5"
            truth = root / "truth.h5"
            names = ["x", "y", "z", "pressure"]
            self._mesh(prediction, {"2": np.zeros((4, 1, 5), dtype=np.float32)}, names, 3, 0, 1)
            self._mesh(truth, {"2": np.zeros((4, 1, 6), dtype=np.float32)}, names, 3, 0, 1)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertFalse(schema["compatible"])
            self.assertEqual(schema["sample_matching"]["incompatible_shape_count"], 1)
            self.assertTrue(any("node counts" in error or "value shapes" in error for error in schema["errors"]))

    def test_table_schema_matches_declared_sample_ids_and_scores_named_columns(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-table-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "prediction.h5"
            truth = root / "truth.h5"
            # Prediction order is deliberately reversed; sample_ids, not row
            # position, must establish the pairing.
            with h5py.File(prediction, "w") as handle:
                handle.create_dataset("predictions", data=np.asarray([[3.1, 4.2], [1.1, 2.2]], dtype=np.float32))
                handle.create_dataset("sample_ids", data=np.asarray([b"b", b"a"]))
            with h5py.File(truth, "w") as handle:
                handle.create_dataset("Y", data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
                handle.create_dataset("sample_ids", data=np.asarray([b"a", b"b"]))
                handle.create_dataset("output_names", data=np.asarray([b"lift", b"drag"]))

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertTrue(schema["compatible"], schema["errors"])
            self.assertEqual(schema["sample_matching"]["strategy"], "id")
            self.assertEqual(schema["sample_matching"]["overlap_count"], 2)
            self.assertEqual(schema["recommended_mapping"]["prediction_array"], "predictions")
            self.assertEqual(schema["recommended_mapping"]["truth_array"], "Y")
            self.assertEqual(
                [item["truth_name"] for item in schema["recommended_mapping"]["field_pairs"]],
                ["lift", "drag"],
            )
            self.assertEqual(schema["recommended_mapping"]["confidence"], "confirm")
            with self.assertRaisesRegex(ValueError, "explicitly confirm"):
                run_field_evaluation({
                    "prediction_path": str(prediction),
                    "truth_path": str(truth),
                })

            report = run_field_evaluation({
                "prediction_path": str(prediction),
                "truth_path": str(truth),
                "field_pairs": schema["recommended_mapping"]["field_pairs"],
            })
            self.assertEqual(report["contract"], "table")
            self.assertEqual(report["truth_source"], "selected")
            self.assertEqual(report["evaluated_samples"], 2)
            self.assertAlmostEqual(report["aggregate"]["mae"]["mean"], 0.15, places=5)

    def test_native_result_and_mlp_prediction_use_exact_embedded_truth(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-embedded-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "sample8.h5"
            truth = root / "truth.h5"
            with h5py.File(prediction, "w") as handle:
                handle.attrs["sample_id"] = 8
                nodes = handle.create_group("nodes")
                nodes.create_dataset("pos", data=np.zeros((4, 3), dtype=np.float32))
                nodes.create_dataset("predicted_denorm", data=np.full((4, 2), 2.5, dtype=np.float32))
                nodes.create_dataset("target_denorm", data=np.full((4, 2), 2.0, dtype=np.float32))
            self._mesh(
                truth,
                {"8": np.zeros((5, 1, 4), dtype=np.float32)},
                ["x", "y", "z", "lift", "drag"],
                3,
                0,
                2,
            )

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertTrue(schema["compatible"], schema["errors"])
            self.assertEqual(schema["prediction"]["contract"], "native_inference_result")
            self.assertEqual(schema["recommended_mapping"]["mode"], "embedded")
            self.assertEqual(schema["recommended_mapping"]["truth_array"], "nodes/target_denorm")
            report = run_field_evaluation({"prediction_path": str(prediction), "truth_path": str(truth)})
            self.assertEqual(report["truth_source"], "embedded")
            self.assertAlmostEqual(report["aggregate"]["mae"]["mean"], 0.5)

            mlp_prediction = root / "mlp-prediction.h5"
            mlp_truth = root / "mlp-truth.h5"
            with h5py.File(mlp_prediction, "w") as handle:
                handle.create_dataset("Y_pred", data=np.asarray([[1.2, 2.4], [3.1, 4.3]], dtype=np.float32))
                handle.create_dataset("Y_true", data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            with h5py.File(mlp_truth, "w") as handle:
                handle.create_dataset("Y", data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            mlp_report = run_field_evaluation({"prediction_path": str(mlp_prediction), "truth_path": str(mlp_truth)})
            self.assertEqual(mlp_report["truth_source"], "embedded")
            self.assertEqual(mlp_report["evaluated_samples"], 2)

    @staticmethod
    def _mesh(
        path: Path,
        samples: dict[str, object],
        feature_names: list[str],
        input_count: int,
        condition_count: int,
        output_count: int,
    ) -> None:
        import h5py
        import numpy as np

        with h5py.File(path, "w") as handle:
            handle.attrs["builder_input_var"] = input_count
            handle.attrs["builder_cond_var"] = condition_count
            handle.attrs["builder_output_var"] = output_count
            metadata = handle.create_group("metadata")
            metadata.create_dataset("feature_names", data=np.asarray([name.encode() for name in feature_names]))
            data = handle.create_group("data")
            for sample_id, values in samples.items():
                data.create_group(sample_id).create_dataset("nodal_data", data=values)


if __name__ == "__main__":
    unittest.main()
