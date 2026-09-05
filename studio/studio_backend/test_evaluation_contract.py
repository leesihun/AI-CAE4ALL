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

        # Row layout follows the suite's mesh HDF5 contract: rows 0:3 are the
        # reference coordinates, rows 3:3+input_var the state the model
        # predicts, and cond_var rows follow. builder_input_var counts the
        # state rows only -- every staged dataset (ex4/ex7/ex8/ex9) is written
        # that way. The earlier version of this fixture treated x/y/z as the
        # "inputs" and put the outputs after the condition row, a layout no
        # builder produces; on ex7 that window would have scored sdf/normal_x/
        # normal_y/node_type as predictions.
        with tempfile.TemporaryDirectory(prefix="evaluation-mesh-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "rollout_sample7.h5"
            truth = root / "truth.h5"
            values = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4)
            names = ["x_coord", "y_coord", "z_coord", "pressure", "velocity", "mach", "node_type"]
            self._mesh(prediction, {"7": values + 0.25}, names, input_count=2, condition_count=1, output_count=2)
            self._mesh(truth, {"7": values}, names, input_count=2, condition_count=1, output_count=2)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertTrue(schema["compatible"], schema["errors"])
            self.assertEqual(schema["prediction"]["contract"], "mesh_state")
            self.assertEqual(schema["sample_matching"]["matched_ids"], ["7"])
            self.assertEqual(schema["truth"]["target_indices"], [3, 4])
            mapping = schema["recommended_mapping"]
            self.assertEqual([item["name"] for item in mapping["field_pairs"]], ["pressure", "velocity"])
            self.assertEqual(mapping["prediction_start"], 3)
            self.assertEqual(mapping["truth_start"], 3)
            self.assertEqual(mapping["num_fields"], 2)

            report = run_field_evaluation({
                "prediction_path": str(prediction),
                "truth_path": str(truth),
                "prediction_start": 3,
                "truth_start": 3,
                "num_fields": 2,
            })
            self.assertEqual(report["contract"], "mesh_state")
            self.assertEqual(report["evaluated_samples"], 1)
            self.assertAlmostEqual(report["aggregate"]["mae"]["mean"], 0.25)

    def test_mesh_schema_never_scores_reference_coordinates(self) -> None:
        """A rollout whose physical channels are named differently from the truth
        file's must not be scored on the only names that do match -- x/y/z_coord.

        That is exactly what a Studio pipeline did: the native writers hardcoded
        NASA-CRM channel names, the by-name pairing matched the three coordinate
        rows, and the report said every metric was 0.0 and R^2 = 1.0 on 87
        held-out samples. Coordinates are copied into every rollout unchanged,
        so a coordinate-only mapping is not a score of anything.
        """
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-coords-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "rollout_sample3.h5"
            truth = root / "truth.h5"
            values = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
            truth_values = np.concatenate([values, np.ones((1, 2, 4), dtype=np.float32)])
            # Rollout: coordinates + two predicted channels under legacy names + node type.
            self._mesh(prediction, {"3": values + 0.5},
                       ["x_coord", "y_coord", "z_coord", "x_disp(mm)", "y_disp(mm)", "Part No."], 0, 0, 0)
            # Truth: the same coordinates, the state under its real names, then a condition row.
            self._mesh(truth, {"3": truth_values},
                       ["x_coord", "y_coord", "z_coord", "ux", "uy", "uz", "die_profile"], 2, 2, 2)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            pairs = schema["recommended_mapping"]["field_pairs"]
            self.assertFalse(any(item["prediction_index"] < 3 for item in pairs), pairs)
            # Row 5 is "Part No." -- the per-node categorical label every mesh
            # rollout writer appends, copied through unchanged like the
            # coordinates. It used to be offered as a scoreable target, which
            # made the prediction look like it had three physical channels.
            self.assertEqual(schema["prediction"]["target_indices"], [3, 4])
            self.assertEqual(schema["truth"]["target_indices"], [3, 4])
            # Two physical rows on each side and no shared names, so a positional
            # mapping is PROPOSED but must be confirmed; it never scores silently.
            self.assertEqual(schema["recommended_mapping"]["confidence"], "confirm")
            with self.assertRaises(ValueError):
                run_field_evaluation({"prediction_path": str(prediction), "truth_path": str(truth)})
            # An explicit coordinate pair is refused too.
            with self.assertRaisesRegex(ValueError, "reference-coordinate"):
                run_field_evaluation({
                    "prediction_path": str(prediction),
                    "truth_path": str(truth),
                    "field_pairs": [{"prediction_index": 0, "truth_index": 0}],
                    "confirm_mapping": True,
                })
            # And so are legacy rows that only cover the coordinates.
            with self.assertRaisesRegex(ValueError, "reference-coordinate"):
                run_field_evaluation({
                    "prediction_path": str(prediction),
                    "truth_path": str(truth),
                    "prediction_start": 0,
                    "truth_start": 0,
                    "num_fields": 3,
                })

    def test_declared_output_var_bounds_the_scoreable_rows(self) -> None:
        """A rollout that declares `output_var` is taken at its word.

        The five native rollout writers now record it beside `num_features`, so
        the evaluator no longer has to infer how many rows are predictions --
        which is how the trailing node-type row and, before that, the reference
        coordinates reached a score.
        """
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-outputvar-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "rollout_sample7.h5"
            values = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
            # Exactly what the five rollout writers now emit: num_features plus
            # output_var, and NO builder_* keys (those belong to a dataset built
            # by the dataset builders, not to a rollout).
            with h5py.File(prediction, "w") as handle:
                handle.attrs["num_samples"] = 1
                handle.attrs["num_features"] = 6
                handle.attrs["num_timesteps"] = 2
                handle.attrs["output_var"] = 2
                handle.create_group("data").create_group("7").create_dataset("nodal_data", data=values)

            schema = evaluation_schema({
                "prediction_path": str(prediction),
                "truth_path": str(prediction),
            })
            self.assertEqual(schema["prediction"]["target_indices"], [3, 4])

    def test_coordinate_rows_named_x_y_z_are_still_excluded(self) -> None:
        """The coordinate check used to match only the `_coord` suffix.

        A file naming its reference rows x/y/z -- the obvious spelling -- turned
        the filter AND the coordinate-only guard off, and the coordinates were
        scored: zero error, R^2 = 1, for any model.
        """
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-xyz-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            path = root / "rollout_xyz.h5"
            values = np.arange(5 * 2 * 4, dtype=np.float32).reshape(5, 2, 4)
            self._mesh(path, {"1": values}, ["x", "y", "z", "ux", "uy"], 0, 0, 0)

            schema = evaluation_schema({"prediction_path": str(path), "truth_path": str(path)})
            self.assertEqual(schema["prediction"]["target_indices"], [3, 4])
            with self.assertRaisesRegex(ValueError, "reference-coordinate"):
                run_field_evaluation({
                    "prediction_path": str(path),
                    "truth_path": str(path),
                    "field_pairs": [{"prediction_index": 1, "truth_index": 1}],
                    "confirm_mapping": True,
                })

    def test_mesh_prediction_without_coordinate_rows_is_still_scoreable(self) -> None:
        """SimulGen-VAE's `reconstruct` writes only its physical channels.

        `data/{id}/nodal_field` holds `num_var` rows and no reference
        coordinates, so a blanket "rows 0:3 are coordinates" rule would discard
        every channel it produces and refuse a scoreable reconstruction. Rows
        are coordinates only when the file's own names or row count say so.
        """
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-nocoords-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "reconstructions.h5"
            truth = root / "truth.h5"
            values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
            # Exactly SimulGen-VAE's layout: two physical rows, nodal_field, no names.
            with h5py.File(prediction, "w") as handle:
                handle.attrs["num_samples"] = 1
                handle.attrs["num_var"] = 2
                handle.create_group("data").create_dataset("5/nodal_field", data=values + 0.5)
            truth_values = np.concatenate([np.zeros((3, 3, 4), dtype=np.float32), values])
            self._mesh(truth, {"5": truth_values},
                       ["x_coord", "y_coord", "z_coord", "ux", "uy"], 2, 0, 2)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertEqual(schema["prediction"]["target_indices"], [0, 1])
            self.assertEqual(schema["truth"]["target_indices"], [3, 4])
            pairs = schema["recommended_mapping"]["field_pairs"]
            self.assertEqual([(p["prediction_index"], p["truth_index"]) for p in pairs], [(0, 3), (1, 4)])

            report = run_field_evaluation({
                "prediction_path": str(prediction),
                "truth_path": str(truth),
                "field_pairs": pairs,
                "confirm_mapping": True,
            })
            self.assertEqual(report["evaluated_samples"], 1)
            self.assertAlmostEqual(report["aggregate"]["mae"]["mean"], 0.5)

    def test_coordinate_less_prediction_with_four_rows_keeps_every_channel(self) -> None:
        """The coordinate rule must key on the array name, not the row count.

        configs/SimulGenVAE/ex3/config_reconstruct.txt sets num_var 4, so that
        reconstruct writes four physical rows and no coordinates. A row-count
        guess ("more than three rows means the first three are coordinates")
        silently dropped three of the four channels and then failed the mapping
        outright on the surviving 1-vs-4 count. `nodal_field` never carries
        coordinates; only `nodal_data` does.
        """
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-nocoords4-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "reconstructions.h5"
            truth = root / "truth.h5"
            values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
            with h5py.File(prediction, "w") as handle:
                handle.attrs["num_samples"] = 1
                handle.attrs["num_var"] = 4
                handle.create_group("data").create_dataset("2/nodal_field", data=values + 0.25)
            truth_values = np.concatenate([np.zeros((3, 3, 5), dtype=np.float32), values])
            self._mesh(truth, {"2": truth_values},
                       ["x_coord", "y_coord", "z_coord", "a", "b", "c", "e"], 4, 0, 4)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertEqual(schema["prediction"]["row_source"], "nodal_field")
            self.assertEqual(schema["prediction"]["target_indices"], [0, 1, 2, 3])
            pairs = schema["recommended_mapping"]["field_pairs"]
            self.assertEqual([(p["prediction_index"], p["truth_index"]) for p in pairs],
                             [(0, 3), (1, 4), (2, 5), (3, 6)])
            report = run_field_evaluation({
                "prediction_path": str(prediction),
                "truth_path": str(truth),
                "field_pairs": pairs,
                "confirm_mapping": True,
            })
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

    def test_mesh_schema_rejects_timestep_mismatch_instead_of_truncating(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="evaluation-time-mismatch-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            prediction = root / "prediction.h5"
            truth = root / "truth.h5"
            names = ["x", "y", "z", "pressure"]
            self._mesh(prediction, {"2": np.zeros((4, 2, 5), dtype=np.float32)}, names, 3, 0, 1)
            self._mesh(truth, {"2": np.zeros((4, 3, 5), dtype=np.float32)}, names, 3, 0, 1)

            schema = evaluation_schema({"prediction_path": str(prediction), "truth_path": str(truth)})

            self.assertFalse(schema["compatible"])
            self.assertEqual(schema["sample_matching"]["compatible_shape_count"], 0)
            self.assertEqual(schema["sample_matching"]["incompatible_shape_count"], 1)
            self.assertTrue(any("timestep/node" in error for error in schema["errors"]))
            with self.assertRaisesRegex(ValueError, "timestep/node"):
                run_field_evaluation({"prediction_path": str(prediction), "truth_path": str(truth)})

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
