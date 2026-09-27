import csv
import json
import tempfile
import unittest
from pathlib import Path

from studio_backend.analysis import (
    comparison_schema,
    optimization_schema,
    run_model_comparison,
    run_optimization,
    write_optimize_summary_table,
    write_screening_table,
)
from studio_backend.paths import RUNTIME_ROOT


class ComparisonSchemaTests(unittest.TestCase):
    def test_common_numeric_schema_drives_multi_csv_ranking(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="comparison-schema-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            first = root / "run-a.csv"
            second = root / "run-b.csv"
            self._write(first, ["case", "relative_l2", "mae"], [["part-1", "0.2", "0.1"]])
            self._write(second, ["case", "relative_l2", "rmse"], [["part-1", "0.3", "0.4"]])

            schema = comparison_schema({"csv_paths": [str(first), str(second)]})

            self.assertEqual(schema["common_columns"], ["case", "relative_l2"])
            self.assertEqual(schema["numeric_columns"], ["relative_l2"])
            self.assertEqual(schema["group_columns"], ["case"])
            report = run_model_comparison({
                "csv_paths": [str(first), str(second)],
                "group_column": "case",
                "metric": "relative_l2",
                "direction": "min",
            })
            self.assertEqual(report["runs"], 2)
            self.assertEqual(report["best"]["source"], schema["sources"][0]["path"])

    def test_comparison_ranks_group_means_not_the_easiest_sample(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="comparison-means-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            first = root / "run-a.csv"
            second = root / "run-b.csv"
            self._write(first, ["prediction_file", "relative_l2"], [["a.h5", 0.01], ["a.h5", 1.0]])
            self._write(second, ["prediction_file", "relative_l2"], [["b.h5", 0.4], ["b.h5", 0.4]])

            report = run_model_comparison({
                "csv_paths": [str(first), str(second)],
                "group_column": "prediction_file",
                "metric": "relative_l2",
                "direction": "min",
            })

            self.assertEqual(report["numeric_rows"], 4)
            self.assertEqual(report["ranked_groups"], 2)
            self.assertEqual(report["best"]["source"], report["sources"][1]["path"])
            self.assertEqual(report["best"]["count"], 2)
            self.assertAlmostEqual(report["best"]["value"], 0.4)
            self.assertIn("mean", report["aggregation"])

    def test_blank_group_column_uses_one_mean_per_csv(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="comparison-runs-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            path = root / "per-sample.csv"
            self._write(path, ["sample", "mae"], [["a", 1.0], ["b", 3.0]])

            report = run_model_comparison({
                "csv_paths": [str(path)],
                "group_column": "",
                "metric": "mae",
                "direction": "min",
            })

            self.assertEqual(report["ranked_groups"], 1)
            self.assertEqual(report["best"]["name"], "per-sample")
            self.assertEqual(report["best"]["count"], 2)
            self.assertAlmostEqual(report["best"]["value"], 2.0)

    @staticmethod
    def _write(path, headers, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)


class OptimizationSchemaTests(unittest.TestCase):
    def test_identifier_columns_are_not_suggested_as_objectives(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-schema-", dir=RUNTIME_ROOT) as directory:
            path = Path(directory) / "candidates.csv"
            ComparisonSchemaTests._write(
                path,
                ["candidate_id", "timesteps", "mass", "peak_stress", "material"],
                [[1, 20, 12.5, 210.0, "steel"], [2, 20, 11.2, 240.0, "aluminum"]],
            )

            schema = optimization_schema({"csv_path": str(path)})

            self.assertEqual(schema["numeric_columns"], ["candidate_id", "timesteps", "mass", "peak_stress"])
            self.assertEqual(schema["identifier_columns"], ["candidate_id", "timesteps"])
            self.assertEqual(schema["objective_columns"], ["mass", "peak_stress"])
            self.assertEqual(schema["rows_sampled"], 2)

    def test_mesh_resolution_counts_are_not_objectives(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-meshcount-", dir=RUNTIME_ROOT) as directory:
            path = Path(directory) / "optimize_summary.csv"
            ComparisonSchemaTests._write(
                path,
                ["id", "mass_kg", "fea_mass_kg", "num_nodes", "num_tets"],
                [["optimized", 0.7, 0.71, 5000, 42000], ["baseline", 0.8, 0.79, 5200, 43000]],
            )

            schema = optimization_schema({"csv_path": str(path)})

            self.assertEqual(schema["identifier_columns"], ["id", "num_nodes", "num_tets"])
            self.assertEqual(schema["objective_columns"], ["mass_kg", "fea_mass_kg"])

    def test_pareto_ignores_non_finite_objective_rows(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-finite-", dir=RUNTIME_ROOT) as directory:
            path = Path(directory) / "candidates.csv"
            ComparisonSchemaTests._write(
                path,
                ["candidate_id", "mass", "peak_stress"],
                [[1, 12.5, 210.0], [2, "nan", 190.0], [3, 11.2, 240.0]],
            )

            report = run_optimization({
                "csv_path": str(path),
                "objectives": "mass,peak_stress",
                "directions": "min,min",
                "top_k": 10,
            })

            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["numeric_candidates"], 2)

    def test_small_pareto_fronts_are_reported_as_boundary_points(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-boundary-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            cases = (
                ("one", [["only", 1.0, 1.0]], 1, 1),
                ("two", [["light", 1.0, 2.0], ["strong", 2.0, 1.0]], 2, 2),
                ("two-top-one", [["light", 1.0, 2.0], ["strong", 2.0, 1.0]], 1, 2),
            )
            for name, rows, top_k, expected_pareto in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    ComparisonSchemaTests._write(path, ["id", "mass", "stress"], rows)
                    report = run_optimization({
                        "csv_path": str(path),
                        "objectives": "mass,stress",
                        "directions": "min,min",
                        "top_k": top_k,
                    })
                    self.assertEqual(report["pareto"], expected_pareto)
                    self.assertEqual(len(report["selected"]), min(top_k, expected_pareto))
                    self.assertTrue(all(item["crowding"] is None for item in report["selected"]))


class OptimizationSummaryTableTests(unittest.TestCase):
    def test_mesh_cardinality_column_matches_analysis_backend(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-summary-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            for backend, mesh_key, other_key, mesh_value in (
                ("surrogate", "num_nodes", "num_tets", 5003),
                ("fea", "num_tets", "num_nodes", 12345),
            ):
                output = root / backend
                output.mkdir()
                (output / "summary.json").write_text(json.dumps({
                    "analysis_backend": backend,
                    "verified": {
                        "baseline": {
                            "mass_kg": 1.0,
                            "peak_von_mises_MPa": 100.0,
                            "max_displacement_mm": 0.1,
                            mesh_key: mesh_value,
                        },
                    },
                }), encoding="utf-8")

                result = write_optimize_summary_table(output)
                self.assertEqual(result["rows"], 1)
                with (output / "optimize_summary.csv").open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                self.assertIn(mesh_key, reader.fieldnames)
                self.assertNotIn(other_key, reader.fieldnames)
                self.assertEqual(rows[0][mesh_key], str(mesh_value))
                # No vertical limit searched, so neither column is invented.
                self.assertNotIn("vertical_displacement_mm", reader.fieldnames)
                self.assertNotIn("vertical_limit_met", reader.fieldnames)

    def test_vertical_limit_columns_and_failed_design(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-summary-", dir=RUNTIME_ROOT) as directory:
            output = Path(directory)
            (output / "summary.json").write_text(json.dumps({
                "analysis_backend": "fea",
                "limits": {"vertical_disp_allow": 0.0002},
                "verified": {
                    # Only the optimized design has a vertical displacement; the
                    # baseline lacks a mass, which must not be written as 0.
                    "baseline": {"peak_von_mises_MPa": 300.0, "max_displacement_mm": 0.3,
                                 "vertical_displacement_mm": 0.25, "num_tets": 10},
                    "optimized": {"mass_kg": 0.8, "peak_von_mises_MPa": 310.0,
                                  "max_displacement_mm": 0.2,
                                  "vertical_displacement_mm": 0.19, "num_tets": 11},
                    "failed": {"typical": "RuntimeError: gmsh failed"},
                    "vertical_limit_met": True,
                },
            }), encoding="utf-8")

            result = write_optimize_summary_table(output)
            self.assertEqual(result["rows"], 3)
            with (output / "optimize_summary.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = {row["id"]: row for row in reader}
            for column in ("vertical_displacement_mm", "vertical_limit_met", "error"):
                self.assertIn(column, reader.fieldnames)
            self.assertEqual(rows["baseline"]["mass_kg"], "")
            self.assertEqual(rows["baseline"]["vertical_limit_met"], "False")
            self.assertEqual(rows["optimized"]["vertical_displacement_mm"], "0.19")
            self.assertEqual(rows["optimized"]["vertical_limit_met"], "True")
            self.assertEqual(rows["typical"]["error"], "RuntimeError: gmsh failed")
            self.assertEqual(rows["typical"]["mass_kg"], "")

    def test_fea_verification_gets_its_own_columns(self):
        """opt_fea_verify's solver numbers sit beside the surrogate's, never in them."""
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimization-summary-", dir=RUNTIME_ROOT) as directory:
            output = Path(directory)
            surrogate = {"peak_von_mises_MPa": 40.0, "max_displacement_mm": 0.1,
                         "vertical_displacement_mm": 0.15, "num_nodes": 5000}
            (output / "summary.json").write_text(json.dumps({
                "analysis_backend": "surrogate",
                "limits": {"vertical_disp_allow": 0.0002},
                "verified": {"optimized": {**surrogate, "mass_kg": 0.8},
                             "baseline": {**surrogate, "mass_kg": 1.0},
                             "vertical_limit_met": True},
                "fea_verification": {"designs": {
                    # The surrogate called it feasible; the solver disagrees.
                    "optimized": {"mass_kg": 0.81, "peak_von_mises_MPa": 310.0,
                                  "max_displacement_mm": 0.3,
                                  "vertical_displacement_mm": 0.26,
                                  "limits": [{"limit": "vertical max |u_z|", "met": False,
                                              "value": 0.26, "allow": 0.2}]},
                    "baseline": {"error": "RuntimeError: gmsh failed"},
                }},
            }), encoding="utf-8")

            write_optimize_summary_table(output)
            with (output / "optimize_summary.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = {row["id"]: row for row in reader}
            self.assertEqual(rows["optimized"]["vertical_limit_met"], "True")
            self.assertEqual(rows["optimized"]["vertical_displacement_mm"], "0.15")
            self.assertEqual(rows["optimized"]["fea_vertical_limit_met"], "False")
            self.assertEqual(rows["optimized"]["fea_vertical_displacement_mm"], "0.26")
            self.assertEqual(rows["optimized"]["fea_mass_kg"], "0.81")
            self.assertEqual(rows["optimized"]["mass_kg"], "0.8")
            # A failed re-solve is reported, not written as zeros.
            self.assertEqual(rows["baseline"]["fea_error"], "RuntimeError: gmsh failed")
            self.assertEqual(rows["baseline"]["fea_mass_kg"], "")
            self.assertEqual(rows["baseline"]["mass_kg"], "1.0")



class ScreeningTableTests(unittest.TestCase):
    """opt_budget 0: every screened design reaches the Optimization block."""

    NATIVE = ["id", "index", "mass_kg", "peak_von_mises_mpa", "max_displacement_mm",
              "vertical_displacement_mm", "vertical_limit_met", "feasible", "score",
              "num_nodes", "role", "path", "error"]

    def _screen(self, output, rows, **summary):
        ComparisonSchemaTests._write(output / "screening.csv", self.NATIVE, rows)
        (output / "summary.json").write_text(json.dumps({
            "analysis_backend": "surrogate",
            "limits": {"vertical_disp_allow": 0.00015},
            "screening": {"table": "screening.csv", "delivered_id": "screen_0001",
                          "typical_id": "screen_0002"},
            **summary,
        }), encoding="utf-8")

    def test_every_screened_design_with_verification_joined(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="screening-table-", dir=RUNTIME_ROOT) as directory:
            output = Path(directory)
            (output / "optimized.stl").write_text("solid x\nendsolid x\n", encoding="utf-8")
            self._screen(output, [
                ["screen_0000", 0, 0.30, 40.0, 0.2, 0.20, "false", "false", 1.9, 5000, "", "", ""],
                ["screen_0001", 1, 0.50, 30.0, 0.1, 0.12, "true", "true", 0.8, 5000,
                 "delivered", "optimized.stl", ""],
                # typical.stl was never written (its re-analysis failed).
                ["screen_0002", 2, 0.70, 20.0, 0.1, 0.09, "true", "true", 1.1, 5000,
                 "typical", "typical.stl", ""],
                ["screen_0003", 3, "", "", "", "", "", "", "", "", "", "",
                 "RuntimeError: no zero crossing"],
            ], verified={
                "optimized": {"mass_kg": 0.51, "peak_von_mises_MPa": 31.0,
                              "max_displacement_mm": 0.11, "vertical_displacement_mm": 0.13},
                "failed": {"typical": "RuntimeError: gmsh failed"},
                "vertical_limit_met": True,
            }, fea_verification={"designs": {
                "optimized": {"mass_kg": 0.52, "peak_von_mises_MPa": 33.0,
                              "max_displacement_mm": 0.12, "vertical_displacement_mm": 0.14,
                              "limits": [{"limit": "vertical max |u_z|", "met": True}]},
            }})

            result = write_screening_table(output)
            self.assertEqual(result["rows"], 4)
            self.assertTrue(result["path"].endswith("optimize_screening.csv"))
            with (output / "optimize_screening.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = {row["id"]: row for row in reader}
            self.assertEqual(reader.fieldnames[:len(self.NATIVE)], self.NATIVE)
            delivered = rows["screen_0001"]
            # The screen's own numbers are untouched; verification sits beside them.
            self.assertEqual(delivered["mass_kg"], "0.5")
            self.assertEqual(delivered["verify_mass_kg"], "0.51")
            self.assertEqual(delivered["verify_vertical_limit_met"], "True")
            self.assertEqual(delivered["fea_vertical_displacement_mm"], "0.14")
            self.assertEqual(delivered["fea_vertical_limit_met"], "True")
            self.assertEqual(delivered["path"], "optimized.stl")
            self.assertEqual(rows["screen_0002"]["verify_error"], "RuntimeError: gmsh failed")
            self.assertEqual(rows["screen_0002"]["path"], "")
            self.assertEqual(rows["screen_0000"]["verify_mass_kg"], "")
            self.assertEqual(rows["screen_0003"]["mass_kg"], "")
            self.assertEqual({row["analysis_backend"] for row in rows.values()}, {"surrogate"})

            # The Optimization block then ranks the whole screen: the lightest
            # design under the 0.15 mm limit, not the lightest overall, and the
            # failed row is skipped rather than ranked as mass 0.
            report = run_optimization({
                "csv_path": str(output / "optimize_screening.csv"),
                "objectives": "mass_kg",
                "directions": "min",
                "constraints": "vertical_displacement_mm <= 0.15",
                "top_k": 5,
            })
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["skipped_rows"], 1)
            self.assertEqual(report["feasible"], 2)
            self.assertEqual([item["id"] for item in report["selected"]], ["screen_0001"])

    def test_not_a_screen_falls_through(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="screening-table-", dir=RUNTIME_ROOT) as directory:
            output = Path(directory)
            self.assertIsNone(write_screening_table(output))
            # A search run's summary has no `screening` block.
            (output / "summary.json").write_text(json.dumps({
                "verified": {"optimized": {"mass_kg": 1.0}}}), encoding="utf-8")
            self.assertIsNone(write_screening_table(output))
            self.assertEqual(write_optimize_summary_table(output)["rows"], 1)
            # A screen whose table is missing publishes nothing, not an empty CSV.
            (output / "summary.json").write_text(json.dumps({
                "verified": {"optimized": {"mass_kg": 1.0}},
                "screening": {"table": "screening.csv"}}), encoding="utf-8")
            self.assertIsNone(write_screening_table(output))
            self.assertFalse((output / "optimize_screening.csv").exists())


if __name__ == "__main__":
    unittest.main()
