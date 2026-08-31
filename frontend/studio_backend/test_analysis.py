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


if __name__ == "__main__":
    unittest.main()
