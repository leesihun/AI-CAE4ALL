from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from studio_backend.hdf5_preview import hdf5_facts, hdf5_sample, hdf5_samples, hdf5_summary
from studio_backend.native_jobs import create_viewer_smoke_fixture
from studio_backend.paths import SUITE_ROOT


class Hdf5SummaryContractTests(unittest.TestCase):
    def test_viewer_fixture_exercises_the_real_operator_grid_contract(self) -> None:
        try:
            fixture = create_viewer_smoke_fixture()
        except RuntimeError as exc:  # pragma: no cover - feature dependency
            self.skipTest(str(exc))
        path = SUITE_ROOT / fixture["operator_grid"]
        catalog = hdf5_samples(path)
        sample = hdf5_sample(path, "0", 0, 0)
        self.assertEqual(catalog["contract"], "operator_grid")
        self.assertEqual(catalog["total_samples"], 2)
        self.assertEqual(sample["returned_points"], 225)
        self.assertFalse(fixture["scientific_use"])

    def test_small_string_contract_is_visible_without_loading_numeric_arrays(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError as exc:  # pragma: no cover - feature dependency
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["provenance"] = "generated-test"
                handle.create_dataset("input_names", data=np.asarray([b"length", b"width"]))
                handle.create_dataset("X", data=np.zeros((1024, 2), dtype=np.float32))

            summary = hdf5_summary(path)
            records = {item["path"]: item for item in summary["items"]}
            self.assertEqual(summary["root_attrs"]["provenance"], "generated-test")
            self.assertEqual(records["input_names"]["values"], ["length", "width"])
            self.assertNotIn("values", records["X"])

    def test_card_facts_come_from_attributes_and_sample_metadata(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError as exc:  # pragma: no cover - feature dependency
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs.update(num_samples=2, num_features=5, num_timesteps=1,
                                    builder_input_var=1, builder_output_var=1, builder_cond_var=1)
                names = handle.create_group("metadata")
                names.create_dataset("feature_names", data=np.asarray([b"x", b"y", b"z", b"p", b"mach"]))
                for sample_id, nodes in (("1", 40), ("2", 55)):
                    group = handle.create_group(f"data/{sample_id}")
                    group.create_dataset("nodal_data", data=np.zeros((5, 1, nodes), dtype=np.float32))
                    group.create_group("metadata").attrs["num_nodes"] = nodes

            facts = hdf5_facts(path)
            self.assertEqual(facts["contract"], "mesh_state")
            self.assertEqual(facts["num_samples"], 2)
            self.assertEqual((facts["nodes_min"], facts["nodes_max"]), (40, 55))
            self.assertEqual(facts["feature_names"], ["x", "y", "z", "p", "mach"])
            self.assertEqual((facts["input_var"], facts["output_var"], facts["cond_var"]), (1, 1, 1))

            table = Path(directory) / "table.h5"
            with h5py.File(table, "w") as handle:
                handle.create_dataset("X", data=np.zeros((7, 2)))
                handle.create_dataset("Y", data=np.zeros((7, 1)))
                handle.create_dataset("output_names", data=np.asarray([b"cd"]))
            facts = hdf5_facts(table)
            self.assertEqual(facts["contract"], "table")
            self.assertEqual(facts["input_names"], ["x 0", "x 1"])
            self.assertEqual(facts["output_names"], ["cd"])


if __name__ == "__main__":
    unittest.main()
