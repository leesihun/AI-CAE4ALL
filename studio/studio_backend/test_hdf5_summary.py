from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from studio_backend.hdf5_preview import hdf5_sample, hdf5_samples, hdf5_summary
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


if __name__ == "__main__":
    unittest.main()
