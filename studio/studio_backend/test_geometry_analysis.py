from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import h5py
import numpy as np

from studio_backend.geometry_analysis import CHECKPOINT_RELATIVE, _read_field, analyze_geometry


class GeometryAnalysisFieldTests(TestCase):
    def write_prediction(self, path: Path, *, declare_uz=True, finite=True) -> None:
        nodal = np.zeros((8, 2, 4), dtype=np.float32)
        nodal[:3, :, :] = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])[:, None, :]
        nodal[4, 0] = 99  # Seed frame must never be displayed as prediction.
        nodal[4, 1] = [-0.2, 0.1, 0.3, 0.0] if finite else [np.nan, 0.1, 0.3, 0.0]
        nodal[5, 1] = 8  # u_z is deliberately not row5 in this valid named contract.
        names = ["x_coord", "y_coord", "z_coord", "u_x", "u_z" if declare_uz else "other", "u_y", "von_mises", "Part No."]
        with h5py.File(path, "w") as handle:
            group = handle.create_group("data/1")
            group.create_dataset("nodal_data", data=nodal)
            group.create_dataset("mesh_edge", data=np.asarray([[0, 1, 2, 0, 1, 2], [1, 2, 0, 3, 3, 3]]))
            handle.create_dataset("metadata/feature_names", data=np.asarray(names, dtype="S12"))

    def test_signed_uz_comes_from_named_channel_and_final_native_frame(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "prediction.h5"
            self.write_prediction(path)
            result = _read_field(path, root)
        np.testing.assert_allclose(result["values"], [-0.2, 0.1, 0.3, 0.0])
        self.assertAlmostEqual(result["stats"]["min"], -0.2)
        self.assertAlmostEqual(result["stats"]["max"], 0.3)
        self.assertEqual(result["feature"], "u_z")
        self.assertEqual(result["unit"], "mm")
        self.assertEqual(len(result["vertices"]), len(result["values"]))
        self.assertEqual(result["prediction_kind"], "surrogate")

    def test_missing_vertical_channel_is_not_replaced_with_coordinates(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.h5"
            self.write_prediction(path, declare_uz=False)
            with self.assertRaisesRegex(ValueError, "does not declare a u_z"):
                _read_field(path, path.parent)

    def test_nonfinite_native_prediction_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.h5"
            self.write_prediction(path, finite=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                _read_field(path, path.parent)


class GeometryAnalysisInterfaceTests(TestCase):
    def test_registration_failure_prevents_native_model_launch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stl = root / "generated.stl"
            stl.touch()
            checkpoint = root / CHECKPOINT_RELATIVE
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            bridge = Mock()
            bridge.mesh_to_ver_record.side_effect = ValueError("frame registration failed: missing bolt interfaces")
            with patch("studio_backend.geometry_analysis._load_bridge", return_value=bridge), \
                 patch("trimesh.load", return_value=object()), \
                 patch("studio_backend.geometry_analysis.subprocess.run") as native:
                with self.assertRaisesRegex(ValueError, "frame registration failed"):
                    analyze_geometry(stl, root / "studio/runtime/geometry_explorer/jobs/test", root)
            native.assert_not_called()

    def test_missing_checkpoint_has_actionable_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stl = root / "generated.stl"
            stl.touch()
            with self.assertRaisesRegex(FileNotFoundError, "DeepJEB HI-MGN checkpoint is unavailable"):
                analyze_geometry(stl, root / "analysis", root)
