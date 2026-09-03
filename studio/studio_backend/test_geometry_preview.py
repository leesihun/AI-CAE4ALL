from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from studio_backend.geometry_preview import geometry_sample


class GeometryPreviewRoutingTests(TestCase):
    def test_step_files_use_cad_tessellator(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "part.step"
            path.write_text("test fixture", encoding="utf-8")
            vertices = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
            faces = np.asarray([[0, 1, 2]], dtype=np.int64)
            with (
                patch(
                    "studio_backend.geometry_preview._load_cad",
                    return_value=(vertices, faces, {"reader": "test"}),
                ) as cad_loader,
                patch("studio_backend.geometry_preview._load_trimesh") as trimesh_loader,
            ):
                result = geometry_sample(path, path.name)

            cad_loader.assert_called_once_with(path)
            trimesh_loader.assert_not_called()
            self.assertEqual(result["preview_kind"], "surface")
            self.assertEqual(result["mesh"]["returned_faces"], 1)
            self.assertEqual(result["metadata"]["reader"], "test")
