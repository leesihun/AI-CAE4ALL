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


class StlWeldingTests(TestCase):
    """An STL is a triangle soup; the preview must see one connected surface."""

    def _sphere_stl(self, directory: str, subdivisions: int) -> Path:
        import trimesh

        path = Path(directory) / "sphere.stl"
        trimesh.creation.icosphere(subdivisions=subdivisions).export(path)
        return path

    def test_stl_corners_are_welded_and_watertightness_is_real(self) -> None:
        try:
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("trimesh is not installed")
        with TemporaryDirectory() as directory:
            path = self._sphere_stl(directory, subdivisions=2)
            result = geometry_sample(path, path.name)

        # 320 faces; the file stores 960 corners, the surface has 162 vertices.
        self.assertEqual(result["mesh"]["returned_faces"], 320)
        self.assertEqual(result["total_points"], 162)
        self.assertEqual(result["returned_points"], 162)
        self.assertEqual(result["metadata"]["file_vertices"], 960)
        self.assertTrue(result["metadata"]["watertight"])

    def test_a_simplified_stl_stays_connected(self) -> None:
        try:
            import fast_simplification  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("trimesh or fast-simplification is not installed")
        with TemporaryDirectory() as directory:
            path = self._sphere_stl(directory, subdivisions=4)
            result = geometry_sample(path, path.name, face_limit=1000)

        mesh = result["mesh"]
        self.assertTrue(mesh["reduced"])
        self.assertEqual(mesh["total_elements"], 5120)
        # A closed triangulated surface has about half as many vertices as
        # faces. Decimating the soup instead kept three private corners per
        # surviving triangle, and the part rendered as scattered fragments.
        self.assertLess(result["returned_points"], mesh["returned_faces"])
