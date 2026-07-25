"""Bounded CAD and mesh extraction for the shared Studio artifact viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio_backend.paths import relative


GEOMETRY_SUFFIXES = {
    ".stl", ".ply", ".obj", ".off",
    ".step", ".stp", ".iges", ".igs", ".brep",
    ".vtk", ".vtu", ".vtp", ".msh",
}
MESHIO_SUFFIXES = {".vtk", ".vtu", ".vtp", ".msh"}
CAD_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep"}


def _imports():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for geometry visualization.") from exc
    return np


def _geometry_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in GEOMETRY_SUFFIXES:
            raise ValueError(f"{path.name} is not a supported geometry file.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Geometry path does not exist: {relative(path)}")
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in GEOMETRY_SUFFIXES
            and not item.name.startswith("._")
        ),
        key=lambda item: item.relative_to(path).as_posix().lower(),
    )


def geometry_samples(path: Path, limit: int = 100) -> dict[str, Any]:
    files = _geometry_paths(path)
    samples = []
    for item in files[:limit]:
        sample_id = item.name if path.is_file() else item.relative_to(path).as_posix()
        samples.append(
            {
                "id": sample_id,
                "label": sample_id,
                "datasets": [
                    {
                        "name": item.name,
                        "shape": [],
                        "dtype": item.suffix.lower().lstrip(".").upper(),
                    }
                ],
                "default_feature": 0,
            }
        )
    return {
        "path": relative(path),
        "source_kind": "geometry",
        "contract": "surface_geometry",
        "default_mode": "mesh",
        "samples": samples,
        "truncated": len(files) > limit,
        "total_samples": len(files),
    }


def _selected_file(path: Path, sample_id: str) -> Path:
    if path.is_file():
        return path
    selected = (path / sample_id).resolve()
    root = path.resolve()
    if root not in selected.parents:
        raise ValueError("Geometry sample path escapes the configured directory.")
    if not selected.is_file() or selected.suffix.lower() not in GEOMETRY_SUFFIXES:
        raise FileNotFoundError(f"Geometry sample was not found: {sample_id}")
    if selected.name.startswith("._"):
        raise ValueError("AppleDouble metadata files are not geometry.")
    return selected


def _load_trimesh(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    np = _imports()
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to preview STL, PLY, OBJ, and OFF files.") from exc
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except BaseException as exc:
        if path.suffix.lower() in CAD_SUFFIXES:
            raise RuntimeError(
                "STEP/IGES preview needs the gmsh/OpenCASCADE reader. "
                "Install gmsh in the Studio interpreter or ingest the CAD to mesh HDF5 first."
            ) from exc
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    vertices = np.asarray(getattr(loaded, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(loaded, "faces", []), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 3 or not vertices.size:
        raise ValueError(f"{path.name} contains no 3D vertices.")
    if faces.size and (faces.ndim != 2 or faces.shape[1] < 3):
        faces = np.empty((0, 3), dtype=np.int64)
    elif faces.size:
        faces = faces[:, :3]
    metadata = {
        "watertight": bool(getattr(loaded, "is_watertight", False)) if faces.size else None,
        "file_size": path.stat().st_size,
    }
    return vertices[:, :3], faces, metadata


def _load_meshio(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    np = _imports()
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to preview VTK/VTU/MSH geometry.") from exc
    try:
        mesh = meshio.read(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    vertices = np.asarray(mesh.points, dtype=np.float64)
    if vertices.ndim != 2 or not vertices.size:
        raise ValueError(f"{path.name} contains no mesh points.")
    if vertices.shape[1] == 2:
        vertices = np.column_stack([vertices, np.zeros(vertices.shape[0])])
    triangles: list[Any] = []
    for cell in mesh.cells:
        data = np.asarray(cell.data, dtype=np.int64)
        if cell.type in {"triangle", "triangle6"} and data.shape[1] >= 3:
            triangles.append(data[:, :3])
        elif cell.type in {"quad", "quad8", "quad9"} and data.shape[1] >= 4:
            triangles.extend([data[:, [0, 1, 2]], data[:, [0, 2, 3]]])
        elif cell.type in {"tetra", "tetra10"} and data.shape[1] >= 4:
            triangles.extend(
                [
                    data[:, [0, 1, 2]],
                    data[:, [0, 1, 3]],
                    data[:, [0, 2, 3]],
                    data[:, [1, 2, 3]],
                ]
            )
    faces = np.concatenate(triangles, axis=0) if triangles else np.empty((0, 3), dtype=np.int64)
    return vertices[:, :3], faces, {
        "watertight": None,
        "file_size": path.stat().st_size,
        "cell_blocks": [cell.type for cell in mesh.cells],
    }


def _finite_list(values: Any) -> list[float | None]:
    np = _imports()
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value) if np.isfinite(value) else None for value in flat]


def _triangle_payload(vertices: Any, faces: Any, face_limit: int) -> dict[str, Any] | None:
    np = _imports()
    if not faces.size:
        return None
    valid = np.all((faces >= 0) & (faces < vertices.shape[0]), axis=1)
    faces = faces[valid]
    total_faces = int(faces.shape[0])
    display_vertices = vertices
    display_faces = faces
    if total_faces > face_limit:
        try:
            import fast_simplification

            display_vertices, display_faces = fast_simplification.simplify(
                vertices,
                faces,
                target_count=face_limit,
            )
        except (ImportError, RuntimeError, ValueError):
            # Keeping a complete surface is preferable to striding faces, which
            # creates fake holes.  The browser can still handle this bounded
            # fallback for moderately oversized meshes.
            if total_faces > face_limit * 2:
                raise RuntimeError(
                    "This mesh is too dense for a faithful browser preview. "
                    "Install fast-simplification or ingest a decimated surface."
                )
    tri = display_vertices[display_faces]
    return {
        "total_edges": 0,
        "returned_edges": 0,
        "total_faces": total_faces,
        "returned_faces": int(display_faces.shape[0]),
        "triangles": {
            "x1": _finite_list(tri[:, 0, 0]),
            "y1": _finite_list(tri[:, 0, 1]),
            "z1": _finite_list(tri[:, 0, 2]),
            "x2": _finite_list(tri[:, 1, 0]),
            "y2": _finite_list(tri[:, 1, 1]),
            "z2": _finite_list(tri[:, 1, 2]),
            "x3": _finite_list(tri[:, 2, 0]),
            "y3": _finite_list(tri[:, 2, 1]),
            "z3": _finite_list(tri[:, 2, 2]),
        },
    }


def geometry_sample(
    path: Path,
    sample_id: str,
    point_limit: int = 3500,
    face_limit: int = 10000,
) -> dict[str, Any]:
    """Normalize one CAD/mesh file into the same payload used for HDF5."""
    np = _imports()
    selected = _selected_file(path, sample_id)
    if selected.suffix.lower() in MESHIO_SUFFIXES:
        vertices, faces, metadata = _load_meshio(selected)
    else:
        vertices, faces, metadata = _load_trimesh(selected)

    count = int(vertices.shape[0])
    stride = max(1, (count + max(1, point_limit) - 1) // max(1, point_limit))
    indices = np.arange(0, count, stride, dtype=np.int64)
    points = vertices[indices]
    mesh = _triangle_payload(vertices, faces, face_limit)
    sample_name = selected.name if path.is_file() else selected.relative_to(path.resolve()).as_posix()
    metadata.update(
        {
            "format": selected.suffix.lower().lstrip(".").upper(),
            "source_file": relative(selected),
            "total_faces": int(faces.shape[0]),
        }
    )
    return {
        "path": relative(path),
        "source_kind": "geometry",
        "preview_kind": "surface" if mesh else "pointcloud",
        "sample": sample_name,
        "dataset": selected.name,
        "shape": [count, 3],
        "feature": 0,
        "feature_count": 1,
        "feature_name": "surface",
        "timestep": 0,
        "timestep_count": 1,
        "total_points": count,
        "returned_points": int(points.shape[0]),
        "x": _finite_list(points[:, 0]),
        "y": _finite_list(points[:, 1]),
        "z": _finite_list(points[:, 2]),
        "values": [0.0] * int(points.shape[0]),
        "mesh": mesh,
        "stats": {"min": None, "max": None, "mean": None, "std": None},
        "supports": {"points": True, "mesh": bool(mesh), "field": False},
        "metadata": metadata,
    }
