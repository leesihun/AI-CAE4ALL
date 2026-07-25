"""Single dispatch surface for every visualizable Studio artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio_backend.geometry_preview import (
    GEOMETRY_SUFFIXES,
    geometry_sample,
    geometry_samples,
)
from studio_backend.hdf5_preview import hdf5_sample, hdf5_samples


HDF5_SUFFIXES = {".h5", ".hdf5"}


def _kind(path: Path) -> str:
    if path.is_dir():
        return "geometry"
    suffix = path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return "hdf5"
    if suffix in GEOMETRY_SUFFIXES:
        return "geometry"
    raise ValueError(
        "The shared viewer supports HDF5 plus STL, PLY, OBJ, OFF, STEP, IGES, "
        "VTK, VTU, VTP, and MSH geometry."
    )


def artifact_samples(path: Path, limit: int = 100) -> dict[str, Any]:
    """Catalog samples through the single viewer entrypoint."""
    if not path.exists():
        raise FileNotFoundError(f"Preview path does not exist: {path}")
    if _kind(path) == "hdf5":
        return hdf5_samples(path, limit=limit)
    return geometry_samples(path, limit=limit)


def artifact_sample(
    path: Path,
    sample_id: str,
    feature: int,
    timestep: int,
) -> dict[str, Any]:
    """Read one sample through the single viewer entrypoint."""
    if not path.exists():
        raise FileNotFoundError(f"Preview path does not exist: {path}")
    if _kind(path) == "hdf5":
        return hdf5_sample(path, sample_id, feature, timestep)
    return geometry_sample(path, sample_id)
