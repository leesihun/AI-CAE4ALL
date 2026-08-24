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
from studio_backend.prediction_preview import (
    is_prediction_file,
    prediction_sample,
    prediction_samples,
    result_flavor,
    rollout_sample,
    rollout_samples,
)


HDF5_SUFFIXES = {".h5", ".hdf5"}


def _kind(path: Path) -> str:
    # Inference results are HDF5 too, but in the per-sample prediction/truth
    # layout rather than the shared dataset contract, so they are checked
    # before both -- including for directories, where a run's whole output
    # folder is one browsable artifact and would otherwise be read as geometry.
    if path.is_dir():
        flavor = result_flavor(path)
        if flavor == "paired":
            return "prediction"
        if flavor == "rollout":
            return "rollout"
        return "geometry"
    suffix = path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return "prediction" if is_prediction_file(path) else "hdf5"
    if suffix in GEOMETRY_SUFFIXES:
        return "geometry"
    raise ValueError(
        "The shared viewer supports HDF5 plus STL, PLY, OBJ, OFF, STEP, IGES, "
        "VTK, VTU, VTP, and MSH geometry."
    )


def artifact_samples(path: Path, limit: int = 100, truth: Path | None = None) -> dict[str, Any]:
    """Catalog samples through the single viewer entrypoint."""
    if not path.exists():
        raise FileNotFoundError(f"Preview path does not exist: {path}")
    kind = _kind(path)
    if kind == "prediction":
        return prediction_samples(path, limit=limit)
    if kind == "rollout":
        return rollout_samples(path, limit=limit, truth_path=truth)
    if kind == "hdf5":
        return hdf5_samples(path, limit=limit)
    return geometry_samples(path, limit=limit)


def artifact_sample(
    path: Path,
    sample_id: str,
    feature: int,
    timestep: int,
    truth: Path | None = None,
) -> dict[str, Any]:
    """Read one sample through the single viewer entrypoint."""
    if not path.exists():
        raise FileNotFoundError(f"Preview path does not exist: {path}")
    kind = _kind(path)
    if kind == "prediction":
        return prediction_sample(path, sample_id, feature)
    if kind == "rollout":
        return rollout_sample(path, sample_id, feature, timestep, truth_path=truth)
    if kind == "hdf5":
        return hdf5_sample(path, sample_id, feature, timestep)
    return geometry_sample(path, sample_id)
