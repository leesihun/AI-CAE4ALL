"""The one place the Studio backend imports cae_suite.

Every launcher-derived fact (registry, specs, preflight, diagnostics,
interpreter settings) flows through this module so the rest of the backend
never needs to know cae_suite's import path or handle its absence twice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from studio_backend.paths import RUNTIME_ROOT, SKIP_DIRS, SUITE_ROOT, safe_repo_path, walk_files

if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

try:
    from cae_suite.config_parser import parse_config
    from cae_suite.diagnostics import Severity
    from cae_suite.preflight import PreflightOptions, run_preflight
    from cae_suite.registry import MethodRegistry
    from cae_suite.settings import LocalSettings
except Exception as import_error:  # pragma: no cover - exposed through /api/health
    SUITE_IMPORT_ERROR: Exception | None = import_error
    parse_config = None  # type: ignore[assignment]
    Severity = None  # type: ignore[assignment]
    PreflightOptions = None  # type: ignore[assignment]
    run_preflight = None  # type: ignore[assignment]
    MethodRegistry = None  # type: ignore[assignment]
    LocalSettings = None  # type: ignore[assignment]
else:
    SUITE_IMPORT_ERROR = None

__all__ = [
    "parse_config",
    "Severity",
    "PreflightOptions",
    "run_preflight",
    "MethodRegistry",
    "LocalSettings",
    "SUITE_IMPORT_ERROR",
    "config_catalog",
    "documentation_catalog",
    "file_catalog",
]


def config_catalog(model_filter: str = "") -> dict[str, Any]:
    roots = (SUITE_ROOT / "configs",)
    result = walk_files(roots, {".txt"}, "config")
    items: list[dict[str, Any]] = []
    for record in result["items"]:
        path = safe_repo_path(record["path"], roots)
        try:
            parsed = parse_config(path) if SUITE_IMPORT_ERROR is None else None
            model = str(parsed.values.get("model", "")) if parsed else ""
            mode = str(parsed.values.get("mode", "")) if parsed else ""
        except Exception:
            model, mode = "", ""
        if model_filter and model.lower() != model_filter.lower():
            continue
        items.append({**record, "model": model, "mode": mode})
    return {**result, "items": items}


def documentation_catalog() -> dict[str, Any]:
    result = walk_files((SUITE_ROOT,), {".md"}, "document")
    items = [
        item
        for item in result["items"]
        if not any(part in SKIP_DIRS for part in Path(item["path"]).parts)
    ]
    return {**result, "items": items}


def file_catalog(kind: str) -> dict[str, Any]:
    if kind == "geometry":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "dataset", SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
            {".stl", ".step", ".stp", ".iges", ".igs", ".brep", ".obj", ".ply", ".off", ".msh", ".vtk", ".vtu", ".vtp"},
            kind,
        )
    if kind == "dataset":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "dataset"),
            {".h5", ".hdf5", ".csv", ".json", ".stl", ".step", ".stp", ".iges", ".igs", ".brep", ".obj", ".ply", ".off", ".vtk", ".vtu", ".vtp", ".msh"},
            kind,
        )
    if kind == "checkpoint":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
            {".pth", ".pt", ".ckpt"},
            kind,
        )
    return walk_files(
        (RUNTIME_ROOT, SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
        {".h5", ".hdf5", ".csv", ".json", ".html", ".png", ".jpg", ".jpeg", ".stl", ".ply", ".obj", ".off", ".vtk", ".vtu", ".vtp", ".msh", ".pth", ".pt", ".txt", ".log"},
        "artifact",
    )
