from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config_parser import ParsedConfig
from .diagnostics import DiagnosticReport, Severity
from .specs import MethodSpec, PathKind


_PATH_SENTINELS = {"", "none", "null", "false"}


def resolve_native_path(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve(strict=False)


def _nearest_existing(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def validate_paths(
    parsed: ParsedConfig,
    spec: MethodSpec,
    mode: str | None,
    repository_root: Path,
    report: DiagnosticReport,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for rule in spec.path_rules:
        if not rule.active(mode) or rule.field not in parsed.values:
            continue
        value: Any = parsed.values[rule.field]
        if not isinstance(value, str):
            report.add(
                "PATH-TYPE-001",
                Severity.ERROR,
                f"{rule.field} must be one path string; got {value!r}.",
                field_name=rule.field,
                location=parsed.location(rule.field),
                hint="Paths containing spaces are not supported by the current native parser.",
            )
            continue
        if value.lower() in _PATH_SENTINELS:
            continue
        path = resolve_native_path(value, repository_root)
        resolved[rule.field] = path

        raw_value = parsed.raw_values.get(rule.field, value)
        raw_path = resolve_native_path(raw_value, repository_root)
        if raw_path.exists() and not path.exists() and raw_path != path:
            report.add(
                "PATH-CASE-001",
                Severity.ERROR,
                f"The original path exists, but the lowercased path seen by the native parser does not: {path}",
                field_name=rule.field,
                location=parsed.location(rule.field),
                hint="Use a case-insensitive location or update the native parser to preserve path case.",
            )
            continue

        if rule.kind in {PathKind.INPUT_FILE, PathKind.INPUT_DIR}:
            expected = "file" if rule.kind is PathKind.INPUT_FILE else "directory"
            valid = path.is_file() if rule.kind is PathKind.INPUT_FILE else path.is_dir()
            if not valid:
                report.add(
                    "PATH-INPUT-001",
                    Severity.ERROR,
                    f"Required input {expected} does not exist: {path}",
                    field_name=rule.field,
                    location=parsed.location(rule.field),
                )
            elif not os.access(path, os.R_OK):
                report.add(
                    "PATH-INPUT-002",
                    Severity.ERROR,
                    f"Required input is not readable: {path}",
                    field_name=rule.field,
                    location=parsed.location(rule.field),
                )
        else:
            target_dir = path if rule.kind is PathKind.OUTPUT_DIR else path.parent
            existing = _nearest_existing(target_dir)
            if existing is None or not existing.is_dir() or not os.access(existing, os.W_OK):
                report.add(
                    "PATH-OUTPUT-001",
                    Severity.ERROR,
                    f"Output location cannot be created or is not writable: {target_dir}",
                    field_name=rule.field,
                    location=parsed.location(rule.field),
                )
            elif path.exists() and rule.kind is PathKind.OUTPUT_FILE:
                report.add(
                    "PATH-OUTPUT-EXISTS",
                    Severity.WARNING,
                    f"Output file already exists and the native run may overwrite it: {path}",
                    field_name=rule.field,
                    location=parsed.location(rule.field),
                )

    if spec.spec_id in {"meshgraphnets", "meshgraphnets_variational"} and mode == "train" and "dataset_dir" in resolved:
        dataset = resolved["dataset_dir"]
        if dataset.exists() and not os.access(dataset, os.W_OK):
            report.add(
                "PATH-MUTATE-002",
                Severity.ERROR,
                "The selected MeshGraphNets training path may write preprocessing statistics, but the HDF5 file is read-only.",
                field_name="dataset_dir",
                location=parsed.location("dataset_dir"),
            )
        else:
            report.add(
                "PATH-MUTATE-001",
                Severity.WARNING,
                "MeshGraphNets training may write preprocessing statistics into the source HDF5 file.",
                field_name="dataset_dir",
                location=parsed.location("dataset_dir"),
            )
    return resolved
