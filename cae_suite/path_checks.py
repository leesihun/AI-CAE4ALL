from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config_parser import ParsedConfig
from .diagnostics import DiagnosticReport, Severity
from .specs import MethodSpec, PathKind


_PATH_SENTINELS = {"", "none", "null", "false"}


def _absolute_native_path(value: str, repository_root: Path) -> Path:
    """Absolutize a native path value, keeping the case the config spelled.

    ``normpath`` collapses the ``../`` segments every native config uses, but
    lexically — unlike ``resolve()`` it never rewrites a component to its
    on-disk case, which is exactly what `_case_mismatch` needs to compare.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return Path(os.path.normpath(path))


def resolve_native_path(value: str, repository_root: Path) -> Path:
    return _absolute_native_path(value, repository_root).resolve(strict=False)


def _case_mismatch(literal: Path, resolved: Path) -> bool:
    """True when the config spells an existing path with the wrong case.

    ``Path.resolve()`` reports a file's real on-disk name on Windows, so a
    literal that differs from it *only* by case names a file the local
    case-insensitive filesystem happily opens but Linux would not find.
    Anything still differing by more than case (a traversed symlink) is not a
    case problem and is ignored.
    """
    literal_str, resolved_str = str(literal), str(resolved)
    return literal_str != resolved_str and literal_str.lower() == resolved_str.lower()


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

        literal = _absolute_native_path(value, repository_root)
        if path.exists() and _case_mismatch(literal, path):
            report.add(
                "PATH-CASE-001",
                Severity.WARNING,
                f"Path case does not match the filesystem: config says {literal}, on disk it is {path}",
                field_name=rule.field,
                location=parsed.location(rule.field),
                hint="This resolves here only because the local filesystem is case-insensitive; "
                "it would fail on Linux. Correct the spelling in the config.",
                promote_in_strict=True,
            )

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
