"""Path resolution, JSON-safety, and the repository-root-relative file catalog helpers.

Every other studio_backend module imports SUITE_ROOT/RUNTIME_ROOT from here so
there is exactly one definition of "where things live" and one definition of
"what counts as inside the repository."
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

FRONTEND_ROOT = Path(__file__).resolve().parent.parent
SUITE_ROOT = FRONTEND_ROOT.parent
RUNTIME_ROOT = FRONTEND_ROOT / "runtime"
CONFIG_RUNTIME = RUNTIME_ROOT / "configs"
JOB_RUNTIME = RUNTIME_ROOT / "jobs"

MAX_BODY = 2 * 1024 * 1024
MAX_UPLOAD = 8 * 1024 * 1024 * 1024
MAX_TEXT = 1024 * 1024
FILE_LIMIT = 750
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", "runtime"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(SUITE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def slug(value: str, default: str = "studio") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned[:80] or default


def safe_repo_path(raw: str, roots: tuple[Path, ...] | None = None) -> Path:
    path = (SUITE_ROOT / unquote(raw)).resolve()
    allowed = roots or (SUITE_ROOT,)
    if not any(path == root.resolve() or root.resolve() in path.parents for root in allowed):
        raise ValueError("Path is outside the allowed AI-CAE4ALL roots.")
    return path


def file_record(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "kind": kind,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


# Files that are derived caches, not artifacts anyone should pick. The
# MeshGraphNets bi-stride coarsening cache sits beside its dataset as
# `<name>.mscache.<hash>.h5`, so it appeared in every dataset picker as if it
# were a real dataset -- selecting it as ground truth yields an empty contract
# and four errors, which is the only thing it can ever do.
DERIVED_FILE_PATTERNS = (re.compile(r"\.mscache\.", re.IGNORECASE),)


def walk_files(
    roots: tuple[Path, ...],
    suffixes: set[str],
    kind: str,
    exclude: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Catalog matching files under `roots`, newest first, capped at FILE_LIMIT.

    The cap is applied *after* ranking by mtime, never during the walk. Cutting
    the walk short instead meant the first directory os.walk happened to reach
    won `FILE_LIMIT` outright: `studio/runtime/configs/` accumulates one .txt
    per config save and per preflight (2 000+ of them in a working session), so
    every artifact picker in the GUI ended up showing nothing but the Studio's
    own scratch configs -- no dataset, no CSV, no report from `output/` could
    ever appear, however recent.
    """
    records: list[dict[str, Any]] = []
    visited = 0
    excluded = {path.resolve() for path in exclude}
    for root in roots:
        if not root.exists():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            if excluded and Path(directory).resolve() in excluded:
                dirs[:] = []
                continue
            for name in files:
                visited += 1
                path = Path(directory) / name
                if suffixes and path.suffix.lower() not in suffixes:
                    continue
                if any(pattern.search(name) for pattern in DERIVED_FILE_PATTERNS):
                    continue
                try:
                    records.append(file_record(path, kind))
                except OSError:
                    continue
    records.sort(key=lambda item: item["modified"], reverse=True)
    matched = len(records)
    truncated = matched > FILE_LIMIT
    return {
        "items": records[:FILE_LIMIT],
        "truncated": truncated,
        "visited": visited,
        "matched": matched,
        "limit": FILE_LIMIT,
    }


def result_roots() -> tuple[Path, ...]:
    """Every directory a run may legitimately have written results into.

    Analysis endpoints used to allow only `dataset/`, `output/`, `outputs/` and
    the runtime dir. Native inference writes to `<MethodRepo>/outputs/...`, which
    is under none of them, so evaluating a real inference run failed with "Path
    is outside the allowed AI-CAE4ALL roots" no matter what the user selected.

    Kept as an explicit allowlist of data locations rather than opening the whole
    suite root: these endpoints read arbitrary caller-supplied paths, and there
    is no reason for them to reach source code or configs.
    """
    roots = [
        SUITE_ROOT / "dataset",
        SUITE_ROOT / "output",
        SUITE_ROOT / "outputs",
        RUNTIME_ROOT,
    ]
    for item in SUITE_ROOT.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS:
            roots.extend([item / "outputs", item / "output"])
    return tuple(root for root in roots if root.is_dir())
