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


def walk_files(roots: tuple[Path, ...], suffixes: set[str], kind: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    visited = 0
    for root in roots:
        if not root.exists():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            for name in files:
                visited += 1
                path = Path(directory) / name
                if suffixes and path.suffix.lower() not in suffixes:
                    continue
                try:
                    records.append(file_record(path, kind))
                except OSError:
                    continue
                if len(records) >= FILE_LIMIT:
                    records.sort(key=lambda item: item["modified"], reverse=True)
                    return {"items": records, "truncated": True, "visited": visited}
    records.sort(key=lambda item: item["modified"], reverse=True)
    return {"items": records, "truncated": False, "visited": visited}
