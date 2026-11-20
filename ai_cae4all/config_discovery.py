from __future__ import annotations

from pathlib import Path


_EXCLUDED_PARTS = frozenset({"outputs", ".git", "__pycache__"})


def checked_in_config_paths(suite_root: Path) -> set[Path]:
    """Return the top-level checked-in native configs audited by CLI and Studio."""
    configs_root = Path(suite_root) / "configs"
    if not configs_root.is_dir():
        return set()

    paths: set[Path] = set()
    for path in configs_root.rglob("config*.txt"):
        lowered = {part.lower() for part in path.parts}
        if lowered.intersection(_EXCLUDED_PARTS):
            continue
        paths.add(path.resolve())
    return paths
