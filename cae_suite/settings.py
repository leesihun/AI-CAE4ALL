from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


@dataclass
class LocalSettings:
    base_dir: Path | None = None
    default_python: str | None = None
    model_pythons: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, suite_root: Path) -> "LocalSettings":
        base_dir = suite_root.resolve()
        path = base_dir / "ai_cae4all.local.toml"
        if not path.exists():
            legacy_path = base_dir / "cae_suite.local.toml"
            if legacy_path.exists():
                path = legacy_path
        if not path.exists():
            return cls(base_dir=base_dir)
        if tomllib is None:
            raise RuntimeError(f"{path.name} requires tomli on Python 3.10")
        with path.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
        python_data = data.get("python", {})
        if not isinstance(python_data, dict):
            return cls()
        default = python_data.get("default")
        models = python_data.get("models", {})
        return cls(
            base_dir=base_dir,
            default_python=str(default) if default else None,
            model_pythons={str(k).lower(): str(v) for k, v in models.items()} if isinstance(models, dict) else {},
        )

    def resolve_python(self, model_id: str, spec_id: str, override: str | None = None) -> Path:
        selected = (
            override
            or self.model_pythons.get(model_id.lower())
            or self.model_pythons.get(spec_id.lower())
            or self.default_python
            or sys.executable
        )
        expanded = Path(selected).expanduser()
        if not expanded.is_absolute():
            discovered = shutil.which(str(expanded))
            if discovered:
                return Path(discovered).resolve()
            if self.base_dir is not None:
                return (self.base_dir / expanded).resolve()
        return expanded.resolve()
