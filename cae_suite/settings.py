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
    default_python: str | None = None
    model_pythons: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, suite_root: Path) -> "LocalSettings":
        path = suite_root / "cae_suite.local.toml"
        if not path.exists():
            return cls()
        if tomllib is None:
            raise RuntimeError("cae_suite.local.toml requires tomli on Python 3.10")
        with path.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
        python_data = data.get("python", {})
        if not isinstance(python_data, dict):
            return cls()
        default = python_data.get("default")
        models = python_data.get("models", {})
        return cls(
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
        return expanded.resolve()
