"""Flat ``key value`` config parsing for the launcher entrypoint.

Faithful to the repository's native config style (shared by the ML methods and
mirrored in ``cae_suite/config_parser.py``): ``%`` starts a full-line comment,
``#`` starts an inline comment, keys are lowercased, and a value is the rest of
the line. This is a small self-contained reader so the entrypoint has no
dependency on ``cae_suite`` (it runs in the method environment).
"""

from __future__ import annotations

from .pipeline import IngestParams


def load_config(path: str) -> dict[str, str]:
    """Parse a flat config file into ``{lowercased_key: raw_value_string}``."""
    values: dict[str, str] = {}
    # utf-8-sig tolerates (strips) a BOM; the launcher treats a BOM as a hard error,
    # so a config that passes preflight will not have one here anyway.
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            body = stripped.split("#", 1)[0].strip()
            if not body:
                continue
            parts = body.split(None, 1)
            if len(parts) != 2 or not parts[1].strip():
                continue
            values[parts[0].strip().lower()] = parts[1].strip()
    return values


def as_list(value: str) -> list[str]:
    """Split a comma- or space-separated value into a list of tokens."""
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value.split()


def params_from_config(cfg: dict[str, str]) -> IngestParams:
    """Map parsed config values onto IngestParams, applying the same defaults."""
    mesh_type = str(cfg.get("mesh_type", "volume")).lower()
    return IngestParams(
        reader=str(cfg.get("reader", "auto")).lower(),
        volume=(mesh_type == "volume"),
        emit=tuple(e.lower() for e in as_list(cfg.get("emit", "graph"))),
        num_fields=int(cfg.get("num_fields", 3)),
        num_points=int(cfg.get("num_points", 0)),
        resample=str(cfg.get("resample_method", "fps")).lower(),
        mesh_size_max=float(cfg.get("mesh_size_max", 0.0)),
        mesh_size_min=float(cfg.get("mesh_size_min", 0.0)),
        seed=int(cfg.get("seed", 42)),
        limit=int(cfg.get("limit", 0)),
    )
