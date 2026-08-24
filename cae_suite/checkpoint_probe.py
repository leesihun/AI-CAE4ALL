#!/usr/bin/env python3
"""Safely inspect basic PyTorch checkpoint metadata with weights_only=True."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _simple(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _describe(value):
    """A config value rendered the way the flat `key value` format writes it.

    `_simple` returns None for anything non-scalar, which is right for the
    identity fields but wrong for `model_config`: half of what a caller wants
    from it (`mp_per_level`, `voronoi_clusters`, `fno_modes`) is a list, and the
    native parsers read those back from a comma-separated line. Rendering them
    that way here means a config rebuilt from a checkpoint round-trips.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        parts = [_describe(item) for item in value]
        if any(part is None for part in parts):
            return None
        return ", ".join(str(part) for part in parts)
    # numpy scalars and 0-d arrays; anything else is not a config value.
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", 0) == 0:
        try:
            return _simple(item())
        except Exception:
            return None
    return None


def _inert_safe_globals():
    """Allowlist the plain-data symbols a real checkpoint unpickles to.

    Every checkpoint this suite writes stores normalization statistics as numpy
    arrays, so `weights_only=True` refuses to load it and the probe reported
    "safe metadata inspection was unavailable" for literally every real
    checkpoint -- the checks downstream of it (model family, stage, presence of
    normalization) had therefore never once run. What is allowed here is the
    array-reconstruction path plus torch's version string: data constructors
    only, so no callable carried by the checkpoint is ever invoked, which is the
    property `weights_only` exists to guarantee.
    """
    allowed = []
    try:
        from torch.torch_version import TorchVersion
    except Exception:
        pass
    else:
        allowed.append(TorchVersion)
    try:
        import numpy as np
    except Exception:
        return allowed
    for module_name, attribute in (
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "scalar"),
    ):
        try:
            module = __import__(module_name, fromlist=["_"])
        except Exception:
            continue
        symbol = getattr(module, attribute, None)
        if symbol is not None and symbol not in allowed:
            allowed.append(symbol)
    allowed.extend(item for item in (np.ndarray, np.dtype) if item not in allowed)
    # NumPy 2 moved the concrete dtypes into np.dtypes and pickles them by class.
    for name in dir(getattr(np, "dtypes", object)):
        if name.endswith("DType"):
            symbol = getattr(np.dtypes, name, None)
            if isinstance(symbol, type) and symbol not in allowed:
                allowed.append(symbol)
    return allowed


def _export(mapping) -> dict:
    """A config-shaped dict of everything in `mapping` that is a config value."""
    exported = {}
    for key, value in (mapping or {}).items():
        described = _describe(value)
        if described is not None:
            exported[str(key)] = described
    return exported


def _load(torch, path: Path):
    allowed = _inert_safe_globals()
    if not allowed:
        return torch.load(path, map_location="cpu", weights_only=True)
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({"ok": False, "error": "usage: checkpoint_probe.py <checkpoint>"}))
        return 2
    path = Path(argv[0])
    try:
        import torch
        checkpoint = _load(torch, path)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoint root is {type(checkpoint).__name__}, expected dict")
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            model_config = checkpoint.get("config")
        if not isinstance(model_config, dict):
            model_config = {}
        exported = _export(model_config)
        # The operator repo splits the contract in two: `model_config` holds the
        # architecture, `data_config` (a DataSpec) holds input_var/output_var and
        # the timestep count -- and its loader applies data_config *after* the
        # model_config overlay, so it is the authority on those. Exporting only
        # model_config left a rebuilt operator config short of two required keys.
        data_config = checkpoint.get("data_config")
        exported_data = _export(data_config) if isinstance(data_config, dict) else {}
        result = {
            "ok": True,
            "top_keys": sorted(str(key) for key in checkpoint.keys()),
            "stage": _simple(checkpoint.get("stage")),
            "selected_model": _simple(checkpoint.get("selected_model")),
            "schema_version": _simple(checkpoint.get("schema_version")),
            "checkpoint_version": _simple(checkpoint.get("checkpoint_version")),
            "model_config_model": _simple(model_config.get("model")),
            "has_model_config": bool(model_config),
            # The architecture the weights were actually fit under. Every native
            # inference path overrides the config file with this, so it is also
            # the only honest way to build a config for a checkpoint whose
            # training config is not on the canvas.
            "model_config": exported,
            "data_config": exported_data,
            "epoch": _simple(checkpoint.get("epoch")),
            "valid_loss": _simple(checkpoint.get("valid_loss")),
            "has_normalization": isinstance(checkpoint.get("normalization"), dict),
            "has_ema": "ema_state_dict" in checkpoint or "ema_state" in checkpoint,
            "has_conditional_prior": (
                "conditional_prior_state_dict" in checkpoint
                or any(str(key).startswith("conditional_prior") for key in checkpoint.keys())
                or bool(model_config.get("use_conditional_prior", False))
            ),
            "linked_vae": _simple(checkpoint.get("vae_modelpath")),
        }
        print(json.dumps(result))
        return 0
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else "unknown checkpoint load error"
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {message}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
