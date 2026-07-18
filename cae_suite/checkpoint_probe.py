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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({"ok": False, "error": "usage: checkpoint_probe.py <checkpoint>"}))
        return 2
    path = Path(argv[0])
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoint root is {type(checkpoint).__name__}, expected dict")
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            model_config = checkpoint.get("config")
        if not isinstance(model_config, dict):
            model_config = {}
        result = {
            "ok": True,
            "top_keys": sorted(str(key) for key in checkpoint.keys()),
            "stage": _simple(checkpoint.get("stage")),
            "selected_model": _simple(checkpoint.get("selected_model")),
            "schema_version": _simple(checkpoint.get("schema_version")),
            "checkpoint_version": _simple(checkpoint.get("checkpoint_version")),
            "model_config_model": _simple(model_config.get("model")),
            "has_model_config": bool(model_config),
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
