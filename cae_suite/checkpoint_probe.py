#!/usr/bin/env python3
"""Safely inspect basic PyTorch checkpoint metadata with weights_only=True.

Where torch's weights_only loader cannot open a checkpoint that holds numpy
arrays (torch < 2.5), a restricted metadata-only unpickler reads it instead;
neither path ever unpickles with weights_only=False.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
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


class _TensorStub:
    """Stands in for a tensor or storage whose bytes the probe never reads."""

    __slots__ = ()

    def __setstate__(self, _state):
        pass

    def __repr__(self):
        return "<tensor>"


_TENSOR = _TensorStub()


def _tensor_stub(*_args, **_kwargs):
    return _TENSOR


class _VersionString(str):
    """TorchVersion's stand-in: a str subclass with a __dict__, like the
    original, so the BUILD step a pickled instance may carry still applies."""


def _device_string(kind, index=None):
    return str(kind) if index is None else f"{kind}:{index}"


_TORCH_DTYPES = frozenset((
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
    "uint8", "int8", "int16", "int32", "int64", "bool",
))


def _metadata_globals() -> dict:
    """What `_MetadataUnpickler` may resolve, spelled as the pickle spells it:
    plain-data constructors (the numpy array path `_inert_safe_globals` hands
    torch, plus the builtins torch's own weights_only table allows) and stubs
    for torch's tensor rebuilders, so no torch code runs and no tensor bytes
    are read."""
    import collections
    import _codecs

    table = {
        "collections.OrderedDict": collections.OrderedDict,
        "builtins.set": set,
        "builtins.frozenset": frozenset,
        "builtins.bytearray": bytearray,
        "builtins.complex": complex,
        "_codecs.encode": _codecs.encode,
        "torch.torch_version.TorchVersion": _VersionString,
        "torch.Size": tuple,
        "torch.device": _device_string,
        "torch._utils._rebuild_tensor_v2": _tensor_stub,
        "torch._utils._rebuild_tensor": _tensor_stub,
        "torch._utils._rebuild_parameter": _tensor_stub,
        "torch._utils._rebuild_parameter_with_state": _tensor_stub,
        "torch._tensor._rebuild_from_type_v2": _tensor_stub,
    }
    try:
        import numpy as np
    except Exception:
        return table
    # NumPy 2 pickles the array path under numpy._core, NumPy 1 under
    # numpy.core; whichever this interpreter has answers for both, so a
    # checkpoint written under one still reads under the other.
    reconstruct = {}
    for module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            module = __import__(module_name, fromlist=["_"])
        except Exception:
            continue
        for attribute in ("_reconstruct", "scalar"):
            symbol = getattr(module, attribute, None)
            if symbol is not None:
                reconstruct.setdefault(attribute, symbol)
    for module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        for attribute, symbol in reconstruct.items():
            table[f"{module_name}.{attribute}"] = symbol
    table["numpy.ndarray"] = np.ndarray
    table["numpy.dtype"] = np.dtype
    for name in dir(getattr(np, "dtypes", object)):
        if name.endswith("DType"):
            symbol = getattr(np.dtypes, name, None)
            if isinstance(symbol, type):
                table[f"numpy.dtypes.{name}"] = symbol
    return table


class _MetadataUnpickler(pickle.Unpickler):
    """Restricted unpickler for checkpoints that torch < 2.5 cannot open safely.

    torch 2.4's weights_only unpickler refuses the BUILD step that rebuilds a
    numpy array or dtype, and its allowlist cannot change that, so every
    checkpoint this suite writes (normalization statistics are numpy arrays)
    failed inspection there. This reader follows the Python docs' "restricting
    globals" pattern: `find_class` resolves only the table above and refuses
    everything else, tensor rebuilders are stubs, and storages come back from
    `persistent_load` as the same stub, so the tensor data is never read.
    """

    def __init__(self, handle):
        # torch.load's own default, which matters only for Python 2 pickles.
        super().__init__(handle, encoding="utf-8")
        self._allowed = _metadata_globals()

    def find_class(self, module, name):
        key = f"{module}.{name}"
        if key in self._allowed:
            return self._allowed[key]
        if module == "torch" and name.endswith("Storage"):
            return _TensorStub
        if module == "torch" and name in _TORCH_DTYPES:
            return key
        raise pickle.UnpicklingError(f"metadata reader refuses global {key}")

    def persistent_load(self, pid):
        return _TENSOR


def _load_metadata_only(path: Path):
    import zipfile

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise pickle.UnpicklingError(
            "metadata reader handles only zip-format checkpoints (torch >= 1.6)"
        ) from exc
    with archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if not names:
            raise pickle.UnpicklingError("zip checkpoint has no data.pkl")
        with archive.open(min(names, key=lambda name: name.count("/"))) as handle:
            return _MetadataUnpickler(handle).load()


def _load(torch, path: Path):
    allowed = _inert_safe_globals()
    serialization = torch.serialization
    if hasattr(serialization, "safe_globals"):
        if not allowed:
            return torch.load(path, map_location="cpu", weights_only=True)
        with serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    # torch < 2.5: no scoped allowlist, and no allowlist at all lets its
    # weights_only unpickler rebuild a numpy array. Its own loader still goes
    # first (it opens a pure-tensor checkpoint); the restricted metadata reader
    # takes over only when it refuses.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as refused:
        try:
            return _load_metadata_only(path)
        except Exception as exc:
            raise pickle.UnpicklingError(
                f"torch {getattr(torch, '__version__', '?')} weights_only refused the checkpoint "
                f"and the metadata reader failed too: {exc}"
            ) from refused


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
