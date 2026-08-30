"""The one place the Studio backend imports cae_suite.

Every launcher-derived fact (registry, specs, preflight, diagnostics,
interpreter settings) flows through this module so the rest of the backend
never needs to know cae_suite's import path or handle its absence twice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from studio_backend.paths import CONFIG_RUNTIME, RUNTIME_ROOT, SKIP_DIRS, SUITE_ROOT, relative, safe_repo_path, walk_files
from studio_backend.system_info import PORTABLE_INFERENCE_MODELS

if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

try:
    from cae_suite.config_parser import parse_config
    from cae_suite.diagnostics import Severity
    from cae_suite.preflight import PreflightOptions, run_preflight
    from cae_suite.registry import MethodRegistry
    from cae_suite.settings import LocalSettings
except Exception as import_error:  # pragma: no cover - exposed through /api/health
    SUITE_IMPORT_ERROR: Exception | None = import_error
    parse_config = None  # type: ignore[assignment]
    Severity = None  # type: ignore[assignment]
    PreflightOptions = None  # type: ignore[assignment]
    run_preflight = None  # type: ignore[assignment]
    MethodRegistry = None  # type: ignore[assignment]
    LocalSettings = None  # type: ignore[assignment]
else:
    SUITE_IMPORT_ERROR = None

__all__ = [
    "parse_config",
    "Severity",
    "PreflightOptions",
    "run_preflight",
    "MethodRegistry",
    "LocalSettings",
    "SUITE_IMPORT_ERROR",
    "config_catalog",
    "documentation_catalog",
    "file_catalog",
    "benchmark_roster",
    "checkpoint_metadata",
    "STANDALONE_INFERENCE_MODELS",
    "PORTABLE_INFERENCE_MODELS",
]


def config_catalog(model_filter: str = "") -> dict[str, Any]:
    roots = (SUITE_ROOT / "configs",)
    result = walk_files(roots, {".txt"}, "config")
    items: list[dict[str, Any]] = []
    for record in result["items"]:
        path = safe_repo_path(record["path"], roots)
        try:
            parsed = parse_config(path) if SUITE_IMPORT_ERROR is None else None
            model = str(parsed.values.get("model", "")) if parsed else ""
            mode = str(parsed.values.get("mode", "")) if parsed else ""
        except Exception:
            model, mode = "", ""
        if model_filter and model.lower() != model_filter.lower():
            continue
        items.append({**record, "model": model, "mode": mode})
    return {**result, "items": items}


def documentation_catalog() -> dict[str, Any]:
    result = walk_files((SUITE_ROOT,), {".md"}, "document")
    items = [
        item
        for item in result["items"]
        if not any(part in SKIP_DIRS for part in Path(item["path"]).parts)
    ]
    return {**result, "items": items}


def file_catalog(kind: str) -> dict[str, Any]:
    if kind == "geometry":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "dataset", SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
            {".stl", ".step", ".stp", ".iges", ".igs", ".brep", ".obj", ".ply", ".off", ".msh", ".vtk", ".vtu", ".vtp"},
            kind,
        )
    if kind == "dataset":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "dataset"),
            {".h5", ".hdf5", ".csv", ".json", ".stl", ".step", ".stp", ".iges", ".igs", ".brep", ".obj", ".ply", ".off", ".vtk", ".vtu", ".vtp", ".msh"},
            kind,
        )
    if kind == "checkpoint":
        return walk_files(
            (RUNTIME_ROOT, SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
            {".pth", ".pt", ".ckpt"},
            kind,
        )
    return walk_files(
        (RUNTIME_ROOT, SUITE_ROOT / "output", SUITE_ROOT / "outputs"),
        {".h5", ".hdf5", ".csv", ".json", ".html", ".png", ".jpg", ".jpeg", ".stl", ".ply", ".obj", ".off", ".vtk", ".vtu", ".vtp", ".msh", ".pth", ".pt", ".txt", ".log"},
        "artifact",
        # Every config save and every preflight writes a .txt here, so this one
        # directory outnumbers all real artifacts by an order of magnitude and,
        # being always the newest, would sit at the top of every picker. It is
        # Studio scratch, not a user artifact.
        exclude=(CONFIG_RUNTIME,),
    )


def benchmark_roster() -> dict[str, Any]:
    """The checked-in benchmark campaign, read from its roster.

    The Benchmarks workspace used to look for configs whose path contained
    "benchmarks/" and reported "0 real configs", which was literally true and
    completely misleading: the campaign was reorganized into
    `configs/benchmarks_all/`, whose roster.tsv names arms that live under each
    method's own `configs/<Method>/<ex>/`. Nothing under a "benchmarks/" path
    remains, so the workspace showed an empty page for a 25-arm campaign.

    Returns roster order (which is the campaign's own grouping by ex slot),
    flagging arms whose config is missing rather than dropping them -- a roster
    pointing at a deleted config is exactly what someone needs to see.
    """
    roster = SUITE_ROOT / "configs" / "benchmarks_all" / "roster.tsv"
    if not roster.is_file():
        return {"items": [], "roster": None}
    rows: list[dict[str, Any]] = []
    lines = roster.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t") if lines else []
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        entry = dict(zip(header, cells))
        config = str(entry.get("train_config", "")).strip()
        path = SUITE_ROOT / config if config else None
        rows.append(
            {
                "label": str(entry.get("label", "")).strip(),
                "path": config,
                "ex_slot": str(entry.get("ex_slot", "")).strip(),
                "light": str(entry.get("light", "")).strip() in {"1", "true", "True"},
                "exists": bool(path and path.is_file()),
                "size": path.stat().st_size if path and path.is_file() else 0,
            }
        )
    return {"items": rows, "roster": relative(roster)}


# The checkpoint families the Studio can build a standalone inference config for.
# SimulGen-VAE and SDFFlow are excluded on purpose: their non-training modes need
# a dozen architecture keys (latent_dim_end, lc_filter, num_filter_enc, ...) that
# their checkpoints do not carry, so those still require the model block.
STANDALONE_INFERENCE_MODELS = (
    "meshgraphnets",
    "meshgraphnets-v",
    "chi-mgnflow",
    "transolver",
    "fno",
    "gino",
    "deeponet",
    "point_deeponet",
    "mlp",
)

CHECKPOINT_PROBE_TIMEOUT = 120

# Repository directory name -> model id, for checkpoints whose own metadata does
# not name the family. Every method writes under its own repo or under
# output/<model>/, so the path is a reliable second source.
_REPOSITORY_MODEL_HINTS = {
    "meshgraphnets": "meshgraphnets",
    "meshgraphnets - variational": "meshgraphnets-v",
    "meshgraphnets-v": "meshgraphnets-v",
    "hi_meshgraphnets": "meshgraphnets",
    "chi-mgnflow": "chi-mgnflow",
    "chi_mgnflow": "chi-mgnflow",
    "geometry_generation": "sdfflow",
    "simulgenvae": "simulgenvae",
    "transolver": "transolver",
    "mlp": "mlp",
    "fno": "fno",
    "gino": "gino",
    "deeponet": "deeponet",
    "point_deeponet": "point_deeponet",
}


def _model_from_path(path: Path) -> str:
    """Guess the model family from where the checkpoint sits."""
    parts = [part.lower() for part in path.parts]
    for part in reversed(parts):
        hint = _REPOSITORY_MODEL_HINTS.get(part)
        if hint:
            return hint
    return ""


def _model_from_architecture(model_config: dict[str, Any]) -> str:
    """Guess the model family from architecture keys only that family writes."""
    keys = set(model_config)
    # cHI-MGNflow shares the whole HI-MGN backbone with MeshGraphNets, so its
    # flow keys must be checked first or an old checkpoint that predates the
    # explicit `model` identity is silently classified as deterministic MGN.
    if {"flow_time_freqs", "flow_t_sampling", "flow_solver"} & keys:
        return "chi-mgnflow"
    if {"slice_num", "attention_kernel"} & keys:
        return "transolver"
    if {"fno_modes", "fno_hidden_channels"} & keys:
        return "fno"
    if {"gino_in_radius", "gino_out_radius", "gino_fno_modes"} & keys:
        return "gino"
    if {"point_sensor_count", "point_feature_dim"} & keys:
        return "point_deeponet"
    if {"deeponet_basis_dim", "deeponet_sensor_resolution"} & keys:
        return "deeponet"
    if {"hidden_layers", "output_activation"} & keys:
        return "mlp"
    if {"edge_var", "message_passing_num"} & keys:
        # The variational tree adds its own keys on top of the vanilla set.
        return "meshgraphnets-v" if {"vae_latent_dim", "num_vae_samples", "prior_type"} & keys else "meshgraphnets"
    return ""


def checkpoint_metadata(path: Path, registry: Any, settings: Any) -> dict[str, Any]:
    """Read a .pth's recorded architecture through the launcher's own probe.

    This is what makes an Inference block runnable without the model block that
    trained it on the canvas: every native inference path overrides the config
    file with the checkpoint's `model_config`, so the checkpoint -- not a config
    the user no longer has -- is the honest source for `input_var`, `edge_var`
    and friends.

    The probe needs torch, so it runs under a method interpreter. Which method
    is what we are trying to find out, so the family is resolved from the path
    first and the probe output is allowed to correct it afterwards.
    """
    import subprocess

    if SUITE_IMPORT_ERROR is not None:
        return {"ok": False, "error": f"cae_suite is unavailable: {SUITE_IMPORT_ERROR}"}
    if not path.is_file():
        return {"ok": False, "error": f"No such checkpoint: {relative(path)}"}

    hinted = _model_from_path(path)
    python = Path(sys.executable)
    repository = SUITE_ROOT
    if hinted and registry is not None and settings is not None:
        try:
            resolved = registry.resolve(hinted, type(registry.validate_installations())())
        except Exception:
            resolved = None
        if resolved is not None:
            python = settings.resolve_python(resolved.model_id, resolved.spec.spec_id)
            repository = resolved.repository_root

    script = SUITE_ROOT / "cae_suite" / "checkpoint_probe.py"
    try:
        completed = subprocess.run(
            [str(python), str(script), str(path)],
            cwd=str(repository),
            capture_output=True,
            text=True,
            timeout=CHECKPOINT_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"Checkpoint probe could not run: {exc}"}

    import json as _json

    try:
        payload = _json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no output"
        return {"ok": False, "error": f"Checkpoint probe returned no result ({detail})"}
    if not payload.get("ok", False):
        return {"ok": False, "error": str(payload.get("error", "unknown probe error")), "path": relative(path)}

    model_config = payload.get("model_config") or {}
    schema_version = str(payload.get("schema_version") or "").strip().lower()
    model_id = str(
        payload.get("selected_model")
        or payload.get("model_config_model")
        or ("sdfflow" if schema_version == "sdfflow_infer_v1" else "")
        or _model_from_architecture(model_config)
        or hinted
        or ""
    ).strip().lower()
    known = set(registry.model_ids) if registry is not None else set()
    if known and model_id not in known:
        model_id = ""
    return {
        "ok": True,
        "path": relative(path),
        "model": model_id,
        "model_source": (
            "checkpoint metadata" if payload.get("selected_model") or payload.get("model_config_model")
            else "checkpoint schema" if schema_version == "sdfflow_infer_v1"
            else "architecture keys" if _model_from_architecture(model_config)
            else "checkpoint location" if hinted
            else ""
        ),
        "standalone_inference": model_id in STANDALONE_INFERENCE_MODELS,
        "portable_inference": model_id in PORTABLE_INFERENCE_MODELS,
        "schema_version": payload.get("schema_version"),
        "model_config": model_config,
        "data_config": payload.get("data_config") or {},
        "epoch": payload.get("epoch"),
        "valid_loss": payload.get("valid_loss"),
        "has_normalization": payload.get("has_normalization", False),
        "has_ema": payload.get("has_ema", False),
        "stage": payload.get("stage"),
    }
