"""Run the repository's native DeepJEB surface-graph and HI-MGN pipeline.

The graph builder is shared with the DeepJEB training data. Prediction runs
the unchanged MeshGraphNets rollout in a separate process: SDFFlow and
MeshGraphNets both use a package named ``general_modules``. Only the browser
preview can be reduced; native inference always uses the complete graph.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from studio_backend.mesh_topology import aggregate_values, build_topology


CHECKPOINT_RELATIVE = Path("output/dataset_matrix/deterministic/ex13/himgn/model.pth")

# The checkpoint owns architecture and normalization. These are the serving
# paths and one-step static contract, not another copy of the architecture.
NATIVE_ENTRY = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from inference_profiles.rollout import run_rollout
config_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding='utf-8'))
run_rollout(config, str(config_path))
"""


def _relative(path: Path, suite_root: Path) -> str:
    try:
        return path.resolve().relative_to(suite_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _load_bridge(suite_root: Path):
    method_root = str(suite_root / "methods" / "SDFFlow")
    if method_root not in sys.path:
        sys.path.insert(0, method_root)
    return importlib.import_module("design_loop.deepjeb_bridge")


def _preview(vertices, edges, *, name: str, source: str, values=None, node_types=None) -> dict[str, Any]:
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    topology = build_topology(vertices, edges, node_budget=5000, edge_limit=50000, face_limit=30000)
    payload: dict[str, Any] = {
        "name": name,
        "vertices": topology["coordinates"].tolist(),
        "faces": topology["faces"].tolist(),
        "edges": topology["edges"].tolist(),
        "unit": "mm",
        "coordinate_unit": "mm",
        "original_nodes": int(len(vertices)),
        "preview_nodes": int(topology["node_count"]),
        "provenance": {"source": source, "preview_reduced": bool(topology["reduced"]),
                       "topology": "Studio surface triangles reconstructed from native graph edges"},
    }
    if node_types is not None:
        inverse = topology["inverse"]
        types = np.asarray(node_types, dtype=np.int64)
        if inverse is not None:
            # Preserve boundary labels in a display cluster. The complete
            # native node-type array remains unchanged in mesh_input.h5.
            reduced_types = np.zeros(topology["node_count"], dtype=np.int64)
            np.maximum.at(reduced_types, inverse, types)
            types = reduced_types
        payload["node_types"] = types.tolist()
    if values is not None:
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Native HI-MGN returned a non-finite u_z prediction.")
        payload.update(
            feature="u_z", feature_name="u_z", unit="mm",
            values=aggregate_values(topology["inverse"], topology["node_count"], values).tolist(),
            stats={"min": float(values.min()), "max": float(values.max()),
                   "mean": float(values.mean()), "max_abs": float(np.abs(values).max())},
        )
    return payload


def _read_field(prediction_path: Path, suite_root: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(prediction_path, "r") as handle:
        group = handle["data/1"]
        nodal = group["nodal_data"][:, -1, :]
        names = [item.decode() if isinstance(item, bytes) else str(item)
                 for item in handle["metadata/feature_names"][:]]
        # Select by the native contract's channel name, never by a displayed
        # magnitude or a positional coordinate.
        if "u_z" not in names:
            raise ValueError("Native prediction does not declare a u_z channel.")
        payload = _preview(nodal[:3].T, group["mesh_edge"][:], name="HI-MGN u_z",
                           source=_relative(prediction_path, suite_root),
                           values=nodal[names.index("u_z")], node_types=nodal[-1])
    payload["computed"] = True
    payload["prediction_kind"] = "surrogate"
    return payload


def _run_native(method_dir: Path, config_path: Path, log_path: Path, cancel_event=None) -> int:
    """Keep the child owned by Studio and interrupt it during server shutdown."""
    def terminate(process) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", NATIVE_ENTRY, str(method_dir), str(config_path)],
            cwd=str(method_dir), stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 600
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Geometry analysis was cancelled during server shutdown.")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Native HI-MGN inference exceeded its 600-second time limit.")
                try:
                    return process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            terminate(process)
            raise


def analyze_geometry(
    stl_path: str | Path,
    output_dir: str | Path,
    suite_root: str | Path,
    graph_nodes: int = 5000,
    progress: Callable[..., Any] | None = None,
    cancel_event=None,
) -> dict[str, Any]:
    """Synchronously build a new graph and predict signed vertical displacement.

    The caller owns GPU serialization and asynchronous job state. ``progress``
    is invoked as ``progress(phase, **data)``; the complete preview graph is
    available during the ``inference`` phase. Generated STL coordinates must
    be in SDFFlow's normalized DeepJEB frame. Native bore registration and its
    rejection gates apply unchanged. This function performs no FEA.
    """
    started = time.perf_counter()
    suite_root, stl_path, output_dir = (Path(item).resolve() for item in (suite_root, stl_path, output_dir))
    graph_nodes = int(graph_nodes)
    if not 1500 <= graph_nodes <= 10000:
        raise ValueError("Surface graph node target must be between 1500 and 10000.")
    if not stl_path.is_file():
        raise FileNotFoundError(f"Generated geometry does not exist: {stl_path}")
    checkpoint = suite_root / CHECKPOINT_RELATIVE
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DeepJEB HI-MGN checkpoint is unavailable: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def notify(phase: str, **data: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Geometry analysis was cancelled during server shutdown.")
        if progress:
            progress(phase, elapsed_s=round(time.perf_counter() - started, 3), **data)

    notify("meshing", detail="Registering DeepJEB interfaces and building a native surface graph")
    import trimesh

    bridge = _load_bridge(suite_root)
    tick = time.perf_counter()
    mesh = trimesh.load(str(stl_path), force="mesh", process=True)
    record = bridge.mesh_to_ver_record(mesh, target_nodes=graph_nodes, name=stl_path.stem)
    graph_path = output_dir / "mesh_input.h5"
    bridge.write_inference_contract(str(graph_path), [record], bridge.VER_FEATURE_NAMES,
                                    bridge.VER_INPUT_VAR, bridge.VER_OUTPUT_VAR, bridge.VER_COND_VAR)
    graph = _preview(record["nodal"][:3, 0].T, record["edges"], name="DeepJEB surface graph",
                     source=_relative(graph_path, suite_root), node_types=record["nodal"][-1, 0])
    graph["registration"] = record["frame"]
    graph_preview_path = output_dir / "graph.json"
    _write_json(graph_preview_path, graph)
    timings["mesh_s"] = round(time.perf_counter() - tick, 3)
    notify("inference", graph=graph, mesh=graph, timings=timings.copy(),
           detail="Running native HI-MGN on the newly generated surface graph")

    config_path, log_path = output_dir / "inference_config.json", output_dir / "native_inference.log"
    config = {
        "model": "meshgraphnets", "mode": "inference", "gpu_ids": [0],
        "modelpath": str(checkpoint), "infer_dataset": str(graph_path),
        "inference_output_dir": str(output_dir / "prediction"), "infer_timesteps": 1,
        "input_var": bridge.VER_INPUT_VAR, "output_var": bridge.VER_OUTPUT_VAR,
        "cond_var": bridge.VER_COND_VAR,
    }
    _write_json(config_path, config)
    tick = time.perf_counter()
    method_dir = suite_root / "methods" / "MeshGraphNets"
    returncode = _run_native(method_dir, config_path, log_path, cancel_event)
    if returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        raise RuntimeError(f"Native HI-MGN inference failed; see {_relative(log_path, suite_root)}. "
                           + " ".join(tail))
    timings["inference_s"] = round(time.perf_counter() - tick, 3)
    notify("reading", timings=timings.copy(), detail="Reading signed u_z from native prediction")
    field_path = output_dir / "prediction" / "rollout_sample1_steps1.h5"
    field = _read_field(field_path, suite_root)
    timings["total_s"] = round(time.perf_counter() - started, 3)
    field["checkpoint"] = _relative(checkpoint, suite_root)
    field["timings"] = timings
    field_preview_path = output_dir / "field.json"
    _write_json(field_preview_path, field)
    result = {
        "graph": graph, "field": field, "result": field,
        "stl_path": _relative(stl_path, suite_root), "graph_path": _relative(graph_path, suite_root),
        "field_path": _relative(field_path, suite_root),
        "graph_preview_path": _relative(graph_preview_path, suite_root),
        "field_preview_path": _relative(field_preview_path, suite_root),
        "checkpoint": _relative(checkpoint, suite_root), "timings": timings,
        "provenance": {"geometry": _relative(stl_path, suite_root), "graph_builder": "mesh_to_ver_record",
                       "predictor": "MeshGraphNets native run_rollout / HI-MGN checkpoint",
                       "prediction_kind": "surrogate", "feature": "u_z", "unit": "mm"},
    }
    _write_json(output_dir / "analysis.json", result)
    notify("completed", **result, detail="Fresh native HI-MGN prediction complete")
    return result
