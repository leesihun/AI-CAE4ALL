"""Reader for the native inference-result HDF5 layout.

The method repos deliberately do NOT write inference output in the shared
dataset contract (`data/{sample}/nodal_data`). They write one file per
predicted sample, holding prediction and ground truth side by side:

    nodes/pos                (N, 3)   float32   node coordinates
    nodes/predicted_denorm   (N, C)   float32   prediction in physical units
    nodes/target_denorm      (N, C)   float32   ground truth in physical units
    nodes/predicted_norm     (N, C)   float32   prediction in normalized units
    nodes/target_norm        (N, C)   float32
    nodes/part_ids           (N,)     int32     optional
    faces/index              (F, 3)   int64     surface triangles
    edges/index              (2, E)   int64

...under `<MethodRepo>/outputs/<split>/<gpu_ids>/<epoch>/<sample>.h5`
(see `MeshGraphNets/general_modules/mesh_utils_fast.py::save_inference_results_fast`).

`hdf5_preview` dispatches on the `data`/`shapes`/`arrays` groups and so falls
all the way through to `_table_sample` for these files, which is why inference
results could not be visualized: nothing here looks like the dataset contract.

The interesting thing about a prediction file is the *comparison*, so each
physical channel is exposed as three selectable viewer features -- prediction,
truth, and signed error -- which lets the existing feature picker drive a
pred/truth/error switch with no new frontend control.

A whole output directory is catalogued as one artifact with one sample per
file, because "browse the samples this run predicted" is the actual task; a
single file is still accepted and behaves like a one-sample catalog.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

from studio_backend.hdf5_preview import (
    _base_payload,
    _finite_list,
    _stats,
    _imports,
    _sort_key,
    hdf5_sample,
    hdf5_samples,
)
from studio_backend.paths import SUITE_ROOT, relative

HDF5_SUFFIXES = {".h5", ".hdf5"}

# Ordered best-first: a prediction is worth showing in physical units when the
# run recorded them, and normalized units are the honest fallback when the
# training statistics were unavailable.
_PREDICTED = ("predicted_denorm", "predicted_norm")
_TARGET = ("target_denorm", "target_norm")

# One physical channel becomes this many viewer features.
_ROLES = ("predicted", "truth", "error")


def _first_present(group: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in group:
            return name
    return None


def is_prediction_file(path: Path) -> bool:
    """Whether `path` is an inference-result file rather than a dataset."""
    if path.suffix.lower() not in HDF5_SUFFIXES or not path.is_file():
        return False
    h5py, _ = _imports()
    try:
        with h5py.File(path, "r") as handle:
            nodes = handle.get("nodes")
            if nodes is None or not hasattr(nodes, "keys"):
                return False
            return "pos" in nodes and _first_present(nodes, _PREDICTED) is not None
    except (OSError, ValueError):
        return False


def prediction_files(path: Path, limit: int | None = None) -> list[Path]:
    """Every prediction file `path` refers to, naturally sorted."""
    if path.is_file():
        return [path]
    candidates = sorted(
        (item for item in path.iterdir() if item.suffix.lower() in HDF5_SUFFIXES),
        key=lambda item: _sort_key(item.stem),
    )
    found = [item for item in candidates if is_prediction_file(item)]
    return found if limit is None else found[:limit]


def is_prediction_dir(path: Path) -> bool:
    """Whether `path` is a directory holding at least one prediction file."""
    if not path.is_dir():
        return False
    for item in sorted(path.iterdir(), key=lambda entry: entry.name):
        if item.suffix.lower() in HDF5_SUFFIXES and is_prediction_file(item):
            return True
    return False


def is_rollout_file(path: Path) -> bool:
    """Whether `path` is a rollout result: the shared dataset contract, as output.

    `mode inference` does not write the paired prediction/truth layout at all --
    it writes an autoregressive rollout back out in the *dataset* contract
    (`data/{id}/nodal_data [F,T,N]` + `mesh_edge`), one sample per file, under
    `<MethodRepo>/outputs/rollout/`. That is why inference output could not be
    found: it does not look like a result file, it looks like a dataset.

    Being indistinguishable from a source dataset by content is exactly why this
    is never used to classify a lone file -- only files inside a run's output
    directory, where the surrounding directory supplies the missing context.
    """
    if path.suffix.lower() not in HDF5_SUFFIXES or not path.is_file():
        return False
    h5py, _ = _imports()
    try:
        with h5py.File(path, "r") as handle:
            group = handle.get("data")
            if group is None or not hasattr(group, "keys"):
                return False
            return any(
                "nodal_data" in group[key] or "nodal_field" in group[key]
                for key in list(group.keys())[:1]
            )
    except (OSError, ValueError):
        return False


def result_flavor(path: Path) -> str | None:
    """"paired", "rollout", or None for a directory of run outputs."""
    if not path.is_dir():
        return "paired" if is_prediction_file(path) else None
    files = [item for item in sorted(path.iterdir()) if item.suffix.lower() in HDF5_SUFFIXES]
    if not files:
        return None
    probe = min(files, key=lambda item: item.name)
    if is_prediction_file(probe):
        return "paired"
    return "rollout" if is_rollout_file(probe) else None


def _rollout_inner_id(handle: Any) -> str | None:
    group = handle.get("data")
    if group is None or not hasattr(group, "keys"):
        return None
    keys = sorted(group.keys(), key=_sort_key)
    return str(keys[0]) if keys else None


def _truth_sample_id(truth_handle: Any, stem: str, inner: str) -> str | None:
    """Find the ground-truth sample matching a rollout file.

    The rollout keeps the source sample id as its inner group, so that matches
    directly; `rollout_sample10_steps19` also carries it in the filename, which
    is the fallback when a run renumbered its groups.
    """
    group = truth_handle.get("data")
    if group is None or not hasattr(group, "keys"):
        return None
    ids = {str(key) for key in group.keys()}
    if str(inner) in ids:
        return str(inner)
    match = re.search(r"sample(\d+)", stem, re.IGNORECASE)
    if match and match.group(1) in ids:
        return match.group(1)
    return None


def rollout_samples(path: Path, limit: int = 100, truth_path: Path | None = None) -> dict[str, Any]:
    """Catalog a rollout directory as one artifact with one sample per file."""
    h5py, _ = _imports()
    if path.is_file() and truth_path is None:
        return hdf5_samples(path, limit=limit)
    files = [path] if path.is_file() else [
        item for item in sorted(path.iterdir(), key=lambda entry: _sort_key(entry.stem))
        if item.suffix.lower() in HDF5_SUFFIXES and is_rollout_file(item)
    ]
    if not files:
        raise ValueError(f"{relative(path)} holds no rollout result files.")

    names: list[str] = []
    samples: list[dict[str, Any]] = []
    for item in files[:limit]:
        inner = hdf5_samples(item, limit=1)
        names = names or inner.get("feature_names") or []
        entry = (inner.get("samples") or [{}])[0]
        samples.append(
            {
                "id": item.stem,
                "label": item.stem,
                "datasets": entry.get("datasets", []),
                "default_feature": entry.get("default_feature", 3),
            }
        )

    # A rollout holds no ground truth of its own -- it is the prediction written
    # back out in the dataset contract. Given the dataset it was predicted from,
    # every channel can be offered three ways, so the error field is visible in
    # the viewer instead of requiring a separate evaluation run.
    if truth_path is not None:
        physical = next((index for index in range(len(names)) if index >= 3), 0)
        return {
            "path": relative(path),
            "source_kind": "hdf5",
            "contract": "inference_rollout",
            "default_mode": "field",
            "feature_names": feature_names(len(names), names),
            "condition_names": [],
            "samples": [dict(item, default_feature=physical * len(_ROLES)) for item in samples],
            "truth_path": relative(truth_path),
            "truncated": len(files) > limit,
            "total_samples": len(files),
        }

    return {
        "path": relative(path),
        "source_kind": "hdf5",
        "contract": "inference_rollout",
        "default_mode": "field",
        "feature_names": names,
        "condition_names": [],
        "samples": samples,
        "truncated": len(files) > limit,
        "total_samples": len(files),
    }


def rollout_sample(
    path: Path,
    sample_id: str,
    feature: int,
    timestep: int,
    truth_path: Path | None = None,
) -> dict[str, Any]:
    """Read one rollout sample, delegating to the shared mesh reader.

    A rollout file *is* the dataset contract, so the existing reader already
    handles its edges, timesteps, and feature names correctly -- only finding
    the right file and its inner sample id is new work.
    """
    h5py, np = _imports()
    if path.is_file() and truth_path is None:
        return hdf5_sample(path, sample_id, feature, timestep)
    files = [path] if path.is_file() else [
        item for item in sorted(path.iterdir(), key=lambda entry: _sort_key(entry.stem))
        if item.suffix.lower() in HDF5_SUFFIXES and is_rollout_file(item)
    ]
    target = next((item for item in files if item.stem == str(sample_id)), files[0] if files else None)
    if target is None:
        raise ValueError(f"{relative(path)} holds no rollout result files.")
    with h5py.File(target, "r") as handle:
        inner = _rollout_inner_id(handle)
    if inner is None:
        raise ValueError(f"{target.name} has no samples under data/.")

    if truth_path is None:
        payload = hdf5_sample(target, inner, feature, timestep)
        payload["sample"] = target.stem
        payload.setdefault("metadata", {})["contract"] = "inference_rollout"
        payload["metadata"]["sample_file"] = relative(target)
        return payload

    # Paired against ground truth: the flat feature index selects a channel and
    # one of prediction / truth / error, exactly as for the paired layout, so the
    # existing feature picker drives the comparison with no new control.
    raw_names = (hdf5_samples(target, limit=1).get("feature_names") or [])
    channel, role = _resolve_feature(feature, max(1, len(raw_names)))
    payload = hdf5_sample(target, inner, channel, timestep)
    payload["sample"] = target.stem
    payload.setdefault("metadata", {})
    payload["metadata"].update(
        {"contract": "inference_rollout", "sample_file": relative(target), "requested_role": role}
    )
    payload["feature"] = int(feature)
    payload["feature_count"] = max(1, len(raw_names) * len(_ROLES))
    payload["feature_names"] = feature_names(len(raw_names), raw_names)
    payload["feature_name"] = payload["feature_names"][
        min(int(feature), len(payload["feature_names"]) - 1)
    ]

    if role == "predicted":
        payload["metadata"]["role"] = "predicted"
        payload["metadata"]["truth_path"] = relative(truth_path)
        return payload

    with h5py.File(truth_path, "r") as truth_handle:
        truth_id = _truth_sample_id(truth_handle, target.stem, inner)
        if truth_id is None:
            payload["metadata"]["role"] = "predicted"
            payload["metadata"]["truth_error"] = (
                f"{relative(truth_path)} has no sample matching {target.stem}"
            )
            return payload
        truth_data = truth_handle[f"data/{truth_id}/nodal_data"]
        step = max(0, min(int(timestep), int(truth_data.shape[1]) - 1))
        row = max(0, min(int(channel), int(truth_data.shape[0]) - 1))
        truth_values = np.asarray(truth_data[row, step, :], dtype=np.float64)

    predicted = np.asarray(payload["values"], dtype=np.float64)
    if truth_values.size != predicted.size:
        payload["metadata"]["role"] = "predicted"
        payload["metadata"]["truth_error"] = (
            f"node count differs: prediction {predicted.size} vs truth {truth_values.size}"
        )
        return payload

    values = truth_values if role == "truth" else predicted - truth_values
    payload["values"] = _finite_list(values)
    payload["stats"] = _stats(values)
    payload["metadata"]["role"] = role
    payload["metadata"]["truth_path"] = relative(truth_path)
    payload["metadata"]["truth_sample"] = truth_id
    return payload


def _channel_count(handle: Any) -> int:
    nodes = handle["nodes"]
    name = _first_present(nodes, _PREDICTED)
    if name is None:
        return 0
    shape = nodes[name].shape
    return int(shape[1]) if len(shape) > 1 else 1


def feature_names(channels: int, channel_names: list[str] | None = None) -> list[str]:
    """Three viewer features -- prediction, truth, error -- per channel."""
    names: list[str] = []
    for channel in range(channels):
        label = (
            channel_names[channel]
            if channel_names and channel < len(channel_names)
            else f"ch{channel}"
        )
        names.extend(f"{role} {label}" for role in _ROLES)
    return names


def _resolve_feature(feature: int, channels: int) -> tuple[int, str]:
    """Split a flat viewer feature index into (channel, role)."""
    total = max(1, channels * len(_ROLES))
    index = max(0, min(int(feature), total - 1))
    return index // len(_ROLES), _ROLES[index % len(_ROLES)]


def _reduce_surface(np: Any, faces: Any, node_count: int, point_limit: int, face_limit: int):
    """Cut the surface down to the viewer's budget without tearing holes in it.

    Dropping nodes first and then discarding any face that referenced one of
    them leaves a shredded surface. Thin the *faces* instead and keep exactly
    the nodes they still reference, so whatever survives is always a coherent
    indexed mesh.
    """
    if faces is None or faces.shape[0] == 0:
        keep = np.arange(min(node_count, point_limit), dtype=np.int64)
        return keep, np.empty((0, 3), dtype=np.int64), False

    reduced = False
    if faces.shape[0] > face_limit:
        step = int(np.ceil(faces.shape[0] / face_limit))
        faces = faces[::step]
        reduced = True

    used = np.unique(faces)
    # Thinning faces does not bound the node count directly, so tighten again
    # until the referenced nodes fit too.
    while used.size > point_limit and faces.shape[0] > 1:
        faces = faces[::2]
        used = np.unique(faces)
        reduced = True

    remap = np.full(node_count, -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return used, remap[faces], reduced


def prediction_samples(path: Path, limit: int = 100) -> dict[str, Any]:
    """Catalog every predicted sample under `path`."""
    h5py, _ = _imports()
    files = prediction_files(path)
    if not files:
        raise ValueError(f"{relative(path)} holds no inference-result files.")

    channels = 0
    samples: list[dict[str, Any]] = []
    for item in files[:limit]:
        with h5py.File(item, "r") as handle:
            channels = max(channels, _channel_count(handle))
            nodes = handle["nodes"]
            predicted = _first_present(nodes, _PREDICTED)
            samples.append(
                {
                    "id": item.stem,
                    "label": item.stem,
                    "datasets": [
                        {"name": f"nodes/{name}", "shape": [int(size) for size in nodes[name].shape]}
                        for name in ("pos", predicted)
                        if name and name in nodes
                    ],
                    "default_feature": 0,
                    "units": "physical" if predicted == "predicted_denorm" else "normalized",
                }
            )

    return {
        "path": relative(path),
        "source_kind": "hdf5",
        "contract": "inference_result",
        "default_mode": "field",
        "feature_names": feature_names(channels),
        "condition_names": [],
        "samples": samples,
        "truncated": len(files) > limit,
        "total_samples": len(files),
    }


def prediction_sample(
    path: Path,
    sample_id: str,
    feature: int,
    point_limit: int = 16000,
    face_limit: int = 40000,
) -> dict[str, Any]:
    """Read one predicted sample as prediction, truth, or signed error."""
    h5py, np = _imports()
    files = prediction_files(path)
    if not files:
        raise ValueError(f"{relative(path)} holds no inference-result files.")
    target_file = next((item for item in files if item.stem == str(sample_id)), files[0])

    with h5py.File(target_file, "r") as handle:
        nodes = handle["nodes"]
        predicted_name = _first_present(nodes, _PREDICTED)
        truth_name = _first_present(nodes, _TARGET)
        if predicted_name is None:
            raise ValueError(f"{target_file.name} has no prediction array.")

        positions = np.asarray(nodes["pos"], dtype=np.float64)
        node_count = int(positions.shape[0])
        channels = _channel_count(handle)
        channel, role = _resolve_feature(feature, channels)

        faces = None
        if "faces" in handle and "index" in handle["faces"]:
            faces = np.asarray(handle["faces"]["index"], dtype=np.int64)
        keep, remapped, reduced = _reduce_surface(np, faces, node_count, point_limit, face_limit)

        predicted = np.asarray(nodes[predicted_name][:, channel], dtype=np.float64)
        if role == "predicted" or truth_name is None:
            values_full = predicted
            # Asking for truth or error on a run that saved no ground truth is
            # a real possibility (inference on an unlabelled dataset); say so
            # in the metadata rather than quietly showing the prediction again.
            resolved_role = "predicted"
        else:
            truth = np.asarray(nodes[truth_name][:, channel], dtype=np.float64)
            values_full = truth if role == "truth" else predicted - truth
            resolved_role = role

        selected = positions[keep]
        return _base_payload(
            path=target_file,
            sample_id=target_file.stem,
            dataset=f"nodes/{predicted_name}",
            shape=nodes[predicted_name].shape,
            feature=feature,
            feature_count=max(1, channels * len(_ROLES)),
            timestep=0,
            timestep_count=1,
            x=selected[:, 0],
            y=selected[:, 1],
            z=selected[:, 2] if selected.shape[1] > 2 else np.zeros(keep.size),
            values=values_full[keep],
            total_points=node_count,
            mesh=(
                {
                    "indexed": True,
                    "element_kind": "tri",
                    "reduced": bool(reduced),
                    "total_edges": int(handle.attrs.get("num_edges", 0) or 0),
                    "returned_edges": 0,
                    "edges": [],
                    "returned_faces": int(remapped.shape[0]),
                    "faces": remapped.reshape(-1).tolist(),
                    "returned_elements": int(remapped.shape[0]),
                    "total_elements": int(faces.shape[0]) if faces is not None else 0,
                }
                if remapped.shape[0]
                else None
            ),
            preview_kind="mesh" if remapped.shape[0] else "points",
            supports_field=True,
            feature_names=feature_names(channels),
            metadata={
                "has_coordinates": True,
                "contract": "inference_result",
                "role": resolved_role,
                "requested_role": role,
                "has_truth": truth_name is not None,
                "channel": int(channel),
                "units": "physical" if predicted_name == "predicted_denorm" else "normalized",
                "sample_file": relative(target_file),
                "node_reduction": "face thinning" if reduced else "none",
            },
        )


def _scan_roots(roots: list[Path], limit: int, since: float | None = None) -> list[dict[str, Any]]:
    """Every prediction directory under `roots`, optionally only recent ones."""
    seen: set[Path] = set()
    runs: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory in [root, *(item for item in root.rglob("*") if item.is_dir())]:
            if directory in seen or len(runs) >= limit:
                continue
            seen.add(directory)
            files = [
                item for item in directory.iterdir()
                if item.is_file() and item.suffix.lower() in HDF5_SUFFIXES
            ]
            if not files:
                continue
            newest = max(item.stat().st_mtime for item in files)
            # Cheap time filter first -- only then pay for opening a file.
            if since is not None and newest < since:
                continue
            probe = min(files, key=lambda item: item.name)
            if not is_prediction_file(probe) and not is_rollout_file(probe):
                continue
            parts = directory.relative_to(SUITE_ROOT).parts
            runs.append(
                {
                    "path": relative(directory),
                    "method": parts[0] if parts else "",
                    "label": "/".join(parts[1:]) or directory.name,
                    "samples": len(files),
                    "modified": newest,
                }
            )
    runs.sort(key=lambda item: item["modified"], reverse=True)
    return runs


def scan_output_dir(directory: Path, limit: int = 12) -> list[dict[str, Any]]:
    """Prediction directories under one explicitly configured output directory.

    Used for inference steps, where `inference_output_dir` names the destination
    outright. No time filter: the directory was named by this run's own config,
    so anything prediction-shaped inside it belongs to this run -- and requiring
    a fresh mtime would drop a rollout that a rerun wrote identically.
    """
    if not directory.is_dir():
        return []
    return _scan_roots([directory], limit)


def outputs_since(repository: Path, since: float, limit: int = 12) -> list[dict[str, Any]]:
    """Prediction directories a just-finished run wrote inside `repository`.

    Used to pin a completed inference job to its own results. Matching on
    "written after this step started, under this step's repository" is what
    makes the link exact -- guessing the path from config keys cannot work,
    because the epoch number in `outputs/<split>/<gpu>/<epoch>/` is only known
    to the training loop.
    """
    roots = [repository / name for name in ("outputs", "output")]
    return _scan_roots([root for root in roots if root.is_dir()], limit, since=since)


# A full scan walks every output tree in the suite and opens one HDF5 per
# candidate directory. That is ~1s today and grows with every run, while the
# answer only changes when a job finishes -- so serve repeat calls from a short
# lived cache. The TTL is deliberately short: a run that just completed must
# show up without the user wondering whether to reload.
_RUNS_CACHE: dict[str, Any] = {"at": 0.0, "limit": 0, "value": None}
_RUNS_CACHE_TTL = 20.0
_RUNS_LOCK = threading.Lock()


def invalidate_prediction_runs() -> None:
    """Drop the cache so the next call rescans. Called when a job writes results."""
    with _RUNS_LOCK:
        _RUNS_CACHE["value"] = None


def prediction_runs(limit: int = 200) -> dict[str, Any]:
    """List every inference-output directory in the suite, newest first.

    Results land under `<MethodRepo>/outputs/...`, which the `artifact` file
    catalog never walked -- it only looks at `SUITE_ROOT/output(s)` -- so a
    finished inference run was invisible to the GUI no matter where it looked.

    Directories are the unit here, not files: one run predicts many samples and
    the user wants to browse the run. Only the first HDF5 in each directory is
    opened, so this stays cheap over a suite with thousands of result files.
    """
    now = time.monotonic()
    with _RUNS_LOCK:
        cached = _RUNS_CACHE["value"]
        if (
            cached is not None
            and _RUNS_CACHE["limit"] == limit
            and now - _RUNS_CACHE["at"] < _RUNS_CACHE_TTL
        ):
            return cached

    roots = [SUITE_ROOT / "output", SUITE_ROOT / "outputs"]
    for item in sorted(SUITE_ROOT.iterdir()):
        if item.is_dir():
            roots.extend([item / "outputs", item / "output"])
    runs = _scan_roots(roots, limit)
    payload = {"items": runs, "truncated": len(runs) >= limit}
    with _RUNS_LOCK:
        _RUNS_CACHE.update({"at": now, "limit": limit, "value": payload})
    return payload
