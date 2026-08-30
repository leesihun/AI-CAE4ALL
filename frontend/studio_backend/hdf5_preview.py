"""Schema-aware HDF5 extraction for the Studio artifact viewer.

All HDF5 visualization flows through :func:`hdf5_samples` and
:func:`hdf5_sample`.  Those functions normalize the suite's mesh-state, SDF
shape, operator-grid, and table contracts into one bounded preview payload.
They never materialize an entire multi-sample dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from studio_backend.mesh_topology import aggregate_values, build_topology
from studio_backend.paths import relative

# Rebuilding the reduced mesh for every timestep or feature change would stall
# the timeline scrubber, and the reduction depends only on geometry.
_TOPOLOGY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TOPOLOGY_CACHE_SIZE = 6


def _imports():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and numpy are required for HDF5 visualization.") from exc
    return h5py, np


def _sort_key(value: str) -> tuple[bool, int | str]:
    text = str(value)
    return (not text.isdigit(), int(text) if text.isdigit() else text.lower())


def _dataset_records(group: Any, prefix: str = "") -> list[dict[str, Any]]:
    h5py, _ = _imports()
    records: list[dict[str, Any]] = []
    for name, item in group.items():
        path = f"{prefix}/{name}".strip("/")
        if isinstance(item, h5py.Dataset):
            records.append({"name": path, "shape": list(item.shape), "dtype": str(item.dtype)})
        elif isinstance(item, h5py.Group):
            records.extend(_dataset_records(item, path))
    return records


def _numeric_datasets(handle: Any) -> list[tuple[str, Any]]:
    h5py, np = _imports()
    datasets: list[tuple[str, Any]] = []

    def visit(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
            datasets.append((name, item))

    handle.visititems(visit)
    return datasets


def _finite_list(values: Any) -> list[float | None]:
    _, np = _imports()
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value) if np.isfinite(value) else None for value in flat]


def _stats(values: Any) -> dict[str, float | None]:
    _, np = _imports()
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
    }


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _text_list(source: Any, key: str) -> list[str]:
    """Read an HDF5 string dataset or attribute as a plain list of names."""
    try:
        if hasattr(source, "attrs") and key in source.attrs:
            raw = source.attrs[key]
        elif key in source:
            raw = source[key][()]
        else:
            return []
    except (KeyError, TypeError, ValueError, OSError):
        return []
    if isinstance(raw, (bytes, str)):
        return [_decode(raw)]
    try:
        return [_decode(item) for item in raw]
    except TypeError:
        return [_decode(raw)]


def _named_features(names: list[str], count: int, prefix: str = "feature") -> list[str]:
    """Pad or trim declared names so index N always resolves to a label."""
    resolved = list(names[:count])
    resolved += [f"{prefix} {index}" for index in range(len(resolved), count)]
    return resolved


def _mesh_feature_names(handle: Any, count: int) -> list[str]:
    metadata = handle.get("metadata") if hasattr(handle, "get") else None
    names = _text_list(metadata, "feature_names") if metadata is not None else []
    if not names:
        names = _text_list(handle, "feature_names")
    return _named_features(names, count)


def _declared_feature_count(samples: list[dict[str, Any]]) -> int:
    """Infer the channel count from the catalogued nodal_data shapes."""
    for sample in samples:
        for record in sample.get("datasets", []):
            if record["name"].rsplit("/", 1)[-1] in {"nodal_data", "nodal_field"} and record["shape"]:
                return int(record["shape"][0])
    return 0


def _operator_feature_names(handle: Any, targets: Any) -> list[str]:
    count = int(targets.shape[1]) if targets.ndim >= 3 else 1
    common = handle.get("common") if hasattr(handle, "get") else None
    if common is not None and "alpha" in common:
        alpha = common["alpha"]
        if alpha.shape and alpha.shape[0] >= count:
            return [f"alpha={float(alpha[index]):.6g}" for index in range(count)]
    return _named_features([f"target {index}" for index in range(count)], count)


def _pair_names(names: list[str], values: list[Any]) -> list[dict[str, Any]]:
    """Zip declared parameter names against a sample's scalar conditions."""
    if not values:
        return []
    labelled = _named_features(names, len(values), prefix="parameter")
    return [{"name": labelled[index], "value": values[index]} for index in range(len(values))]


def _table_column_names(handle: Any) -> list[str]:
    """Name the flattened X (then Y) columns of a tabular parametric dataset."""
    inputs = _text_list(handle, "input_names")
    outputs = _text_list(handle, "output_names")
    return inputs + outputs


def _sample_indices(count: int, limit: int) -> Any:
    _, np = _imports()
    stride = max(1, (count + max(1, limit) - 1) // max(1, limit))
    return np.arange(0, count, stride, dtype=np.int64)


def _base_payload(
    *,
    path: Path,
    sample_id: str,
    dataset: str,
    shape: Iterable[int],
    feature: int,
    feature_count: int,
    timestep: int,
    timestep_count: int,
    x: Any,
    y: Any,
    z: Any,
    values: Any,
    total_points: int,
    mesh: dict[str, Any] | None,
    preview_kind: str,
    supports_field: bool,
    feature_name: str | None = None,
    feature_names: list[str] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    returned_points = len(x)
    names = feature_names or []
    return {
        "path": relative(path),
        "source_kind": "hdf5",
        "preview_kind": preview_kind,
        "sample": str(sample_id),
        "dataset": dataset,
        "shape": [int(value) for value in shape],
        "feature": int(feature),
        "feature_count": max(1, int(feature_count)),
        "feature_name": feature_name or (names[feature] if feature < len(names) else f"feature {feature}"),
        "feature_names": names,
        "parameters": parameters or [],
        "timestep": int(timestep),
        "timestep_count": max(1, int(timestep_count)),
        "total_points": int(total_points),
        "returned_points": int(returned_points),
        "x": _finite_list(x),
        "y": _finite_list(y),
        "z": _finite_list(z),
        "values": _finite_list(values),
        "mesh": mesh,
        "stats": _stats(values),
        "supports": {
            "points": True,
            "mesh": bool(mesh),
            "field": bool(supports_field),
        },
        "metadata": metadata or {},
    }


def hdf5_summary(path: Path) -> dict[str, Any]:
    h5py, _ = _imports()
    items: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        root_attrs = {str(key): value for key, value in handle.attrs.items()}

        def visit(name: str, obj: Any) -> None:
            if len(items) >= 300:
                return
            record: dict[str, Any] = {"path": name or "/", "type": "group"}
            if isinstance(obj, h5py.Dataset):
                record.update(type="dataset", shape=list(obj.shape), dtype=str(obj.dtype))
                # Names and split/provenance labels are commonly stored as tiny
                # string datasets. Shape + dtype alone hides the actual data
                # contract, so expose only these bounded values while continuing
                # to avoid materializing numeric training arrays.
                if obj.size <= 128 and h5py.check_string_dtype(obj.dtype) is not None:
                    raw = obj.asstr()[()]
                    values = raw.tolist() if hasattr(raw, "tolist") else raw
                    record["values"] = values if isinstance(values, list) else [values]
            if obj.attrs:
                record["attrs"] = {str(key): value for key, value in obj.attrs.items()}
            items.append(record)

        handle.visititems(visit)
    return {
        "path": relative(path),
        "size": path.stat().st_size,
        "root_attrs": root_attrs,
        "items": items,
        "truncated": len(items) >= 300,
    }


def _catalog_group_samples(
    path: Path,
    root_name: str,
    root: Any,
    limit: int,
    contract: str,
    default_mode: str,
    feature_names: list[str] | None = None,
    condition_names: list[str] | None = None,
) -> dict[str, Any]:
    sample_ids = sorted(root.keys(), key=_sort_key)
    samples: list[dict[str, Any]] = []
    for sample_id in sample_ids[:limit]:
        item = root[sample_id]
        datasets = _dataset_records(item) if hasattr(item, "items") else []
        samples.append(
            {
                "id": str(sample_id),
                "label": f"{root_name}/{sample_id}",
                "datasets": datasets,
                "default_feature": 3 if any(record["name"] == "nodal_data" and record["shape"][0] > 3 for record in datasets) else 0,
            }
        )
    return {
        "path": relative(path),
        "source_kind": "hdf5",
        "contract": contract,
        "default_mode": default_mode,
        "feature_names": feature_names or [],
        "condition_names": condition_names or [],
        "samples": samples,
        "truncated": len(sample_ids) > limit,
        "total_samples": len(sample_ids),
    }


def hdf5_samples(path: Path, limit: int = 100) -> dict[str, Any]:
    """Return a bounded sample catalog for every supported HDF5 contract."""
    h5py, _ = _imports()
    with h5py.File(path, "r") as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Group):
            catalog = _catalog_group_samples(path, "data", handle["data"], limit, "mesh_state", "field")
            declared = int(handle.attrs.get("num_features", 0) or 0) or _declared_feature_count(catalog["samples"])
            catalog["feature_names"] = _mesh_feature_names(handle, declared) if declared else []
            return catalog

        if "shapes" in handle and isinstance(handle["shapes"], h5py.Group):
            return _catalog_group_samples(
                path,
                "shapes",
                handle["shapes"],
                limit,
                "sdf_shapes",
                "points",
                condition_names=_text_list(handle, "cond_names"),
            )

        if (
            "arrays" in handle
            and isinstance(handle["arrays"], h5py.Group)
            and "targets" in handle["arrays"]
            and "common" in handle
            and "query_xy" in handle["common"]
        ):
            targets = handle["arrays/targets"]
            count = int(targets.shape[0]) if targets.ndim else 1
            samples = [
                {
                    "id": str(index),
                    "label": f"arrays/targets[{index}]",
                    "datasets": [
                        {"name": "arrays/targets", "shape": list(targets.shape), "dtype": str(targets.dtype)},
                        {
                            "name": "common/query_xy",
                            "shape": list(handle["common/query_xy"].shape),
                            "dtype": str(handle["common/query_xy"].dtype),
                        },
                    ],
                    "default_feature": 0,
                }
                for index in range(min(count, limit))
            ]
            return {
                "path": relative(path),
                "source_kind": "hdf5",
                "contract": "operator_grid",
                "default_mode": "field",
                "feature_names": _operator_feature_names(handle, targets),
                "condition_names": [],
                "samples": samples,
                "truncated": count > limit,
                "total_samples": count,
            }

        datasets = _numeric_datasets(handle)
        if not datasets:
            return {
                "path": relative(path),
                "source_kind": "hdf5",
                "contract": "unsupported",
                "default_mode": "points",
                "samples": [],
                "truncated": False,
                "total_samples": 0,
            }

        preferred_name, preferred = next(
            ((name, item) for name, item in datasets if name.rsplit("/", 1)[-1] in {"X", "Y", "targets", "predictions"}),
            datasets[0],
        )
        count = int(preferred.shape[0]) if preferred.ndim >= 2 else 1
        records = [
            {"name": name, "shape": list(item.shape), "dtype": str(item.dtype)}
            for name, item in datasets[:30]
        ]
        input_names = _text_list(handle, "input_names")
        output_names = _text_list(handle, "output_names")
        column_names = input_names + output_names
        value_sources = [
            handle[name]
            for name in ("X", "Y")
            if name in handle and getattr(handle[name], "ndim", 0) >= 2 and int(handle[name].shape[0]) == count
        ]
        samples = []
        for index in range(min(count, limit)):
            sample = {
                "id": str(index),
                "label": f"row {index}",
                "datasets": records,
                "default_feature": 0,
            }
            if column_names and value_sources:
                values: list[float | None] = []
                for source in value_sources:
                    values.extend(_finite_list(source[index]))
                sample["parameter_values"] = values[: len(column_names)]
            samples.append(sample)
        return {
            "path": relative(path),
            "source_kind": "hdf5",
            "contract": "table",
            "default_mode": "field",
            "feature_names": column_names,
            "condition_names": input_names,
            "output_names": output_names,
            "samples": samples,
            "truncated": count > limit,
            "total_samples": count,
            "primary_dataset": preferred_name,
        }


def _cached_topology(
    path: Path,
    group: Any,
    sample_id: str,
    coordinates: Any,
    point_limit: int,
    edge_limit: int,
    face_limit: int,
) -> dict[str, Any]:
    _, np = _imports()
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    key = (str(path), stamp, str(sample_id), point_limit, edge_limit, face_limit)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached
    raw_edges = np.asarray(group["mesh_edge"]) if "mesh_edge" in group else None
    topology = build_topology(
        coordinates,
        raw_edges,
        node_budget=point_limit,
        edge_limit=edge_limit,
        face_limit=face_limit,
    )
    if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_SIZE:
        _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
    _TOPOLOGY_CACHE[key] = topology
    return topology


def _mesh_payload(topology: dict[str, Any], declared_cells: int | None) -> dict[str, Any] | None:
    """Describe the reduced mesh as indexed geometry over the returned nodes."""
    edges = topology["edges"]
    faces = topology["faces"]
    if edges.shape[0] == 0 and faces.shape[0] == 0:
        return None
    # A reconstructed quad is emitted as two triangles, so element counts and
    # the renderer's triangle count are not the same number.
    per_element = 2 if topology["element_kind"] == "quad" else 1
    return {
        "indexed": True,
        "element_kind": topology["element_kind"],
        "reduced": bool(topology["reduced"]),
        "total_edges": int(topology["total_edges"]),
        "returned_edges": int(edges.shape[0]),
        "edges": edges.reshape(-1).tolist(),
        "returned_faces": int(faces.shape[0]),
        "faces": faces.reshape(-1).tolist(),
        "returned_elements": int(faces.shape[0] // per_element),
        "total_elements": int(declared_cells) if declared_cells else int(topology["total_faces"] // per_element),
    }


def _mesh_state_sample(
    path: Path,
    handle: Any,
    group: Any,
    sample_id: str,
    feature: int,
    timestep: int,
    point_limit: int,
    edge_limit: int,
    face_limit: int,
) -> dict[str, Any]:
    _, np = _imports()
    dataset_name = "nodal_data" if "nodal_data" in group else "nodal_field" if "nodal_field" in group else None
    if dataset_name is None:
        raise ValueError(f"Sample {sample_id} has neither nodal_data nor nodal_field.")
    data = group[dataset_name]
    if data.ndim != 3:
        raise ValueError(f"{dataset_name} must be rank 3 [F,T,N], got {data.shape}.")

    selected_feature = max(0, min(int(feature), int(data.shape[0]) - 1))
    selected_time = max(0, min(int(timestep), int(data.shape[1]) - 1))
    node_count = int(data.shape[2])
    values_full = np.asarray(data[selected_feature, selected_time, :], dtype=np.float64)
    feature_names = _mesh_feature_names(handle, int(data.shape[0]))

    if dataset_name != "nodal_data" or data.shape[0] < 3:
        # No coordinate rows: fall back to an index/value series, which has no
        # topology to preserve.
        indices = _sample_indices(node_count, point_limit)
        return _base_payload(
            path=path,
            sample_id=sample_id,
            dataset=dataset_name,
            shape=data.shape,
            feature=selected_feature,
            feature_count=int(data.shape[0]),
            timestep=selected_time,
            timestep_count=int(data.shape[1]),
            x=np.arange(node_count, dtype=np.float64)[indices],
            y=values_full[indices],
            z=np.zeros(indices.size, dtype=np.float64),
            values=values_full[indices],
            total_points=node_count,
            mesh=None,
            preview_kind="series",
            supports_field=True,
            feature_names=feature_names,
            metadata={"has_coordinates": False},
        )

    coordinates = np.stack([np.asarray(data[axis, 0, :], dtype=np.float64) for axis in range(3)], axis=1)
    topology = _cached_topology(
        path, group, sample_id, coordinates, point_limit, edge_limit, face_limit
    )
    nodes = topology["coordinates"]
    values = aggregate_values(topology["inverse"], topology["node_count"], values_full)
    declared_cells = None
    if "metadata" in group and hasattr(group["metadata"], "attrs"):
        declared_cells = group["metadata"].attrs.get("num_cells")

    return _base_payload(
        path=path,
        sample_id=sample_id,
        dataset=dataset_name,
        shape=data.shape,
        feature=selected_feature,
        feature_count=int(data.shape[0]),
        timestep=selected_time,
        timestep_count=int(data.shape[1]),
        x=nodes[:, 0],
        y=nodes[:, 1],
        z=nodes[:, 2],
        values=values,
        total_points=node_count,
        mesh=_mesh_payload(topology, declared_cells),
        preview_kind="mesh",
        supports_field=True,
        feature_names=feature_names,
        metadata={
            "has_coordinates": True,
            "node_reduction": "vertex clustering" if topology["reduced"] else "none",
            "declared_cells": int(declared_cells) if declared_cells else None,
        },
    )


def _sdf_shape_sample(
    path: Path,
    handle: Any,
    group: Any,
    sample_id: str,
    point_limit: int,
) -> dict[str, Any]:
    _, np = _imports()
    if "surface_points" not in group:
        raise ValueError(f"Shape {sample_id} has no surface_points dataset.")
    points = np.asarray(group["surface_points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"surface_points must have shape [N,3], got {points.shape}.")
    indices = _sample_indices(int(points.shape[0]), point_limit)
    values = np.zeros(indices.size, dtype=np.float64)
    metadata: dict[str, Any] = {}
    conditions: list[float | None] = []
    if "cond" in group:
        conditions = _finite_list(group["cond"][()])
        metadata["conditions"] = conditions
    return _base_payload(
        path=path,
        sample_id=sample_id,
        dataset="surface_points",
        shape=points.shape,
        feature=0,
        feature_count=1,
        timestep=0,
        timestep_count=1,
        x=points[indices, 0],
        y=points[indices, 1],
        z=points[indices, 2],
        values=values,
        total_points=int(points.shape[0]),
        mesh=None,
        preview_kind="pointcloud",
        supports_field=False,
        feature_name="surface",
        feature_names=["surface"],
        parameters=_pair_names(_text_list(handle, "cond_names"), conditions),
        metadata=metadata,
    )


def _operator_grid_sample(path: Path, handle: Any, sample_id: str, feature: int, point_limit: int) -> dict[str, Any]:
    _, np = _imports()
    targets = handle["arrays/targets"]
    row = max(0, min(int(sample_id), int(targets.shape[0]) - 1))
    feature_count = int(targets.shape[1]) if targets.ndim >= 3 else 1
    selected_feature = max(0, min(int(feature), feature_count - 1))
    values_full = np.asarray(
        targets[row, selected_feature, :] if targets.ndim >= 3 else targets[row, :],
        dtype=np.float64,
    ).reshape(-1)
    query = np.asarray(handle["common/query_xy"], dtype=np.float64)
    if query.ndim != 2 or query.shape[1] < 2 or query.shape[0] != values_full.size:
        raise ValueError("common/query_xy does not align with arrays/targets.")
    indices = _sample_indices(int(query.shape[0]), point_limit)
    feature_names = _operator_feature_names(handle, targets)
    return _base_payload(
        path=path,
        sample_id=str(row),
        dataset="arrays/targets",
        shape=targets.shape,
        feature=selected_feature,
        feature_count=feature_count,
        timestep=0,
        timestep_count=1,
        x=query[indices, 0],
        y=query[indices, 1],
        z=np.zeros(indices.size, dtype=np.float64),
        values=values_full[indices],
        total_points=int(query.shape[0]),
        mesh=None,
        preview_kind="field_points",
        supports_field=True,
        feature_names=feature_names,
    )


def _table_sample(path: Path, handle: Any, sample_id: str, feature: int, point_limit: int) -> dict[str, Any]:
    _, np = _imports()
    datasets = _numeric_datasets(handle)
    if not datasets:
        raise ValueError("No numeric dataset was found.")
    preferred_name, preferred = next(
        ((name, item) for name, item in datasets if name.rsplit("/", 1)[-1] in {"X", "targets", "predictions", "Y"}),
        datasets[0],
    )
    row_count = int(preferred.shape[0]) if preferred.ndim >= 2 else 1
    row = max(0, min(int(sample_id), row_count - 1))

    if preferred.ndim >= 2:
        raw = np.asarray(preferred[row], dtype=np.float64)
    else:
        raw = np.asarray(preferred[()], dtype=np.float64)

    feature_count = int(raw.shape[0]) if raw.ndim >= 2 else 1
    selected_feature = max(0, min(int(feature), feature_count - 1))
    values_full = np.asarray(raw[selected_feature] if raw.ndim >= 2 else raw, dtype=np.float64).reshape(-1)
    dataset_label = preferred_name

    if preferred_name.rsplit("/", 1)[-1] == "X" and "Y" in handle and handle["Y"].ndim >= 2:
        output = np.asarray(handle["Y"][row], dtype=np.float64).reshape(-1)
        if raw.ndim <= 1:
            values_full = np.concatenate([values_full, output])
            dataset_label = "X + Y"

    indices = _sample_indices(int(values_full.size), point_limit)
    x = np.arange(values_full.size, dtype=np.float64)
    columns = _table_column_names(handle)
    return _base_payload(
        path=path,
        sample_id=str(row),
        dataset=dataset_label,
        shape=preferred.shape,
        feature=selected_feature,
        feature_count=feature_count,
        timestep=0,
        timestep_count=1,
        x=x[indices],
        y=values_full[indices],
        z=np.zeros(indices.size, dtype=np.float64),
        values=values_full[indices],
        total_points=int(values_full.size),
        mesh=None,
        preview_kind="series",
        supports_field=True,
        feature_name=f"row values{f' / slice {selected_feature}' if feature_count > 1 else ''}",
        feature_names=_named_features([], feature_count, prefix="slice"),
        parameters=_pair_names(columns, _finite_list(values_full)) if columns else [],
    )


def hdf5_sample(
    path: Path,
    sample_id: str,
    feature: int,
    timestep: int,
    point_limit: int = 16000,
    edge_limit: int = 60000,
    face_limit: int = 40000,
) -> dict[str, Any]:
    """Normalize one requested HDF5 sample into the shared viewer payload."""
    h5py, _ = _imports()
    with h5py.File(path, "r") as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Group) and sample_id in handle["data"]:
            return _mesh_state_sample(
                path,
                handle,
                handle["data"][sample_id],
                sample_id,
                feature,
                timestep,
                point_limit,
                edge_limit,
                face_limit,
            )
        if "shapes" in handle and isinstance(handle["shapes"], h5py.Group) and sample_id in handle["shapes"]:
            return _sdf_shape_sample(path, handle, handle["shapes"][sample_id], sample_id, 4000)
        if (
            "arrays" in handle
            and isinstance(handle["arrays"], h5py.Group)
            and "targets" in handle["arrays"]
            and "common" in handle
            and "query_xy" in handle["common"]
        ):
            return _operator_grid_sample(path, handle, sample_id, feature, point_limit)
        return _table_sample(path, handle, sample_id, feature, point_limit)
