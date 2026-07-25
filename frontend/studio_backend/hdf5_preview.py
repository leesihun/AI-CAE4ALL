"""Schema-aware HDF5 extraction for the Studio artifact viewer.

All HDF5 visualization flows through :func:`hdf5_samples` and
:func:`hdf5_sample`.  Those functions normalize the suite's mesh-state, SDF
shape, operator-grid, and table contracts into one bounded preview payload.
They never materialize an entire multi-sample dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from studio_backend.paths import relative


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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    returned_points = len(x)
    return {
        "path": relative(path),
        "source_kind": "hdf5",
        "preview_kind": preview_kind,
        "sample": str(sample_id),
        "dataset": dataset,
        "shape": [int(value) for value in shape],
        "feature": int(feature),
        "feature_count": max(1, int(feature_count)),
        "feature_name": feature_name or f"feature {feature}",
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
        "samples": samples,
        "truncated": len(sample_ids) > limit,
        "total_samples": len(sample_ids),
    }


def hdf5_samples(path: Path, limit: int = 100) -> dict[str, Any]:
    """Return a bounded sample catalog for every supported HDF5 contract."""
    h5py, _ = _imports()
    with h5py.File(path, "r") as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Group):
            return _catalog_group_samples(path, "data", handle["data"], limit, "mesh_state", "field")

        if "shapes" in handle and isinstance(handle["shapes"], h5py.Group):
            return _catalog_group_samples(path, "shapes", handle["shapes"], limit, "sdf_shapes", "points")

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
        samples = [
            {
                "id": str(index),
                "label": f"row {index}",
                "datasets": records,
                "default_feature": 0,
            }
            for index in range(min(count, limit))
        ]
        return {
            "path": relative(path),
            "source_kind": "hdf5",
            "contract": "table",
            "default_mode": "field",
            "samples": samples,
            "truncated": count > limit,
            "total_samples": count,
            "primary_dataset": preferred_name,
        }


def _edge_payload(group: Any, coords: list[Any], values: Any, node_count: int, edge_limit: int) -> dict[str, Any] | None:
    _, np = _imports()
    if "mesh_edge" not in group:
        return None
    raw_edges = np.asarray(group["mesh_edge"])
    if raw_edges.ndim == 2 and raw_edges.shape[0] == 2:
        raw_edges = raw_edges.T
    if raw_edges.ndim != 2 or raw_edges.shape[1] < 2:
        return None
    raw_edges = np.asarray(raw_edges[:, :2], dtype=np.int64)
    valid = (
        (raw_edges[:, 0] >= 0)
        & (raw_edges[:, 0] < node_count)
        & (raw_edges[:, 1] >= 0)
        & (raw_edges[:, 1] < node_count)
    )
    raw_edges = raw_edges[valid]
    stride = max(1, (raw_edges.shape[0] + max(1, edge_limit) - 1) // max(1, edge_limit))
    sampled = raw_edges[::stride]
    source = sampled[:, 0]
    target = sampled[:, 1]
    edge_values = (values[source] + values[target]) * 0.5
    return {
        "total_edges": int(raw_edges.shape[0]),
        "returned_edges": int(sampled.shape[0]),
        "x1": _finite_list(coords[0][source]),
        "y1": _finite_list(coords[1][source]),
        "z1": _finite_list(coords[2][source]),
        "x2": _finite_list(coords[0][target]),
        "y2": _finite_list(coords[1][target]),
        "z2": _finite_list(coords[2][target]),
        "values": _finite_list(edge_values),
        "triangles": None,
    }


def _mesh_state_sample(
    path: Path,
    group: Any,
    sample_id: str,
    feature: int,
    timestep: int,
    point_limit: int,
    edge_limit: int,
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
    indices = _sample_indices(node_count, point_limit)

    has_coordinates = dataset_name == "nodal_data" and data.shape[0] >= 3
    if has_coordinates:
        coords = [np.asarray(data[axis, 0, :], dtype=np.float64) for axis in range(3)]
        preview_kind = "mesh"
    else:
        coords = [
            np.arange(node_count, dtype=np.float64),
            values_full,
            np.zeros(node_count, dtype=np.float64),
        ]
        preview_kind = "series"

    mesh = _edge_payload(group, coords, values_full, node_count, edge_limit) if has_coordinates else None
    return _base_payload(
        path=path,
        sample_id=sample_id,
        dataset=dataset_name,
        shape=data.shape,
        feature=selected_feature,
        feature_count=int(data.shape[0]),
        timestep=selected_time,
        timestep_count=int(data.shape[1]),
        x=coords[0][indices],
        y=coords[1][indices],
        z=coords[2][indices],
        values=values_full[indices],
        total_points=node_count,
        mesh=mesh,
        preview_kind=preview_kind,
        supports_field=True,
        metadata={"has_coordinates": has_coordinates},
    )


def _sdf_shape_sample(path: Path, group: Any, sample_id: str, point_limit: int) -> dict[str, Any]:
    _, np = _imports()
    if "surface_points" not in group:
        raise ValueError(f"Shape {sample_id} has no surface_points dataset.")
    points = np.asarray(group["surface_points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"surface_points must have shape [N,3], got {points.shape}.")
    indices = _sample_indices(int(points.shape[0]), point_limit)
    values = np.zeros(indices.size, dtype=np.float64)
    metadata: dict[str, Any] = {}
    if "cond" in group:
        metadata["conditions"] = _finite_list(group["cond"][()])
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
    feature_name = f"target {selected_feature}"
    if "alpha" in handle["common"] and selected_feature < handle["common/alpha"].shape[0]:
        feature_name = f"alpha={float(handle['common/alpha'][selected_feature]):.6g}"
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
        feature_name=feature_name,
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
    )


def hdf5_sample(
    path: Path,
    sample_id: str,
    feature: int,
    timestep: int,
    point_limit: int = 2500,
    edge_limit: int = 6000,
) -> dict[str, Any]:
    """Normalize one requested HDF5 sample into the shared viewer payload."""
    h5py, _ = _imports()
    with h5py.File(path, "r") as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Group) and sample_id in handle["data"]:
            return _mesh_state_sample(
                path,
                handle["data"][sample_id],
                sample_id,
                feature,
                timestep,
                point_limit,
                edge_limit,
            )
        if "shapes" in handle and isinstance(handle["shapes"], h5py.Group) and sample_id in handle["shapes"]:
            return _sdf_shape_sample(path, handle["shapes"][sample_id], sample_id, point_limit)
        if (
            "arrays" in handle
            and isinstance(handle["arrays"], h5py.Group)
            and "targets" in handle["arrays"]
            and "common" in handle
            and "query_xy" in handle["common"]
        ):
            return _operator_grid_sample(path, handle, sample_id, feature, point_limit)
        return _table_sample(path, handle, sample_id, feature, point_limit)
