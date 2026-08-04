#!/usr/bin/env python3
"""Create a transient-safe ex3 feature layout without modifying the source HDF5.

The source NASA-CRM files store rows as::

    xyz | normal_xyz, cp, surface_area, cf_xyz | six flight/control scalars

The AI-CAE mesh loaders require predicted state rows first, followed by rows
that are inputs only.  This utility writes::

    xyz | cp, cf_xyz | six flight/control scalars | normal_xyz, surface_area

With that order, mesh methods use ``input_var=4``, ``output_var=4`` and
``cond_var=10``.  SimulGenVAE can use the same file with ``num_var=4`` and
``cond_var=6`` because its latent conditioner consumes only the leading six
constant condition rows and ignores the trailing spatial geometry descriptors.

The copy is written to a sibling ``.partial`` file and atomically renamed only
after validation.  Existing destinations are preserved unless ``--replace`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


SOURCE_FEATURES = (
    "x_coord",
    "y_coord",
    "z_coord",
    "normal_x",
    "normal_y",
    "normal_z",
    "cp",
    "surface_area",
    "cf_x",
    "cf_y",
    "cf_z",
    "mach",
    "aoa_deg",
    "aileron_inboard_deg",
    "aileron_outboard_deg",
    "elevator_deg",
    "htp_deg",
)

TARGET_FEATURES = (
    "x_coord",
    "y_coord",
    "z_coord",
    "cp",
    "cf_x",
    "cf_y",
    "cf_z",
    "mach",
    "aoa_deg",
    "aileron_inboard_deg",
    "aileron_outboard_deg",
    "elevator_deg",
    "htp_deg",
    "normal_x",
    "normal_y",
    "normal_z",
    "surface_area",
)

ROW_ORDER = tuple(SOURCE_FEATURES.index(name) for name in TARGET_FEATURES)


def _decode_names(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    )


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def _dataset_options(source: h5py.Dataset) -> dict[str, object]:
    options: dict[str, object] = {}
    if source.chunks is not None:
        options["chunks"] = source.chunks
    if source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    if source.scaleoffset is not None:
        options["scaleoffset"] = source.scaleoffset
    return options


def _is_feature_vector(path: str, source: h5py.Dataset) -> bool:
    if source.ndim < 1 or source.shape[0] != len(SOURCE_FEATURES):
        return False
    normalized = path.replace("\\", "/")
    return (
        normalized.endswith("/feature_names")
        or "/normalization_params/" in normalized
        or ("/metadata/feature_" in normalized and "/data/" in normalized)
    )


def _copy_tree(
    source_group: h5py.Group,
    destination_group: h5py.Group,
    *,
    path_prefix: str,
) -> None:
    _copy_attrs(source_group.attrs, destination_group.attrs)
    for name, source_obj in source_group.items():
        object_path = f"{path_prefix}/{name}"
        if isinstance(source_obj, h5py.Group):
            child = destination_group.create_group(name)
            _copy_tree(source_obj, child, path_prefix=object_path)
            continue

        if _is_feature_vector(object_path, source_obj):
            values = source_obj[...][list(ROW_ORDER), ...]
            destination = destination_group.create_dataset(
                name, data=values, dtype=source_obj.dtype, **_dataset_options(source_obj)
            )
            _copy_attrs(source_obj.attrs, destination.attrs)
        else:
            source_group.copy(source_obj, destination_group, name=name)


def _copy_nodal_data(source: h5py.Dataset, destination_group: h5py.Group) -> None:
    expected_rows = len(SOURCE_FEATURES)
    if source.ndim != 3 or source.shape[0] != expected_rows:
        raise ValueError(
            f"{source.name}: expected nodal_data [{expected_rows}, T, N], got {source.shape}"
        )
    destination = destination_group.create_dataset(
        "nodal_data",
        shape=source.shape,
        dtype=source.dtype,
        **_dataset_options(source),
    )
    _copy_attrs(source.attrs, destination.attrs)
    for destination_row, source_row in enumerate(ROW_ORDER):
        destination[destination_row, :, :] = source[source_row, :, :]


def _validate_source(source: h5py.File) -> None:
    if "data" not in source or "metadata/feature_names" not in source:
        raise ValueError("source must contain data/{sample}/nodal_data and metadata/feature_names")
    actual = _decode_names(source["metadata/feature_names"][...])
    if actual != SOURCE_FEATURES:
        raise ValueError(
            "source feature order does not match the expected ex3 schema:\n"
            f"  expected={SOURCE_FEATURES}\n  actual={actual}"
        )


def _validate_output(source_path: Path, output_path: Path) -> None:
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r") as output:
        names = _decode_names(output["metadata/feature_names"][...])
        if names != TARGET_FEATURES:
            raise ValueError(f"output feature order mismatch: {names}")
        if dict(source.attrs) != dict(output.attrs):
            raise ValueError("root attributes changed during copy")

        source_ids = sorted(source["data"].keys(), key=lambda value: int(value))
        output_ids = sorted(output["data"].keys(), key=lambda value: int(value))
        if source_ids != output_ids:
            raise ValueError("sample IDs changed during copy")

        # The source stores one mesh_edge object and hard-links it into every
        # sample. Preserve that deduplication instead of materializing the same
        # ~7 MB edge array once per case.
        source_edge_addresses = {
            h5py.h5o.get_info(source[f"data/{sample_id}/mesh_edge"].id).addr
            for sample_id in source_ids
        }
        output_edge_addresses = {
            h5py.h5o.get_info(output[f"data/{sample_id}/mesh_edge"].id).addr
            for sample_id in output_ids
        }
        if len(source_edge_addresses) != len(output_edge_addresses):
            raise ValueError(
                "mesh_edge hard-link topology changed during copy: "
                f"source objects={len(source_edge_addresses)}, "
                f"output objects={len(output_edge_addresses)}"
            )

        probe_ids = sorted({source_ids[0], source_ids[len(source_ids) // 2], source_ids[-1]}, key=int)
        for sample_id in probe_ids:
            source_data = source[f"data/{sample_id}/nodal_data"]
            output_data = output[f"data/{sample_id}/nodal_data"]
            if source_data.shape != output_data.shape:
                raise ValueError(f"sample {sample_id}: nodal_data shape changed")
            node_indices = sorted({0, source_data.shape[2] // 2, source_data.shape[2] - 1})
            for output_row, source_row in enumerate(ROW_ORDER):
                expected = source_data[source_row, :, node_indices]
                actual = output_data[output_row, :, node_indices]
                if not np.array_equal(expected, actual):
                    raise ValueError(
                        f"sample {sample_id}: row {source_row}->{output_row} failed value validation"
                    )


def reorder_file(source_path: Path, output_path: Path, *, replace: bool = False) -> None:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different files")
    if output_path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with h5py.File(source_path, "r") as source:
            _validate_source(source)
            with h5py.File(partial_path, "w") as output:
                _copy_attrs(source.attrs, output.attrs)
                shared_edge_paths: dict[int, str] = {}
                for name, source_obj in source.items():
                    if name != "data":
                        if isinstance(source_obj, h5py.Group):
                            destination_group = output.create_group(name)
                            _copy_tree(source_obj, destination_group, path_prefix=f"/{name}")
                        else:
                            source.copy(source_obj, output, name=name)
                        continue

                    output_data = output.create_group("data")
                    _copy_attrs(source_obj.attrs, output_data.attrs)
                    sample_ids = sorted(source_obj.keys(), key=lambda value: int(value))
                    for index, sample_id in enumerate(sample_ids, start=1):
                        source_sample = source_obj[sample_id]
                        output_sample = output_data.create_group(sample_id)
                        _copy_attrs(source_sample.attrs, output_sample.attrs)
                        for child_name, child in source_sample.items():
                            if child_name == "nodal_data":
                                _copy_nodal_data(child, output_sample)
                            elif isinstance(child, h5py.Group):
                                child_output = output_sample.create_group(child_name)
                                _copy_tree(
                                    child,
                                    child_output,
                                    path_prefix=f"/data/{sample_id}/{child_name}",
                                )
                            elif child_name == "mesh_edge":
                                address = h5py.h5o.get_info(child.id).addr
                                existing_path = shared_edge_paths.get(address)
                                if existing_path is not None:
                                    output_sample[child_name] = output[existing_path]
                                else:
                                    source_sample.copy(child, output_sample, name=child_name)
                                    shared_edge_paths[address] = f"/data/{sample_id}/{child_name}"
                            else:
                                source_sample.copy(child, output_sample, name=child_name)
                        print(f"[{index:3d}/{len(sample_ids):3d}] copied sample {sample_id}", flush=True)

        _validate_output(source_path, partial_path)
        os.replace(partial_path, output_path)
        print(f"validated and wrote {output_path}")
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source ex3 HDF5")
    parser.add_argument("output", type=Path, help="new reordered HDF5")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing derived output after validation",
    )
    args = parser.parse_args()
    reorder_file(args.source, args.output, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
