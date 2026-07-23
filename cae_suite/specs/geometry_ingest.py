from __future__ import annotations

from ..diagnostics import Severity
from .base import (
    MethodSpec,
    PathKind,
    PathRule,
    SpecValidationContext,
    as_list,
    integer,
    numeric,
    validate_common_values,
    validate_nonnegative_int_fields,
)


# The launcher validates against these; an unlisted key becomes CFG-UNKNOWN-001.
GEOMETRY_INGEST_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "log_file_dir",
        "input_geometry", "output_dataset",
        "reader", "mesh_type", "emit",
        "num_fields", "num_points", "resample_method",
        "mesh_size_max", "mesh_size_min",
        "seed", "limit",
    }
)

_READERS = {"auto", "trimesh", "gmsh"}
_MESH_TYPES = {"surface", "volume"}
_EMIT = {"graph", "pointcloud"}
_RESAMPLE = {"fps", "random"}


def validate_geometry_ingest(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values

    reader = str(values.get("reader", "auto")).lower()
    if reader not in _READERS:
        ctx.add("GEOM-READER-001", Severity.ERROR,
                "reader must be auto, trimesh, or gmsh.", field_name="reader")

    mesh_type = str(values.get("mesh_type", "volume")).lower()
    if mesh_type not in _MESH_TYPES:
        ctx.add("GEOM-MESH-001", Severity.ERROR,
                "mesh_type must be surface or volume.", field_name="mesh_type")

    emit = [str(e).lower() for e in as_list(values.get("emit", "graph"))]
    bad = sorted(set(emit) - _EMIT)
    if bad:
        ctx.add("GEOM-EMIT-001", Severity.ERROR,
                f"emit accepts graph and/or pointcloud; got unknown: {bad}.", field_name="emit")

    resample = str(values.get("resample_method", "fps")).lower()
    if resample not in _RESAMPLE:
        ctx.add("GEOM-RESAMPLE-001", Severity.ERROR,
                "resample_method must be fps or random.", field_name="resample_method")

    validate_nonnegative_int_fields(
        ctx, ("num_fields", "num_points", "seed", "limit"), "GEOM-INT-001")

    for name in ("mesh_size_max", "mesh_size_min"):
        if name in values:
            value = numeric(values[name])
            if value is None or value < 0:
                ctx.add("GEOM-SIZE-001", Severity.ERROR,
                        f"{name} must be a nonnegative number; got {values[name]!r}.", field_name=name)

    # trimesh reads only pre-triangulated surfaces; it cannot tet-mesh a volume.
    if reader == "trimesh" and mesh_type == "volume":
        ctx.add("GEOM-READER-002", Severity.ERROR,
                "reader=trimesh cannot produce a volume mesh; use reader=gmsh (or auto) for mesh_type=volume.",
                field_name="mesh_type")

    # gmsh is imported lazily at runtime; surface trimesh runs do not need it.
    if mesh_type == "volume" or reader == "gmsh":
        ctx.add("GEOM-GMSH-001", Severity.NOTICE,
                "Volume/CAD meshing requires the 'gmsh' package in the method environment (pip install gmsh).",
                field_name="mesh_type")

    if "pointcloud" in emit and integer(values.get("num_points", 0)) == 0:
        ctx.add("GEOM-PC-001", Severity.NOTICE,
                "emit includes pointcloud with num_points=0; the full node set is used as the point cloud.",
                field_name="num_points")


def build_geometry_ingest_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="geometry_ingest",
        display_name="Geometry Ingest",
        model_ids=("geometry_ingest",),
        repository="dataset/geometry_ingest",
        entrypoint="main.py",
        valid_modes=("ingest", "inspect"),
        known_keys=GEOMETRY_INGEST_KEYS,
        # This tool needs no GPU; drop gpu_ids from the common required set.
        required_common=frozenset({"model", "mode"}),
        required_by_mode={
            "ingest": frozenset({"input_geometry", "output_dataset"}),
            "inspect": frozenset({"input_geometry"}),
        },
        recommended_by_mode={"ingest": frozenset({"reader", "mesh_type", "emit"})},
        defaults={
            "reader": "auto", "mesh_type": "volume", "emit": "graph",
            "num_fields": 3, "num_points": 0, "resample_method": "fps", "seed": 42,
        },
        path_rules=(
            PathRule("input_geometry", PathKind.INPUT_DIR),
            PathRule("output_dataset", PathKind.OUTPUT_FILE, frozenset({"ingest"})),
        ),
        validators=(validate_geometry_ingest,),
        # numpy/h5py/trimesh are always needed; gmsh is checked lazily at runtime.
        import_modules=("numpy", "h5py", "trimesh"),
        dataset_kind=None,        # input is geometry files, not an HDF5 dataset
        native_probe=False,       # no native flat-config validator to probe
    )
