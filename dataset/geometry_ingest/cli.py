"""geometry_ingest developer CLI (argparse front end over the shared pipeline).

For launcher-driven runs use ``AI_CAE4ALL_main.py --config <file>`` instead; this
argparse interface is handy for quick, ad-hoc ingests and stats.

Examples:
    # Volume tet mesh from STEP (needs gmsh), graph + resampled point cloud:
    python -m geometry_ingest.cli part.step --volume --emit graph,pointcloud \
        --num-points 8192 --mesh-size-max 5.0 --output out/part.h5

    # Surface mesh from a directory of STL/PLY (trimesh), stats only:
    python -m geometry_ingest.cli ./stl_dir --reader trimesh --dry-run
"""

from __future__ import annotations

import argparse

from .pipeline import IngestParams, gather_paths, pointcloud_output_path, run_ingest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Geometry -> shared mesh HDF5 (volume + point cloud)")
    ap.add_argument("input", help="A geometry file or a directory of them")
    ap.add_argument("--output", default=None, help="Output HDF5 path (graph emit)")
    ap.add_argument("--reader", choices=["auto", "trimesh", "gmsh"], default="auto")
    ap.add_argument("--volume", action="store_true", help="gmsh: volume tet mesh (dim=3)")
    ap.add_argument("--emit", default="graph", help="Comma list of graph,pointcloud (default graph)")
    ap.add_argument("--num-fields", type=int, default=3,
                    help="Zero-filled field rows after xyz (default 3 = ux,uy,uz)")
    ap.add_argument("--num-points", type=int, default=0,
                    help="Point-cloud resample size (0 = keep all nodes)")
    ap.add_argument("--resample", choices=["fps", "random"], default="fps")
    ap.add_argument("--mesh-size-max", type=float, default=0.0, help="gmsh Mesh.MeshSizeMax")
    ap.add_argument("--mesh-size-min", type=float, default=0.0, help="gmsh Mesh.MeshSizeMin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Cap number of inputs (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print stats; do not write")
    args = ap.parse_args(argv)

    emit = tuple(e.strip().lower() for e in args.emit.split(",") if e.strip())
    bad = set(emit) - {"graph", "pointcloud"}
    if bad:
        raise SystemExit(f"--emit accepts graph,pointcloud; got unknown: {sorted(bad)}")
    if not args.dry_run and not args.output:
        raise SystemExit("Provide --output, or pass --dry-run for stats only")

    params = IngestParams(
        reader=args.reader,
        volume=args.volume,
        emit=emit,
        num_fields=args.num_fields,
        num_points=args.num_points,
        resample=args.resample,
        mesh_size_max=args.mesh_size_max,
        mesh_size_min=args.mesh_size_min,
        seed=args.seed,
        limit=args.limit,
    )
    paths = gather_paths(args.input, params.limit)
    if not paths:
        raise SystemExit(f"No geometry files found under {args.input}")
    return run_ingest(paths, params, args.output, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
