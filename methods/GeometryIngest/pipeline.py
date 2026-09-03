"""Shared ingest pipeline, driven by either front end (argparse cli.py or the
launcher entrypoint main.py). Keeps the read -> graph/pointcloud -> HDF5 flow in
one place so both entrypoints behave identically.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from . import readers, to_graph, to_pointcloud, writer

_MESH_GLOBS = ("*.stl", "*.ply", "*.obj", "*.off",
               "*.step", "*.stp", "*.igs", "*.iges", "*.brep")


@dataclass
class IngestParams:
    reader: str = "auto"          # auto | trimesh | gmsh
    volume: bool = True           # True -> gmsh tet mesh (dim 3); False -> surface
    emit: tuple[str, ...] = ("graph",)   # subset of {graph, pointcloud}
    num_fields: int = 3           # zero-filled solution-field rows after xyz
    num_points: int = 0           # point-cloud resample size (0 = keep all nodes)
    resample: str = "fps"         # fps | random
    mesh_size_max: float = 0.0    # gmsh Mesh.MeshSizeMax (0 = gmsh default)
    mesh_size_min: float = 0.0
    seed: int = 42
    limit: int = 0                # cap number of inputs (0 = all)


def gather_paths(root: str, limit: int) -> list[str]:
    if os.path.isfile(root):
        return [root]
    paths = sorted(
        p for pattern in _MESH_GLOBS
        for p in glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if not os.path.basename(p).startswith("._")  # skip AppleDouble junk
    )
    return paths[:limit] if limit > 0 else paths


def process_one(path: str, params: IngestParams) -> dict:
    if params.reader == "gmsh":
        raw = readers.read_gmsh(path, volume=params.volume,
                                size_max=params.mesh_size_max, size_min=params.mesh_size_min)
    elif params.reader == "trimesh":
        raw = readers.read_trimesh(path)
    else:
        raw = readers.read_auto(path, params.volume, params.mesh_size_max, params.mesh_size_min)

    raw["source"] = os.path.basename(path)
    if "graph" in params.emit:
        raw["mesh_edge"] = to_graph.edges_from_connectivity(raw["conn"], raw["nodes_per_elem"])
    if "pointcloud" in params.emit:
        raw["pc_coords"] = to_pointcloud.resample(
            raw["coords"], params.num_points, method=params.resample, seed=params.seed)
    return raw


def summarize(s: dict) -> str:
    c = s["coords"]
    bbox = c.max(axis=0) - c.min(axis=0)
    wt = s["watertight"]
    wt_str = "n/a" if wt is None else ("yes" if wt else "NO")
    edge_str = f"edges={s['mesh_edge'].shape[1]:>9d}  " if s.get("mesh_edge") is not None else ""
    pc_str = f"pc={s['pc_coords'].shape[0]:>7d}  " if s.get("pc_coords") is not None else ""
    return (f"nodes={c.shape[0]:>8d}  cells={s['num_cells']:>8d}  {edge_str}{pc_str}"
            f"dim={s['dim']}  watertight={wt_str}  "
            f"bbox=[{bbox[0]:.3g},{bbox[1]:.3g},{bbox[2]:.3g}]")


def pointcloud_output_path(graph_out: str) -> str:
    stem, ext = os.path.splitext(graph_out)
    return f"{stem}_pointcloud{ext or '.h5'}"


def run_ingest(paths: list[str], params: IngestParams, output: str | None,
               dry_run: bool) -> int:
    """Process every path, then write the requested emit(s). Returns an exit code."""
    samples = []
    for path in paths:
        try:
            s = process_one(path, params)
        except Exception as exc:  # report and continue over a batch
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        print(f"  {os.path.basename(path):<28s} {summarize(s)}")
        samples.append(s)

    if not samples:
        print("Nothing ingested.")
        return 1
    if dry_run:
        print(f"\n[inspect] {len(samples)} sample(s) ok; no file written.")
        return 0

    if "graph" in params.emit:
        writer.write_contract(output, samples, params.num_fields, with_edges=True)
        print(f"\nWrote graph -> {output}")
    if "pointcloud" in params.emit:
        pc_path = pointcloud_output_path(output)
        pc_samples = [{"coords": s["pc_coords"], "num_cells": 0, "source": s["source"]}
                      for s in samples]
        writer.write_contract(pc_path, pc_samples, params.num_fields, with_edges=False)
        print(f"Wrote point cloud -> {pc_path}")
    return 0
