"""Geometry readers: CAD/mesh file -> welded nodes + element connectivity.

Two backends behind one dict-returning interface:

* ``read_trimesh`` -- STL / PLY / OBJ / OFF surface meshes (triangles). Pure
  ``trimesh``; installed today.
* ``read_gmsh``    -- STEP / IGES / BREP / STL via OpenCASCADE. Emits a surface
  triangulation (``volume=False``, dim 2) or a **volume tet mesh**
  (``volume=True``, dim 3). Requires ``gmsh`` (imported lazily).

Returned dict (a "raw mesh"):
    coords          (N, 3) float64   physical-unit node coordinates
    conn            (C, k) int64     element -> node-row indices (k=3 tri, 4 tet)
    nodes_per_elem  int              3 (surface) or 4 (volume)
    num_cells       int
    watertight      bool | None      None when not cheaply known (gmsh volume)
    dim             int              2 or 3
"""

from __future__ import annotations

import numpy as np

from .deps import ensure

_CAD_EXTS = (".step", ".stp", ".igs", ".iges", ".brep")


def read_trimesh(path: str) -> dict:
    trimesh = ensure("trimesh")  # installs from bundled wheels/ if missing

    mesh = trimesh.load(path, force="mesh", process=True)  # process=True welds vertices
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("empty mesh after load")
    return {
        "coords": verts,
        "conn": faces,
        "nodes_per_elem": 3,
        "num_cells": int(faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "dim": 2,
    }


def read_gmsh(path: str, volume: bool = True,
              size_max: float = 0.0, size_min: float = 0.0) -> dict:
    """STEP/IGES/BREP/STL -> surface (dim2) or volume tet (dim3) mesh.

    Deterministic: single-threaded, terminal silenced. gmsh node tags are
    1-based and non-contiguous, so remap to dense 0..N-1 rows with a sorted
    ``searchsorted`` (order-independent, vectorised).
    """
    gmsh = ensure("gmsh")  # installs from bundled wheels/ if missing

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)  # reproducible meshing
        gmsh.open(path)
        if size_max > 0:
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        if size_min > 0:
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)

        dim = 3 if volume else 2
        gmsh.model.mesh.generate(dim)

        node_tags, coord, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coord, dtype=np.float64).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)

        etype, npe = (4, 4) if volume else (2, 3)  # gmsh element types: 4=tet4, 2=tri3
        _, elem_nodes = gmsh.model.mesh.getElementsByType(etype)
        elem_nodes = np.asarray(elem_nodes, dtype=np.int64).reshape(-1, npe)
        if elem_nodes.size == 0:
            raise ValueError(f"gmsh produced no dim-{dim} elements for {path}")

        order = np.argsort(node_tags)
        pos = np.searchsorted(node_tags[order], elem_nodes.reshape(-1))
        conn = order[pos].reshape(-1, npe)  # tags -> dense row indices

        return {
            "coords": coords,
            "conn": conn,
            "nodes_per_elem": npe,
            "num_cells": int(conn.shape[0]),
            "watertight": None,
            "dim": dim,
        }
    finally:
        gmsh.finalize()


def read_auto(path: str, volume: bool, size_max: float, size_min: float) -> dict:
    """Route by extension: CAD or any volume request -> gmsh; else trimesh."""
    import os

    ext = os.path.splitext(path)[1].lower()
    if ext in _CAD_EXTS or volume:
        return read_gmsh(path, volume=volume, size_max=size_max, size_min=size_min)
    return read_trimesh(path)
