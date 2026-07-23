"""Surface-mesh cleanup / watertight repair.

Ported from ``Geometry_generation/build_dataset.py``'s repair path. This matters
mainly for the *volume* path: gmsh tet meshing needs a watertight, non-degenerate
boundary or it produces slivers (or fails). For dirty STL, run this first, export
a clean surface, then hand that to gmsh for volume meshing.

Kept module-level and dependency-lazy so ProcessPoolExecutor (Windows spawn) can
pickle it.
"""

from __future__ import annotations


def repair_surface(mesh, fill: bool = True):
    """Best-effort make a trimesh surface watertight. Returns the (maybe) fixed mesh.

    Mirrors the conservative sequence already trusted in the SDFFlow builder:
    drop unreferenced verts, merge coincident verts, dedupe/clean faces, fix
    normals, fill small holes, and finally fall back to pymeshfix if present.
    """
    from .deps import ensure

    trimesh = ensure("trimesh")  # installs from bundled wheels/ if missing

    if fill and not mesh.is_watertight:
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fill_holes(mesh)

    if fill and not mesh.is_watertight:
        try:
            import pymeshfix

            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(joincomp=True, remove_smallest_components=False)
            mesh = trimesh.Trimesh(vertices=meshfix.points, faces=meshfix.faces, process=True)
            trimesh.repair.fix_normals(mesh, multibody=True)
        except (ImportError, RuntimeError, ValueError):
            pass  # pymeshfix is optional; report watertightness downstream

    return mesh
