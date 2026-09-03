"""Marching-cubes surface -> CAE tetrahedral volume mesh via gmsh.

The recipe recorded in GEOMETRY_UPGRADE_MESHING_SEMANTIC_2026-08.md routes MC
output through `classifySurfaces(..., forReparametrization=True)` +
`createGeometry()`. That path is **not safe to run unattended**: on shapes
carrying a near-degenerate sliver (here one face of area 1e-8 against a median
of 1e-4) gmsh's reparametrization recurses without bound -- observed splitting
the same 3-triangle patch past level 557,000, i.e. a hang no Python-level
timeout can interrupt. Deleting the sliver stops the hang but opens a hole, so
`generate(3)` then yields zero tets.

This module takes the reparametrization-free route instead:

    decimate -> merge -> removeDuplicateNodes
    -> classifySurfaces(angle, forReparametrization=False) -> createTopology
    -> surface loop -> volume -> generate(3)

Because nothing is reparametrized, the trap against decimating first no longer
applies -- and decimation is now *useful*: it removes the slivers, repairs the
MC surface (measured watertight after, 0.1% volume error at 12k faces), and is
what sets the resulting tet count. Note the corollary: gmsh keeps the surface it
is handed and never remeshes it, so `mesh_size_max` sizes the *interior* only.
`target_faces` is the knob that moves the tet count (12k -> 24.8k tets,
25k -> 61.2k at an unchanged `mesh_size_max`); a coarse surface stays coarse. The other two traps still hold and are baked
in below: `MeshSizeFromCurvature` stays 0, and second-order elements need
`ElementOrder = 2` before `generate(3)` plus `HighOrderOptimize = 2`.
"""

import os
import tempfile

import numpy as np

CLASSIFY_ANGLES = (40.0, 60.0, 30.0)


class MeshingError(RuntimeError):
    pass


def prepare_surface(mesh, target_faces=12000):
    """Decimate the raw MC surface to the face budget that sets the tet count."""
    surface = mesh.copy()
    if target_faces and len(surface.faces) > target_faces:
        surface = surface.simplify_quadric_decimation(face_count=int(target_faces))
    surface.remove_unreferenced_vertices()
    if len(surface.faces) == 0:
        raise MeshingError('surface decimation removed every face')
    return surface


# gmsh 4.15.2 fails to load a *binary* STL below roughly a thousand facets
# ("Error loading ..."); the same geometry as ASCII loads fine, and both load
# fine once the mesh is large. A decimated MC surface is far above that, but a
# collapsed design is not -- and an unreadable file is a far more confusing
# failure than a genuinely unmeshable one.
ASCII_STL_FACE_LIMIT = 2000


def _export_stl(surface, path):
    if len(surface.faces) < ASCII_STL_FACE_LIMIT:
        import trimesh
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(trimesh.exchange.stl.export_stl_ascii(surface))
    else:
        surface.export(path)


def tet_mesh_from_surface(mesh, mesh_size_max=0.05, target_faces=12000,
                          angles=CLASSIFY_ANGLES, verbose=False, second_order=False):
    """Tetrahedralize a closed surface mesh. Returns (nodes[N,3], tets[E,4|10], info)."""
    surface = prepare_surface(mesh, target_faces)
    stl_path = os.path.join(tempfile.mkdtemp(prefix='sdfflow_mesh_'), 'shape.stl')
    _export_stl(surface, stl_path)

    last_error = None
    for angle in angles:
        try:
            nodes, tets, info = _attempt(stl_path, angle, mesh_size_max, verbose, second_order)
            info.update(classify_angle=angle,
                        surface_faces=int(len(surface.faces)),
                        surface_watertight=bool(surface.is_watertight))
            return nodes, tets, info
        except Exception as exc:  # gmsh raises plain Exception for PLC/topology errors
            last_error = exc
            _finalize_quietly()
    raise MeshingError(f'gmsh failed at every classify angle: {last_error}')


def _finalize_quietly():
    import gmsh
    try:
        gmsh.finalize()
    except Exception:
        pass


def _attempt(stl_path, angle, mesh_size_max, verbose, second_order):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
        gmsh.option.setNumber('General.Verbosity', 5 if verbose else 0)
        gmsh.merge(stl_path)
        gmsh.model.mesh.removeDuplicateNodes()
        # forReparametrization=False: keep the discrete surface, skip the hang.
        gmsh.model.mesh.classifySurfaces(np.deg2rad(angle), True, False, np.deg2rad(180.0))
        gmsh.model.mesh.createTopology()

        surfaces = [s[1] for s in gmsh.model.getEntities(2)]
        if not surfaces:
            raise MeshingError('classifySurfaces produced no surfaces')
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)      # trap: hangs otherwise
        gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_max)
        gmsh.option.setNumber('Mesh.Optimize', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        if second_order:                                            # trap: order before gen
            gmsh.option.setNumber('Mesh.ElementOrder', 2)
            gmsh.option.setNumber('Mesh.HighOrderOptimize', 2)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        remap = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        remap[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))

        etype = 11 if second_order else 4                            # tet10 / tet4
        elem_tags, node_tags = gmsh.model.mesh.getElementsByType(etype)
        if len(elem_tags) == 0:
            raise MeshingError('no tetrahedra generated')
        npe = 10 if second_order else 4
        tets = remap[np.asarray(node_tags, dtype=np.int64).reshape(-1, npe)]

        quality = np.asarray(gmsh.model.mesh.getElementQualities(elem_tags, 'minSICN'),
                             dtype=np.float64)
        info = {
            'num_nodes': int(coords.shape[0]),
            'num_tets': int(tets.shape[0]),
            'min_sicn': float(quality.min()),
            'p1_sicn': float(np.percentile(quality, 1)),
            'median_sicn': float(np.median(quality)),
            'negative_jacobians': int((quality <= 0).sum()),
            'second_order': bool(second_order),
        }
        return coords, tets, info
    finally:
        gmsh.finalize()
