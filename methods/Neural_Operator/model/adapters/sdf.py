"""SDF sourcing and validation (IMPLEMENTATION_PLAN.md section 7.5).

No field in the shipped HDF5 files provides a signed distance function, so
`sdf_source none` remains the default. `dataset` and `sidecar` read a stored
field; `mesh` derives one from `mesh_edge` and the reference coordinates at
load time, so datasets that ship no SDF can still supply one.

`sdf_from_mesh` signs all eleven shipped arms. It takes the exact route where
one exists -- planar face tracing in 2-D, the reconstructed shell in 3-D -- and
falls back to an alpha complex of the node cloud where the edge graph carries
no boundary (a 3-D volume mesh, or a k-nearest-neighbour graph over a point
cloud). The fallback is an *estimate* of the boundary, not a reading of it; see
that module's docstring for which arm takes which route and what the estimate
assumes. Each route still verifies itself before it is used, so a mesh whose
geometry admits no boundary at all still raises rather than returning a sign
that is quietly inverted.
"""

import h5py
import numpy as np

from general_modules.dataset_stats import resolve_active_axes
from model.adapters.sdf_from_mesh import signed_distance_from_mesh

# Reconstructing the boundary and evaluating it at every node costs 0.02-0.3 s
# on the small arms, 4-19 s on the two largest exact ones (ex3_mid, ex3_full),
# and 29 s on the worst case, ex2_dynamic_contact -- 200k nodes whose boundary
# has to be estimated from a 1.3M-tetrahedron Delaunay tessellation, 17 s of
# which is the tessellation itself. It depends only on the reference
# coordinates, which are static for the whole trajectory, so recomputing it per
# epoch or per timestep would dominate the input pipeline; it is memoized per
# (file, sample).
_MESH_SDF_CACHE = {}


def mesh_sdf_cache_clear() -> None:
    """Drop the memoized per-sample reconstructions (used by the tests)."""
    _MESH_SDF_CACHE.clear()


def _mesh_positions_and_edges(h5_file: str, sample_id: int,
                              dimension_tolerance: float = 1e-4):
    """Reference coordinates (active axes only) and edges for one sample."""
    with h5py.File(h5_file, 'r') as f:
        group = f[f'data/{sample_id}']
        coords = np.asarray(group['nodal_data'][0:3, 0, :], dtype=np.float64).T
        edges = np.asarray(group['mesh_edge'][:], dtype=np.int64)
    # the same axis rule the rest of the pipeline uses, so a 2-D dataset is
    # 2-D here too instead of a degenerate 3-D one with a zero-extent axis
    active, _ = resolve_active_axes(coords.min(axis=0), coords.max(axis=0),
                                    dimension_tolerance)
    return coords[:, list(active)], edges


def mesh_sdf_at_nodes(h5_file: str, sample_id: int,
                      dimension_tolerance: float = 1e-4) -> np.ndarray:
    """Signed distance evaluated at the sample's own nodes, memoized.

    On a 2-D domain mesh this is the distance to the wall, which is a real
    feature. On a 3-D *shell* every node lies on the surface, so it is ~0 by
    construction and carries no geometry -- there the useful query points are
    off-surface ones (e.g. a latent grid), which `signed_distance_from_mesh`
    accepts directly.
    """
    key = (str(h5_file), int(sample_id))
    if key not in _MESH_SDF_CACHE:
        positions, edges = _mesh_positions_and_edges(h5_file, sample_id,
                                                     dimension_tolerance)
        _MESH_SDF_CACHE[key] = signed_distance_from_mesh(positions, edges, positions)
    return _MESH_SDF_CACHE[key]


def sdf_available(h5_file: str, sample_id: int, source: str, sidecar_path: str = None) -> bool:
    if source == 'none':
        return False
    if source == 'dataset':
        with h5py.File(h5_file, 'r') as f:
            return f'data/{sample_id}/sdf' in f
    if source == 'sidecar':
        if not sidecar_path or sidecar_path == 'none':
            return False
        with h5py.File(sidecar_path, 'r') as f:
            return f'data/{sample_id}/sdf' in f
    if source == 'mesh':
        # "available" here means the boundary can actually be reconstructed
        # or estimated and signed; a geometry that admits neither reports False
        # and the caller raises with the reconstruction's own diagnosis.
        try:
            mesh_sdf_at_nodes(h5_file, sample_id)
        except (ValueError, ImportError):
            return False
        return True
    raise ValueError(f"Unknown sdf_source '{source}'")


def load_sdf(h5_file: str, sample_id: int, source: str, sidecar_path: str = None,
             time_idx: int = None) -> np.ndarray:
    """Return [N] float32 SDF values for one sample, or raise if unavailable/invalid."""
    if source == 'dataset':
        with h5py.File(h5_file, 'r') as f:
            key = f'data/{sample_id}/sdf'
            if key not in f:
                raise ValueError(
                    f"sdf_source=dataset but '{key}' is missing in {h5_file}. "
                    "Set sdf_source none or add the field (IMPLEMENTATION_PLAN.md section 4.1)."
                )
            arr = f[key][:]
    elif source == 'sidecar':
        if not sidecar_path or sidecar_path == 'none':
            raise ValueError("sdf_source=sidecar requires sdf_sidecar to name a file.")
        with h5py.File(sidecar_path, 'r') as f:
            key = f'data/{sample_id}/sdf'
            if key not in f:
                raise ValueError(f"Sample {sample_id} missing from SDF sidecar '{sidecar_path}'.")
            arr = f[key][:]
    elif source == 'mesh':
        arr = mesh_sdf_at_nodes(h5_file, sample_id)
    else:
        raise ValueError(f"Unknown sdf_source '{source}'")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:  # [T, N] or [1, N]
        idx = time_idx if (time_idx is not None and arr.shape[0] > 1) else 0
        arr = arr[idx]
    if arr.ndim != 1:
        raise ValueError(f"Sample {sample_id}: SDF must reduce to shape [N], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Sample {sample_id}: SDF contains non-finite values.")
    return arr
