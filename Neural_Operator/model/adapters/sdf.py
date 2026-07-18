"""SDF sourcing and validation (IMPLEMENTATION_PLAN.md section 7.5).

No field in the current ex1/ex2/hex HDF5 files provides a signed distance
function, so `sdf_source none` is the default and only path exercised by the
shipped configs. `dataset` and `sidecar` sources are implemented and tested
against synthetic fixtures so a future dataset with real SDF data works
without code changes. Both fail loudly instead of estimating a signed
distance from `mesh_edge` alone, which the plan explicitly forbids.
"""

import h5py
import numpy as np


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
