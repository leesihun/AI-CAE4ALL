"""Lazy dependency bootstrap.

If a required third-party module (gmsh, trimesh, ...) is not importable when it is
first needed, install it **offline** from the wheels committed under ``wheels/``
(no PyPI index), then retry. This is what lets an airgapped clone run the volume
path without a separate install step: the platform wheels travel with the repo.

Notes:
* ``--no-deps`` is used deliberately -- numpy/h5py are already present in the
  method environments, and the bundled gmsh wheel is self-contained.
* If no bundled wheel matches this platform/Python (e.g. the Linux gmsh wheel on a
  Windows box), pip fails and a clear ImportError is raised with a manual-install
  hint instead of a cryptic traceback.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

WHEELS_DIR = Path(__file__).resolve().parent / "wheels"


def ensure(module_name: str, package: str | None = None):
    """Import ``module_name``, installing it from bundled wheels if necessary."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    package = package or module_name
    if WHEELS_DIR.is_dir():
        print(f"geometry_ingest: '{module_name}' not installed; installing from "
              f"bundled wheels ({WHEELS_DIR}) ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index",
             "--find-links", str(WHEELS_DIR), "--no-deps", package],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            reason = tail[-1] if tail else "no compatible wheel found"
            raise ImportError(
                f"Could not install '{package}' offline from {WHEELS_DIR}: {reason}. "
                f"Install it manually (pip install {package}) or add a wheel matching "
                f"this platform/Python.")
        importlib.invalidate_caches()

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"'{module_name}' is required but unavailable, and no bundled wheel in "
            f"{WHEELS_DIR} matched this environment. Install it with: pip install {package}"
        ) from exc
