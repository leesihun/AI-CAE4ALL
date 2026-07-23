"""Family -> driver dispatch, with the one-family-per-process rule enforced.

Every family folder under `families/` keeps its original internal package
roots (`model.*`, `general_modules.*`) so vendored files needed zero import
rewriting. Because `model`/`general_modules` mean different things per
family, only ONE family may ever be imported into a given Python process:
`load_driver` prepends that family's directory (plus `cae_infer/` itself, so
`common.*` stays reachable) to `sys.path` and raises if a different family
was already loaded in this process.
"""

import importlib
import sys
from pathlib import Path

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # PyInstaller build: families/common ship as raw data (see pyinstaller.spec)
    # under sys._MEIPASS/cae_infer/..., not frozen into the PYZ archive -- see
    # that spec file's comment for why (duplicate `model`/`general_modules`
    # module names across families can't coexist in one frozen import cache).
    CAE_INFER_DIR = Path(sys._MEIPASS) / "cae_infer"
else:
    CAE_INFER_DIR = Path(__file__).resolve().parent
FAMILIES_DIR = CAE_INFER_DIR / "families"

# family key -> subdirectory name under families/
FAMILY_DIRS = {
    "neural_operator": "neural_operator",
    "transolver": "transolver",
    "meshgraphnets": "meshgraphnets",
    "meshgraphnets_v": "meshgraphnets_v",
    "geometry": "geometry",
}

_loaded_family = None


def load_driver(family: str):
    """Import and return `families/<family>/driver.py`'s module, after
    prepending the right directories to `sys.path`. Raises if a different
    family was already loaded in this process (see module docstring)."""
    global _loaded_family
    if family not in FAMILY_DIRS:
        raise ValueError(f"Unknown family '{family}'; expected one of {sorted(FAMILY_DIRS)}.")
    if _loaded_family is not None and _loaded_family != family:
        raise RuntimeError(
            f"cae_infer already loaded family '{_loaded_family}' in this process; "
            f"cannot also load '{family}'. Each family owns the top-level `model`/"
            "`general_modules` module names, so only one may be imported per "
            "process. Run each family in its own process (this is what the CLI "
            "and `infer()` already do -- one checkpoint per invocation)."
        )

    family_dir = FAMILIES_DIR / FAMILY_DIRS[family]
    if not family_dir.is_dir():
        raise FileNotFoundError(f"Family directory not found: {family_dir}")

    for p in (str(CAE_INFER_DIR), str(family_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    driver = importlib.import_module("driver")
    _loaded_family = family
    return driver
