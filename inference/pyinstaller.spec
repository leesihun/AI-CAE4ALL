# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the AI-CAE4ALL inference bundle.

    pyinstaller pyinstaller.spec

Produces a one-folder build at dist/run_inference/ -- copy that whole folder
to hand off; it needs no Python install on the target machine.

## Why families/ and common/ ship as raw `datas`, not frozen code

Every family folder under cae_infer/families/ keeps the SAME top-level module
names (`model`, `general_modules`, `driver`, ...) so the vendored files needed
zero import rewriting from their source repos (see cae_infer/registry.py's
docstring). PyInstaller's frozen import mechanism has one flat sys.modules
namespace -- five different `model` packages cannot coexist there. So
families/ and common/ are shipped as plain files on disk (via `datas`) and
imported at runtime through registry.py's sys.path manipulation, exactly as
they would be from an unfrozen checkout. This also means PyInstaller's static
analysis never sees what those files import -- every third-party package used
anywhere in a family driver (torch, torch_geometric, scipy, scikit-image,
trimesh, h5py) is listed explicitly below via collect_all, since there is no
import-graph tracing to catch it automatically.
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ("torch", "torch_geometric", "scipy", "skimage", "trimesh", "h5py"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# numpy is a dependency of everything above and normally auto-collected, but
# list it explicitly too -- it's cheap insurance against a partial hook.
extra_datas, extra_binaries, extra_hidden = collect_all("numpy")
datas += extra_datas
binaries += extra_binaries
hiddenimports += extra_hidden

datas += [
    ("cae_infer/common", "cae_infer/common"),
    ("cae_infer/families", "cae_infer/families"),
]

block_cipher = None

a = Analysis(
    ["run_inference.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "jupyter", "tkinter"],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run_inference",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="run_inference",
)
