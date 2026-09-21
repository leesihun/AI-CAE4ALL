"""Renders original-vs-deformed shell geometry for the first 10 training shapes
(one representative draw each, draw_000) using the real OpenRadioss ANIM output.

Pipeline: drawA001 (t=0, original) / drawA002 (t=final, deformed) -> vendor
`anim_to_vtk_win64.exe` converter -> ASCII VTK -> pyvista.

The deformed panel is drawn with the displacement field WARPED (exaggerated)
by a per-shape scale factor so the circumferential buckling waves (mode
number K) are visible -- true-scale post-buckling radial deflection is only
a few percent of the shell radius. The warp factor is reported in each
subplot's title; the color scale (radial displacement r(t=final) - r(t=0))
is not rescaled, only the geometry is.

Run once: python gen_sample_geometries.py
Writes sample_geometries_orig_deformed.png into this directory.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
PROD_ROOT = REPO_ROOT / "buckling" / "runs" / "production"
ANIM_TO_VTK = pathlib.Path("C:/tmp/OpenRadioss/win64/OpenRadioss/exec/anim_to_vtk_win64.exe")

N_SHAPES = 10
DRAW = "draw_000"
TARGET_WARP_FRAC = 0.12  # warped max |radial disp| as a fraction of R, for visual clarity


def shape_dirs(n):
    dirs = sorted(
        d for d in PROD_ROOT.iterdir()
        if d.is_dir() and d.name[:3].isdigit() and int(d.name[:3]) < n
    )
    return dirs


def anim_to_vtk(anim_path, out_path):
    with open(out_path, "w") as f:
        subprocess.run([str(ANIM_TO_VTK), str(anim_path)], cwd=anim_path.parent,
                        stdout=f, check=True)


def load_pair(shape_dir, tmpdir):
    draw_dir = shape_dir / DRAW
    a001, a002 = draw_dir / "drawA001", draw_dir / "drawA002"
    v001 = tmpdir / f"{shape_dir.name}_a001.vtk"
    v002 = tmpdir / f"{shape_dir.name}_a002.vtk"
    anim_to_vtk(a001, v001)
    anim_to_vtk(a002, v002)
    g0, g1 = pv.read(v001), pv.read(v002)
    v001.unlink(missing_ok=True)
    v002.unlink(missing_ok=True)
    return g0, g1


def build_panels(g0, g1, R):
    pts0, pts1 = g0.points, g1.points
    disp = pts1 - pts0
    r0 = np.hypot(pts0[:, 0], pts0[:, 1])
    r1 = np.hypot(pts1[:, 0], pts1[:, 1])
    dr = r1 - r0

    lo, hi = np.percentile(dr, [1, 99])
    vmax = max(abs(lo), abs(hi), 1e-9)
    scale = TARGET_WARP_FRAC * R / vmax

    g1w = g1.copy()
    g1w.points = pts0 + scale * disp
    g1w.point_data["radial_disp"] = dr

    quad0 = np.where(g0.celltypes == 9)[0]
    quad1 = np.where(g1w.celltypes == 9)[0]
    shell0 = g0.extract_cells(quad0)
    shell1 = g1w.extract_cells(quad1)
    return shell0, shell1, vmax, scale


def main():
    dirs = shape_dirs(N_SHAPES)
    assert len(dirs) == N_SHAPES, f"expected {N_SHAPES} shapes, found {len(dirs)}: {dirs}"

    rows = (N_SHAPES + 1) // 2
    pl = pv.Plotter(shape=(rows, 4), off_screen=True, window_size=(2000, 420 * rows), border=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        for i, shape_dir in enumerate(dirs):
            summary = json.loads((shape_dir / DRAW / "summary.json").read_text())
            name = shape_dir.name  # e.g. 000_train_rt165.2_lr1.251
            _, _, rt_tag, lr_tag = name.split("_")
            rt = rt_tag[2:]
            lr = lr_tag[2:]
            K = summary["K"]

            g0, g1 = load_pair(shape_dir, tmpdir)
            R = np.hypot(g0.points[:, 0], g0.points[:, 1]).mean()
            shell0, shell1, vmax, scale = build_panels(g0, g1, R)

            row = i // 2
            col_base = (i % 2) * 2

            pl.subplot(row, col_base)
            pl.add_mesh(shell0, color="#c9ccd1", show_edges=False, smooth_shading=True)
            pl.add_text(f"#{i}  R/t={rt}  L/R={lr}\noriginal", font_size=9)
            pl.camera_position = "iso"
            pl.camera.azimuth = 25
            pl.camera.elevation = 15

            pl.subplot(row, col_base + 1)
            pl.add_mesh(shell1, scalars="radial_disp", cmap="RdBu_r", clim=[-vmax, vmax],
                        show_edges=False, smooth_shading=True, show_scalar_bar=False)
            pl.add_text(f"K={K} waves\ndeformed, warp x{scale:.0f}", font_size=9)
            pl.camera_position = "iso"
            pl.camera.azimuth = 25
            pl.camera.elevation = 15

            print(f"[{i+1}/{N_SHAPES}] {name}: K={K}, warp x{scale:.0f}", file=sys.stderr)

    out = HERE / "sample_geometries_orig_deformed.png"
    pl.screenshot(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
