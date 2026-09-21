"""Exports decimated, indexed mesh data for the first 10 training shapes
(one representative draw each, draw_000) for an interactive Three.js viewer.

Pipeline, per shape:
  drawA001 (t=0, original) / drawA002 (t=final, true deformed) -> vendor
  `anim_to_vtk_win64.exe` -> ASCII VTK -> pyvista -> quad-only shell ->
  triangulate -> decimate_pro(0.85, preserve_topology=True) on the TRUE
  (unwarped) deformed mesh (decimating the warped mesh instead over-preserves
  triangles because the exaggerated curvature looks like real geometric
  detail to decimate_pro -- confirmed empirically: 0.85 and 0.95 both landed
  on the same triangle count on the warped mesh, vs. a clean ~85% reduction
  on the true-scale mesh).

decimate_pro with preserve_topology=True does not reposition surviving
vertices (confirmed: exact 0.0 nearest-neighbor distance against the
full-resolution point cloud), so each decimated vertex is recovered by an
exact nearest-neighbor lookup against the ORIGINAL full point arrays, giving
robust index correspondence between the original and deformed frames without
depending on the raw (and seam-quirky) node storage order.

Geometric warping (for visual clarity -- true post-buckling radial
displacement is only a few percent of R) is applied client-side in the
viewer via a slider, using the same per-shape scale factor computed here
(TARGET_WARP_FRAC of R, from a robust 1st/99th percentile |radial disp|
estimate) as the slider's "fully exaggerated" endpoint.

Writes shapes_data.js (defines `window.SHAPES_DATA`) into this directory.

Run once: python export_interactive_geometries.py
"""
import base64
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
PROD_ROOT = REPO_ROOT / "buckling" / "runs" / "production"
ANIM_TO_VTK = pathlib.Path("C:/tmp/OpenRadioss/win64/OpenRadioss/exec/anim_to_vtk_win64.exe")

N_SHAPES = 10
DRAW = "draw_000"
# 0.85 was too aggressive for a K~16-21 circumferential wave count: decimate_pro
# under-resolves the wave crests and introduces visible faceting/aliasing across
# the whole wall (confirmed by comparing against the full-resolution render --
# the true surface is smooth there). 0.5 keeps 2x more triangles and matches the
# full-resolution look while still cutting the payload ~2x from full res.
DECIMATE_REDUCTION = 0.5
TARGET_WARP_FRAC = 0.12  # warped max |radial disp| as a fraction of R, for visual clarity


def shape_dirs(n):
    return sorted(
        d for d in PROD_ROOT.iterdir()
        if d.is_dir() and d.name[:3].isdigit() and int(d.name[:3]) < n
    )


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


def b64_f32(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def b64_u16(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.uint16).tobytes()).decode("ascii")


def export_shape(shape_dir, tmpdir):
    summary = json.loads((shape_dir / DRAW / "summary.json").read_text())
    name = shape_dir.name  # e.g. 000_train_rt165.2_lr1.251
    _, _, rt_tag, lr_tag = name.split("_")
    rt = float(rt_tag[2:])
    lr = float(lr_tag[2:])
    K = summary["K"]

    g0, g1 = load_pair(shape_dir, tmpdir)

    quad0 = np.where(g0.celltypes == 9)[0]
    shell1 = g1.extract_cells(quad0)
    tri = shell1.extract_surface().triangulate()

    dec = tri.decimate_pro(DECIMATE_REDUCTION, preserve_topology=True)

    full_pts = g1.points
    tree = cKDTree(full_pts)
    dist, idx = tree.query(dec.points, k=1)
    assert dist.max() < 1e-6, f"{name}: decimated vertex not an exact subset (max dist {dist.max()})"

    orig_pos = g0.points[idx]
    true_def_pos = g1.points[idx]

    r0 = np.hypot(orig_pos[:, 0], orig_pos[:, 1])
    r1 = np.hypot(true_def_pos[:, 0], true_def_pos[:, 1])
    dr = r1 - r0

    faces = dec.faces.reshape(-1, 4)[:, 1:4]
    assert faces.max() < len(idx), f"{name}: face index out of range after decimation"
    assert len(idx) < 65536, f"{name}: {len(idx)} verts exceeds uint16 range"

    R = float(r0.mean())
    L = float(orig_pos[:, 2].max() - orig_pos[:, 2].min())
    lo, hi = np.percentile(dr, [1, 99])
    vmax = float(max(abs(lo), abs(hi), 1e-9))
    scale = TARGET_WARP_FRAC * R / vmax

    print(f"  {name}: n_verts={len(idx)} n_tris={len(faces)} K={K} R={R:.1f} "
          f"vmax={vmax:.4f} warp x{scale:.0f}", file=sys.stderr)

    return {
        "name": name,
        "rt": rt,
        "lr": lr,
        "K": K,
        "R": R,
        "L": L,
        "vmax": vmax,
        "warpScale": scale,
        "nVerts": len(idx),
        "nTris": len(faces),
        "faces_u16": b64_u16(faces.reshape(-1)),
        "pos0_f32": b64_f32(orig_pos.reshape(-1)),
        "pos1_f32": b64_f32(true_def_pos.reshape(-1)),
        "dr_f32": b64_f32(dr),
    }


def main():
    dirs = shape_dirs(N_SHAPES)
    assert len(dirs) == N_SHAPES, f"expected {N_SHAPES} shapes, found {len(dirs)}: {dirs}"

    # Split each shape into a small metadata record (loaded eagerly, for the
    # sidebar) and a separate per-shape file carrying the heavy base64 arrays
    # (loaded lazily on first selection). A single combined file at this
    # resolution runs ~15MB, uncomfortably close to the artifact host's
    # per-file text cap and slow to parse/decode up front for shapes the
    # viewer may never show.
    meta = []
    total_bytes = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        for i, shape_dir in enumerate(dirs):
            print(f"[{i+1}/{N_SHAPES}] {shape_dir.name}", file=sys.stderr)
            rec = export_shape(shape_dir, tmpdir)
            raw = {k: rec.pop(k) for k in ("faces_u16", "pos0_f32", "pos1_f32", "dr_f32")}
            meta.append(rec)
            total_bytes += sum(len(v) for v in raw.values())

            data_out = HERE / f"shape_data_{i}.js"
            data_out.write_text(f"window.SHAPE_RAW_{i} = {json.dumps(raw, separators=(',', ':'))};\n")
            print(f"  wrote {data_out.name} ({data_out.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)

    (HERE / "shapes_data.js").unlink(missing_ok=True)  # stale monolithic export from the previous layout

    meta_out = HERE / "shapes_meta.js"
    meta_out.write_text(f"window.SHAPES_META = {json.dumps(meta, separators=(',', ':'))};\n")
    print(f"wrote {meta_out} ({meta_out.stat().st_size / 1e3:.1f} KB) + {N_SHAPES} shape_data_*.js files "
          f"(total base64 payload {total_bytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
