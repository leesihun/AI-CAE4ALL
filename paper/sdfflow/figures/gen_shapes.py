"""Shape figures for paper/sdfflow/sdfflow.tex.

Every panel is rendered off-screen with pyvista from a fixed camera over the [-1, 1]^3 box (so sizes are
comparable across panels), then composed in matplotlib so every label is Korean. Body counts and the hole
count of the largest body are measured here, from the same meshes that are drawn, and written to
data/shape_panels.json.

  fig_data_ex3.png      ex3: the five nut classes, each in its common and a less common orientation
  fig_data_jeb.png      DeepJEB: siblings of one parent vs. five different parents
  fig_interp_kl.png     one noise-to-noise path (seed 42, rows 0 -> 1) for each penalty weight
  fig_endpoints.png     original shapes vs. the stage-1 reconstruction of the same held-out shapes
  fig_interp_methods.png  penalty 1e-8: three ways to go between two real shapes of one class
  fig_cross.png         penalty 1e-8: two real shapes of different classes
  fig_cond_fixed.png    penalty 1e-6: the same noise path, empty condition vs. one fixed condition
  fig_csweep.png        one fixed noise, only the condition moves from shape A to shape B

Inputs (read-only): dataset/geometry_generation/{ex3_mcb,ex1_deepjeb}.h5, D:/CAE_datasets_raw/{mcb,deepjeb},
junk/sdfflow_interp_raw/, junk/sdfflow_followup/{eval2/vols,vols}/.
Run with a scratch directory as cwd (VTK drops a cache directory into cwd):
  cd <scratch> && python <repo>/paper/sdfflow/figures/gen_shapes.py
"""
import json
import pathlib

import h5py
import numpy as np
import trimesh
import pyvista as pv
from skimage import measure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
EX3 = REPO / "dataset/geometry_generation/ex3_mcb.h5"
JEB = REPO / "dataset/geometry_generation/ex1_deepjeb.h5"
RAW_MCB = pathlib.Path("D:/CAE_datasets_raw/mcb/nuts_subset")
RAW_JEB = pathlib.Path("D:/CAE_datasets_raw/deepjeb/FieldMesh")
INTERP = REPO / "junk/sdfflow_interp_raw"
VOLS = [REPO / "junk/sdfflow_followup/eval2/vols", REPO / "junk/sdfflow_followup/vols"]

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
TEAL, RED, GT = "#2aa198", "#dc322f", "#9aa0a6"
KCLS = {"Hexagonal nuts": "육각 너트", "Locknuts": "잠금 너트", "Castle nuts": "캐슬 너트",
        "Slotted nuts": "홈 너트", "Square nuts": "사각 너트"}
ARM_LAB = {"baseline_kl1e-4": "벌점 1e-4\n(처음 설정)", "kl1e-6": "벌점 1e-6", "kl1e-8": "벌점 1e-8",
           "kl1e-10": "벌점 1e-10"}
TILE = 220
LOG = {}


# ------------------------------------------------------------------ meshes
def mc(vol):
    """Value-0 surface of a decoded 128^3 volume, split into bodies, largest first (no keep_largest)."""
    vol = vol.astype(np.float32)
    if vol.min() > 0 or vol.max() < 0:
        return []
    sp = 2.0 / (vol.shape[0] - 1)
    v, f, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(sp,) * 3)
    return split(trimesh.Trimesh(v - 1.0, f, process=True))


def split(m):
    return sorted(m.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True)


def holes(m):
    return int(round((2 - m.euler_number) / 2)) if m.is_watertight else None


def to_pv(m):
    f = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64).ravel()
    return pv.PolyData(np.asarray(m.vertices, dtype=np.float64), f)


def vol(arm, name):
    for d in VOLS:
        p = d / arm / f"{name}.npz"
        if p.exists():
            return np.load(p)["vols"]
    raise FileNotFoundError(f"{arm}/{name}")


def gt_ex3(sid):
    with h5py.File(EX3, "r") as f:
        g = f[f"shapes/{sid:05d}"]
        src = g.attrs["source"].replace("/data/Lee/CAE_datasets_raw/mcb/", "")
        c, s = g.attrs["center"], float(g.attrs["scale"])
        sp = g["surface_points"][:]
    m = trimesh.load(RAW_MCB / src, force="mesh")
    if not m.is_watertight:  # replay build_dataset.py --repair, which can drop stray faces of the raw obj
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        m.update_faces(m.unique_faces())
        m.update_faces(m.nondegenerate_faces())
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fill_holes(m)
    if not m.is_watertight:
        import pymeshfix
        fix = pymeshfix.MeshFix(np.asarray(m.vertices), np.asarray(m.faces))
        fix.repair(joincomp=True, remove_smallest_components=False)
        m = trimesh.Trimesh(fix.points, fix.faces, process=True)
    m = trimesh.Trimesh((m.vertices - c) * s, m.faces, process=True)
    err = np.abs(m.bounds - np.stack([sp.min(0), sp.max(0)])).max()
    assert err < 0.02, (sid, err)  # the raw mesh and the stored surface points must agree
    return m


def gt_jeb(key):
    with h5py.File(JEB, "r") as f:
        g = f[f"shapes/{key}"]
        stem = g.attrs["source"].split("\\")[-1][:-4]
        c, s = g.attrs["center"], float(g.attrs["scale"])
        sp = g["surface_points"][:]
    with h5py.File(RAW_JEB / f"{stem}.h5", "r") as h:  # the stored 'faces' cover only one patch:
        v, tet = h["vertices"][:], h["cells"][:, :4]     # take the outer skin of the tet10 corner nodes
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet.astype(np.int64)}, ((v - c) * s).astype(np.float64))
    surf = grid.extract_surface().triangulate()
    m = trimesh.Trimesh(np.asarray(surf.points), surf.faces.reshape(-1, 4)[:, 1:], process=True)
    err = np.abs(m.bounds - np.stack([sp.min(0), sp.max(0)])).max()
    assert err < 0.03, (key, stem, err)
    return m


# ------------------------------------------------------------------ rendering
DIR = np.array([1.0, -1.3, 1.1]) / np.linalg.norm([1.0, -1.3, 1.1])
BOX = pv.Box(bounds=(-0.9, 0.9, -0.9, 0.9, -0.9, 0.9))


def render(cells, box=True, scale=1.22):
    """cells[r][c] = list of (trimesh, colour) or None. Returns the same grid of RGB tiles."""
    nr, nc = len(cells), max(len(r) for r in cells)
    pl = pv.Plotter(shape=(nr, nc), off_screen=True, window_size=(TILE * nc, TILE * nr), border=False)
    for r, row in enumerate(cells):
        for c in range(nc):
            pl.subplot(r, c)
            pl.set_background("white")
            items = row[c] if c < len(row) else None
            if items is None:
                continue
            if box:
                pl.add_mesh(BOX, style="wireframe", color="#d5d5d5", line_width=1)
            for m, col in items:
                pl.add_mesh(to_pv(m), color=col, smooth_shading=True, specular=0.15)
            pl.enable_parallel_projection()
            cam = pl.camera
            cam.focal_point = (0.0, 0.0, 0.0)
            cam.position = tuple(DIR * 10.0)
            cam.up = (0.0, 0.0, 1.0)
            cam.parallel_scale = scale
            cam.clipping_range = (1.0, 25.0)
    img = pl.screenshot(return_img=True)
    pl.close()
    assert img.shape[:2] == (TILE * nr, TILE * nc), img.shape
    return [[img[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for c in range(nc)] for r in range(nr)]


def coloured(parts):
    return [(p, TEAL if i == 0 else RED) for i, p in enumerate(parts)] if parts else []


def label(parts):
    if not parts:
        return "비어 있음", True
    g = holes(parts[0])
    return f"조각 {len(parts)}\n구멍 {'?' if g is None else g}", len(parts) != 1


def compose(tiles, out, row_labels, cell_labels=None, red=None, col_labels=None, W=7.1, lw=0.95,
            lab_h=0.24, head_h=0.2, row_fs=7.0, cell_fs=6.0):
    nr, nc = len(tiles), len(tiles[0])
    cw = (W - lw) / nc
    rh = cw + (lab_h if cell_labels else 0.04)
    H = (head_h if col_labels else 0.04) + nr * rh + 0.04
    fig = plt.figure(figsize=(W, H))
    y0 = head_h if col_labels else 0.04
    for r in range(nr):
        top = y0 + r * rh + (lab_h if cell_labels else 0.0)
        for c in range(nc):
            if tiles[r][c] is None:
                continue
            left = lw + c * cw
            ax = fig.add_axes([left / W, 1 - (top + cw) / H, cw / W, cw / H])
            ax.imshow(tiles[r][c])
            ax.set_axis_off()
            if cell_labels and cell_labels[r][c]:
                fig.text((left + cw / 2) / W, 1 - (top - 0.01) / H, cell_labels[r][c], ha="center", va="bottom",
                         fontsize=cell_fs, linespacing=1.0,
                         color="#b8261c" if (red and red[r][c]) else "#333333")
        fig.text((lw - 0.05) / W, 1 - (top + cw / 2) / H, row_labels[r], ha="right", va="center",
                 fontsize=row_fs, linespacing=1.15)
    if col_labels:
        for c, t in enumerate(col_labels):
            fig.text((lw + c * cw + cw / 2) / W, 1 - 0.02 / H, t, ha="center", va="top", fontsize=row_fs,
                     weight="bold")
    fig.savefig(HERE / out, dpi=250)
    plt.close(fig)
    print("written", out)


def path_rows(vol_rows, out, row_labels, col_labels, **kw):
    """vol_rows: list of [steps, n, n, n] volumes -> one strip per row with body / hole labels."""
    parts = [[mc(v) for v in V] for V in vol_rows]
    tiles = render([[coloured(p) for p in row] for row in parts])
    labs = [[label(p) for p in row] for row in parts]
    LOG[out] = [[l[0].replace("\n", " ") for l in row] for row in labs]
    compose(tiles, out, row_labels, [[l[0] for l in row] for row in labs], [[l[1] for l in row] for row in labs],
            col_labels, **kw)


# ------------------------------------------------------------------ figures
def axis_of(pts):
    """Hole axis = principal axis whose spread differs most from the other two; 'obl' if > 18 deg off x/y/z."""
    w, v = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
    gap = [min(abs(w[i] - w[j]) for j in range(3) if j != i) for i in range(3)]
    a = v[:, int(np.argmax(gap))]
    k = int(np.argmax(np.abs(a)))
    return "xyz"[k] if abs(a[k]) > 0.95 else "obl"


def fig_data_ex3():
    with h5py.File(EX3, "r") as f:
        keys = sorted(f["shapes"].keys())
        info = []
        for k in keys:
            g = f[f"shapes/{k}"]
            cls = g.attrs["source"].split("/")[-2]
            # picks examples only; the per-class counts quoted in the paper are the aarl ones in gen_concept.py
            info.append((int(k), cls, axis_of(g["surface_points"][::4])))
    order = ["Hexagonal nuts", "Locknuts", "Castle nuts", "Slotted nuts", "Square nuts"]
    want = {"Hexagonal nuts": ("z", "x"), "Locknuts": ("z", "x"), "Castle nuts": ("z", "obl"),
            "Slotted nuts": ("x", "y"), "Square nuts": ("z", "x")}
    count = {c: sum(1 for _, cc, _ in info if cc == c) for c in order}
    LOG.pop("fig_data_ex3.axis_counts", None)
    cells, labs, pick = [[], []], [[], []], []
    for c in order:
        for r, a in enumerate(want[c]):
            sid = next(i for i, cc, aa in info if cc == c and aa == a)
            pick.append((c, a, sid))
            cells[r].append([(gt_ex3(sid), GT)])
            name = {"z": "구멍 축 z (세움)", "x": "구멍 축 x (누움)", "y": "구멍 축 y (누움)", "obl": "비스듬함"}[a]
            labs[r].append(name)
    LOG["fig_data_ex3.picks"] = pick
    tiles = render(cells)
    compose(tiles, "fig_data_ex3.png", ["흔한 방향", "다른 방향"], labs, None,
            [f"{KCLS[c]} ({count[c]})" for c in order], W=6.3, lw=0.75, lab_h=0.16, cell_fs=6.5)


def fig_data_jeb():
    with h5py.File(JEB, "r") as f:
        stem2key = {f[f"shapes/{k}"].attrs["source"].split("\\")[-1][:-4]: k for k in f["shapes"]}
    sib = ["421_152", "421_229", "421_233", "421_482", "421_61"]
    oth = ["290_14", "86_371", "142_233", "513_249", "14_107"]
    tiles = render([[[(gt_jeb(stem2key[s]), GT)] for s in sib], [[(gt_jeb(stem2key[s]), GT)] for s in oth]])
    n421 = sum(1 for s in stem2key if s.split("_")[0] == "421")
    labs = [[f"부모 421 · {s.split('_')[1]}번" for s in sib], [f"부모 {s.split('_')[0]}" for s in oth]]
    LOG["fig_data_jeb"] = dict(siblings=sib, others=oth, n_parent_421=n421)
    compose(tiles, "fig_data_jeb.png", [f"같은 부모의\n형제 형상\n(부모 421,\n형제 {n421}개)", "서로 다른\n부모"],
            labs, None, None, W=6.3, lw=0.95, lab_h=0.16, cell_fs=6.5)


ALPHA11 = [f"α {a:g}" for a in np.round(np.linspace(0, 1, 11), 2)]


def fig_interp_kl():
    rows, labs = [], []
    for arm in ["baseline_kl1e-4", "kl1e-6", "kl1e-8", "kl1e-10"]:
        d = INTERP / f"ex3_mcb_{arm}"
        names = ["sample_42_000.stl"] + [f"sample_42_000_001_alpha0p{k}.stl" for k in range(1, 10)] + ["sample_42_001.stl"]
        rows.append([split(trimesh.load(d / n, force="mesh", process=True)) for n in names])
        labs.append(ARM_LAB[arm])
        meta = [json.loads((d / f"interpolation_000_001_alpha0p{k}_meta.json").read_text()) for k in range(1, 10)]
        LOG[f"fig_interp_kl.meta.{arm}"] = [m.get("body_count_raw") for m in meta]
    tiles = render([[coloured(p) for p in row] for row in rows])
    cl = [[label(p) for p in row] for row in rows]
    for arm, row in zip(["baseline_kl1e-4", "kl1e-6", "kl1e-8", "kl1e-10"], cl):
        # quote the pipeline's own count (the STL re-split can differ by one where two bodies touch at a point)
        for k, bc in enumerate(LOG[f"fig_interp_kl.meta.{arm}"], start=1):
            n = bc[1]
            row[k] = (f"조각 {n}\n" + row[k][0].split("\n")[1], n != 1)
    LOG["fig_interp_kl.png"] = [[l[0].replace("\n", " ") for l in row] for row in cl]
    compose(tiles, "fig_interp_kl.png", labs, [[l[0] for l in row] for row in cl], [[l[1] for l in row] for row in cl],
            ALPHA11)


def fig_endpoints():
    cols = [(443, "SAME_lerp_p0", 0), (938, "SAME_lerp_p0", -1), (210, "SAME_lerp_p1", 0), (1080, "SAME_lerp_p1", -1),
            (460, "CROSS_lerp_p0", 0), (383, "CROSS_lerp_p0", -1), (494, "CROSS_lerp_p1", 0), (205, "CROSS_lerp_p1", -1)]
    with h5py.File(EX3, "r") as f:
        cls = {s: KCLS[f[f"shapes/{s:05d}"].attrs["source"].split("/")[-2]] for s, _, _ in cols}
    arms = ["baseline_kl1e-4", "kl1e-6", "kl1e-8", "kl1e-10"]
    cells = [[[(gt_ex3(s), GT)] for s, _, _ in cols]]
    labs = [[""] * len(cols)]
    red = [[False] * len(cols)]
    for arm in arms:
        ps = [mc(vol(arm, vf)[k]) for _, vf, k in cols]
        cells.append([coloured(p) for p in ps])
        ll = [label(p) for p in ps]
        labs.append([l[0] for l in ll])
        red.append([l[1] for l in ll])
    LOG["fig_endpoints.png"] = [[x.replace("\n", " ") for x in row] for row in labs]
    tiles = render(cells)
    compose(tiles, "fig_endpoints.png", ["원래 형상"] + [ARM_LAB[a] + "\n복원" for a in arms], labs, red,
            [f"{cls[s]} {s}" for s, _, _ in cols], W=6.6, lw=1.0, cell_fs=6.3)


def fig_interp_methods():
    arm = "kl1e-8"
    rows = [vol(arm, f"SAME_{m}_p{p}") for p in (0, 1) for m in ("lerp", "inv", "invc")]
    labs = []
    for p, pair in ((0, "육각 443→938\n(세움→누움)"), (1, "잠금 210→1080\n(둘 다 세움)")):
        for m in ("① 코드 직선 섞기", "② 거꾸로 걷기,\n빈 조건", "③ 거꾸로 걷기,\n조건도 섞기"):
            labs.append(f"{pair}\n{m}")
    path_rows(rows, "fig_interp_methods.png", labs, ALPHA11, lw=1.15, row_fs=6.2)


def fig_cross():
    arm = "kl1e-8"
    rows = [vol(arm, f"CROSS_{m}_p{p}") for p in (0, 1) for m in ("lerp", "inv")]
    labs = []
    for pair in ("육각 460→캐슬 383", "육각 494→잠금 205"):
        for m in ("① 코드 직선 섞기", "② 거꾸로 걷기,\n빈 조건"):
            labs.append(f"{pair}\n{m}")
    path_rows(rows, "fig_cross.png", labs, ALPHA11, lw=1.15, row_fs=6.2)


def fig_cond_fixed():
    arm = "kl1e-6"
    rows = [vol(arm, "D1_p0"), vol(arm, "D1c_p0"), vol(arm, "D1_p1"), vol(arm, "D1c_p1")]
    labs = ["잡음 쌍 1\n빈 조건", "잡음 쌍 1\n조건 고정\n(육각 443)", "잡음 쌍 2\n빈 조건", "잡음 쌍 2\n조건 고정\n(잠금 210)"]
    path_rows(rows, "fig_cond_fixed.png", labs, ALPHA11, lw=1.0, row_fs=6.4)


def fig_csweep():
    rows, labs = [], []
    for arm in ("kl1e-6", "kl1e-8", "kl1e-10"):
        for cfg in (1, 2):
            rows.append(vol(arm, f"CSW_p0_r0_cfg{cfg}"))
            labs.append(f"{ARM_LAB[arm]}\n조건 강조 {cfg}")
    cols = [f"α {a:g}" for a in np.linspace(0, 1, 9)]
    path_rows(rows, "fig_csweep.png", labs, cols, lw=1.0, row_fs=6.4)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig_data_ex3", "fig_data_jeb", "fig_interp_kl", "fig_endpoints", "fig_interp_methods",
                            "fig_cross", "fig_cond_fixed", "fig_csweep"]
    out = HERE / "data/shape_panels.json"
    if out.exists():
        LOG.update(json.loads(out.read_text(encoding="utf-8")))
    for name in todo:
        globals()[name]()
        out.write_text(json.dumps(LOG, ensure_ascii=False, indent=1), encoding="utf-8")
