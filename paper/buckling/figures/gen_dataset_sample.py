"""One real ex3 sample, drawn as two slides: (1) what goes in and what comes
out, (2) what the HDF5 record holds.

Everything is read from dataset/probabilistic/ex3_shell_buckling.h5, sample
2133: reference coordinates, node types, the thickness / Delta rows, the final
displacement field and the per-sample attrs (dimple a / theta / z / sigma,
P_max, P_cr, ...). Two drawing choices are stated on the slide: the seeded
dimple is drawn x20 (it is 0.7 mm deep and is not part of the stored mesh),
and the deformed shape is drawn with the displacement x3.

Run once:  python gen_dataset_sample.py
Writes:    dataset_sample_view.png, dataset_sample_variables.png
"""
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, FancyBboxPatch

pv.OFF_SCREEN = True
HERE = Path(__file__).resolve().parent
H5 = HERE.parents[2] / "dataset" / "probabilistic" / "ex3_shell_buckling.h5"
SID = "2133"
DENT = 20.0          # the seeded dimple, drawn exaggerated
WARP = 3.0           # the displacement field, drawn exaggerated

plt.rcParams.update({"font.family": "Calibri", "mathtext.fontset": "custom",
                     "mathtext.rm": "Calibri", "mathtext.it": "Calibri:italic",
                     "mathtext.bf": "Calibri:bold"})

# ----------------------------------------------------------------- the sample
with h5py.File(H5, "r") as f:
    g = f["data"][SID]
    nd = g["nodal_data"][:, -1, :]                       # [9, N]
    A = {k: g.attrs[k] for k in g.attrs}
R, T, L, D = float(A["R"]), float(A["t"]), float(A["L"]), float(A["delta"])
a_mm, th_d, z_d, sig = (float(A[k]) for k in ("a_mm", "theta_dimple", "z_dimple", "sigma"))
xyz, u, ntype = nd[0:3].T, nd[3:6].T, nd[8]
assert np.allclose(nd[6], T) and np.allclose(nd[7], D)

NJ = int(np.unique(np.round(xyz[:, 2], 6)).size)         # nodes along z
NI = xyz.shape[0] // NJ                                  # nodes around
assert NI * NJ == xyz.shape[0]
ELEM = L / (NJ - 1)                                      # element edge, mm

theta = np.arctan2(xyz[:, 1], xyz[:, 0]) % (2 * np.pi)
rhat = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
dth = np.abs(theta - th_d)
dth = np.minimum(dth, 2 * np.pi - dth) * R
w0 = a_mm * np.exp(-(dth ** 2 + (xyz[:, 2] - z_d) ** 2) / sig ** 2)   # seeded dimple, mm
ur = (u[:, :2] * rhat[:, :2]).sum(1)                                   # radial displacement


def shell(pts, scal):
    P = pts.reshape(NI, NJ, 3)
    P = np.concatenate([P, P[:1]], axis=0)               # close the seam
    S = scal.reshape(NI, NJ)
    S = np.concatenate([S, S[:1]], axis=0)
    gd = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])
    gd["s"] = S.ravel(order="F")
    return gd


# ------------------------------------------------------------------ rendering
AZ = th_d + np.radians(32)                               # dimple left of centre, well lit
EL = np.radians(17)
view = np.array([np.cos(AZ) * np.cos(EL), np.sin(AZ) * np.cos(EL), np.sin(EL)])
fwd = np.array([np.cos(AZ), np.sin(AZ), 0.0])            # horizontal view direction
side = np.array([-np.sin(AZ), np.cos(AZ), 0.0])          # screen-right in world
WIN = (1500, 1250)
PLATE_R, PLATE_H, RING_W, RING_H = 1.12 * R, 6.0, 7.0, 6.0
KEEP = []                                                # live renderers for projection


def render(pts, scal, cmap, clim, ring_z):
    pl = pv.Plotter(window_size=WIN, off_screen=True, lighting="three lights")
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(shell(pts, scal), scalars="s", cmap=cmap, clim=clim, smooth_shading=True,
                show_scalar_bar=False, ambient=0.28, diffuse=0.75, specular=0.08)
    plate = pv.Cylinder(center=(0, 0, -PLATE_H / 2), direction=(0, 0, 1), radius=PLATE_R,
                        height=PLATE_H, resolution=240)
    pl.add_mesh(plate, color="#4d4d4d", ambient=0.3, diffuse=0.7, smooth_shading=True)
    ring = pv.Disc(center=(0, 0, ring_z), inner=R - RING_W, outer=R + RING_W,
                   normal=(0, 0, 1), r_res=2, c_res=240).extrude((0, 0, RING_H), capping=True)
    pl.add_mesh(ring, color="#3e6fae", ambient=0.3, diffuse=0.7, smooth_shading=True)
    pl.enable_parallel_projection()
    c = np.array([0, 0, L / 2])
    pl.camera_position = [tuple(c + view * 1500), tuple(c), (0, 0, 1)]
    aspect = WIN[0] / WIN[1]
    pl.camera.parallel_scale = max(0.5 * (L * np.cos(EL) + 2.1 * PLATE_R * np.sin(EL)) + 18,
                                   (PLATE_R + 12) / aspect)
    img = pl.screenshot(return_img=True)
    ren, h = pl.renderer, WIN[1]

    def proj(p):
        out = []
        for q in np.atleast_2d(p):
            ren.SetWorldPoint(*q, 1.0)
            ren.WorldToDisplay()
            x, y, _ = ren.GetDisplayPoint()
            out.append((x, h - y))
        return np.array(out)
    KEEP.append(pl)
    return img, proj


dent_cmap = LinearSegmentedColormap.from_list(
    "dent", [(0.0, "#d9d9d9"), (0.25, "#d9d9d9"), (1.0, "#c9735a")])
xyz_in = xyz + (DENT * w0)[:, None] * rhat
img_in, proj_in = render(xyz_in, w0 / a_mm, dent_cmap, (0, 1), L)
ulim = float(np.ceil(np.percentile(np.abs(ur), 99.5)))
xyz_out = xyz + WARP * u
img_out, proj_out = render(xyz_out, ur, "RdBu_r", (-ulim, ulim), L - WARP * D)

# ------------------------------------------------------------------ slide 1
FW, FH = 16.0, 9.0
INK, GRAY, BLUE, RED, GREEN = "#222222", "#6b6b6b", "#2f62a8", "#b8432f", "#2e7d32"
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_axis_off()


def place(img, x0, y0, w):
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), zorder=1, interpolation="lanczos")
    return (lambda p: np.column_stack([x0 + p[:, 0] / img.shape[1] * w,
                                       y0 + h - p[:, 1] / img.shape[0] * h])), h


def leader(anchor, text_xy, text, color=INK, ha="left", va="center", fs=11, weight="normal"):
    ax.plot(*anchor, "o", ms=4, color=color, zorder=7)
    ax.annotate(text, anchor, text_xy, fontsize=fs, color=color, ha=ha, va=va, zorder=8,
                fontweight=weight, linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=0, shrinkB=3))


def dim(p, q, label, off, rot=0, fs=11):
    ax.annotate("", p, q, zorder=6, arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0,
                                                    shrinkA=0, shrinkB=0, mutation_scale=11))
    m = (np.asarray(p) + np.asarray(q)) / 2 + off
    ax.text(*m, label, ha="center", va="center", fontsize=fs, rotation=rot, color=INK,
            zorder=7, bbox=dict(fc="white", ec="none", pad=1.5))


def ext(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=5)


# header
ax.text(0.5, 8.5, "Shell-buckling dataset (ex3) — one sample", fontsize=22, fontweight="bold",
        color=INK, va="center")
ax.text(0.5, 8.05,
        f"sample {SID}: geometry {int(A['shape_idx'])} of 150, imperfection draw "
        f"{int(A['seed'])} of 64   |   {int(A['n_nodes']):,} nodes, {int(A['n_elements']):,} "
        f"4-node shell elements ({ELEM:.1f} mm)   |   steel, E = 210 GPa, ν = 0.3   |   "
        "OpenRadioss explicit", fontsize=12, color=GRAY, va="center")
ax.plot([0.5, 15.5], [7.75, 7.75], color="#c8c8c8", lw=0.8)
ax.text(0.5, 7.42, "(a)  Input — reference shell, support, loading", fontsize=13.5,
        fontweight="bold", color=INK, va="center")
ax.text(8.75, 7.42, "(b)  Output — final displacement of every node", fontsize=13.5,
        fontweight="bold", color=INK, va="center")

# ---- (a)
X0A, Y0A, WA = 2.65, 1.8, 4.8
TXL = 2.45                                               # right edge of the left text column
P_in, HA = place(img_in, X0A, Y0A, WA)
to_in = lambda p: P_in(proj_in(np.atleast_2d(p)))

ring_top = L + RING_H
sil_l = to_in(-side * R + (0, 0, 0.5 * L))[0]            # left silhouette, mid height
sil_r = to_in(side * R + (0, 0, 0.5 * L))[0]
# Delta: flat arrows pressing the far side of the ring, text above them
arrow_top = 0.0
for ang in (-40.0, 0.0, 40.0):
    a = AZ + np.pi + np.radians(ang)
    p = to_in((R * np.cos(a), R * np.sin(a), ring_top))[0]
    ax.add_patch(FancyArrow(p[0], p[1] + 0.7, 0, -0.6, width=0.07, head_width=0.2,
                            head_length=0.18, color=BLUE, zorder=6, length_includes_head=True))
    arrow_top = max(arrow_top, p[1] + 0.7)
ptop = to_in((0, 0, ring_top))[0]
ax.text(ptop[0], arrow_top + 0.38, f"Δ = {D:.2f} mm end shortening imposed on a rigid top ring",
        ha="center", va="bottom", fontsize=12, color=BLUE, fontweight="bold")
ax.text(ptop[0], arrow_top + 0.12, f"at v = {A['v']:.2f} mm/ms  →  {A['final_time']:.1f} ms;  "
        "ring: in-plane motion and rotations fixed", ha="center", va="bottom", fontsize=10.5,
        color=BLUE)
# L on the right, 2R below the plate
xr = sil_r[0] + 0.4
dim((xr, to_in(side * R)[0, 1]), (xr, to_in(side * R + (0, 0, L))[0, 1]),
    f"L = {L:.1f} mm", (0.0, 0.0), rot=90)
ext(to_in(side * R)[0], (xr + 0.12, to_in(side * R)[0, 1]))
ext(to_in(side * R + (0, 0, L))[0], (xr + 0.12, to_in(side * R + (0, 0, L))[0, 1]))
pl_near = to_in(fwd * PLATE_R - (0, 0, PLATE_H))[0]
yb = pl_near[1] - 0.42
dim((to_in(-side * R)[0, 0], yb), (to_in(side * R)[0, 0], yb), f"2R = {2 * R:.1f} mm",
    (0.0, -0.22))
ext(to_in(-side * R)[0], (to_in(-side * R)[0, 0], yb - 0.1))
ext(to_in(side * R)[0], (to_in(side * R)[0, 0], yb - 0.1))
# wall thickness, upper left
pt = to_in(-side * R + (0, 0, 0.86 * L))[0]
leader(pt, (TXL, pt[1] + 0.3), f"wall thickness  t = {T:.2f} mm\n(R / t = {R / T:.0f})",
       ha="right", va="bottom")
# clamped base, lower left
a = AZ - np.radians(62)
pb = to_in((PLATE_R * np.cos(a), PLATE_R * np.sin(a), -PLATE_H / 2))[0]
leader(pb, (TXL, pb[1] - 0.3), "clamped base plate\n(all 6 DOF fixed)", ha="right", va="top")
# the dimple: dashed circle of radius sigma, drawn on the dented surface
phi = np.linspace(0, 2 * np.pi, 160)
circ = []
for p in phi:
    th = th_d + sig * np.cos(p) / R
    rr = R + DENT * a_mm * np.exp(-1.0)
    circ.append((rr * np.cos(th), rr * np.sin(th), z_d + sig * np.sin(p)))
rp = to_in(np.array(circ))
ax.plot(rp[:, 0], rp[:, 1], color=RED, lw=1.5, ls=(0, (5, 3)), zorder=6)
pc = to_in(np.array(circ)[np.argmin(rp[:, 0])])[0]
leader(pc, (TXL, pc[1] - 0.3),
       f"seeded dimple  (drawn ×{DENT:.0f})\n"
       f"a = {a_mm:+.2f} mm, inward\n"
       f"σ = {sig:.1f} mm\n"
       f"θ = {np.degrees(th_d):.0f}°,  z = {z_d:.0f} mm\n"
       "not in the stored mesh:\nthe model never sees it",
       color=RED, ha="right", va="top", fs=10.5)
NOTE_Y = 0.7
ax.text(X0A + WA / 2, NOTE_Y, "no field yet: a node holds only (x, y, z), t, Δ "
        "and its support type", ha="center", fontsize=10.5, color=GRAY, style="italic")

# ---- solver arrow
xa, ya = 7.8, Y0A + HA / 2
ax.add_patch(FancyArrow(xa, ya, 1.25, 0, width=0.75, head_width=1.2, head_length=0.45,
                        color="#dedede", zorder=2, length_includes_head=True))
ax.text(xa + 0.4, ya, "OpenRadioss\nexplicit FEA", ha="center", va="center", fontsize=11,
        fontweight="bold", color=INK, zorder=3, linespacing=1.25)
ax.text(xa + 0.62, ya - 0.9, f"{A['final_time']:.1f} ms of load", ha="center",
        fontsize=10.5, color=GRAY)

# ---- (b)
X0B, Y0B, WB = 9.2, Y0A, WA
P_out, HB = place(img_out, X0B, Y0B, WB)
to_out = lambda p: P_out(proj_out(np.atleast_2d(p)))
cax = fig.add_axes([(X0B + 0.6) / FW, (Y0B - 0.2) / FH, (WB - 1.2) / FW, 0.14 / FH])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-ulim, ulim)),
                  cax=cax, orientation="horizontal")
cb.set_label("radial displacement  $u_r$  [mm]   (+ outward)", fontsize=10.5, labelpad=3)
cb.ax.tick_params(labelsize=9.5)
ax.text(X0B + WB / 2, NOTE_Y, f"shape drawn with the displacement ×{WARP:.0f}; "
        "colour is the true value", ha="center", fontsize=10.5, color=GRAY, style="italic")
xk = X0B + WB + 0.35
ax.text(xk, Y0B + HB - 0.05, "$P_{max}$ / $P_{cr}$ = " + f"{A['knockdown']:.3f}",
        fontsize=14, fontweight="bold", color=RED, va="top")
ax.text(xk, Y0B + HB - 0.5,
        f"$P_{{max}}$ = {A['p_max']:.0f} kN\n"
        f"$P_{{cr}}$ = {A['p_cr']:.0f} kN (classical)\n"
        f"→ buckled,  λ = {A['lam']:.2f}",
        fontsize=10.5, color=INK, va="top", linespacing=1.35)
ax.text(xk, Y0B + 0.75, f"max |u| = {A['max_deform']:.1f} mm\n= {A['max_deform_over_t']:.1f} t",
        fontsize=12, color=INK, va="bottom", fontweight="bold", linespacing=1.3)
ax.text(xk, Y0B + 0.4, f"rms |u| = {A['rms_deform_over_t']:.1f} t", fontsize=10.5,
        color=INK, va="bottom")

fig.savefig(HERE / "dataset_sample_view.png", dpi=200, facecolor="white")

# ============================================================ raw geometry only: input | output
def trim(img, pad=12):
    rgb = img[..., :3].astype(int)
    m = np.where((rgb < 245).any(axis=2))
    r0, r1 = max(m[0].min() - pad, 0), min(m[0].max() + pad, img.shape[0])
    c0, c1 = max(m[1].min() - pad, 0), min(m[1].max() + pad, img.shape[1])
    return img[r0:r1, c0:c1]


raw_l, raw_r = trim(img_in), trim(img_out)
hh = max(raw_l.shape[0], raw_r.shape[0])
fig, axs = plt.subplots(1, 2, figsize=(12, 12 * hh / (raw_l.shape[1] + raw_r.shape[1]) + 0.2),
                        gridspec_kw=dict(width_ratios=[raw_l.shape[1], raw_r.shape[1]], wspace=0.04))
for a_, im in zip(axs, (raw_l, raw_r)):
    a_.imshow(im, interpolation="lanczos")
    a_.axis("off")
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
fig.savefig(HERE / "dataset_sample_view_raw.png", dpi=200, facecolor="white")
plt.close(fig)
plt.close(fig)

# ------------------------------------------------------------------ slide 2
fig = plt.figure(figsize=(FW, 6.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, 6.4)
ax.set_axis_off()
ax.text(0.5, 5.9, "What one sample stores", fontsize=22, fontweight="bold", color=INK,
        va="center")
ax.text(0.5, 5.45, f"nodal_data  [9 rows × 1 step × {int(A['n_nodes']):,} nodes]   +   "
        f"mesh_edge  [2 × {int(A['n_elements']) * 2 + NI:,}]   +   per-sample attrs",
        fontsize=12, color=GRAY, va="center")
ax.plot([0.5, 15.5], [5.15, 5.15], color="#c8c8c8", lw=0.8)


def chip(x, y, w, h, fc, ec, name, sub, val):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h * 0.66, name, ha="center", va="center", fontsize=16,
            fontweight="bold", color=ec, zorder=5)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=9.5, color=GRAY,
            zorder=5)
    ax.text(x + w / 2, y - 0.26, val, ha="center", va="center", fontsize=10, color=INK)


def brace(x0, x1, y, label, color, note=""):
    ax.plot([x0, x0, x1, x1], [y + 0.1, y, y, y + 0.1], color=color, lw=1.4)
    ax.text((x0 + x1) / 2, y - 0.2, label, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=color)
    if note:
        ax.text((x0 + x1) / 2, y - 0.5, note, ha="center", va="top", fontsize=9.5, color=GRAY,
                linespacing=1.3)


CIN, COUT, CCOND, CTYPE = ("#e3edf8", BLUE), ("#f9e3df", RED), ("#e4f2e4", GREEN), ("#ececec", "#555555")
rng_txt = lambda v: f"{v.min():.1f} … {v.max():.1f}"
rows = [("$x$", "row 0", rng_txt(nd[0]), CIN), ("$y$", "row 1", rng_txt(nd[1]), CIN),
        ("$z$", "row 2", rng_txt(nd[2]), CIN),
        ("$u_x$", "row 3", rng_txt(nd[3]), COUT), ("$u_y$", "row 4", rng_txt(nd[4]), COUT),
        ("$u_z$", "row 5", rng_txt(nd[5]), COUT),
        ("$t$", "row 6", f"{T:.2f}  (every node)", CCOND),
        ("Δ", "row 7", f"{D:.2f}  (every node)", CCOND),
        ("type", "row 8", f"{(ntype == 0).sum():,} / {(ntype == 1).sum()} / {(ntype == 2).sum()}",
         CTYPE)]
x, wc, gap, yc, hc = 0.5, 1.58, 0.1, 3.55, 1.0
for i, (name, sub, val, (fc, ec)) in enumerate(rows):
    chip(x + i * (wc + gap), yc, wc, hc, fc, ec, name, sub, val)
sx = lambda i: x + i * (wc + gap)
ex_ = lambda i: x + (i + 1) * (wc + gap) - gap
brace(sx(0), ex_(2), 3.0, "reference geometry  →  model input", BLUE,
      "the nominal cylinder: the dimple is not in it")
brace(sx(3), ex_(5), 3.0, "final displacement  →  model output", RED, "mm, at the end of the run")
brace(sx(6), ex_(7), 3.0, "conditions  →  input only", GREEN, "cond_var = 2")
brace(sx(8), ex_(8), 3.0, "support type", "#555555", "0 free / 1 clamped / 2 loaded")

# attrs
ax.add_patch(FancyBboxPatch((0.5, 0.35), 7.3, 1.55, boxstyle="round,pad=0,rounding_size=0.1",
                            fc="#f6f6f6", ec="#c8c8c8", lw=1.2))
ax.text(0.72, 1.65, "fixed for the geometry (shared by its 64 draws)", fontsize=12,
        fontweight="bold", color=INK, va="center")
ax.text(0.72, 0.95,
        f"R = {R:.1f} mm      t = {T:.2f} mm      L = {L:.1f} mm      Δ = {D:.2f} mm\n"
        f"$P_{{cr}}$ = 3.803 E t² = {A['p_cr']:.0f} kN       λ = Δ / $d_{{cr}}$ = {A['lam']:.2f}"
        f"       R/t = {R / T:.0f},  L/R = {L / R:.2f}",
        fontsize=11, color=INK, va="center", linespacing=1.6)
ax.add_patch(FancyBboxPatch((8.2, 0.35), 7.3, 1.55, boxstyle="round,pad=0,rounding_size=0.1",
                            fc="#fdf4f2", ec=RED, lw=1.2))
ax.text(8.42, 1.65, "drawn per sample  (the model never sees these)", fontsize=12,
        fontweight="bold", color=RED, va="center")
ax.text(8.42, 0.95,
        f"dimple  a = {a_mm:+.2f} mm ~ N(0, 0.4²)      θ = {np.degrees(th_d):.0f}°      "
        f"z = {z_d:.1f} mm      σ = {sig:.1f} mm\n"
        f"result  $P_{{max}}$ / $P_{{cr}}$ = {A['knockdown']:.3f},   max |u| = "
        f"{A['max_deform']:.1f} mm,   buckled = {bool(A['buckled'])}",
        fontsize=11, color=INK, va="center", linespacing=1.6)
fig.savefig(HERE / "dataset_sample_variables.png", dpi=200, facecolor="white")
plt.close(fig)
print("wrote dataset_sample_view.png, dataset_sample_view_raw.png, dataset_sample_variables.png")
