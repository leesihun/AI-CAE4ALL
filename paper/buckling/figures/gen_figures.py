"""Generates the schematic figures for paper/buckling/buckling.tex.

Run once: python gen_figures.py
Writes geometry_schematic.png, design_box.png and imperfection_dimples.png
into this directory.

Every number drawn here is pulled from the PRODUCTION generator at
D:/CAE_datasets_raw/shell_buckling/src -- the sampler, the mesh and the
imperfection law themselves, not a hand-typed illustrative value. A setup
figure that quietly disagrees with the deck it claims to document is worse
than no figure, and the previous revision of this script did exactly that
(it drew sigma = 0.11 R and K = 4 while the generator used l_c(R,t)).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

SRC = Path(r"D:\CAE_datasets_raw\shell_buckling\src")
sys.path.insert(0, str(SRC))
import geometry
import imperfection
import sampling

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 200,
})

# The centre of the design box -- used wherever one concrete shell is drawn.
R0, T0, L0, D0 = 150.0, 1.5, 225.0, 2.0
MESH0 = geometry.ShellMesh(R=R0, t=T0, L=L0)
SIGMA0 = imperfection.l_c(R0, T0)

# ---------------------------------------------------------------------------
# Figure 1: the geometry + boundary conditions + what is imposed vs measured
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 6.0))
# drawn to scale: half-width R0, height L0, wall t0 (exaggerated x8 to be visible)
s = 1.0 / R0                       # scale so the drawing is ~1 unit wide
R, L, t = R0 * s, L0 * s, T0 * s * 8.0
ax.add_patch(Polygon([(-R, 0), (R, 0), (R, L), (-R, L)],
                     closed=True, fill=False, lw=2, edgecolor="black"))
ax.plot([R, R - t], [L * 0.62, L * 0.62], color="crimson", lw=3, solid_capstyle="butt")
ax.annotate(f"$t$ = {T0:.1f} mm\n(drawn $\\times$8)", (R - t - 0.06, L * 0.62),
            color="crimson", ha="right", va="center", fontsize=8.6)
ax.annotate("", xy=(-R, -0.44), xytext=(R, -0.44), arrowprops=dict(arrowstyle="<->"))
ax.annotate(f"$2R$   ($R$ = {R0:.0f} mm)", (0, -0.60), ha="center", fontsize=9.5)
ax.annotate("", xy=(R + 0.20, 0), xytext=(R + 0.20, L),
            arrowprops=dict(arrowstyle="<->"))
ax.annotate(f"$L$ = {L0:.0f} mm", (R + 0.34, L / 2), va="center", fontsize=9.5,
            rotation=90, ha="center")

# clamped base
ax.plot([-R - 0.05, R + 0.05], [0, 0], color="dimgray", lw=6, solid_capstyle="butt")
for x in np.linspace(-R, R, 11):
    ax.plot([x, x - 0.07], [0, -0.10], color="dimgray", lw=1.0)
ax.annotate("clamped rim: all 6 DOF fixed\n(/RBODY master + /BCS)", (R + 0.30, -0.16),
            ha="left", va="top", fontsize=8.4, color="dimgray")

# imposed end-shortening at the top -- the INPUT
ax.add_patch(Rectangle((-R - 0.05, L), 2 * R + 0.10, 0.035,
                       facecolor="tab:blue", edgecolor="none", alpha=0.75))
for x in (-R * 0.55, 0.0, R * 0.55):
    ax.annotate("", xy=(x, L + 0.05), xytext=(x, L + 0.30),
                arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.6))
ax.annotate(r"imposed end-shortening  $\Delta$" + f"  (= {D0:.1f} mm here)",
            (0, L + 0.56), ha="center", fontsize=9.6, color="tab:blue")
ax.annotate("INPUT: /IMPVEL, velocity-ramped, $u_z$ prescribed",
            (0, L + 0.42), ha="center", fontsize=8.0, color="tab:blue")

# measured reaction -- the OUTPUT
ax.annotate("", xy=(-R - 0.26, 0.02), xytext=(-R - 0.26, 0.46),
            arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=1.8))
ax.annotate("axial load $P$\nOUTPUT: read back from\nthe /TH/RBODY trace [kN]",
            (-R - 0.40, 0.50), ha="right", va="bottom", fontsize=8.6,
            color="darkgreen")
ax.annotate(f"$P_{{cr}} = 3.803\\,E t^2$ = {MESH0.P_cr_classical:.0f} kN;\n"
            f"the label is $P_{{max}}/P_{{cr}}$",
            (-R - 0.40, 0.34), ha="right", va="top", fontsize=8.0,
            color="darkgreen")

ax.set_xlim(-2.75, 2.05)
ax.set_ylim(-0.72, L + 0.74)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "Circular cylindrical shell: the only geometry family.\n"
    r"$R$, $t$, $L$ and the imposed $\Delta$ are the four design-point knobs"
    "\n" r"(the fifth, the dimple depth $a$, is redrawn per draw -- Fig. 3);"
    "\n" r"$P$ is measured, never prescribed.",
    fontsize=10.2,
)
fig.tight_layout()
fig.savefig("geometry_schematic.png", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the design box -- the actual 158-point roster from sampling.py
# ---------------------------------------------------------------------------
rows = sampling.production_tiers()
tr = np.array([[r[1], r[2], r[3], r[4]] for r in rows if r[0] == "train"])
od = np.array([[r[1], r[2], r[3], r[4]] for r in rows if r[0] != "train"])

pairs = [(0, 1, "$R$  [mm]", "$t$  [mm]"), (2, 3, "$L$  [mm]", r"$\Delta$  [mm]")]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
handles = None
for ax, (a, b, la, lb) in zip(axes, pairs):
    (lo_a, hi_a), (lo_b, hi_b) = sampling.KNOB_RANGES[a], sampling.KNOB_RANGES[b]
    ax.add_patch(Rectangle((lo_a, lo_b), hi_a - lo_a, hi_b - lo_b,
                           facecolor="tab:blue", alpha=0.07, edgecolor="tab:blue",
                           lw=1.2, ls="--", zorder=0,
                           label="the design box (train support)"))
    ax.scatter(tr[:, a], tr[:, b], s=17, color="tab:blue", zorder=3,
               label=f"train, {len(tr)} points x {sampling.DRAWS_PER_POINT} draws")
    ax.scatter(od[:, a], od[:, b], s=48, marker="D", facecolor="none",
               edgecolor="crimson", lw=1.5, zorder=4,
               label=f"OOD, {len(od)} points x {sampling.DRAWS_PER_OOD_POINT} draws")
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.grid(alpha=0.25, lw=0.5)
    handles = ax.get_legend_handles_labels()
axes[0].set_title("geometry knobs", fontsize=10.5)
axes[1].set_title("length and load knobs", fontsize=10.5)
fig.legend(*handles, fontsize=8.8, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.755), frameon=False)
fig.suptitle(
    "The design-point roster actually run: a 4-D Latin hypercube over ONE box, "
    f"{sum(r[5] for r in rows)} draws in total.\n"
    "No rejection and no conditional bounds -- every corner of the box is RSA-feasible "
    "by construction,\nso the four knobs stay genuinely independent. Each OOD point "
    "moves exactly ONE knob out of the box\n(the four whose moved knob is not on a "
    "given pair of axes therefore sit stacked at the box centre).",
    fontsize=9.6, y=0.995, va="top",
)
# explicit, not tight_layout: tight_layout treats the figure legend as content
# and shrinks the panels away from it, leaving a band of dead space.
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.10, top=0.70, wspace=0.20)
fig.savefig("design_box.png", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: the imperfection law -- ONE Gaussian dimple, drawn by the real
#           sampler at the real sigma = l_c(R, t)
# ---------------------------------------------------------------------------
# The x axis is ARCLENGTH s = R*theta, not theta, and the panels are drawn at
# equal aspect -- a dimple is isotropic on the surface, and a theta-in-radians
# axis makes it look like an axial crease, which is a different defect entirely.
n_s, n_z = 620, 220
s_arc = np.linspace(0.0, 2.0 * np.pi * R0, n_s)
theta = s_arc / R0
zc = np.linspace(0, L0, n_z)
TH, Z = np.meshgrid(theta, zc)

# seeds chosen to span the law: near-perfect, strong inward, strong outward.
draws = []
for seed in (39, 6, 34):
    rng = np.random.default_rng((seed, 0))
    centers, amps, sigma, K = imperfection.draw_imperfection(rng, R0, T0, L0)
    field = imperfection.evaluate(TH, Z, R0, centers, amps, sigma)
    draws.append((field, centers, amps, sigma))

vmax = max(np.max(np.abs(f)) for f, _, _, _ in draws)

# Size the canvas FROM the equal-aspect panels, instead of letting three
# 4.19:1 panels rattle around inside a guessed figure box.
_PW = 8.0                                  # panel width, inches
_PH = _PW * L0 / (2.0 * np.pi * R0)        # equal aspect -> panel height
_HSP, _ML, _MR, _MB, _MT = 0.12, 0.80, 1.65, 0.62, 1.05     # gap + margins, inches
_AH = 3.0 * _PH + 2.0 * _HSP * _PH
_FW, _FH = _PW + _ML + _MR, _AH + _MB + _MT

fig, axes = plt.subplots(3, 1, figsize=(_FW, _FH), sharex=True)
im = None
for ax, (field, centers, amps, sigma), k in zip(axes, draws, (1, 2, 3)):
    im = ax.pcolormesh(s_arc, zc, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto", rasterized=True)
    (th_k, z_k), a_k = centers[0], amps[0]
    ax.plot(th_k * R0, z_k, "+", color="black", ms=9, mew=1.4)
    for edge in (2.0 * sigma, L0 - 2.0 * sigma):
        ax.axhline(edge, color="0.25", lw=0.9, ls=":")
    ax.set_aspect("equal")
    ax.set_ylabel(r"$z$  [mm]", fontsize=9.5)
    ax.set_yticks([0, 100, 200])
    ax.annotate(f"draw #{k}:   $a$ = {a_k:+.3f} mm   ($a/t$ = {a_k / T0:+.2f}),   "
                rf"$\theta$ = {th_k:.2f} rad,   $z$ = {z_k:.1f} mm",
                (0.008, 0.90), xycoords="axes fraction", va="top", fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6))
axes[-1].set_xlabel(r"circumferential arclength $R\theta$  [mm]   "
                    r"(one full turn; drawn at equal aspect, so a dimple is round)",
                    fontsize=9.5)
axes[0].annotate(r"$2\sigma$ edge exclusion", (8, 2.0 * SIGMA0 + 6), fontsize=8.0,
                 color="0.25")

fig.subplots_adjust(left=_ML / _FW, right=(_ML + _PW) / _FW,
                    bottom=_MB / _FH, top=(_MB + _AH) / _FH, hspace=_HSP)
cax = fig.add_axes(((_ML + _PW + 0.28) / _FW, (_MB + 0.25 * _AH) / _FH,
                    0.15 / _FW, 0.5 * _AH / _FH))
cb = fig.colorbar(im, cax=cax)
cb.set_label("radial push $w$  [mm]   (inward $-$, outward $+$)", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

fig.suptitle(
    "The imperfection law, unrolled over the box-centre shell "
    rf"($R$ = {R0:.0f}, $t$ = {T0:.1f}, $L$ = {L0:.0f} mm, $\sigma = l_c(R,t)$ = {SIGMA0:.1f} mm)."
    "\n"
    r"ONE Gaussian dimple per draw, signed depth $a \sim \mathcal{N}(0,\ 0.4^2)$ mm, "
    r"position uniform inside the $2\sigma$ band."
    "\n"
    r"The hidden latent is exactly 3-D -- $(a, z, \theta)$ -- and the model never sees "
    "any of it; it sees only the mesh those three numbers produced.",
    fontsize=9.6,
)
fig.savefig("imperfection_dimples.png", bbox_inches="tight")
plt.close(fig)

print(f"box centre: n_nodes={MESH0.n_nodes}  sigma=l_c={SIGMA0:.2f} mm  "
      f"P_cr={MESH0.P_cr_classical:.0f} kN  d_cr={MESH0.end_shortening_cr:.3f} mm")
print("wrote geometry_schematic.png, design_box.png and imperfection_dimples.png")
