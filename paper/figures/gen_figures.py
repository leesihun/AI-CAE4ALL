"""Generates the schematic figures for paper/src/main.tex.

Run once: python gen_figures.py
Writes geometry_schematic.png and imperfection_dimples.png into this directory.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 200,
})


# ---------------------------------------------------------------------------
# Figure 1: the one geometry (cylinder), with R, t, L as the only parameters
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 5.2))
R, t, L = 1.0, 0.09, 3.4
outer = Polygon(
    [(-R, 0), (R, 0), (R, L), (-R, L)],
    closed=True, fill=False, lw=2, edgecolor="black",
)
ax.add_patch(outer)
ax.plot([R, R - t * 4], [L * 0.5, L * 0.5], color="crimson", lw=2)
ax.annotate("$t$", (R - t * 2, L * 0.5 + 0.12), color="crimson", ha="center")
ax.annotate("", xy=(-R, -0.16), xytext=(R, -0.16), arrowprops=dict(arrowstyle="<->"))
ax.annotate("$2R$", (0, -0.42), ha="center")
ax.annotate("", xy=(R + 0.22, 0), xytext=(R + 0.22, L), arrowprops=dict(arrowstyle="<->"))
ax.annotate("$L$", (R + 0.42, L / 2), va="center")
ax.plot([-R - 0.05, R + 0.05], [0, 0], color="dimgray", lw=5, solid_capstyle="butt")
ax.annotate("clamped base", (0, -0.66), ha="center", fontsize=9.5, color="dimgray")
ax.annotate(r"axial load $P$", (0, L + 0.30), ha="center", fontsize=9.5, color="dimgray")
ax.annotate("", xy=(0, L + 0.04), xytext=(0, L + 0.24), arrowprops=dict(arrowstyle="->", color="dimgray"))
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-0.9, L + 0.55)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "The one geometry: a circular cylindrical shell.\n"
    r"Only $R$, $t$, $L$ change between geometries" + "\n"
    r"(swept as $R/t$ and $L/R$; see Table 1).",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig("geometry_schematic.png", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the imperfection law -- RSA-placed Gaussian dimples, three draws
# ---------------------------------------------------------------------------
R = 1.0
L_over_R = 4.0
Lc = L_over_R * R
sigma = 0.11 * R          # dimple width == l_c = pi*[12(1-nu^2)]^-1/4 * sqrt(R t)
                          # (schematic uses a fixed illustrative value; the real
                          # generator ties sigma to l_c(R,t), see main.tex Sec 2.1)
min_sep_factor = 2.0       # RSA minimum-separation multiple of sigma (== 2 l_c)
K = 4                      # dimples per draw -- simplified for legibility only;
                           # the real design draws K ~ round(clip(Normal(20,7^2),0,40))
                           # per draw (see main.tex Sec 2.2), still far below the
                           # paper's saturation N (~190-570 for our geometries)
edge_margin = 3.0 * sigma  # keep dimples away from the clamped/loaded ends

n_theta, n_z = 420, 260
theta = np.linspace(0, 2 * np.pi, n_theta)
zc = np.linspace(0, Lc, n_z)
TH, Z = np.meshgrid(theta, zc)


def rsa_dimples(seed, k=K, max_tries=500):
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(k):
        for _try in range(max_tries):
            th = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(edge_margin, Lc - edge_margin)
            ok = True
            for (th2, z2) in centers:
                dth = np.minimum(np.abs(th - th2), 2 * np.pi - np.abs(th - th2)) * R
                dz = abs(z - z2)
                if np.hypot(dth, dz) < min_sep_factor * sigma:
                    ok = False
                    break
            if ok:
                centers.append((th, z))
                break
    signs = rng.choice([-1.0, 1.0], size=len(centers))
    # moment-matched to the paper's <delta_bar>=0.15, Delta delta_bar=0.05 (Sec VI)
    mags = rng.lognormal(mean=np.log(0.142), sigma=0.325, size=len(centers))
    amps = signs * mags
    return centers, amps


def dimple_field(centers, amps):
    field = np.zeros_like(TH)
    for (th_k, z_k), a_k in zip(centers, amps):
        dth = np.minimum(np.abs(TH - th_k), 2 * np.pi - np.abs(TH - th_k)) * R
        dz = Z - z_k
        d2 = dth ** 2 + dz ** 2
        field += a_k * np.exp(-d2 / sigma ** 2)
    return field, centers


fields = []
for seed in (11, 12, 13):
    centers, amps = rsa_dimples(seed)
    field, centers = dimple_field(centers, amps)
    fields.append((field, centers))

vmax = max(np.max(np.abs(f)) for f, _ in fields)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
im = None
for ax, (field, centers), k in zip(axes, fields, (1, 2, 3)):
    im = ax.pcolormesh(theta, zc, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    for (th_k, z_k) in centers:
        ax.plot(th_k, z_k, "+", color="black", ms=8, mew=1.3)
    ax.set_title(f"draw #{k}: {len(centers)} dimples\n(RSA-placed, independent of the others)", fontsize=9.8)
    ax.set_xlabel(r"circumferential angle $\theta$", fontsize=9.5)

axes[0].set_ylabel(r"axial coordinate $z$")
fig.subplots_adjust(left=0.07, right=0.90, top=0.72, bottom=0.16, wspace=0.12)
cax = fig.add_axes((0.925, 0.16, 0.014, 0.52))
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"radial push $w/t$ (in $-$, out $+$)", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

fig.suptitle(
    "The imperfection law, unrolled over the shell surface: a handful of localized\n"
    "Gaussian dimples, placed by Random Sequential Adsorption (RSA) so they never overlap,\n"
    "each pushing the surface in or out by a small log-normal amount ($+$ marks a dimple center).\n"
    "This is the ONE thing that differs between draws of the same geometry -- and the model never sees it.",
    fontsize=9.6,
)
fig.savefig("imperfection_dimples.png", bbox_inches="tight")
plt.close(fig)

print("wrote geometry_schematic.png and imperfection_dimples.png")
