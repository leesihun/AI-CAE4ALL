"""The dataset-overview figure.

Four panels answering "what is this dataset": the mechanical setup, the
imperfection law that supplies the randomness, what the model is and is not
shown, and the design space.

Panels (a), (b) and (d) are drawn from the real generator -- the same mesh, the
same imperfection code, the same plan file the production run consumed. Only
panel (c) is a diagram, because it describes a file layout rather than a
measurement.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_data import SNAPSHOT
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
BUCK = os.path.join(ROOT, "buckling")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(BUCK, "src"))

from geometry import ShellMesh                                     # noqa: E402
from imperfection import (SpectralImperfection, process_signature,  # noqa: E402
                          critical_wavenumber)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
C_IN, C_OUT, C_HID = "#1f4e79", "#c1440e", "#2e7d32"
C_TAP, C_GREY = "#6a1b9a", "#666666"


# ----------------------------------------------------------------- panel (a)
def panel_setup(ax):
    """Real mesh, drawn coarse for legibility, with the boundary conditions."""
    m = ShellMesh(110, 1.0, elems_per_half_wave=0.6)
    X, Y, Z = m.coords[:, 0], m.coords[:, 1], m.coords[:, 2]
    az, el = np.deg2rad(28.0), np.deg2rad(16.0)
    px = X * np.cos(az) - Y * np.sin(az)
    py = (X * np.sin(az) + Y * np.cos(az)) * np.sin(el) + Z * np.cos(el)
    depth = X * np.sin(az) + Y * np.cos(az)

    order = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    segs = []
    for e in m.elements:
        ring = [e[k] for k in order]
        for a, b in zip(ring[:-1], ring[1:]):
            segs.append((a, b, 0.5 * (depth[a] + depth[b])))
    segs.sort(key=lambda s: s[2])                      # painter's algorithm
    dmin, dmax = min(s[2] for s in segs), max(s[2] for s in segs)
    for a, b, d in segs:
        f = (d - dmin) / max(dmax - dmin, 1e-9)
        ax.plot([px[a], px[b]], [py[a], py[b]], lw=0.45,
                color=str(0.80 - 0.55 * f), zorder=1 + f, solid_capstyle="round")

    top = m.top_nodes
    ax.plot(px[top], py[top], lw=1.6, color=C_OUT, zorder=6)
    for t in np.linspace(0, len(top) - 1, 7).astype(int):
        i = top[t]
        ax.add_patch(FancyArrowPatch((px[i], py[i] + 0.30), (px[i], py[i] + 0.06),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=C_OUT, lw=1.0, zorder=7))
    bot = m.bottom_nodes
    ax.plot(px[bot], py[bot], lw=1.8, color="k", zorder=6)
    yb = py[bot].min()
    for xh in np.linspace(px[bot].min(), px[bot].max(), 16):
        ax.plot([xh, xh - 0.10], [yb - 0.045, yb - 0.155], lw=0.7, color="k",
                zorder=6)

    ax.text(0.50, 1.05, r"prescribed $\delta = 1.5\,\delta_{\mathrm{cr}}$",
            transform=ax.transAxes, ha="center", color=C_OUT, fontsize=8)
    ax.text(0.50, -0.05, "clamped, all 6 DOF", transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.text(0.50, -0.17, r"$R/t \in [95,245]$,  $L/R \in [0.6,1.5]$",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=C_GREY)
    ax.set_title("(a) axial compression of a thin shell", pad=18)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------- panel (b)
def panel_imperfection(ax_a, ax_b, ax_bar):
    """Component A and Component B on ONE shared colour scale.

    The point is that B is invisible at A's scale and yet decides the outcome,
    so the two must not be plotted with independent normalisation.
    """
    m = ShellMesh(110, 1.0, elems_per_half_wave=3.0)
    wA, _ = process_signature(m.theta, m.z, m.L, m.t, 8, 0.15, seed=8,
                              critical_n=critical_wavenumber(m.r_over_t))
    fld = SpectralImperfection(m.R, m.L, m.t, 1e-3, 0.70, 1.5,
                               rng=np.random.default_rng(1))
    wB = fld.evaluate(m.theta, m.z)

    idx = m.node_index[:, ::2]                         # even rows are complete
    gA, gB = wA[idx] / m.t, wB[idx] / m.t
    v = float(np.abs(gA).max())

    for ax, g, ttl, sub, col in (
            (ax_a, gA, "Component A", r"known, RMS $0.15\,t$", C_IN),
            (ax_b, gB, "Component B", r"withheld, nominal $10^{-3}t$", C_HID)):
        im = ax.imshow(g.T, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-v, vmax=v, extent=[0, 360, 0, 1])
        ax.set_title(ttl, fontsize=8.5, color=col, pad=2)
        ax.text(0.5, 1.10, sub, transform=ax.transAxes, ha="center",
                fontsize=7.2, color=col)
        ax.set_xticks([0, 180, 360])
        ax.set_xlabel(r"$\theta$ [deg]", fontsize=8, labelpad=1)
        ax.set_yticks([0, 1])
    ax_a.set_ylabel(r"$z/L$", fontsize=8, labelpad=1)

    ax_b.text(0.5, 0.5, "(flat at this scale)", transform=ax_b.transAxes,
              ha="center", va="center", fontsize=7, color=C_GREY, style="italic")
    cb = plt.colorbar(im, cax=ax_bar)
    cb.set_label(r"$w/t$", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=6.5)


# ----------------------------------------------------------------- panel (c)
def panel_contract(ax):
    """What one stored sample contains, and which side of the wall each row is on."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(c) what one sample stores", pad=4)

    ax.text(0.15, 9.75, r"$\mathtt{data/\{id\}/nodal\_data}$  $[11,1,N]$"
                        r"   $+$  $\mathtt{mesh\_edge}$  $[2,E]$",
            fontsize=7.2, va="center", color=C_GREY)

    rows = [
        ("0:3", "reference coordinates", C_IN, "input"),
        ("3:6", r"displacement $u_x,u_y,u_z$", C_OUT, "OUTPUT"),
        ("6:9", "Component A offset", C_IN, "input"),
        ("9", r"local thickness $t(z)$", C_IN, "input"),
        ("10", "node type", C_IN, "input"),
    ]
    y = 8.85
    for tag, txt, col, role in rows:
        ax.add_patch(Rectangle((0.15, y - 0.70), 9.7, 0.76, facecolor=col,
                               alpha=0.15, edgecolor=col, lw=0.9))
        ax.text(0.95, y - 0.32, tag, fontsize=7.5, color=col, va="center",
                ha="center", fontweight="bold")
        ax.text(1.85, y - 0.32, txt, fontsize=8, va="center")
        ax.text(9.65, y - 0.32, role, fontsize=6.8, va="center", ha="right",
                color=col, style="italic")
        y -= 0.96

    y -= 0.52
    ax.plot([0.15, 9.85], [y, y], lw=1.2, ls="--", color=C_HID)
    ax.text(5.0, y + 0.22, "information wall", fontsize=7.2, ha="center",
            va="bottom", color=C_HID, style="italic")
    y -= 1.05

    ax.add_patch(Rectangle((0.15, y - 0.70), 9.7, 0.80, facecolor=C_HID,
                           alpha=0.15, edgecolor=C_HID, lw=0.9))
    ax.text(0.95, y - 0.30, r"$\mathtt{latent/}$", fontsize=7.2, color=C_HID,
            va="center", ha="center", fontweight="bold")
    ax.text(1.85, y - 0.30, "Component B coefficients", fontsize=8, va="center")
    ax.text(9.65, y - 0.30, "WITHHELD", fontsize=6.8, va="center", ha="right",
            color=C_HID, style="italic")
    ax.text(0.15, y - 1.05,
            "kept for diagnostics; a model reading the documented\n"
            "input rows cannot read these coefficients",
            fontsize=7.0, va="top", color=C_GREY)


# ----------------------------------------------------------------- panel (d)
def panel_design_space(ax):
    """The plan file, plotted. Tiers, not a cartoon."""
    plan = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["plan"]
    style = {
        "train": dict(c=C_IN, m="o", s=46, lbl="train  (12 geom, 576 draws)"),
        "t1": dict(c=C_OUT, m="s", s=46, lbl="t1  parameter extrapolation"),
        "t2": dict(c=C_HID, m="^", s=56, lbl=r"t2  cones $\alpha=10^\circ,20^\circ$"),
        "t3": dict(c=C_TAP, m="D", s=40, lbl=r"t3  taper $\gamma=0.15,0.25$"),
    }
    ax.add_patch(Rectangle((110, 0.8), 90, 0.5, facecolor=C_IN, alpha=0.07,
                           edgecolor=C_IN, ls="--", lw=0.9, zorder=1))
    seen = set()
    for e in plan:
        st = style[e["tier"]]
        lab = st["lbl"] if e["tier"] not in seen else None
        seen.add(e["tier"])
        # nudge the coincident cone/taper markers apart so all are visible
        dx = {"t2": -7.0, "t3": 7.0}.get(e["tier"], 0.0)
        dy = {"t2": -0.05, "t3": 0.05}.get(e["tier"], 0.0)
        ax.scatter(e["r_over_t"] + dx, e["l_over_r"] + dy, c=st["c"],
                   marker=st["m"], s=st["s"], label=lab, zorder=3,
                   edgecolors="white", lw=0.6)

    ax.text(155, 0.845, "training box", fontsize=7.2, ha="center", color=C_IN)
    ax.set_xlabel(r"$R/t$")
    ax.set_ylabel(r"$L/R$")
    ax.set_title("(d) design space: 22 geometries, 896 draws", pad=4)
    ax.set_xlim(78, 264)
    ax.set_ylim(0.42, 1.95)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=6.8, handletextpad=0.3,
              borderpad=0.15, labelspacing=0.28, ncol=1)


def main():
    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95],
                          hspace=0.40, wspace=0.28)

    panel_setup(fig.add_subplot(gs[0, 0]))

    sub = gs[0, 1].subgridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.34)
    aA = fig.add_subplot(sub[0, 0])
    aB = fig.add_subplot(sub[0, 1], sharey=aA)
    plt.setp(aB.get_yticklabels(), visible=False)
    panel_imperfection(aA, aB, fig.add_subplot(sub[0, 2]))
    aA.text(1.12, 1.34, "(b) the imperfection law", transform=aA.transAxes,
            ha="center", fontsize=9)
    aA.text(1.12, -0.30,
            r"Nominal A/B scale ratio: $150$; B varies across solver draws",
            transform=aA.transAxes, ha="center", fontsize=7.4, color=C_GREY,
            style="italic")

    panel_contract(fig.add_subplot(gs[1, 0]))
    panel_design_space(fig.add_subplot(gs[1, 1]))

    p = os.path.join(FIG, "dataset_overview.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
