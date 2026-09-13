"""Generate measured plots from frozen data and the analytical Gaussian example.

SAOI is a historical note transcription, not a rerun of missing predictions.
Missing sources fail the build instead of leaving a stale plot in place.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_data import saoi_rows, SNAPSHOT, sha

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
BUCK = os.path.join(ROOT, "buckling")
PROD = os.path.join(BUCK, "work", "production")
sys.path.insert(0, os.path.join(BUCK, "src"))

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})
C_MGNV, C_FLOW, C_REF = "#1f4e79", "#c1440e", "#555555"


# ---------------------------------------------------------------- SAOI tables
# Transcribed from docs/research/SAOI_PROBABILISTIC_SWEEP_2026-09.md.
# Mean over the three eval sets. arm 2 of MGN-V died in training.
_SAOI = saoi_rows()
MGNV, FLOW = _SAOI["mgnv"], _SAOI["flow"]


def fig_saoi_ranking():
    """Per-arm W1/sd for both models, and the dispersion deficit."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    a = ax[0]
    mv = sorted([r[2] for r in MGNV])
    fl = sorted([r[2] for r in FLOW])
    a.plot(range(1, len(mv) + 1), mv, "o-", color=C_MGNV, label="MeshGraphNets-V", ms=4)
    a.plot(range(1, len(fl) + 1), fl, "s-", color=C_FLOW, label="cHI-MGNflow", ms=4)
    a.set_xlabel("arm, sorted by score")
    a.set_ylabel(r"$W_1/\mathrm{sd}$   (0 is ideal)")
    a.set_title("(a) distribution error per arm")
    a.legend(frameon=False)
    a.set_ylim(0, 1.75)

    b = ax[1]
    mv = [r[4] for r in MGNV]
    fl = [r[4] for r in FLOW]
    b.axhline(1.0, color="k", lw=1.0, ls="--")
    b.text(1.5, 1.02, "reference width", fontsize=7.5, color="k")
    parts = b.boxplot([mv, fl], widths=0.5, patch_artist=True,
                      tick_labels=["MGN-V", "flow"])
    for p, c in zip(parts["boxes"], (C_MGNV, C_FLOW)):
        p.set_facecolor(c)
        p.set_alpha(0.35)
    for k in ("medians",):
        for ln in parts[k]:
            ln.set_color("k")
    b.set_ylabel(r"$\mathrm{sd}_{\mathrm{gen}}/\mathrm{sd}_{\mathrm{ref}}$")
    b.set_title("(b) mean dispersion ratio across three parts")
    b.set_ylim(0, 1.15)

    fig.tight_layout()
    p = os.path.join(FIG, "saoi_ranking.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p)


def fig_failure_modes():
    """The two failure modes are orthogonal: width vs location."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter([r[3] for r in MGNV], [r[4] for r in MGNV], s=34, color=C_MGNV,
               label="MeshGraphNets-V", zorder=3)
    ax.scatter([r[3] for r in FLOW], [r[4] for r in FLOW], s=34, marker="s",
               color=C_FLOW, label="cHI-MGNflow", zorder=3)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.axvline(0.0, color="k", lw=0.9, ls=":")
    ax.plot([0], [1], marker="*", ms=14, color="darkgreen", zorder=4)
    ax.annotate("matching mean and width", (0, 1), textcoords="offset points",
                xytext=(8, -10), fontsize=8, color="darkgreen")
    ax.set_xlabel(r"$|\Delta\mathrm{mean}|/\mathrm{sd}$   (location error)")
    ax.set_ylabel(r"$\mathrm{sd}_{\mathrm{gen}}/\mathrm{sd}_{\mathrm{ref}}$   (width)")
    ax.set_title("Location and width diagnostics")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(0.35, 1.1)
    fig.tight_layout()
    p = os.path.join(FIG, "failure_modes.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p)


def fig_shrinkage():
    """The CFM loss floor is linear in the trainable target scale."""
    from scipy.integrate import quad
    a = np.linspace(0.02, 3.0, 200)
    num = [quad(lambda t, aa=aa: aa ** 2 / ((1 - t) ** 2 + aa ** 2 * t ** 2), 0, 1)[0]
           for aa in a]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))

    ax[0].plot(a, num, lw=2, color=C_MGNV, label=r"$\int_0^1\mathrm{Var}(U|Z_t)\,dt$")
    ax[0].plot(a, np.pi * a / 2, "--", lw=1.2, color="k", label=r"$\pi a/2$")
    ax[0].set_xlabel(r"posterior scale $a$")
    ax[0].set_ylabel("optimal CFM loss")
    ax[0].set_title("(a) loss floor is linear in $a$")
    ax[0].legend(frameon=False)

    for aa, c in ((0.3, "#7fb3d5"), (1.0, C_MGNV), (2.0, "#0b2b45")):
        t = np.linspace(0, 1, 400)
        ax[1].plot(t, aa ** 2 / ((1 - t) ** 2 + aa ** 2 * t ** 2), color=c,
                   label=r"$a=%.1f$" % aa)
    ax[1].set_xlabel(r"flow time $t$")
    ax[1].set_ylabel(r"$\mathrm{Var}(U\mid Z_t)$")
    ax[1].set_title("(b) irreducible conditional variance")
    ax[1].legend(frameon=False)

    fig.tight_layout()
    p = os.path.join(FIG, "shrinkage.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------ buckling figures
def _load_production():
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    train = {"%s_rt%d_lr%03d_a%02d_g%02d" % (
        x["shape"][:3], x["r_over_t"], round(x["l_over_r"] * 100),
        round(x["alpha_deg"] * 10), round(x["thickness_gamma"] * 100))
        for x in snapshot["plan"] if x["tier"] == "train"}
    recs = [x for x in snapshot["records"] if x.get("ok")
            and x["key"].rsplit("_s", 1)[0] in train]
    g = defaultdict(list)
    for x in recs:
        g[x["key"].rsplit("_s", 1)[0]].append(x)
    if len(g) != 12 or any(len(v) != 48 for v in g.values()):
        raise ValueError("Figures require the complete 12 x 48 training-tier snapshot")
    return {k: sorted(v, key=lambda r: r["seed"]) for k, v in g.items()}


def fig_buckling_modes():
    g = _load_production()
    if not g:
        print("SKIP buckling modes: no production data")
        return
    # Show the R/t progression at fixed L/R = 1.0: the mode distribution both
    # SHIFTS (centre tracks 0.86*sqrt(R/t)) and WIDENS (top-mode share falls)
    # as the shell thins. An arbitrary first-four would show only the shift.
    pref = [k for k in sorted(g) if k.endswith("_lr100_a00_g00")]
    ks = pref[:4] if len(pref) >= 4 else sorted(g)[:4]
    fig, ax = plt.subplots(1, len(ks), figsize=(1.85 * len(ks), 2.3), sharey=True)
    if len(ks) == 1:
        ax = [ax]
    for a, k in zip(ax, ks):
        ns = [x["n_dominant"] for x in g[k]]
        c = Counter(ns)
        xs = sorted(c)
        a.bar(xs, [c[x] / len(ns) for x in xs], color=C_MGNV, alpha=0.8, width=0.7)
        ncrit = .86 * np.sqrt(g[k][0]["r_over_t"])
        if ncrit:
            a.axvline(ncrit, color=C_FLOW, lw=1.4, ls="--")
        rt = int(g[k][0]["r_over_t"])
        lr = float(g[k][0]["l_over_r"])
        a.set_title(r"$R/t=%d$, $L/R=%.1f$" % (rt, lr), fontsize=8)
        a.set_xlabel("circumferential mode $n$")
        a.set_xticks(xs)
    ax[0].set_ylabel("probability")
    fig.tight_layout()
    p = os.path.join(FIG, "buckling_modes.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p, "(%d geometries)" % len(ks))


def fig_buckling_fields():
    """Real deformed shapes: same geometry, different withheld realisation.

    The contour panels alone do not make the mode difference legible (a mode n
    shows 2n alternating lobes, so 8 and 10 look similar at a glance), so the
    circumferential profile at the most-deformed axial station is plotted
    underneath, where the wavenumber is directly countable.
    """
    g = _load_production()
    if not g:
        print("SKIP buckling fields: no production data")
        return
    from geometry import ShellMesh
    from analyse import to_grid
    k = sorted(g)[1] if len(g) > 1 else sorted(g)[0]
    recs = g[k]
    m = ShellMesh(recs[0]["r_over_t"], recs[0]["l_over_r"], elems_per_half_wave=3.0)
    by_mode = defaultdict(list)
    for x in recs:
        by_mode[x["n_dominant"]].append(x)
    picks = [by_mode[n][0] for n in sorted(by_mode)]

    npick = len(picks)
    fig = plt.figure(figsize=(1.95 * npick, 4.1))
    gs = fig.add_gridspec(2, npick, height_ratios=[1.25, 1.0], hspace=0.55,
                          wspace=0.30)

    prof = []
    for c, x in enumerate(picks):
        w = np.load(os.path.join(PROD, x["key"], "draw.npz"))["w"]
        gr = to_grid(m, w)
        # Only EVEN lattice rows are complete: S8R is serendipity, so the
        # odd-odd lattice points carry no node. Zero-filling them (the obvious
        # thing) injects an alternating real/zero sawtooth that swamps the
        # circumferential signal and makes every mode look identical.
        gr = gr[:, ::2]
        a = fig.add_subplot(gs[0, c])
        v = np.abs(gr).max()
        a.imshow(gr.T, aspect="auto", origin="lower", cmap="RdBu_r",
                 vmin=-v, vmax=v, extent=[0, 360, 0, 1])
        a.set_title(r"$n=%d$" % x["n_dominant"], fontsize=8.5)
        a.set_xticks([0, 180, 360])
        a.set_xlabel(r"$\theta$ [deg]", fontsize=8)
        a.grid(False)
        if c == 0:
            a.set_ylabel(r"$z/L$")
        j = int(np.argmax(np.max(np.abs(gr), axis=0)))
        prof.append((x["n_dominant"], gr[:, j]))

    b = fig.add_subplot(gs[1, :])
    th = np.linspace(0, 360, prof[0][1].size, endpoint=False)
    for (n, p), col in zip(prof, ("#1f4e79", "#c1440e", "#2e7d32", "#6a1b9a")):
        b.plot(np.r_[th, 360.], np.r_[p, p[0]], lw=1.1, color=col, label=r"$n=%d$" % n)
    b.set_xlim(0, 360)
    b.set_xticks([0, 90, 180, 270, 360])
    b.set_xlabel(r"circumferential angle $\theta$ [deg]")
    b.set_ylabel(r"$w/t$")
    lo, hi = b.get_ylim()
    b.set_ylim(lo, hi + 0.42 * (hi - lo))      # headroom so the legend clears the curves
    b.legend(frameon=False, ncol=len(prof), loc="upper center", fontsize=8)
    b.set_title("circumferential profile at the most-deformed station",
                fontsize=8.5)

    fig.suptitle(r"One geometry ($R/t=%d$, $L/R=%.1f$): %d withheld realisations"
                 % (int(recs[0]["r_over_t"]), float(recs[0]["l_over_r"]), npick),
                 fontsize=9.5, y=0.99)
    p = os.path.join(FIG, "buckling_fields.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", p, "(%d modes)" % npick)


def main():
    # the dataset-overview figure lives in its own module
    import fig_dataset
    fig_dataset.main()
    fig_saoi_ranking()
    fig_failure_modes()
    fig_shrinkage()
    fig_buckling_modes()
    fig_buckling_fields()


if __name__ == "__main__":
    main()
