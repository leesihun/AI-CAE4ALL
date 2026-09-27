"""Stage-1 (VAE) size figures for paper/sdfflow/sdfflow.tex:

  fig_vae_shapes.png  what goes in and out of every step of the encoder and the decoder, with the real sizes
                      of the DeepJEB configs (encoder_dim 256, latent_dim 32, fourier_bands 8, 4 blocks each)
  fig_vae_points.png  one real DeepJEB shape: the 6144 surface points the encoder reads, the T anchor points
                      picked from them (T = 32 and T = 512, same farthest-point rule as model/sdf_vae.py), and
                      the question points the decoder answers during training

Sizes follow methods/SDFFlow/model/sdf_vae.py (PointCloudEncoder, SDFDecoderAttention) and
general_modules/sdf_dataset.py (num_encoder_points / num_query_points drawn per access).
Run from any directory:  python paper/sdfflow/figures/gen_vae_shapes.py
"""
import pathlib

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
JEB = REPO / "dataset/geometry_generation/ex1_deepjeb.h5"
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, GREY, TEAL, GOLD = "#3b6fd0", "#c0392b", "#8a8f94", "#1f7a78", "#b07d12"

# ------------------------------------------------------------------ figure 1: sizes
ROWS_H = {"N": 3.2, "T": 1.1, "M": 3.2}          # drawn height of a table with that many rows
COLS_W = {3: 0.32, 6: 0.45, 51: 0.85, 54: 0.9, 256: 1.7, 64: 0.8, 32: 0.55, 1: 0.2}


def table(ax, x, yc, rows, cols, color, top, bottom, hl_rows=None):
    """A table of `rows` x `cols` numbers drawn as a box with faint grid lines; returns its box."""
    h, w = ROWS_H[rows], COLS_W[cols]
    y0 = yc - h / 2
    ax.add_patch(Rectangle((x, y0), w, h, facecolor=color, edgecolor="#333", lw=0.9, zorder=2))
    n_r = 9 if rows != "T" else 4
    for k in range(1, n_r):
        ax.plot([x, x + w], [y0 + h * k / n_r] * 2, color="white", lw=0.6, zorder=3)
    n_c = max(1, min(8, int(w / 0.2)))
    for k in range(1, n_c):
        ax.plot([x + w * k / n_c] * 2, [y0, y0 + h], color="white", lw=0.6, zorder=3)
    if hl_rows:
        for k in hl_rows:
            ax.add_patch(Rectangle((x, y0 + h * k / n_r), w, h / n_r, facecolor=RED, alpha=0.75,
                                   edgecolor="none", zorder=4))
    ax.text(x + w / 2, y0 + h + 0.18, top, ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.text(x + w / 2, y0 - 0.15, bottom, ha="center", va="top", fontsize=6.6, color="#333", linespacing=1.25)
    return x, y0, w, h


def arrow(ax, p, q, label=None, color="#333", ls="-", lab_dy=0.2, fs=6.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=5))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + lab_dy, label, ha="center", va="bottom",
                fontsize=fs, color=color, linespacing=1.2)


def block(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                facecolor=color, edgecolor=TEAL, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4, linespacing=1.3, zorder=6)


def tab(ax, xc, yc, rows, cols, color, top, bottom, hl_rows=None):
    """`table` placed by its centre x."""
    return table(ax, xc - COLS_W[cols] / 2, yc, rows, cols, color, top, bottom, hl_rows)


def fig_sizes(out):
    fig = plt.figure(figsize=(6.9, 5.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 25.6)
    ax.axis("off")
    NA, TA, ZC, QM = "#b8c7e0", "#e8b27a", "#e07b39", "#c9d6c3"

    # ---------------- encoder (top) ----------------
    ax.text(0.2, 25.3, "(가) 요약부: 표면 점 6144개 → 형상 코드 (대표점 T개 × 숫자 32개)", fontsize=9,
            fontweight="bold", color=TEAL, va="top")
    yc = 21.0
    a = tab(ax, 1.3, yc, "N", 6, NA, "6144 × 6", "표면 점마다\n좌표 3\n+ 면 방향 3")
    b = tab(ax, 4.1, yc, "N", 54, NA, "6144 × 54", "좌표를 사인·\n코사인 51개로\n+ 면 방향 3")
    c = tab(ax, 7.2, yc, "N", 256, NA, "6144 × 256", "표면 점마다\n숫자 256개\n(표면 점 표)",
            hl_rows=[1, 4, 7])
    arrow(ax, (a[0] + a[2] + 0.1, yc), (b[0] - 0.1, yc))
    arrow(ax, (b[0] + b[2] + 0.1, yc), (c[0] - 0.1, yc))
    d = tab(ax, 10.9, yc + 1.0, "T", 256, TA, "T × 256", "대표점 T개의\n줄만 복사", hl_rows=[0, 1, 2, 3])
    arrow(ax, (c[0] + c[2] + 0.1, yc + 1.0), (d[0] - 0.1, yc + 1.0), "T줄 고름", color=RED, lab_dy=0.12)
    bx0, bw = 12.8, 5.4
    block(ax, bx0, yc - 1.7, bw, 3.4,
          "4번 반복\n(1) 대표점 T개 각각이\n표면 점 6144개를 모두\n훑어 정보를 모음\n(2) 대표점끼리\n정보를 나눔",
          "#e3f2f1")
    arrow(ax, (d[0] + d[2] + 0.1, yc + 1.0), (bx0 - 0.1, yc + 1.0))
    arrow(ax, (c[0] + c[2] + 0.1, yc - 1.1), (bx0 - 0.1, yc - 1.1), color=BLUE, ls="--")
    ax.text(10.5, yc - 1.3, "훑어볼 대상:\n6144줄 전체", fontsize=6.6, color=BLUE, ha="center", va="top")
    e = tab(ax, 19.6, yc, "T", 256, TA, "T × 256", "대표점마다\n숫자 256개")
    arrow(ax, (bx0 + bw + 0.15, yc), (e[0] - 0.1, yc))
    f = tab(ax, 22.1, yc, "T", 64, TA, "T × 64", "64개로\n줄임")
    arrow(ax, (e[0] + e[2] + 0.1, yc), (f[0] - 0.1, yc))
    g1 = tab(ax, 24.1, yc + 1.3, "T", 32, TA, "평균 T × 32", "")
    g2 = tab(ax, 24.1, yc - 1.3, "T", 32, TA, "", "퍼짐 T × 32")
    arrow(ax, (f[0] + f[2] + 0.1, yc + 0.2), (g1[0] - 0.1, yc + 1.1))
    arrow(ax, (f[0] + f[2] + 0.1, yc - 0.2), (g2[0] - 0.1, yc - 1.1))
    z = tab(ax, 26.8, yc, "T", 32, ZC, "형상 코드\nT × 32", "J3: 1,024개\nJ1·J2: 16,384개")
    arrow(ax, (g1[0] + g1[2] + 0.1, yc + 1.1), (z[0] - 0.1, yc + 0.25))
    arrow(ax, (g2[0] + g2[2] + 0.1, yc - 1.1), (z[0] - 0.1, yc - 0.25))
    ax.text(24.6, yc - 3.1, "학습 중: 평균에 퍼짐만큼 흔들림을 섞음\n학습 뒤: 평균만 씀", fontsize=6.4,
            ha="center", va="top", color="#555")

    # ---------------- decoder (bottom) ----------------
    ax.text(0.2, 15.3, "(나) 복원부: 질문 점 M개 + 형상 코드 → 질문 점마다 부호 거리 1개", fontsize=9,
            fontweight="bold", color=BLUE, va="top")
    yc2 = 8.6
    h = tab(ax, 1.3, yc2, "M", 3, QM, "M × 3", "질문 점마다\n좌표 3\n(공간 어디든)")
    i = tab(ax, 4.1, yc2, "M", 51, QM, "M × 51", "사인·코사인\n51개로 늘림")
    j = tab(ax, 7.2, yc2, "M", 256, QM, "M × 256", "질문 점마다\n숫자 256개")
    arrow(ax, (h[0] + h[2] + 0.1, yc2), (i[0] - 0.1, yc2))
    arrow(ax, (i[0] + i[2] + 0.1, yc2), (j[0] - 0.1, yc2))
    bx1 = 9.2
    block(ax, bx1, yc2 - 1.7, bw, 3.4,
          "4층\n질문 점 M개 각각이\n대표점 T개(참고표)를\n훑어 자기 위치에\n필요한 정보를 모음", "#e3ecf8")
    arrow(ax, (j[0] + j[2] + 0.1, yc2), (bx1 - 0.1, yc2))
    yr = 12.7
    zr = tab(ax, 10.3, yr, "T", 32, ZC, "", "")
    ax.text(zr[0] - 0.15, yr, "형상 코드 T × 32\n((가)의 결과)", fontsize=6.6, ha="right", va="center",
            color="#8a3d10")
    zr2 = tab(ax, 13.2, yr, "T", 256, TA, "참고표 T × 256", "")
    arrow(ax, (zr[0] + zr[2] + 0.1, yr), (zr2[0] - 0.1, yr))
    arrow(ax, (13.2, zr2[1] - 0.1), (13.2, yc2 + 1.8), color=GOLD, ls="--")
    ax.text(13.45, zr2[1] - 0.25, "훑어볼 대상:\n대표점 T줄", fontsize=6.6, color=GOLD, ha="left", va="top")
    k = tab(ax, 16.7, yc2, "M", 256, QM, "M × 256", "")
    arrow(ax, (bx1 + bw + 0.15, yc2), (k[0] - 0.1, yc2))
    s = tab(ax, 19.2, yc2, "M", 1, "#d9534f", "M × 1", "질문 점마다\n부호 거리 1개")
    arrow(ax, (k[0] + k[2] + 0.1, yc2), (s[0] - 0.1, yc2))
    ax.text(19.9, yc2 + 0.3,
            "M = 학습할 때 8,192\n     (형상마다 새로 뽑음)\nM = STL 만들 때 128³ = 2,097,152\n     (65,536개씩 나눠 물음)",
            fontsize=6.8, ha="left", va="center", linespacing=1.35)

    # ---------------- take-away box ----------------
    ax.add_patch(FancyBboxPatch((0.4, 0.3), 29.1, 3.7, boxstyle="round,pad=0.08,rounding_size=0.2",
                                facecolor="#fbf3e6", edgecolor=GOLD, lw=1.0))
    ax.text(0.9, 3.75, "헷갈리기 쉬운 점", fontsize=8, fontweight="bold", color="#7a5608", va="top")
    ax.text(0.9, 3.0, "✗  (T × 256) → 요약부 → 형상 코드 → 복원부 → (T × 1)", fontsize=7.6, va="top",
            color=RED)
    ax.text(0.9, 2.3, "✓  질문 점 (M × 3)  +  형상 코드 (T × 32)  →  복원부  →  부호 거리 (M × 1)",
            fontsize=7.6, va="top", color=TEAL)
    ax.text(0.9, 1.55,
            "대표점 T개는 답을 내는 자리가 아니라, 질문 점이 훑어보는 참고 줄이다. 답의 개수는 질문 점 수 M이고 T와 상관없다.\n"
            "T가 32든 512든 복원부는 공간의 아무 점에서나 답하므로, 128³ 격자 전체의 부호 거리를 얻어 STL 표면을 뽑는다.",
            fontsize=7.0, va="top", color="#333", linespacing=1.45)
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2: one real shape
def fps(points, k):
    """Same rule as model/sdf_vae.py::farthest_point_sample: start farthest from the centroid."""
    cur = int(((points - points.mean(0)) ** 2).sum(1).argmax())
    idx = np.empty(k, dtype=int)
    dmin = np.full(len(points), np.inf)
    for i in range(k):
        idx[i] = cur
        dmin = np.minimum(dmin, ((points - points[cur]) ** 2).sum(1))
        cur = int(dmin.argmax())
    return idx


def fig_points(out, shape_id="00000"):
    with h5py.File(JEB, "r") as f:
        g = f["shapes"][shape_id]
        surf = g["surface_points"][()]
        sdf_p = g["sdf_points"][()]
        sdf_v = g["sdf_values"][()]
    rng = np.random.default_rng(0)
    enc = surf[rng.choice(len(surf), 6144, replace=False)]
    qi = rng.choice(len(sdf_p), 8192, replace=False)
    qp, qv = sdf_p[qi], np.clip(sdf_v[qi], -0.1, 0.1)

    # view along the axis of smallest extent so the bracket is seen flat
    ext = surf.max(0) - surf.min(0)
    ax_h, ax_v = [a for a in np.argsort(ext)[::-1][:2]]
    lim_lo = np.minimum(surf.min(0), -0.05)[[ax_h, ax_v]] - 0.05
    lim_hi = np.maximum(surf.max(0), 0.05)[[ax_h, ax_v]] + 0.05

    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.0))
    titles = ["① 요약부 입력: 표면 점\n6144개 (저장된 8192개 중)",
              "② 대표점 T = 32\n(J3, ex3)",
              "③ 대표점 T = 512\n(J0·J1·J2)",
              "④ 복원부 학습 질문 점\n8192개, 색 = 정답 부호 거리"]
    for a in axs:
        a.set_aspect("equal")
        a.set_xlim(lim_lo[0], lim_hi[0])
        a.set_ylim(lim_lo[1], lim_hi[1])
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#bbb")
    axs[0].scatter(enc[:, ax_h], enc[:, ax_v], s=0.4, c=BLUE, lw=0)
    for a, T in ((axs[1], 32), (axs[2], 512)):
        a.scatter(enc[:, ax_h], enc[:, ax_v], s=0.3, c="#c8ccd2", lw=0)
        sel = enc[fps(enc, T)]
        a.scatter(sel[:, ax_h], sel[:, ax_v], s=9 if T == 32 else 2.2, c=RED, lw=0)
    sc = axs[3].scatter(qp[:, ax_h], qp[:, ax_v], s=0.5, c=qv, cmap="coolwarm", vmin=-0.1, vmax=0.1, lw=0)
    for a, t in zip(axs, titles):
        a.set_title(t, fontsize=7.2, linespacing=1.25)
    cb = fig.colorbar(sc, ax=axs[3], fraction=0.05, pad=0.03, shrink=0.62, ticks=[-0.1, 0, 0.1])
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_yticklabels(["안 −0.1", "표면 0", "밖 +0.1"])
    fig.text(0.5, 0.02, f"DeepJEB 형상 {shape_id}. 두 방향을 펼쳐 본 그림(가장 얇은 방향으로 눌러 봄). "
             "④의 점은 표면 근처 8192개 + 상자 전체 2048개 중에서 뽑으며, 정답은 ±0.1에서 자른다.",
             ha="center", fontsize=6.4, color="#444")
    fig.subplots_adjust(left=0.01, right=0.90, top=0.78, bottom=0.10, wspace=0.08)
    fig.savefig(out, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_sizes(HERE / "fig_vae_shapes.png")
    fig_points(HERE / "fig_vae_points.png")
    print("ok")
