"""Concept figures for paper/sdfflow/sdfflow.tex (no model output):

  fig_sdf2d.png      signed distance of a hex-nut cross-section, and the +-clamp_dist cut
  fig_schedule.png   stage-1 training schedule, read from the checked-in ex3 train config
  fig_kl_budget.png  penalty-weight budget map with every arm that was actually run
  fig_orient.png     symmetry-axis direction per ex3 class (counts measured on aarl, 2026-09-25)

Run from any directory:  python paper/sdfflow/figures/gen_concept.py
"""
import math
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
EX3_TRAIN = REPO / "configs/SDFFlow/geometry_generation/ex3/baseline/config_train_sdfflow.txt"
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, GREY, TEAL, GOLD = "#3b6fd0", "#c0392b", "#8a8f94", "#1f7a78", "#b07d12"


def read_config(path):
    vals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("%", 1)[0].strip()
        if line:
            k, _, v = line.partition(" ")
            vals[k] = v.strip()
    return vals


def sci(v, _=None):
    if v <= 0:
        return ""
    e = math.floor(math.log10(v) + 1e-9)
    m = v / 10 ** e
    return f"1e{e}" if abs(m - 1) < 1e-6 else f"{m:g}e{e}"


# ------------------------------------------------------------------ signed distance, 2-D
def sd_hexagon(px, py, r):
    """Exact signed distance to a flat-topped regular hexagon of apothem r."""
    kx, ky, kz = -0.866025404, 0.5, 0.577350269
    px, py = np.abs(px), np.abs(py)
    d = np.minimum(kx * px + ky * py, 0.0)
    px, py = px - 2 * d * kx, py - 2 * d * ky
    px = px - np.clip(px, -kz * r, kz * r)
    py = py - r
    return np.hypot(px, py) * np.sign(py)


def fig_sdf2d(cfg):
    clamp = float(cfg["clamp_dist"])
    n = 600
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    hole = 0.30
    D = np.maximum(sd_hexagon(X, Y, 0.62), hole - np.hypot(X, Y))  # nut = hexagon minus hole
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    lim = 0.45
    im = axs[0].imshow(D, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[0].contour(X, Y, D, levels=np.arange(-0.4, 0.45, 0.1), colors="k", linewidths=0.4, alpha=0.5)
    axs[0].contour(X, Y, D, levels=[0], colors="k", linewidths=2.2)
    axs[0].set_title("(a) 부호 거리: 표면까지의 거리에 부호를 붙인 값", fontsize=10.5, loc="left")
    axs[0].text(0.0, 0.47, "재료 안쪽: 음수 (파랑)", ha="center", fontsize=9, color="#123f8c",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    axs[0].text(0.0, 0.0, "구멍 = 바깥\n양수 (빨강)", ha="center", va="center", fontsize=8.5, color="#8c1d12",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    axs[0].text(0.0, -0.88, "바깥: 양수 (빨강),  굵은 선 = 표면 (값 0)", ha="center", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.02)
    cb.set_label("부호 거리", labelpad=6)
    Dc = np.clip(D, -clamp, clamp)
    im2 = axs[1].imshow(Dc, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r", vmin=-clamp, vmax=clamp)
    axs[1].contour(X, Y, D, levels=[0], colors="k", linewidths=2.2)
    axs[1].contour(X, Y, D, levels=[-clamp, clamp], colors="k", linewidths=0.8, linestyles="--")
    axs[1].set_title(f"(b) 학습에서는 ±{clamp:g}에서 자른다 (clamp_dist {clamp:g})", fontsize=10.5, loc="left")
    axs[1].text(0.0, -0.88, "점선 사이 띠 안에서만 값이 변한다", ha="center", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.02)
    cb2.set_label("잘라낸 부호 거리", labelpad=10)
    for ax in axs[:2]:
        ax.axhline(0, color="#f0a030", lw=1.2, ls=":")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_aspect("equal")
    row = D[n // 2]
    axs[2].plot(xs, row, color=GREY, lw=1.2, label="부호 거리 (원래 값)")
    axs[2].plot(xs, np.clip(row, -clamp, clamp), color=BLUE, lw=2.4, label=f"학습 목표 (±{clamp:g}에서 자름)")
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].fill_between(xs, -0.5, 0.5, where=row < 0, color="#3b6fd0", alpha=0.08, lw=0)
    axs[2].set_ylim(-0.4, 0.5); axs[2].set_xlim(-1, 1)
    axs[2].set_xlabel("x (주황 점선을 따라감)")
    axs[2].set_ylabel("부호 거리")
    axs[2].set_title("(c) 주황 점선 위의 값: 표면을 지날 때 부호가 바뀐다", fontsize=10.5, loc="left")
    for x0, txt in ((-0.47, "재료"), (0.47, "재료"), (0.0, "구멍"), (-0.86, "바깥"), (0.86, "바깥")):
        axs[2].text(x0, 0.42, txt, ha="center", fontsize=9)
    axs[2].legend(fontsize=8.5, loc="lower center")
    axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_sdf2d.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ stage-1 schedule
def ramp(e, start, length):
    """training_profiles/train_vae.py::_warmup_scale."""
    e = np.asarray(e, float)
    if length <= 0:
        return (e >= start).astype(float)
    return np.where(e < start, 0.0, np.minimum((e - start + 1) / length, 1.0))


def fig_schedule(cfg):
    E = int(cfg["vae_training_epochs"])
    det = int(cfg["deterministic_warmup_epochs"])
    nz = int(cfg["posterior_noise_warmup_epochs"])
    klw = int(cfg["kl_warmup_epochs"])
    nmax = float(cfg["posterior_noise_max_scale"])
    lr0 = float(cfg["vae_learningr"])
    lrw = int(cfg["vae_warmup_epochs"])
    e = np.arange(E)
    noise = nmax * ramp(e, det, nz)
    kl = ramp(e, det, klw)
    # training_profiles/setup.py::build_optimizer_scheduler: LinearLR(0.01 -> 1, lrw) then cosine to 1e-8
    t0 = max(E - lrw, 1)
    lr = np.where(e < lrw, 0.01 + 0.99 * e / lrw,
                  (1e-8 + (lr0 - 1e-8) * 0.5 * (1 + np.cos(np.pi * (e - lrw) / t0))) / lr0)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axvspan(0, det, color=GREY, alpha=0.18, lw=0)
    ax.axvspan(det, det + nz, color="#f0a030", alpha=0.20, lw=0)
    ax.axvspan(det + nz, det + klw, color="#f0a030", alpha=0.08, lw=0)
    ax.plot(e, noise, color=RED, lw=2.4, label=f"섞는 잡음의 크기 (0 → {nmax:g})")
    ax.plot(e, kl, color=BLUE, lw=2.4, label=f"퍼짐 벌점의 적용 비율 (0 → 1, 실제 가중치는 × kl_weight)")
    ax.plot(e, lr, color=GREY, lw=1.6, ls="--", label=f"학습 속도 / 최대값 {lr0:g}  (처음 {lrw}바퀴 동안 올린 뒤 서서히 줄임)")
    best = det + max(nz, klw)
    ax.axvline(best, color="k", lw=1, ls=":")
    ax.text(best + 8, 0.52, f"{best}바퀴부터 점검 오차를\n서로 비교할 수 있다\n(그 전에는 잡음과 벌점이\n아직 커지는 중)", fontsize=8.8, va="center")
    ax.axvline(500, color=TEAL, lw=1, ls=":")
    ax.text(505, 0.14, "DeepJEB 실험(J1–J3)은 총 500바퀴\n(학습 속도도 500에 맞춰 줄어듦)", fontsize=8.8, color=TEAL)
    for y, x0, x1, txt, c in ((1.40, 0, det, f"0–{det}바퀴: 그대로 복원만", "k"),
                              (1.27, det, det + nz, f"{det}–{det + nz}바퀴: 잡음을 서서히 키움", "#8a4f00"),
                              (1.14, det, det + klw, f"{det}–{det + klw}바퀴: 벌점을 서서히 키움", "#8a4f00")):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="|-|", color=c, lw=1.0, shrinkA=0, shrinkB=0))
        ax.text(x1 + 6, y, txt, fontsize=8.8, va="center", color=c)
    ax.set_xlim(0, E); ax.set_ylim(0, 1.5)
    ax.set_xlabel("학습 바퀴 수 (epoch)")
    ax.set_ylabel("배율")
    ax.legend(fontsize=8.8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.25)
    ax.set_title(f"1단계(압축 모델) 학습 일정 — ex3 설정 파일 값 그대로 (총 {E}바퀴)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_schedule.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ penalty budget map
def fig_kl_budget():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    xs = np.array([500, 3e5])
    ax.fill_between(xs, 1.6e-2 / xs, 1, color=RED, alpha=0.08, lw=0)
    ax.fill_between(xs, 1e-3 / xs, 1.6e-2 / xs, color=GOLD, alpha=0.10, lw=0)
    ax.fill_between(xs, 1e-14, 1e-3 / xs, color=TEAL, alpha=0.07, lw=0)
    ax.plot(xs, 1e-3 / xs, "-", color=TEAL, lw=1.2)
    ax.plot(xs, 1.6e-2 / xs, "--", color=RED, lw=1.0)
    ax.text(150000, 1e-3 / 150000 * 0.45, "가중치 × 숫자 개수 = 1e-3\n(이 선 아래는 ex3에서\n정상 확인)", fontsize=8.5, color=TEAL, va="top", ha="center")
    ax.text(150000, 1.6e-2 / 150000 * 1.5, "가중치 × 숫자 개수 = 1.6e-2\n(J0에서 붕괴 확인)", fontsize=8.5, color=RED, ha="center")
    ax.text(150000, 3.2e-8, "두 선 사이는\n아직 확인 안 됨", fontsize=8.5, color=GOLD, ha="center", va="center")
    pts = [(1024, 1e-4, "ex3 1e-4 (처음 설정): 붕괴\n살아 있는 칸 2/1024", RED, "x"),
           (1024, 1e-6, "ex3 1e-6: 정상", TEAL, "o"),
           (1024, 1e-8, "ex3 1e-8: 정상\nJ3 (DeepJEB, 대표점 32개): 학습 중", TEAL, "o"),
           (1024, 1e-10, "ex3 1e-10: 정상", TEAL, "o"),
           (16384, 1e-6, "J0 (DeepJEB 1e-6): 붕괴\n살아 있는 칸 7/16384", RED, "x"),
           (16384, 6.25e-8, "J4 제안 6.25e-8\n(ex3 1e-6과 같은 벌점 크기)", GOLD, "D"),
           (16384, 1e-8, "J1: 학습 중", BLUE, "s"),
           (16384, 1e-10, "J2: 학습 중", BLUE, "s")]
    for x, y, lab, c, m in pts:
        ax.scatter([x], [y], color=c, marker=m, s=95 if m == "x" else 60, zorder=3, linewidths=2.4 if m == "x" else 1)
        ax.text(x * 1.3, y, lab, fontsize=8.5, va="center", color=c)
    ax.scatter([1024], [1e-8], facecolors="none", edgecolors=BLUE, marker="s", s=240, linewidths=1.8, zorder=4)
    ax.set_xlim(500, 3e5); ax.set_ylim(3e-11, 5e-4)
    ax.set_xticks([1024, 16384])
    ax.set_xticklabels(["1024\n(대표점 32 × 32)", "16384\n(대표점 512 × 32)"])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(sci))
    ax.set_xlabel("형상 코드의 숫자 개수 (벌점이 이 개수만큼 더해진다)")
    ax.set_ylabel("퍼짐 벌점 가중치 (kl_weight)")
    ax.grid(alpha=0.25, which="major")
    ax.set_title("같은 가중치라도 코드 숫자가 16배면 벌점도 16배다", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_kl_budget.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ orientation per class
MCB_AXIS = {  # class: (z, x, y, oblique) -- aarl ex3_mcb.h5; axis = PCA eigenvector with the most distinct eigenvalue, |cos| > 0.95
    "육각 너트 (824)": (484, 229, 99, 12),
    "잠금 너트 (196)": (144, 28, 9, 15),
    "캐슬 너트 (181)": (87, 8, 11, 75),
    "홈 너트 (59)": (1, 54, 4, 0),
    "사각 너트 (45)": (26, 11, 5, 3),
}


def fig_orient():
    cols = [TEAL, BLUE, GOLD, "#9aa5a2"]
    labs = ["구멍 축이 z (세움)", "구멍 축이 x (누움)", "구멍 축이 y (누움)", "비스듬함 (좌표축에서 18° 넘게 기움)"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.0, 1]})
    names = list(MCB_AXIS)
    arr = np.array([MCB_AXIS[n] for n in names], float)
    frac = arr / arr.sum(1, keepdims=True)
    left = np.zeros(len(names))
    for j in range(4):
        ax.barh(names, frac[:, j], left=left, color=cols[j], label=labs[j], edgecolor="white")
        for i in range(len(names)):
            if arr[i, j] >= 8:
                ax.text(left[i] + frac[i, j] / 2, i, f"{int(arr[i, j])}", ha="center", va="center", color="white", fontsize=9.5)
        left += frac[:, j]
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("클래스 안에서의 비율")
    tot = arr.sum(0).astype(int)
    ax.set_title(f"(a) ex3: 너트 구멍 축의 방향 (전체 1305개: z {tot[0]}, x {tot[1]}, y {tot[2]}, 비스듬 {tot[3]})",
                 loc="left", fontsize=10.5)
    ax.legend(ncol=4, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    bx.axis("off")
    bx.set_title("(b) DeepJEB: 모든 형상이 같은 방향", loc="left", fontsize=10.5)
    txt = ("가장 긴 방향 → y축 : 2138 / 2138\n"
           "중간 방향 → x축 : 2137,  가장 짧은 방향 → z축 : 2137\n\n"
           "ex3는 같은 너트가 세워져 있기도, 누워 있기도 하다.\n"
           "모델은 방향마다 따로 배우므로, 방향이 다른\n"
           "두 형상 사이를 섞으면 중간에서 한 번에 뒤집힌다.\n\n"
           "구멍 축은 표면 점들이 퍼진 세 주축 가운데 길이가\n"
           "나머지 둘과 가장 다른 축으로 잡았다. 그 축이 x·y·z\n"
           "어느 것과도 18° 넘게 어긋나면 '비스듬함'으로 셌다.")
    bx.text(0.0, 0.92, txt, va="top", fontsize=9.6, linespacing=1.55)
    fig.tight_layout()
    fig.savefig(HERE / "fig_orient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = read_config(EX3_TRAIN)
    fig_sdf2d(cfg)
    fig_schedule(cfg)
    fig_kl_budget()
    fig_orient()
    print("written:", sorted(p.name for p in HERE.glob("fig_*.png")))
