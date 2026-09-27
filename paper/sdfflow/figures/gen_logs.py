"""Training-log figures for paper/sdfflow/sdfflow.tex, drawn from data/curves.json
(the real aarl logs, parsed by the scratchpad extract_curves.py):

  fig_collapse.png  live code channels and held-out error, every stage-1 run
  fig_epochs.png    is the epoch count enough? stage 1 and stage 2 held-out curves

Run from any directory:  python paper/sdfflow/figures/gen_logs.py
"""
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = pathlib.Path(__file__).resolve().parent
D = json.loads((HERE / "data/curves.json").read_text())
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, TEAL, BLUE, GREY, GOLD = "#c0392b", "#1f7a78", "#3b6fd0", "#8a8f94", "#b07d12"
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
PCT = FuncFormatter(lambda v, _: f"{100 * v:g}%")

RUNS = [  # key, label, colour, line style, width
    ("ex3_kl1e-4_baseline", "ex3 · 1e-4 (처음 설정)", RED, "-", 2.2),
    ("ex3_kl1e-6", "ex3 · 1e-6", TEAL, "-", 2.0),
    ("ex3_kl1e-8", "ex3 · 1e-8", "#2e9e7a", "--", 1.6),
    ("ex3_kl1e-10", "ex3 · 1e-10", "#71b8a4", ":", 1.8),
    ("jeb_J0_kl1e-6_t512", "DeepJEB J0 · 1e-6 · 대표점 512", "#e67e22", "-", 2.2),
    ("jeb_J1_kl1e-8_t512", "DeepJEB J1 · 1e-8 · 대표점 512", BLUE, "-", 2.0),
    ("jeb_J2_kl1e-10_t512", "DeepJEB J2 · 1e-10 · 대표점 512", "#6c8ee0", "--", 1.8),
    ("jeb_J3_kl1e-8_t32", "DeepJEB J3 · 1e-8 · 대표점 32", "#1b3f8b", ":", 2.2),
]


def bands(ax):
    ax.axvspan(0, 50, color=GREY, alpha=0.13, lw=0)
    ax.axvspan(50, 150, color="#f0a030", alpha=0.18, lw=0)
    ax.axvspan(150, 250, color="#f0a030", alpha=0.07, lw=0)


def fig_collapse():
    logs = D["vae_logs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.0))
    for key, lab, c, ls, lw in RUNS:
        rows = [r for r in logs.get(key, []) if r["snr"] is not None]
        if not rows:
            continue
        ep = np.array([r["ep"] for r in rows])
        fr = np.array([max(r["snr"], 0.5) / r["dims"] for r in rows])
        a.plot(ep, fr, ls, color=c, lw=lw, label=lab)
        va = [(r["ep"], r["valid"]) for r in rows if r["valid"] is not None]
        if va:
            b.plot(*zip(*va), ls, color=c, lw=lw, label=lab)
        last = rows[-1]
        if key in ("jeb_J0_kl1e-6_t512", "ex3_kl1e-4_baseline"):
            a.annotate(f"{last['snr']}/{last['dims']}", (last["ep"], fr[-1]), xytext=(6, 0),
                       textcoords="offset points", fontsize=8.5, color=c, va="center")
    for ax in (a, b):
        bands(ax)
        ax.set_yscale("log")
        ax.set_xlim(0, 1000)
        ax.set_xlabel("학습 바퀴 수 (epoch)")
        ax.grid(alpha=0.25, which="both")
    a.set_ylim(2e-4, 1.6)
    a.yaxis.set_major_formatter(PCT); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_ylabel("살아 있는 칸의 비율")
    a.set_title("(a) 형상 코드에서 살아 있는 칸의 비율 (로그 눈금)", loc="left", fontsize=10.5)
    b.yaxis.set_major_formatter(PLAIN); b.yaxis.set_minor_formatter(PLAIN)
    b.set_ylabel("점검용 형상의 부호 거리 오차")
    b.set_title("(b) 점검용 형상 오차 (ex3와 DeepJEB는 데이터가 달라 서로 높이 비교는 하지 않는다)", loc="left", fontsize=10.5)
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(HERE / "fig_collapse.png", dpi=150)
    plt.close(fig)


def fig_epochs():
    ea = D["epoch_analysis"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.2))
    for key, lab, c, ls in (("kl1e-6", "1e-6", TEAL, "-"), ("kl1e-8", "1e-8", "#2e9e7a", "--"),
                            ("kl1e-10", "1e-10", "#71b8a4", ":")):
        v = ea["vae"][key]
        a.plot(v["vep"], v["valid"], ls, color=c, lw=2, label=f"점검용 · {lab}")
        a.plot(v["ep"], v["train"], ls, color=c, lw=0.9, alpha=0.55)
        a.scatter([v["best_ep"]], [v["best"]], color=c, s=55, zorder=4, edgecolor="k")
        off = {"kl1e-6": (0, 12), "kl1e-8": (22, -20), "kl1e-10": (-28, -20)}[key]
        a.annotate(f"가장 좋음 {v['best_ep']}", (v["best_ep"], v["best"]), xytext=off,
                   textcoords="offset points", fontsize=8.5, color=c, ha="center")
    a.axvspan(500, 1000, color=TEAL, alpha=0.06, lw=0)
    a.text(750, 0.0091, "500바퀴 뒤로는 거의 평탄", ha="center", fontsize=9, color=TEAL)
    a.text(640, 0.0046, "옅은 선 = 학습용 형상 오차", fontsize=8.5, color=GREY)
    a.axvline(250, color="k", lw=0.8, ls=":")
    a.text(255, 0.0128, "250 전에는 비교 불가\n(잡음·벌점이 커지는 중)", fontsize=8.3, va="top")
    a.set_xlim(0, 1000); a.set_ylim(0.002, 0.014)
    a.yaxis.set_major_formatter(PLAIN)
    a.set_xlabel("학습 바퀴 수 (epoch)")
    a.set_ylabel("부호 거리 오차")
    a.set_title("(a) 1단계 압축 모델 (ex3, 1000바퀴): 충분하다", loc="left", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    a.grid(alpha=0.25, which="both")
    fm = ea["fm"]
    for key, lab, c, ls, lw in (("kl1e-8", "500바퀴 · 1e-8", "#2e9e7a", "-", 1.8),
                                ("kl1e-10", "500바퀴 · 1e-10", "#71b8a4", "-", 1.8),
                                ("F0_long (kl1e-6, 2000ep)", "2000바퀴 · 1e-6 · 기본 설정", TEAL, "-", 1.4),
                                ("F1_logit (2000ep)", "2000바퀴 · 1e-6 · 시간 뽑기 방식 변경", BLUE, "--", 1.3),
                                ("F2_logit_ema999 (2000ep)", "2000바퀴 · 1e-6 · 위 + 느린 평균본", "#1b3f8b", ":", 1.6)):
        v = fm[key]
        b.plot(v["vep"], v["valid"], ls, color=c, lw=lw, label=f"{lab} (가장 좋음 {v['best_ep']})")
        b.scatter([v["best_ep"]], [v["best"]], color=c, s=50, zorder=4, edgecolor="k")
    b.axvline(500, color="k", lw=0.8, ls=":")
    b.text(510, 1.3, "500바퀴 실행은 끝날 때까지\n오차가 줄고 있었다 (455–499)", fontsize=8.5)
    b.axvspan(530, 680, color=GOLD, alpha=0.15, lw=0)
    b.text(605, 0.80, "2000바퀴 실행의\n최저점 530–680", fontsize=8.5, ha="center", color="#7a5608")
    b.set_xlim(0, 2000); b.set_ylim(0.4, 2.1)
    b.set_xlabel("학습 바퀴 수 (epoch)")
    b.set_ylabel("점검용 형상의 이동 방향 오차")
    b.set_title("(b) 2단계 생성 모델 (ex3): 500은 조금 모자람, 약 700 권장", loc="left", fontsize=10.5)
    b.legend(fontsize=8.2, loc="upper right")
    b.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig_epochs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_collapse()
    fig_epochs()
    print("written: fig_collapse.png fig_epochs.png  (", D["source"], ")")
