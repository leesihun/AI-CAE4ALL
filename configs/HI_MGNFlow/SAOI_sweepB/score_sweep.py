#!/usr/bin/env python
"""Score every arm of the cHI-MGNflow SAOI Wave-B sweep into ONE report.

Two independent signals are collected per arm, because neither alone is enough:

  * From the TRAINING LOG (config's log_file_dir + the run's stdout
    transcript): best CRPS and the final `Train fm=` / `Valid fm=` losses.
    CRPS is the sampling-path score -- the number `best_by crps` selects on,
    and the only training-time metric that mirrors what inference does.
  * From the INFERENCE stage: the warpage spread distribution, i.e.
    max(z_disp) - min(z_disp) per realization, generated against the held-out
    eval set's ground truth. A model can improve its one-step regression loss
    every epoch while the sampled distribution rots, and peak-to-valley
    warpage is the number the part is actually judged on.

There is no rank-histogram / verification-rank stage here (unlike the
variational tree's misc/eval_distribution.py): that tool samples through a
learned latent prior, which this method does not have. The warpage comparison
below is this sweep's calibration signal instead.

Design: a 2^(4-1) resolution-IV half fraction, 8 arms, ONE per GPU. Defining
relation I = ABCD -- see gen_sweep_configs.py for the factor table.

Outputs (into --out-dir, default output/chi-mgnflow/saoi_sweepB):
    sweep_results.md    compact table -- this is the file to read/paste
    sweep_results.json  everything, including per-arm training-log detail
    warpage_<tag>.png   GT + all arms overlaid on one axis, per eval set

Usage:
    python configs/HI_MGNFlow/SAOI_sweepB/score_sweep.py
    python .../score_sweep.py --arms 1 8
"""
import argparse
import itertools
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
METHOD_REPO = REPO_ROOT / "methods" / "HI_MGNFlow"

# Keep in design order (A, B, C, D); arm N is design point N-1: <batch>_<t-sampling>_<capacity>_<lr>.
AXES = [
    ("batch_size",      ["b16", "b32"], ["16", "32"]),
    ("flow_t_sampling", ["tu", "tl"],   ["uniform", "logitnormal"]),
    ("capacity",        ["k0", "k1"],   ["128 / 4,6,8,6,4", "192 / 6,8,12,8,6"]),
    ("learningr",       ["lr1", "lr3"], ["1e-4", "3e-4"]),
]


def _design_tags():
    """Level tags for arms 1..8, in arm order.

    Same construction as gen_sweep_configs.arms(): D = A xor B xor C. The arm
    name is now just its 1-based index, so this -- not the name -- is what
    carries the design point. If one changes, the other must.
    """
    out = []
    for i in range(8):
        free = [(i >> (2 - k)) & 1 for k in range(3)]
        bits = free + [free[0] ^ free[1] ^ free[2]]      # D = A xor B xor C
        out.append([AXES[k][1][b] for k, b in enumerate(bits)])
    return out


DESIGN = _design_tags()
DEFAULT_ARMS = [str(i + 1) for i in range(8)]

# Confounded 2-factor pairs at resolution IV, by axis index (0=A .. 3=D).
# AB=CD, AC=BD, AD=BC -- an "effect" computed for (0,1) is really AB+CD, etc.
CONFOUND = {(0, 1): (2, 3), (0, 2): (1, 3), (0, 3): (1, 2)}


def arm_tags(arm):
    """Arm '3' -> its design point's level tags, e.g. ['b16', 'tl', 'k0', 'lr3'].

    Arms used to be named after their levels ('b16_tl_k0_lr3') and this was a
    string split. They are numbered now, so the levels come from the arm's
    position in the half fraction instead. Returns [] for anything outside
    1..8; callers already gate on len(...) == len(AXES).
    """
    try:
        idx = int(arm) - 1
    except (TypeError, ValueError):
        return []
    return DESIGN[idx] if 0 <= idx < len(DESIGN) else []


# Eval sets each arm is inferred on (gen_sweep_configs.INFER_SOURCES).
INFER_TAGS = ["s26fe_main", "s26fe_sec", "sm_l345u_main"]


def parse_config(path):
    """Minimal reader for the native flat `key value` format.

    Only the handful of keys this report needs. Mirrors the native quirks that
    matter here: `%` starts a comment line, `#` starts an inline comment, and
    key/value are split on the first run of whitespace.
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0].strip().lower()] = parts[1].strip()
    return out


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_training_log(log_path, transcript_path=None):
    """Pull best CRPS / final losses / ensemble-spread ratio out of a run's logs.

    The trainer splits these across TWO files, so both are read:

      * `log_path` -- the config's `log_file_dir`, one `f.write()` line per
        epoch: "Elapsed: ...s Epoch N LR: ... | Train fm=... | Valid fm=...
        | CRPS ... spread ...". The minimum CRPS across lines is the best the
        run achieved.
      * `transcript_path` -- run_sweep.sh's per-arm stdout capture.
        `[FlowDiag]` lines are emitted with `tqdm.write`, i.e. to STDOUT ONLY;
        they never reach the epoch log, so the one-step deterministic MSE
        (`det(1fwd) mse=`) comes from here and is simply absent without a
        transcript.

    Every field degrades to None rather than raising: an arm that died early,
    logged nothing, or ran with `val_interval > 1` (most epoch lines then carry
    no CRPS at all) must still produce a report row.
    """
    res = {"best_crps": None, "final_train_fm": None, "final_valid_fm": None,
           "final_epoch": None, "min_spread_ratio": None, "det_mse": None,
           "sources": [], "searched": []}

    num = r"([-+0-9.eE]+)"
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    crps_re = re.compile(r"CRPS\s+" + num)
    train_re = re.compile(r"Train\s+fm=" + num)
    valid_re = re.compile(r"Valid\s+fm=" + num)
    spread_re = re.compile(r"spread(?:/gt)?[= ]" + num)
    det_re = re.compile(r"det\(1fwd\)\s+mse=" + num)

    def as_float(m):
        # `search` returns None on every epoch line without the field -- with
        # val_interval > 1 that is most of them. `None.group()` raises
        # AttributeError, which a plain TypeError/ValueError guard does not
        # catch, and one such line would abort the entire report.
        if m is None:
            return None
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return None

    def readable(path):
        return path is not None and path.exists()

    crps_vals, spread_vals, last = [], [], None

    # Per-epoch numbers: the epoch log is authoritative. Fall through on a
    # file that EXISTS but holds no epoch line, not just on a missing one --
    # `init_log_file` creates the epoch log (header + embedded config) before
    # the first epoch runs, so a stale or misdirected log is present-but-empty
    # and would otherwise shadow a perfectly good transcript.
    for path in (log_path, transcript_path):
        if not readable(path):
            if path is not None:
                res["searched"].append(f"{path}  (missing)")
            continue
        res["searched"].append(str(path))
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "Epoch" not in line:
                    continue
                c = as_float(crps_re.search(line))
                if c is not None:
                    crps_vals.append(c)
                last = line
        if last is not None:
            res["sources"].append(path.name)
            break
        crps_vals.clear()

    # FlowDiag: stdout-only (tqdm.write), so the transcript is the real source.
    for path in (transcript_path, log_path):
        if not readable(path):
            continue
        found = False
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "FlowDiag" not in line:
                    continue
                m = spread_re.search(line)
                if m and _is_float(m.group(1)):
                    spread_vals.append(float(m.group(1)))
                    found = True
                m = det_re.search(line)
                if m and _is_float(m.group(1)):
                    res["det_mse"] = float(m.group(1))
                    found = True
        if found:
            if path.name not in res["sources"]:
                res["sources"].append(path.name)
            break

    if crps_vals:
        res["best_crps"] = min(crps_vals)
    if spread_vals:
        res["min_spread_ratio"] = min(spread_vals)
    if last:
        m = epoch_re.search(last)
        res["final_epoch"] = int(m.group(1)) if m else None
        res["final_train_fm"] = as_float(train_re.search(last))
        res["final_valid_fm"] = as_float(valid_re.search(last))
    return res


def fmt(v, spec=".2e", dash="-"):
    return dash if v is None else format(v, spec)


def _w1(a, b, n=512):
    """1-Wasserstein distance between two 1-D empirical distributions.

    Computed as the mean |difference| of their quantile functions, which is the
    exact 1-D form of W1 and needs no equal sample sizes -- ground truth has one
    value per eval scene while the generated side has scene x draws.
    """
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def load_spreads(cfg_dir, arm, tag):
    """(gt, gen) arrays written by the rollout, or None.

    Reads inference_output_dir out of the generated inference config rather
    than rebuilding the path, so a change in the generator cannot silently
    desync this script from it.
    """
    cfg_path = cfg_dir / f"config_infer_{arm}_{tag}.txt"
    if not cfg_path.exists():
        return None
    out_dir = parse_config(cfg_path).get("inference_output_dir")
    if not out_dir:
        return None
    npz = (METHOD_REPO / out_dir / "spread_values.npz").resolve()
    if not npz.exists():
        return None
    try:
        with np.load(npz) as z:
            return np.asarray(z["gt"], float), np.asarray(z["gen"], float)
    except (OSError, KeyError, ValueError) as exc:
        print(f"  [warn] unreadable {npz}: {exc}", flush=True)
        return None


def collect_warpage(rows):
    """Attach {tag: stats} to each row from the inference spread dumps."""
    for r in rows:
        w = {}
        for tag in INFER_TAGS:
            got = load_spreads(HERE, r["arm"], tag)
            if got is None:
                continue
            gt, gen = got
            if gt.size == 0 or gen.size == 0:
                continue
            sd = float(gt.std()) or 1.0
            w[tag] = {
                "gt_mean": float(gt.mean()), "gt_std": float(gt.std()),
                "gen_mean": float(gen.mean()), "gen_std": float(gen.std()),
                "n_gt": int(gt.size), "n_gen": int(gen.size),
                # Normalized by the GT spread's own std, so the three eval
                # sets (different parts, different magnitudes) are comparable.
                "w1_norm": _w1(gt, gen) / sd,
                "dmean_norm": (float(gen.mean()) - float(gt.mean())) / sd,
                "std_ratio": float(gen.std()) / sd,
            }
        if w:
            r["warpage"] = w


def render_warpage(rows):
    """Table of the physical quantity the sweep is actually for.

    spread = max(z_disp) - min(z_disp) per realization, i.e. peak-to-valley
    warpage. Ground truth gives one value per eval scene; the model gives
    num_vae_samples values per scene. A model can have good CRPS and still get
    this distribution wrong, and peak-to-valley is the number the part is
    judged on, so this table is the one to read alongside CRPS.

      W1/sd     1-Wasserstein between the two distributions, in units of the GT
                spread's own std. 0 = identical. THE ranking column.
      dmean/sd  systematic bias: >0 the model warps more than reality.
      sd ratio  generated spread-of-spreads over GT's. 1 = right dispersion,
                <1 under-dispersed (the classic failure), >1 over-dispersed.
    """
    have = [r for r in rows if r.get("warpage")]
    if not have:
        return ("## Warpage spread (max - min z_disp)\n\n"
                "No inference spread dumps found -- run with INFER=1, or check "
                "`[SPREAD]` lines in the per-arm inference logs.")
    L = ["## Warpage spread (max - min z_disp), generated vs ground truth", ""]
    hdr = "| arm |" + "".join(f" W1/sd {t} | dmean/sd {t} | sd ratio {t} |"
                              for t in INFER_TAGS)
    L.append(hdr)
    L.append("|" + "---|" * (1 + 3 * len(INFER_TAGS)))

    def key(r):
        v = [d["w1_norm"] for d in r["warpage"].values()]
        return sum(v) / len(v)

    for r in sorted(have, key=key):
        cells = []
        for t in INFER_TAGS:
            d = r["warpage"].get(t)
            cells += ["-", "-", "-"] if d is None else [
                f"{d['w1_norm']:.3f}", f"{d['dmean_norm']:+.3f}", f"{d['std_ratio']:.3f}"]
        L.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Sorted by mean W1/sd across eval sets: best first.")
    L.append("Per-arm plots: output/chi-mgnflow/saoi_sweepB/infer/<arm>/<tag>/histogram_compare.png")
    return "\n".join(L)


def plot_warpage_overlay(rows, out_dir):
    """One figure per eval set: GT filled, every arm's generated density on top.

    Eight separate PNGs cannot be compared by eye; this puts them on one axis,
    ordered and coloured by W1 so the ranking is visible without reading numbers.
    """
    have = [r for r in rows if r.get("warpage")]
    if not have:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for tag in INFER_TAGS:
        arms = [r for r in have if tag in r["warpage"]]
        if not arms:
            continue
        loaded = [(r["arm"], load_spreads(HERE, r["arm"], tag)) for r in arms]
        loaded = [(a, g) for a, g in loaded if g is not None]
        if not loaded:
            continue
        gt = loaded[0][1][0]
        ranked = sorted(
            loaded,
            key=lambda ag: next(r["warpage"][tag]["w1_norm"]
                                for r in arms if r["arm"] == ag[0]))
        lo = min([gt.min()] + [g.min() for _, (_, g) in ranked])
        hi = max([gt.max()] + [g.max() for _, (_, g) in ranked])
        edges = np.linspace(lo, hi, 61)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(gt, bins=edges, density=True, color="0.4", alpha=0.55,
                label=f"GROUND TRUTH (n={gt.size:,})")
        cmap = plt.get_cmap("viridis")
        for i, (arm, (_, gen)) in enumerate(ranked):
            w1 = next(r["warpage"][tag]["w1_norm"] for r in arms if r["arm"] == arm)
            ax.hist(gen, bins=edges, density=True, histtype="step", linewidth=1.3,
                    color=cmap(i / max(len(ranked) - 1, 1)),
                    label=f"{arm}  W1/sd={w1:.3f}")
        ax.set_title(f"warpage spread  max(z_disp) - min(z_disp)   --   eval set: {tag}\n"
                     "grey = ground truth; lines = arms, dark->light = best->worst W1")
        ax.set_xlabel("max(z_disp) - min(z_disp) per realization")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        path = pathlib.Path(out_dir) / f"warpage_{tag}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def render_main_effects(rows):
    """Average best CRPS over the 4 arms at each level of each factor.

    This is what a FRACTIONAL FACTORIAL buys over a one-factor-at-a-time scan:
    every arm contributes to all four comparisons, so each main effect is a
    4-vs-4 difference rather than a 1-vs-1 one. `delta` is (level 1 - level 0);
    lower CRPS is better, so a negative delta means level 1 wins.

    Read a main effect as provisional if the two levels' spreads overlap -- with
    one run per cell there is no replication, and a large interaction (see the
    table below) shows up here as a main effect averaging two opposite
    behaviours, which must not be read alone.
    """
    def crps(r):
        return (r.get("train_log") or {}).get("best_crps")

    L = ["## Main effects (4 arms per level)", ""]
    L.append("| factor | level 0 | level 1 | mean CRPS @0 | mean CRPS @1 | delta | n |")
    L.append("|" + "---|" * 7)
    for k, (key, tags, vals) in enumerate(AXES):
        cell = [[], []]
        for r in rows:
            try:
                lvl = tags.index(arm_tags(r["arm"])[k])
            except (ValueError, IndexError):
                continue
            v = crps(r)
            if v is not None:
                cell[lvl].append(float(v))
        if not cell[0] or not cell[1]:
            continue
        m0 = sum(cell[0]) / len(cell[0])
        m1 = sum(cell[1]) / len(cell[1])
        L.append(
            f"| `{key}` | {vals[0]} | {vals[1]} | "
            f"{m0:.4g} | {m1:.4g} | {m1 - m0:+.4g} | "
            f"{len(cell[0])}v{len(cell[1])} |"
        )
    return "\n".join(L)


def render_interactions(rows):
    """The three confounded 2-factor pairs on best CRPS, largest first.

    At resolution IV, the six C(4,2) pairs collapse into three ESTIMABLE
    quantities: AB+CD, AC+BD, AD+BC. What is printed here for e.g. `batch_size
    x flow_t_sampling` is actually that sum -- a large value means EITHER (or
    both) of the confounded pair is real, and telling them apart needs a
    follow-up run that breaks the alias.

    Effect for factors (A, B), standard two-level definition:
        AB = ( mean(A1B1) + mean(A0B0) - mean(A1B0) - mean(A0B1) ) / 2
    Negative = the (level-1, level-1) corner is better than additivity predicts.
    """
    def crps(r):
        return (r.get("train_log") or {}).get("best_crps")

    seen = set()
    out = []
    for a, b in itertools.combinations(range(len(AXES)), 2):
        pair = tuple(sorted((a, b)))
        if pair in seen:
            continue
        seen.add(pair)
        confound = CONFOUND.get(pair)
        if confound:
            seen.add(confound)
        ta, tb = AXES[a][1], AXES[b][1]
        cell, ok = {}, True
        for ia in (0, 1):
            for ib in (0, 1):
                vals = [
                    float(crps(r)) for r in rows
                    if crps(r) is not None
                    and len(arm_tags(r["arm"])) == len(AXES)
                    and arm_tags(r["arm"])[a] == ta[ia]
                    and arm_tags(r["arm"])[b] == tb[ib]
                ]
                if not vals:
                    ok = False
                cell[(ia, ib)] = sum(vals) / len(vals) if vals else None
        if not ok:
            continue
        eff = (cell[(1, 1)] + cell[(0, 0)] - cell[(1, 0)] - cell[(0, 1)]) / 2
        cname = f"{AXES[confound[0]][0]} x {AXES[confound[1]][0]}" if confound else "-"
        out.append((abs(eff), eff, AXES[a][0], AXES[b][0], cname, cell, ta, tb))
    if not out:
        return ""
    out.sort(reverse=True)

    L = ["## Confounded two-factor effects on best CRPS (largest first)", ""]
    L.append("| factor A | factor B | confounded with | effect | A0B0 | A0B1 | A1B0 | A1B1 |")
    L.append("|" + "---|" * 8)
    for _, eff, na, nb, cname, cell, ta, tb in out:
        L.append(
            f"| `{na}` | `{nb}` | `{cname}` | {eff:+.4g} | "
            f"{fmt(cell[(0, 0)])} | {fmt(cell[(0, 1)])} | "
            f"{fmt(cell[(1, 0)])} | {fmt(cell[(1, 1)])} |"
        )
    L.append("")
    L.append("Level order is the one in the axis legend above (A0 = first level).")
    L.append("Each printed value is the SUM of the named pair and its confound --")
    L.append("resolution IV cannot separate them without another run. Read one as")
    L.append("real only if it is large next to the spread of the four cell means")
    L.append("it is built from -- with one run per cell there is no replication,")
    L.append("so small values here are noise.")
    return "\n".join(L)


def render_markdown(rows):
    L = []
    L.append("# cHI-MGNflow SAOI Wave B -- 2^(4-1) resolution IV (8 arms, 1 per GPU)")
    L.append("")
    L.append("| tag | key | level 0 | level 1 |")
    L.append("|---|---|---|---|")
    for key, tags, vals in AXES:
        L.append(f"| {tags[0]} / {tags[1]} | `{key}` | {vals[0]} | {vals[1]} |")
    L.append("")
    L.append("Defining relation I = ABCD: `learningr` is not free, it is")
    L.append("`batch_size` xor `flow_t_sampling` xor `capacity`. The 4 main")
    L.append("effects are clean, but 2-factor effects come in CONFOUNDED PAIRS --")
    L.append("AB=CD, AC=BD, AD=BC -- so a large one cannot be attributed to a")
    L.append("single pair without another run (see the table below).")
    L.append("")
    L.append("**1000 epochs at a measured 500 s/epoch is a BUDGET-LIMITED")
    L.append("comparison, NOT a converged one.** Read every number here as")
    L.append("\"best at this budget\", not as an asymptotic ranking.")
    L.append("")
    L.append("`voronoi_clusters` and `flow_steps`/`flow_solver` are held fixed --")
    L.append("the former to avoid a second coarsening cache, the latter because")
    L.append("they are sampling-time choices answered by Wave A, not by training")
    L.append("runs. See gen_sweep_configs.py for the full rationale.")
    L.append("")
    L.append("**Read best CRPS, not the one-step `Valid fm` loss.** `Valid fm` is")
    L.append("the deterministic-ish one-step regression the trainer optimizes")
    L.append("directly; CRPS is the ensemble-sampling score that mirrors what")
    L.append("inference actually does, and the two can diverge as training")
    L.append("progresses. `spread ratio` is `[FlowDiag]`'s ensemble-spread /")
    L.append("ground-truth-std at validation time -- <1 under-dispersed, >1")
    L.append("over-dispersed, same reading as the warpage table's `sd ratio`")
    L.append("below but measured on the TRAIN split during training, not on a")
    L.append("held-out eval set.")
    L.append("")

    hdr = ("| arm | batch | t-sched | capacity | lr | best CRPS | valid fm | "
           "spread ratio | det mse |")
    L.append(hdr)
    L.append("|" + "---|" * 9)
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['arm']} | | | | | **{r['error']}** | | | |")
            continue
        tl = r.get("train_log", {})
        b, t, k, lr = arm_tags(r["arm"])
        L.append(
            f"| {r['arm']} | {b} | {t} | {k} | {lr} | "
            f"{fmt(tl.get('best_crps'))} | {fmt(tl.get('final_valid_fm'))} | "
            f"{fmt(tl.get('min_spread_ratio'), '.2f')} | "
            f"{fmt(tl.get('det_mse'))} |"
        )

    L.append("")
    L.append(render_main_effects(rows))
    L.append("")
    L.append(render_interactions(rows))
    L.append("")
    L.append(render_warpage(rows))
    L.append("")
    L.append("## Per-arm detail")
    for r in rows:
        L.append("")
        L.append(f"### {r['arm']}")
        if r.get("error"):
            L.append(f"- ERROR: {r['error']}")
            continue
        tl = r.get("train_log", {})
        # Name the files that were actually read: a row of '-' means "no log
        # found", which looks identical to "the run produced no numbers".
        L.append(f"- final epoch {fmt(tl.get('final_epoch'), '.0f')}, "
                 f"train fm {fmt(tl.get('final_train_fm'))}, "
                 f"valid fm {fmt(tl.get('final_valid_fm'))} "
                 f"[logs read: {', '.join(tl.get('sources') or []) or 'NONE FOUND'}]")
        if not tl.get("sources"):
            # Print the exact paths, because "no numbers" and "wrong path" look
            # identical in the table and only one of them is a training problem.
            L.append("  - no epoch lines found. Searched: "
                     + ("; ".join(tl.get("searched") or []) or "(nothing)"))
        if r.get("warpage"):
            for tag, d in r["warpage"].items():
                L.append(f"- `{tag}`: W1/sd {d['w1_norm']:.3f}, "
                         f"dmean/sd {d['dmean_norm']:+.3f}, "
                         f"sd ratio {d['std_ratio']:.3f} "
                         f"(n_gt={d['n_gt']}, n_gen={d['n_gen']})")
        else:
            L.append("- no warpage spread dump found for this arm")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--python", default=sys.executable,
                    help="unused by this script; accepted so run_sweep.sh can "
                         "pass the same $PYTHON to every stage uniformly")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweepB"))
    ap.add_argument("--run-logs",
                    default=str(REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweepB" / "run_logs"),
                    help="run_sweep.sh's per-arm stdout transcripts; the only "
                         "place [FlowDiag] lines are recorded")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_logs = pathlib.Path(args.run_logs)

    rows = []
    for arm in args.arms:
        cfg_path = HERE / f"config_train_{arm}.txt"
        row = {"arm": arm, "config": str(cfg_path)}
        if not cfg_path.exists():
            row["error"] = "config not found"
            rows.append(row)
            print(f"[{arm}] SKIP: no config", flush=True)
            continue

        cfg = parse_config(cfg_path)
        row.update({
            "batch_size": cfg.get("batch_size"),
            "flow_t_sampling": cfg.get("flow_t_sampling"),
            "latent_dim": cfg.get("latent_dim"),
            "mp_per_level": cfg.get("mp_per_level"),
            "learningr": cfg.get("learningr"),
            "training_epochs": cfg.get("training_epochs"),
        })

        ckpt = (METHOD_REPO / cfg.get("modelpath", "")).resolve()
        row["checkpoint"] = str(ckpt)
        if not ckpt.exists():
            row["error"] = "checkpoint missing (arm did not finish?)"
            rows.append(row)
            print(f"[{arm}] SKIP: no checkpoint at {ckpt}", flush=True)
            continue

        # log_file_dir is a plain cwd-relative path like every other path key
        # (training_profiles/setup.py::init_log_file), so it resolves against
        # the method repo directly. It used to be relative to <method>/outputs/,
        # and prepending that segment here made the lookup land in
        # methods/output/... -- the epoch log was never found and this silently
        # fell back to run_sweep.sh's transcript, which carries the same numbers
        # at lower precision. Keep the two in sync if init_log_file changes.
        log_path = (METHOD_REPO / cfg.get("log_file_dir", "")).resolve()
        transcript = run_logs / f"{arm}.log"
        row["train_log"] = parse_training_log(log_path, transcript)
        rows.append(row)
        print(f"[{arm}] logs parsed", flush=True)

    # Warpage spread (max - min z_disp) from the inference stage. Attached
    # before the JSON dump so the raw numbers land there too, and plotted so
    # all 8 arms can be compared on one axis instead of 8 separate PNGs.
    collect_warpage(rows)
    for path in plot_warpage_overlay(rows, out_dir):
        print(f"  warpage overlay -> {path}", flush=True)

    json_path = out_dir / "sweep_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")

    md = render_markdown(rows)
    md_path = out_dir / "sweep_results.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")

    # The report file is always UTF-8; the CONSOLE may not be (a cp949/cp1252
    # Windows terminal raises UnicodeEncodeError and would lose the whole run's
    # summary over a stray character). Degrade the echo, never the file.
    try:
        print("\n" + md)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print("\n" + md.encode(enc, "replace").decode(enc))
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
