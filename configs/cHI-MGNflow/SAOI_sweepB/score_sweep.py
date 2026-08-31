#!/usr/bin/env python
"""Score every arm of the cHI-MGNflow SAOI Wave-B sweep into ONE report.

Two independent signals are collected per arm, because neither alone is enough:

  * From the TRAINING LOG (config's log_file_dir): best CRPS and the final
    validation losses. CRPS is the sampling-path score -- the number
    `best_by crps` selects on, and the only training-time metric that mirrors
    what inference does.
  * From the INFERENCE stage: the warpage spread distribution, i.e.
    max(z_disp) - min(z_disp) per realization, generated against the held-out
    eval set's ground truth. A model can improve its one-step regression loss
    every epoch while the sampled distribution rots, and peak-to-valley warpage
    is the number the part is actually judged on.

There is no rank-histogram stage here: the variational tree's
misc/eval_distribution.py samples through a latent prior that this method does
not have. Porting it means reimplementing the draw loop against the ODE
sampler; until then the spread comparison plus CRPS is what the report carries.

Outputs (into --out-dir, default outputs/saoi_sweep3):
    sweep_results.md    compact table -- this is the file to read/paste
    sweep_results.json  everything, including full rank histograms

Usage:
    python configs/MeshGraphNets-V/SAOI_sweep3/score_sweep.py
    python .../score_sweep.py --arms ad_g1_z16_c1_r100 --k 20
"""
import argparse
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
METHOD_REPO = REPO_ROOT / "cHI-MGNflow"
# No eval_distribution.py in this tree (see the module docstring).

# Wave 3: a 2^(5-1) resolution-V half fraction. Keep AXES in the order the arm
# name encodes, so a name splits straight into its cell coordinates.
AXES = [
    ("batch_size",       ["b16", "b32"],  ["16", "32"]),
    ("flow_t_sampling",  ["tu", "tl"],    ["uniform", "logitnormal"]),
    ("voronoi_clusters", ["c1k", "c2k"],  ["1000, 100", "2000, 250"]),
    ("capacity",         ["k0", "k1"],    ["128 / 4,6,8,6,4", "192 / 6,8,12,8,6"]),
    ("learningr",        ["lr1", "lr3"],  ["1e-4", "3e-4"]),
]
def _half_fraction():
    """The 16 cells of the 2^(5-1) resolution-V design: E = A xor B xor C xor D.

    NOT the full product of the five axes -- that would be 32 runs. Mirrors
    gen_sweep_configs.arms(); if one changes, the other must.
    """
    out = []
    for i in range(16):
        free = [(i >> (3 - k)) & 1 for k in range(4)]
        bits = free + [free[0] ^ free[1] ^ free[2] ^ free[3]]
        out.append("_".join(AXES[k][1][b] for k, b in enumerate(bits)))
    return out


DEFAULT_ARMS = _half_fraction()


def arm_tags(arm):
    """'b32_tl_c2k_k1_lr1' -> ['b32', 'tl', 'c2k', 'k1', 'lr1']."""
    return arm.split("_")

# Eval sets each arm is inferred on (gen_sweep_configs.INFER_SOURCES).
INFER_TAGS = ["s26fe_main", "s26fe_sec", "sm_l345u_main"]

# Graph counts tried in order until one does not OOM.
NGRAPH_LADDER = [0, 64, 32, 16, 8]


def parse_config(path):
    """Minimal reader for the native flat `key value` format.

    Only the handful of keys this report needs. Mirrors the native quirks that
    matter here: `%` starts a comment line, `#` starts an inline comment, and
    key/value are split on the first run of whitespace.
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("%") or line == "'":
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0].strip().lower()] = parts[1].strip()
    return out


def parse_training_log(log_path, transcript_path=None):
    """Pull best CRPS / final losses / spread ratio out of a run's logs.

    The trainer splits these across TWO files, so both are read:

      * `log_path` -- the config's `log_file_dir`, one `f.write()` line per
        epoch carrying recon and CRPS. The minimum CRPS across lines is the
        best the run achieved.
      * `transcript_path` -- run_sweep.sh's per-arm stdout capture. `[PriorDiag]`
        and `[PriorTail]` are emitted with `tqdm.write`, i.e. to STDOUT ONLY;
        they never reach the epoch log, so spread ratio and p99 coverage come
        from here and are simply absent when no transcript is available.

    Every field degrades to None rather than raising: an arm that died early,
    logged nothing, or ran with `val_interval > 1` (most epoch lines then read
    "Valid skipped" with no CRPS at all) must still produce a report row.
    """
    res = {"best_crps": None, "final_recon": None, "final_valid": None,
           "final_epoch": None, "min_spread_ratio": None, "amp_p99_cov": None,
           "sources": [], "searched": []}

    num = r"([-+0-9.eE]+)"
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    crps_re = re.compile(r"CRPS\s+" + num)
    train_re = re.compile(r"Train\s+recon=" + num)
    valid_re = re.compile(r"Valid\s+recon=" + num)
    ratio_re = re.compile(r"ratio=([-+0-9.eE]+)")
    cov_re = re.compile(r"p99_cov=([-+0-9.eE]+)")

    def as_float(m):
        # `search` returns None on every epoch line without the field -- with
        # val_interval > 1 that is most of them. `None.group()` raises
        # AttributeError, which the TypeError/ValueError guard does not catch,
        # and one such line aborted the entire report.
        if m is None:
            return None
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return None

    def readable(path):
        return path is not None and path.exists()

    crps_vals, spread_vals, last = [], [], None

    # Per-epoch numbers: the epoch log is authoritative (.4e). The transcript's
    # console lines carry the same values at .2e, so it is the fallback.
    # Fall through on a file that EXISTS but holds no epoch line, not just on a
    # missing one -- `init_log_file` creates the epoch log (header + embedded
    # config) before the first epoch runs, so a stale or misdirected log is
    # present-but-empty and would otherwise shadow a perfectly good transcript.
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

    # Prior diagnostics: stdout-only, so the transcript is the real source.
    for path in (transcript_path, log_path):
        if not readable(path):
            continue
        found = False
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PriorDiag" in line:
                    spread_vals += [float(v) for v in ratio_re.findall(line)
                                    if _is_float(v)]
                    found = True
                elif "PriorTail" in line:
                    m = cov_re.search(line)
                    if m and _is_float(m.group(1)):
                        res["amp_p99_cov"] = float(m.group(1))
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
        res["final_recon"] = as_float(train_re.search(last))
        res["final_valid"] = as_float(valid_re.search(last))
    return res


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


EVAL_PATTERNS = {
    "n_graphs": re.compile(r"n_graphs=(\d+)"),
    "k": re.compile(r"K=(\d+)"),
    "wild_pct": re.compile(r"WILD RATE = \d+ / \d+ \(([-+0-9.eE]+)%\)"),
    # margin 0.0 = "did the draw leave the observed data envelope at all?".
    # The headline WILD RATE uses margin 0.5 (half the whole range on each
    # side), which almost nothing trips -- it detects blow-ups, not calibration.
    "wild0_pct": re.compile(r"WILD LADDER.*?0\.00=([-+0-9.eE]+)%"),
    "chi2": re.compile(r"chi2 = ([-+0-9.eE]+)"),
    "crit": re.compile(r"critical ~([-+0-9.eE]+)"),
    # The K+1-bin chi2 above is unusable at these sample sizes (K=50 puts 51
    # bins against ~n observations). eval_distribution.py also reports a 5-bin
    # version, which is the one the table should show.
    "chi2_5": re.compile(r"chi2_5 = ([-+0-9.eE]+)"),
    "crit_5": re.compile(r"5-bin critical ([-+0-9.eE]+)"),
    "shape": re.compile(r"shape: (.+)"),
}
HIST_RE = re.compile(r"RANK HISTOGRAM \(\d+ bins, expect ~[-+0-9.eE]+ each\): (\[.*\])")


def parse_eval_stdout(text):
    """Extract the diagnostic numbers from eval_distribution.py's printout."""
    out = {"raw_tail": text.strip().splitlines()[-8:]}
    for key, rx in EVAL_PATTERNS.items():
        m = rx.search(text)
        if m:
            val = m.group(1).strip()
            out[key] = val if key == "shape" else (
                float(val) if _is_float(val) else None)
    m = HIST_RE.search(text)
    if m:
        try:
            hist = json.loads(m.group(1))
            out["hist"] = hist
            out["hist5"] = rebin(hist, 5)
        except json.JSONDecodeError:
            pass
    return out


def rebin(hist, nbins):
    """Collapse a K+1-bin rank histogram into `nbins` contiguous groups, as %.

    The tool's own `shape:` label uses hardcoded thresholds; a coarse rebin lets
    a reader judge U / dome / skew directly instead of trusting that heuristic.
    """
    total = sum(hist) or 1
    n = len(hist)
    edges = [round(i * n / nbins) for i in range(nbins + 1)]
    return [round(100.0 * sum(hist[edges[i]:edges[i + 1]]) / total, 1)
            for i in range(nbins)]


def run_eval(cfg_path, split, k, sampler, python, timeout, eval_log, arm):
    """Run eval_distribution.py, stepping down --n-graphs until it fits.

    One invocation is a many-minute, silent job: it re-fits the train-split
    normalization statistics, then runs `--k` full forward passes over the whole
    requested split. Two things follow from that, and both are deliberate here:

      * The child's output is TEE'd to `eval_log` instead of being captured into
        memory, so `tail -f` shows progress while it runs. `capture_output=True`
        made the whole ladder invisible until it finished.
      * Each attempt announces itself with its elapsed time, so a run that looks
        hung can be told apart from one that is merely slow.
    """
    attempts = []
    for n_graphs in NGRAPH_LADDER:
        cmd = [python, str(EVAL_SCRIPT), "--config", str(cfg_path),
               "--split", split, "--k", str(k), "--sampler", sampler,
               "--n-graphs", str(n_graphs)]
        label = f"n_graphs={n_graphs or 'all'}"
        print(f"  [{arm}] {sampler} {label} ... (tail -f {eval_log})",
              flush=True)
        start = time.time()
        offset = eval_log.stat().st_size if eval_log.exists() else 0
        timed_out = False
        with eval_log.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n===== {sampler} {label} @ "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            fh.flush()
            try:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        elapsed = time.time() - start
        # Read back only what this attempt appended.
        with eval_log.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            text = fh.read()

        if timed_out:
            print(f"  [{arm}] {sampler} {label} TIMEOUT after {elapsed:.0f}s",
                  flush=True)
            attempts.append(f"{label}: timeout after {timeout}s")
            continue
        if proc.returncode == 0 and "WILD RATE" in text:
            print(f"  [{arm}] {sampler} {label} ok ({elapsed:.0f}s)", flush=True)
            res = parse_eval_stdout(text)
            res["ok"] = True
            res["attempts"] = attempts
            res["log"] = str(eval_log)
            return res
        tail = text.strip().splitlines()[-3:]
        print(f"  [{arm}] {sampler} {label} rc={proc.returncode} "
              f"({elapsed:.0f}s): {' | '.join(tail)}", flush=True)
        attempts.append(f"{label}: rc={proc.returncode} :: " + " | ".join(tail))
        # Only an OOM is worth retrying smaller; anything else will just repeat.
        if "out of memory" not in text.lower():
            break
    return {"ok": False, "attempts": attempts, "log": str(eval_log)}


def fmt(v, spec=".2e", dash="-"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "saoi_sweepB"))
    ap.add_argument("--run-logs",
                    default=str(REPO_ROOT / "outputs" / "saoi_sweepB" / "run_logs"),
                    help="run_sweep.sh's per-arm stdout transcripts; the only "
                         "place [PriorDiag]/[PriorTail] are recorded")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per eval invocation, seconds")
    ap.add_argument("--samplers", nargs="*", default=["prior", "normal"],
                    choices=["auto", "prior", "normal"])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_logs = pathlib.Path(args.run_logs)
    eval_logs = out_dir / "eval_logs"
    eval_logs.mkdir(parents=True, exist_ok=True)
    print(f"Live eval output: {eval_logs}/<arm>.<sampler>.log  (tail -f these)\n",
          flush=True)

    # A missing hierarchy cache makes the FIRST eval silently rebuild the whole
    # thing -- the same build run_sweep.sh budgets 6 hours for -- before it can
    # score anything. run_sweep.sh tells you to delete the cache when the sweep
    # ends; that has to happen AFTER scoring, not before.
    for probe_arm in args.arms:
        probe_cfg = HERE / f"config_{probe_arm}.txt"
        if not probe_cfg.exists():
            continue
        ds = (METHOD_REPO / parse_config(probe_cfg).get("dataset_dir", "")).resolve()
        if ds.exists() and not list(ds.parent.glob(f"{ds.stem}.mscache.*.h5")):
            print(f"NOTE: no hierarchy cache next to {ds.name} -- the first eval "
                  f"has to rebuild it before it can score anything.\n", flush=True)
        break

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
            "z": cfg.get("vae_latent_dim"),
            "alpha_recon": cfg.get("alpha_recon"),
            "lambda_mmd": cfg.get("lambda_mmd"),
            "prior_nll_weight": cfg.get("prior_nll_weight"),
            "flow_t_sampling": cfg.get("flow_t_sampling"),
            "voronoi_clusters": cfg.get("voronoi_clusters"),
            "use_conditional_prior": cfg.get("use_conditional_prior"),
            "latent_dim": cfg.get("latent_dim"),
            "mp_per_level": cfg.get("mp_per_level"),
            "batch_size": cfg.get("batch_size"),
        })

        ckpt = (METHOD_REPO / cfg.get("modelpath", "")).resolve()
        row["checkpoint"] = str(ckpt)
        if not ckpt.exists():
            row["error"] = "checkpoint missing (arm did not finish?)"
            rows.append(row)
            print(f"[{arm}] SKIP: no checkpoint at {ckpt}", flush=True)
            continue

        # The trainer writes the epoch log to `outputs/<log_file_dir>` relative
        # to the method repo (training_profiles/setup.py::init_log_file), which
        # is exactly why the configs' log_file_dir carries a leading '../..'
        # that `modelpath` does not. Resolving it without that 'outputs/' prefix
        # lands one directory ABOVE the monorepo, and every training-log column
        # then renders as '-' with no error anywhere.
        log_path = (METHOD_REPO / "outputs" / cfg.get("log_file_dir", "")).resolve()
        transcript = run_logs / f"{arm}.log"
        row["train_log"] = parse_training_log(log_path, transcript)

        print(f"[{arm}] evaluating ({', '.join(args.samplers)})...", flush=True)
        # No rank-histogram stage in this tree; the report is built from the
        # training log plus the inference spread dumps.
        row["eval"] = {}
        rows.append(row)

    # Warpage spread (max - min z_disp) from the inference stage. Attached
    # before the JSON dump so the raw numbers land there too, and plotted so the
    # 16 arms can be compared on one axis instead of 16 separate PNGs.
    collect_warpage(rows)
    for path in plot_warpage_overlay(rows, out_dir):
        print(f"  warpage overlay -> {path}", flush=True)

    json_path = out_dir / "sweep_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")

    md = render_markdown(rows, args)
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
    print(f"Wrote {json_path}  (full rank histograms)")


def _metric_by_cell(rows, k, fn):
    """{level tag: [values]} for factor index k."""
    cells = {}
    for r in rows:
        try:
            tag = arm_tags(r["arm"])[k]
        except (IndexError, KeyError):
            continue
        v = fn(r)
        if v is not None:
            cells.setdefault(tag, []).append(float(v))
    return cells


def render_interactions(rows):
    """All ten 2-factor interactions on best CRPS, largest first.

    A resolution-V design estimates every one of these clean, but a main-effects
    table cannot show them -- and this grid's sharpest PREDICTION is an
    interaction: under `cc` each extra processor block compounds the concat
    fuser's ~1.33x gain, under `ad` it does not, so capacity should hurt one half
    and help the other. A large interaction also means the corresponding MAIN
    effect is an average over two opposite behaviours and must not be read alone.

    Effect for factors (A, B), standard two-level definition:
        AB = ( mean(A1B1) + mean(A0B0) - mean(A1B0) - mean(A0B1) ) / 2
    Negative = the (level-1, level-1) corner is better than additivity predicts.
    """
    def crps(r):
        return (r.get("train_log") or {}).get("best_crps")

    out = []
    for a, b in itertools.combinations(range(len(AXES)), 2):
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
        out.append((abs(eff), eff, AXES[a][0], AXES[b][0], cell, ta, tb))
    if not out:
        return ""
    out.sort(reverse=True)

    L = ["## Two-factor interactions on best CRPS (largest first)", ""]
    L.append("| factor A | factor B | AB effect | A0B0 | A0B1 | A1B0 | A1B1 |")
    L.append("|" + "---|" * 7)
    for _, eff, na, nb, cell, ta, tb in out:
        L.append(
            f"| `{na}` | `{nb}` | {eff:+.4g} | "
            f"{fmt(cell[(0, 0)])} | {fmt(cell[(0, 1)])} | "
            f"{fmt(cell[(1, 0)])} | {fmt(cell[(1, 1)])} |"
        )
    L.append("")
    L.append("Level order is the one in the axis legend above (A0 = first level).")
    L.append("Read an interaction as real only if it is large next to the spread")
    L.append("of the four cell means it is built from -- with one run per cell")
    L.append("there is no replication, so small values here are noise.")
    return "\n".join(L)


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

    Reads inference_output_dir out of the generated inference config rather than
    rebuilding the path, so a change in the generator cannot silently desync.
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
        # Narrow on purpose: a bare except here hid a missing numpy import.
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
                # Everything normalized by the GT spread's own std, so the three
                # eval sets (different parts, different magnitudes) are comparable.
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
    this distribution wrong, and peak-to-valley is the number the part is judged
    on, so this table is the one to read alongside CRPS.

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
    L.append("Per-arm plots: output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/histogram_compare.png")
    return "\n".join(L)


def plot_warpage_overlay(rows, out_dir):
    """One figure per eval set: GT filled, every arm's generated density on top.

    Sixteen separate PNGs cannot be compared by eye; this puts them on one axis,
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
    """Average each metric over the 8 arms at each level of each factor.

    This is what a FULL FACTORIAL buys over a one-factor-at-a-time scan: every
    arm contributes to all four comparisons, so each main effect is an 8-vs-8
    difference rather than a 1-vs-1 one. `delta` is (level 1 - level 0); for
    CRPS and wild% NEGATIVE is better, so a negative delta means level 1 wins.

    Read a main effect as provisional if the two levels' spreads overlap -- with
    n=1 per cell there is no replication, and a large interaction shows up here
    as a main effect that does not reproduce in the per-arm table.
    """
    def crps(r):
        return (r.get("train_log") or {}).get("best_crps")

    def wild(r):
        return ((r.get("eval") or {}).get("prior") or {}).get("wild0_pct")

    metrics = [("best CRPS", crps, ".4g"), ("wild0% prior", wild, ".1f")]
    L = ["## Main effects (8 arms per level)", ""]
    L.append("| factor | level 0 | level 1 | metric | mean @0 | mean @1 | delta | n |")
    L.append("|" + "---|" * 8)
    for k, (key, tags, vals) in enumerate(AXES):
        for label, fn, fmt_spec in metrics:
            cell = [[], []]
            for r in rows:
                try:
                    lvl = tags.index(arm_tags(r["arm"])[k])
                except (ValueError, IndexError):
                    continue
                v = fn(r)
                if v is not None:
                    cell[lvl].append(float(v))
            if not cell[0] or not cell[1]:
                continue
            m0 = sum(cell[0]) / len(cell[0])
            m1 = sum(cell[1]) / len(cell[1])
            L.append(
                f"| `{key}` | {vals[0]} | {vals[1]} | {label} | "
                f"{m0:{fmt_spec}} | {m1:{fmt_spec}} | {m1 - m0:+{fmt_spec}} | "
                f"{len(cell[0])}v{len(cell[1])} |"
            )
    return "\n".join(L)


def render_markdown(rows, args):
    L = []
    L.append("# cHI-MGNflow SAOI Wave B -- 2^(5-1) resolution V (16 arms, 2 per GPU)")
    L.append("")
    L.append(f"split={args.split}  K={args.k}  samplers={','.join(args.samplers)}")
    L.append("")
    L.append("| tag | key | level 0 | level 1 |")
    L.append("|---|---|---|---|")
    for key, tags, vals in AXES:
        L.append(f"| {tags[0]} / {tags[1]} | `{key}` | {vals[0]} | {vals[1]} |")
    L.append("")
    L.append("cc = the legacy concat fuser Linear([x, z]); ad = AdaLN-Zero,")
    L.append("identity at init. g0 = the prior's flow-matching gradient is")
    L.append("detached from the encoder (no CVAE rate term); g1 = end-to-end.")
    L.append("beta_aux is fixed at 1.0 in every arm -- it is the I(z;y) floor")
    L.append("that guards g1 against collapsing to a deterministic z = h(g).")
    L.append("")
    L.append("r001/r100 move lambda_mmd AND prior_nll_weight together. With")
    L.append("alpha_recon 1000 both are ~0.1% of the objective at r001, so the")
    L.append("r001 half is the \"regularizers effectively off\" control -- and the")
    L.append("g0/g1 contrast can only show force in the r100 half.")
    L.append("")
    L.append("Defining relation I = ABCDE: the regularizer scale is not free,")
    L.append("it is A xor B xor C xor D. All 5 main effects and all 10 two-factor")
    L.append("interactions are clean; 3-factor and higher alias with them.")
    L.append("The interaction to read first is z_conditioning x capacity: under")
    L.append("cc every extra block compounds the fuser's ~1.33x gain, under ad")
    L.append("it does not -- so depth should HURT one half and HELP the other.")
    L.append("")
    L.append("**Read CRPS and wild%, not recon.** Reconstruction measures the")
    L.append("posterior path; the generative path is what inference uses.")
    L.append("Lower CRPS = better. wild0% counts (graph, draw) pairs whose field")
    L.append("leaves the observed data envelope AT ALL -- that is the")
    L.append("discriminating one. wild% uses the old margin of half the whole")
    L.append("range on each side, which only catches outright blow-ups.")
    L.append("chi2/crit < 1 means the rank histogram is consistent with uniform")
    L.append("(calibrated). This is the 5-BIN test: the raw K+1-bin one puts 51")
    L.append("bins against ~n observations and cannot be trusted -- measured, its")
    L.append("shape label called over-dispersion \"roughly flat\" about half the time.")
    L.append("rank% is that 5-bin histogram: flat ~ [20,20,20,20,20];")
    L.append("both ends heavy = under-dispersed; middle heavy = over-dispersed;")
    L.append("one end heavy = biased location.")
    L.append("")

    hdr = ("| arm | t sched | voronoi | z | mmd | best CRPS | valid recon | spread | "
           "wild0% prior | wild0% N(0,I) | wild% prior | chi2/crit | "
           "rank% (prior) | n |")
    L.append(hdr)
    L.append("|" + "---|" * 14)

    for r in rows:
        if r.get("error"):
            # arm, zcond, grad, error + 10 blanks + close = 14 columns.
            L.append(f"| {r['arm']} | {r.get('z_conditioning','?')} | "
                     f"{r.get('prior_grad_to_encoder','?')} | **{r['error']}** | "
                     + "| " * 10 + "|")
            continue
        tl = r.get("train_log", {})
        ev = r.get("eval", {})
        ep, en = ev.get("prior", {}), ev.get("normal", {})
        # Prefer the 5-bin test; fall back to the fine one only for logs from
        # an eval_distribution.py that predates it.
        num, den = ep.get("chi2_5"), ep.get("crit_5")
        if num is None or not den:
            num, den = ep.get("chi2"), ep.get("crit")
        ratio = num / den if num is not None and den else None
        L.append(
            f"| {r['arm']} | {r.get('z_conditioning','?')} | "
            f"{r.get('prior_grad_to_encoder','?')} | {r.get('z','?')} | "
            f"{r.get('lambda_mmd','?')} | "
            f"{fmt(tl.get('best_crps'))} | {fmt(tl.get('final_valid'))} | "
            f"{fmt(tl.get('min_spread_ratio'), '.2f')} | "
            f"{fmt(ep.get('wild0_pct'), '.1f')} | {fmt(en.get('wild0_pct'), '.1f')} | "
            f"{fmt(ep.get('wild_pct'), '.1f')} | "
            f"{fmt(ratio, '.2f')} | {ep.get('hist5', '-')} | "
            f"{fmt(ep.get('n_graphs'), '.0f')} |"
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
                 f"train recon {fmt(tl.get('final_recon'))}, "
                 f"amp_p99_cov {fmt(tl.get('amp_p99_cov'), '.2f')} "
                 f"[logs read: {', '.join(tl.get('sources') or []) or 'NONE FOUND'}]")
        if not tl.get("sources"):
            # Print the exact paths, because "no numbers" and "wrong path" look
            # identical in the table and only one of them is a training problem.
            L.append("  - no epoch lines found. Searched: "
                     + ("; ".join(tl.get("searched") or []) or "(nothing)"))
        for s, ev in r.get("eval", {}).items():
            if not ev.get("ok"):
                L.append(f"- sampler `{s}`: FAILED -- {ev.get('attempts')}")
                continue
            L.append(f"- sampler `{s}`: wild {fmt(ev.get('wild_pct'), '.1f')}%, "
                     f"chi2 {fmt(ev.get('chi2'), '.1f')} vs crit "
                     f"{fmt(ev.get('crit'), '.1f')}, shape \"{ev.get('shape')}\", "
                     f"rank5 {ev.get('hist5')}")
            if ev.get("attempts"):
                L.append(f"  - retries: {ev['attempts']}")
    L.append("")
    L.append("Full rank histograms are in sweep_results.json.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
