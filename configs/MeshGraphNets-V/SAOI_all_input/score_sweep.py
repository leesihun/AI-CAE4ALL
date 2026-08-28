#!/usr/bin/env python
"""Score every arm of the SAOI latent x regularization sweep into ONE report.

Two independent signals are collected per arm, because neither alone is enough:

  * From the TRAINING LOG (config's log_file_dir): best CRPS, final validation
    reconstruction, and the [PriorDiag] prior/posterior spread ratio. CRPS is
    the inference-mirroring score -- it is the number `best_by crps` selects on.
  * From misc/eval_distribution.py: wild rate and the verification-rank
    histogram on held-out geometries. Reconstruction loss cannot see either;
    a model can improve its recon every epoch while the generative path rots.

Both samplers are evaluated (`--sampler prior` and `--sampler normal`) because
which one wins is itself the open question -- the conditional prior has already
measured WORSE than plain N(0,I) once, on b8.

eval_distribution.py batches the whole split at once, so on large SAOI meshes
`--n-graphs 0` can OOM. This script retries with progressively fewer graphs and
records which count actually ran, rather than silently reporting nothing.

Outputs (into --out-dir, default outputs/saoi_sweep):
    sweep_results.md    compact table -- this is the file to read/paste
    sweep_results.json  everything, including full rank histograms

Usage:
    python configs/MeshGraphNets-V/SAOI_all_input/score_sweep.py
    python .../score_sweep.py --arms sweep_z8_r1 sweep_z8_r4 --k 20
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
METHOD_REPO = REPO_ROOT / "MeshGraphNets - variational"
EVAL_SCRIPT = METHOD_REPO / "misc" / "eval_distribution.py"

Z_LEVELS = [128, 64, 16, 8]
R_LEVELS = ["r1", "r2", "r3", "r4"]
DEFAULT_ARMS = [f"sweep_z{z}_{r}" for z in Z_LEVELS for r in R_LEVELS]

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
    "chi2": re.compile(r"chi2 = ([-+0-9.eE]+)"),
    "crit": re.compile(r"critical ~([-+0-9.eE]+)"),
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
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "saoi_sweep"))
    ap.add_argument("--run-logs",
                    default=str(REPO_ROOT / "outputs" / "saoi_sweep" / "run_logs"),
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
        cfg_path = HERE / f"config_{arm}.txt"
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
        row["eval"] = {
            s: run_eval(cfg_path, args.split, args.k, s, args.python,
                        args.timeout, eval_logs / f"{arm}.{s}.log", arm)
            for s in args.samplers
        }
        rows.append(row)

    json_path = out_dir / "sweep_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md = render_markdown(rows, args)
    md_path = out_dir / "sweep_results.md"
    md_path.write_text(md, encoding="utf-8")

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


def render_markdown(rows, args):
    L = []
    L.append("# SAOI latent x regularization sweep -- results")
    L.append("")
    L.append(f"split={args.split}  K={args.k}  samplers={','.join(args.samplers)}")
    L.append("")
    L.append("AXIS 1 = vae_latent_dim (z). AXIS 2 = alpha_recon:lambda_mmd ratio.")
    L.append("r1=1000:0.1  r2=1000:1  r3=100:1  r4=10:1")
    L.append("")
    L.append("**Read CRPS and wild%, not recon.** Reconstruction measures the")
    L.append("posterior path; the generative path is what inference uses.")
    L.append("Lower CRPS = better. Lower wild% = fewer out-of-envelope fields.")
    L.append("chi2/crit < 1 means the rank histogram is consistent with uniform")
    L.append("(calibrated). rank% is the rank histogram rebinned into 5 groups:")
    L.append("flat ~ [20,20,20,20,20]; U-shape = under-dispersed; dome = over-dispersed.")
    L.append("")

    hdr = ("| arm | z | a:mmd | best CRPS | valid recon | spread | "
           "wild% prior | wild% N(0,I) | chi2/crit | rank% (prior) | n |")
    L.append(hdr)
    L.append("|" + "---|" * 11)

    for r in rows:
        if r.get("error"):
            # arm, z, a:mmd, error + 6 blanks + close = the header's 11 columns.
            L.append(f"| {r['arm']} | {r.get('z','?')} | | **{r['error']}** | "
                     + "| " * 6 + "|")
            continue
        tl = r.get("train_log", {})
        ev = r.get("eval", {})
        ep, en = ev.get("prior", {}), ev.get("normal", {})
        ratio = (ep.get("chi2") / ep["crit"]
                 if ep.get("chi2") is not None and ep.get("crit") else None)
        L.append(
            f"| {r['arm']} | {r.get('z','?')} | "
            f"{r.get('alpha_recon','?')}:{r.get('lambda_mmd','?')} | "
            f"{fmt(tl.get('best_crps'))} | {fmt(tl.get('final_valid'))} | "
            f"{fmt(tl.get('min_spread_ratio'), '.2f')} | "
            f"{fmt(ep.get('wild_pct'), '.1f')} | {fmt(en.get('wild_pct'), '.1f')} | "
            f"{fmt(ratio, '.2f')} | {ep.get('hist5', '-')} | "
            f"{fmt(ep.get('n_graphs'), '.0f')} |"
        )

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
