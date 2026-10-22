"""Loss curves and per-geometry histograms for every job of a sweep round.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/plots.py r3 [job ...]

Reads <round>/jobs.txt and, for each job (or the jobs named), writes:

  loss_<log>.png beside each epoch log of the run (MGN-V train.log; SAR
      train_vae.log / train_sar.log; LDGN train_vae.log / train_ldgn.log).
      One panel per loss component plus the total and the learning rate,
      train as a line and validation as dots. A component validation does not
      compute is drawn train-only and named in the footer: MGN-V validation
      scores recon + the latent regularizer, never aux or the prior loss, so
      its val `total` is alpha*recon + lambda*reg while the train `total` also
      carries those terms.
      Runs started before the logs carried every component (r3) miss the MGN-V
      aux term in train.log; it is taken from the job's stdout
      (run_logs/<node>/<job>.train.out) when that file exists.

  infer/histogram_compare_<attr>_<value>.png + spread_metrics_by_group.json,
      one histogram per held-out geometry, redrawn from infer/spread_values.npz
      with the inference code's own helper. New inference runs write these
      themselves; this covers runs scored before they did. --force redraws.

  infer/spread_w2.json, the per-geometry 1-D W2 between the FE results and
      the draws, over that geometry's own FE sd, split exactly as
      w2^2 = bias^2 + (sd_ratio - 1)^2 + shape^2, plus the root mean square
      over geometries. brief.py ranks by it. Rewritten when spread_values.npz
      is newer.

Then prints, per job, the per-geometry W2 parts, CRPS and generated/true
width (x100).

Needs numpy, h5py, matplotlib and torch: run it with a method interpreter
(the MGN-V, SAR or LDGN venv), from anywhere.
"""
import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

SWEEP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SWEEP_ROOT.parents[2]
METHOD_DIRS = {"mgnv": "methods/MeshGraphNets_Variational", "sar": "methods/SAR",
               "ldgn": "methods/LDGN"}

NUM = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
LOG_EPOCH = re.compile(r"^(?:\[[^\]]+\] )?Elapsed:? [0-9.]+s Epoch (\d+) (.*)$")
OUT_EPOCH = re.compile(r"^Epoch (\d+)/\d+ (.*)$")
PAIR = re.compile(r"([A-Za-z_][\w]*)=" + NUM)
BARE_TRAIN = re.compile(r"\bTrain:? " + NUM)
BARE_VALID = re.compile(r"\bValid:? " + NUM)
LR = re.compile(r"\bLR:? " + NUM)


# ----------------------------------------------------------------- configs
def cfg_values(path):
    """The flat `key value` config, first token per key, comments stripped."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("%", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] not in values:
            values[parts[0]] = parts[1].split(",")[0].split()[0]
    return values


def read_jobs(sweep_dir):
    jobs = []
    for line in (sweep_dir / "jobs.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        node, jid, method, tcfg, icfg = line.split()[:5]
        # train config "-": an inference-only job on another job's checkpoint
        jobs.append(dict(node=node, job=jid, method=method,
                         train=None if tcfg == "-" else REPO_ROOT / tcfg, infer=REPO_ROOT / icfg))
    return jobs


def method_path(method, value):
    """A config path is spelled from the method directory (../../output/...)."""
    return (REPO_ROOT / METHOD_DIRS[method] / value).resolve()


# -------------------------------------------------------------- epoch logs
def _num(text):
    return float(text)


def parse_epoch(rest):
    """One epoch line (after `Epoch k`) -> ({train term: v}, {val term: v})."""
    train, val = {}, {}
    m = LR.search(rest)
    if m:
        train["lr"] = _num(m.group(1))
    for seg in (s.strip() for s in rest.split("|")):
        if not seg or seg.startswith("VRAM"):
            continue
        if seg.startswith("CRPS"):
            m = re.search(NUM, seg[4:])
            if m:
                val["crps (learned prior)"] = _num(m.group(1))
            continue
        if seg.startswith("PRIOR-FIT"):   # MGN-V stdout, prior-only tail
            for k, v in PAIR.findall(seg):
                train["prior"] = _num(v)
            continue
        if seg.startswith("Valid"):
            if "skipped" in seg:
                continue
            target = val
        elif seg.startswith("Train") or seg.startswith("LR"):
            target = train
        else:                              # SAR/LDGN extras: recon=.. val_recon=..
            target = None
        # SAR / LDGN / plain MGN: `Train X Valid Y LR Z`, the stage's total loss
        m = BARE_TRAIN.search(seg)
        if m:
            train["loss"] = _num(m.group(1))
        m = BARE_VALID.search(seg)
        if m:
            val["loss"] = _num(m.group(1))
        for k, v in PAIR.findall(seg):
            if k in ("peak", "reserved"):
                continue
            if target is None:
                (val if k.startswith("val_") else train)[k[4:] if k.startswith("val_") else k] = _num(v)
            else:
                target[k] = _num(v)
    return train, val


def read_curves(path, pattern):
    """{epoch: (train, val)}; a resumed run repeats epochs, the later line wins."""
    curves = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in re.split(r"[\r\n]+", text):
        m = pattern.match(line.strip())
        if not m:
            continue
        rest = m.group(2)
        if pattern is OUT_EPOCH and "Train" not in rest and "PRIOR-FIT" not in rest:
            continue   # a progress bar, not the epoch summary
        curves[int(m.group(1))] = parse_epoch(rest)
    return curves


def merge_missing(curves, extra):
    """Add terms the log lacks from another source (stdout), never overwrite."""
    for ep, (tr, va) in extra.items():
        if ep not in curves:
            continue
        for src, dst in ((tr, curves[ep][0]), (va, curves[ep][1])):
            for k, v in src.items():
                dst.setdefault(k, v)


def plot_curves(curves, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = []
    for ep in sorted(curves):
        for d in curves[ep]:
            for k in d:
                if k not in order:
                    order.append(k)
    first = [k for k in ("total", "loss") if k in order]
    last = [k for k in ("crps (learned prior)", "lr") if k in order]
    keys = first + [k for k in order if k not in first + last] + last
    if not keys:
        return False
    ncol = min(3, len(keys))
    nrow = math.ceil(len(keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow), squeeze=False)
    train_only = []
    for ax, key in zip(axes.flat, keys):
        tr = [(ep, curves[ep][0][key]) for ep in sorted(curves) if key in curves[ep][0]]
        va = [(ep, curves[ep][1][key]) for ep in sorted(curves) if key in curves[ep][1]]
        if tr:
            ax.plot(*zip(*tr), lw=1.2, label="train")
        if va:
            ax.plot(*zip(*va), "o-", ms=2.5, lw=0.8, color="C1", label="val")
        if tr and not va and key != "lr":
            train_only.append(key)
        vals = [v for _, v in tr + va if math.isfinite(v)]
        if vals and min(vals) > 0 and (key == "lr" or max(vals) / min(vals) > 20):
            ax.set_yscale("log")
        name = {"loss": "total (stage loss)", "lr": "learning rate"}.get(key, key)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if tr and va:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    foot = ""
    if train_only:
        foot = "train only (validation does not compute it): " + ", ".join(train_only)
    if "total" in keys and "aux" in train_only:
        foot += "\nval total = alpha*recon + lambda*reg; train total also adds aux / prior"
    fig.suptitle(title, fontsize=11)
    if foot:
        fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if foot else 0, 1, 0.95))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def job_losses(job, out_root):
    cfg = cfg_values(job["train"])
    if "log_file_dir" not in cfg:
        return []
    run_dir = method_path(job["method"], cfg["log_file_dir"]).parent
    written = []
    for log in sorted(run_dir.glob("*.log")):
        curves = read_curves(log, LOG_EPOCH)
        if not curves:
            continue   # infer.log and other logs without epochs
        if job["method"] == "mgnv":
            for out in sorted(out_root.glob(f"run_logs/*/{job['job']}.train.out")):
                merge_missing(curves, read_curves(out, OUT_EPOCH))
        png = run_dir / f"loss_{log.stem}.png"
        if plot_curves(curves, png, f"{job['job']}  {log.name}  ({len(curves)} epochs)"):
            written.append(png)
    return written


# ----------------------------------------------------- per-geometry spread
def load_spread_module():
    path = REPO_ROOT / "methods/SAR/inference_profiles/spread.py"
    spec = importlib.util.spec_from_file_location("sweep_spread", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_groups(job, spread, force):
    cfg = cfg_values(job["infer"])
    out_dir = cfg.get("inference_output_dir")
    attr = cfg.get("split_group_attr")
    eval_h5 = cfg.get("eval_dataset") or cfg.get("infer_dataset")
    if not (out_dir and attr and eval_h5):
        return None
    out_dir = method_path(job["method"], out_dir)
    npz, by_group = out_dir / "spread_values.npz", out_dir / "spread_metrics_by_group.json"
    if not npz.is_file():
        return None
    if force or not by_group.is_file() or by_group.stat().st_mtime < npz.stat().st_mtime:
        import numpy as np
        data = np.load(npz)
        eval_h5 = method_path(job["method"], eval_h5)
        groups = spread._eval_dataset_groups(str(eval_h5), attr)
        if not groups:
            print(f"  {job['job']}: no `{attr}` attribute in {eval_h5}; no per-geometry histograms")
            return None
        label = "spread"
        metrics = out_dir / "spread_metrics.json"
        if metrics.is_file():
            label = json.loads(metrics.read_text(encoding="utf-8")).get("label", label)
        spread._spread_by_group(
            data["gt"], data["gt_scene"], data["gen"], data["gen_scene"], groups, attr,
            str(out_dir), eval_path=str(eval_h5), bins=int(cfg.get("histogram_bins", 60)),
            clip_quantile=float(cfg.get("histogram_clip_quantile", 0.0)),
            label=label, stat=cfg.get("spread_stat", spread.SPREAD_STAT_DEFAULT))
    return json.loads(by_group.read_text(encoding="utf-8"))


def w2_parts(gt, gen):
    """Exact 1-D W2 between two empirical samples, and its three parts.

    W2^2 = (mean gap)^2 + (sd gap)^2 + shape, shape = 2 sd_gt sd_gen (1 - rho),
    where rho correlates the two quantile functions. Moments are population
    moments, so the identity is exact. Returns (w2^2, mean gap, sd_gen, sd_gt, shape).
    """
    import numpy as np
    x = np.sort(np.asarray(gen, dtype=np.float64))
    y = np.sort(np.asarray(gt, dtype=np.float64))
    n, m = x.size, y.size
    # Both quantile functions are steps; between the union of their breakpoints
    # each is constant, so a weighted sum over those pieces is the exact integral.
    cuts = np.union1d(np.arange(n + 1) / n, np.arange(m + 1) / m)
    w = np.diff(cuts)
    mid = 0.5 * (cuts[:-1] + cuts[1:])
    qx = x[np.minimum((mid * n).astype(int), n - 1)]
    qy = y[np.minimum((mid * m).astype(int), m - 1)]
    mx, my = float((w * qx).sum()), float((w * qy).sum())
    sx = math.sqrt(float((w * (qx - mx) ** 2).sum()))
    sy = math.sqrt(float((w * (qy - my) ** 2).sum()))
    cov = float((w * (qx - mx) * (qy - my)).sum())
    return float((w * (qx - qy) ** 2).sum()), mx - my, sx, sy, max(2.0 * (sx * sy - cov), 0.0)


def job_w2(job, spread, force):
    """Per-geometry W2 / FE sd, split into bias, sd and shape; writes spread_w2.json.

    Each geometry is normalized by its own FE sd, so per geometry
    w2^2 = bias^2 + (sd - 1)^2 + shape^2. The summary takes the root mean square
    over geometries (the same identity holds for it), plus the geometric mean of
    the sd ratios, which keeps the direction (below 1 = too narrow).
    """
    cfg = cfg_values(job["infer"])
    out_dir, attr = cfg.get("inference_output_dir"), cfg.get("split_group_attr")
    eval_h5 = cfg.get("eval_dataset") or cfg.get("infer_dataset")
    if not (out_dir and attr and eval_h5):
        return None
    out_dir = method_path(job["method"], out_dir)
    npz, path = out_dir / "spread_values.npz", out_dir / "spread_w2.json"
    if not npz.is_file():
        return None
    if path.is_file() and not force and path.stat().st_mtime >= npz.stat().st_mtime:
        return json.loads(path.read_text(encoding="utf-8"))
    import numpy as np
    data = np.load(npz)
    groups = spread._eval_dataset_groups(str(method_path(job["method"], eval_h5)), attr)
    if not groups:
        return None
    gt, gen = np.asarray(data["gt"], dtype=np.float64), np.asarray(data["gen"], dtype=np.float64)
    gt_group = np.asarray([groups.get(str(s), "") for s in data["gt_scene"]])
    gen_group = np.asarray([groups.get(str(s), "") for s in data["gen_scene"]])
    pooled_sd = float(gt.std()) or 1.0

    def score(keep):
        rows, skipped = {}, []
        for value in spread._group_order(v for v in gt_group if v):
            g, p = gt[gt_group == value], gen[keep & (gen_group == value)]
            if not p.size:
                continue
            w2sq, gap, sx, sy, shape = w2_parts(g, p)
            if g.size < 2 or sy <= 1e-9 * pooled_sd:
                skipped.append(value)   # one FE result, or all identical: nothing to normalize by
                continue
            rows[value] = {"w2": math.sqrt(w2sq) / sy, "bias": gap / sy, "sd_ratio": sx / sy,
                           "shape": math.sqrt(shape) / sy, "gt_sd_over_pooled": sy / pooled_sd,
                           "n_gt": int(g.size), "n_gen": int(p.size)}
        if not rows:
            return None, rows, skipped

        def rms(key, shift=0.0):
            return math.sqrt(sum((r[key] - shift) ** 2 for r in rows.values()) / len(rows))

        summary = {"w2": rms("w2"), "bias": rms("bias"), "sd_gap": rms("sd_ratio", 1.0),
                   "shape": rms("shape"), "n_groups": len(rows),
                   "sd_ratio_geomean": math.exp(sum(math.log(max(r["sd_ratio"], 1e-12))
                                                    for r in rows.values()) / len(rows))}
        return summary, rows, skipped

    # A `latent_inflation` list tags every draw with its factor (gen_lam). Each
    # factor is scored on its own; the headline is factor 1 when it was drawn.
    lams = np.asarray(data["gen_lam"], dtype=np.float64) if "gen_lam" in data.files else None
    every = np.ones(gen.shape, dtype=bool)
    by_inflation = {}
    if lams is not None and np.unique(lams).size > 1:
        for lam in np.unique(lams):
            s, _, _ = score(lams == lam)
            if s:
                by_inflation[f"{lam:g}"] = s
        keep = (lams == 1.0) if np.any(lams == 1.0) else every
    else:
        keep = every
    summary, rows, skipped = score(keep)
    if summary is None:
        return None
    out = {"attr": attr, "norm": "each geometry's own FE sd",
           "identity": "w2^2 = bias^2 + (sd_ratio - 1)^2 + shape^2 (per geometry and for the rms summary)",
           "summary": summary, "groups": rows, "skipped_zero_sd": skipped}
    if by_inflation:
        out["summary_draws"] = "latent_inflation 1" if keep is not every else "every latent_inflation pooled"
        out["by_inflation"] = by_inflation
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def shown(path):
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def fmt(v):
    return "-" if v is None or not math.isfinite(v) else f"{100 * v:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("round", help="round folder beside this script (r3, r4, ...) or its path")
    ap.add_argument("jobs", nargs="*", help="only these jobs")
    ap.add_argument("--force", action="store_true", help="redraw the per-geometry histograms")
    ap.add_argument("--no-loss", action="store_true")
    ap.add_argument("--no-hist", action="store_true")
    args = ap.parse_args()

    sweep_dir = Path(args.round)
    if not sweep_dir.is_dir():
        sweep_dir = SWEEP_ROOT / args.round
    jobs = [j for j in read_jobs(sweep_dir) if not args.jobs or j["job"] in args.jobs]
    spread = None if args.no_hist else load_spread_module()

    table, w2_table = [], []
    for job in jobs:
        cfg = cfg_values(job["train"] or job["infer"])
        out_root = method_path(job["method"], cfg.get("log_file_dir", ".")).parent.parent
        if not args.no_loss and job["train"] is not None:
            for png in job_losses(job, out_root):
                print(f"  {job['job']:<18} {shown(png)}")
        if spread is not None:
            try:
                by = job_groups(job, spread, args.force)
            except Exception as exc:   # one broken run never stops the rest
                print(f"  {job['job']}: per-geometry histograms failed: {exc}")
                by = None
            if by:
                table.append((job["job"], by["attr"], by["groups"]))
            try:
                w2 = job_w2(job, spread, args.force)
            except Exception as exc:
                print(f"  {job['job']}: per-geometry W2 failed: {exc}")
                w2 = None
            if w2:
                w2_table.append((job["job"], w2))

    if w2_table:
        print("\nPer geometry W2 (x100, each geometry over its own FE sd; lower is better)")
        print("  w2^2 = bias^2 + (sd - 100)^2 + shape^2;  'fe sd' = that geometry's FE sd / pooled FE sd")
        for jid, w2 in w2_table:
            s = w2["summary"]
            print(f"  {jid:<18} rms w2 {fmt(s['w2'])}  bias {fmt(s['bias'])}  sd gap {fmt(s['sd_gap'])}  "
                  f"shape {fmt(s['shape'])}  sd (geometric mean) {fmt(s['sd_ratio_geomean'])}")
            print(f"  {'':<18} {w2['attr']:>8} {'w2':>6} {'bias':>6} {'sd':>6} {'shape':>6} {'fe sd':>6}")
            for k, r in w2["groups"].items():
                print(f"  {'':<18} {k:>8} {fmt(r['w2']):>6} {fmt(r['bias']):>6} {fmt(r['sd_ratio']):>6} "
                      f"{fmt(r['shape']):>6} {fmt(r['gt_sd_over_pooled']):>6}")
            for lam, b in w2.get("by_inflation", {}).items():
                print(f"  {'':<18} latent_inflation {lam:>5}: w2 {fmt(b['w2'])}  bias {fmt(b['bias'])}  "
                      f"sd gap {fmt(b['sd_gap'])}  shape {fmt(b['shape'])}  sd (geometric mean) "
                      f"{fmt(b['sd_ratio_geomean'])}")
            if w2.get("skipped_zero_sd"):
                print(f"  {'':<18} skipped (one FE result, or FE sd 0): {', '.join(w2['skipped_zero_sd'])}")

    if table:
        print("\nPer geometry (x100): crps (lower is better) / sd (100 = right width)")
        for jid, attr, groups in table:
            keys = list(groups)
            print(f"  {jid:<18} {attr} {' '.join(f'{k:>7}' for k in keys)}")
            print(f"  {'':<18} crps{' ' * (len(attr) - 4)} "
                  + " ".join(f"{fmt(groups[k].get('crps_norm')):>7}" for k in keys))
            print(f"  {'':<18} sd{' ' * (len(attr) - 2)} "
                  + " ".join(f"{fmt(groups[k].get('sd_ratio')):>7}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
