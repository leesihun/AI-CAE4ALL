#!/usr/bin/env python3
"""Rank the Phase-A compressors and say whether they were trained long enough.

    python configs/HI_MGNFlow/hyperparameter_sweep/ae_report.py
    python configs/HI_MGNFlow/hyperparameter_sweep/ae_report.py --half bot --probe-epoch 600

Reads the epoch log each `mode train_ae` run writes to its `log_file_dir` and
prints, per arm, the best validation reconstruction, where it happened, and how
much the run improved after `--probe-epoch`.

WHY THE PROBE COLUMN EXISTS. build_optimizer_scheduler sets
cosine_T0 = training_epochs - warmup_epochs with T_mult 1, so the learning rate
anneals to eta_min = 1e-8 exactly at the final epoch and no warm restart ever
lands inside a run. Every run therefore looks converged at its own end,
whatever its length, and you cannot decide "was 600 epochs enough" by staring
at the tail of a 600-epoch run -- it is flat by construction. Running long once
and reading the curve is the only cheap way to answer it.

READ THE PROBE COLUMN AS AN UPPER BOUND. At epoch 600 of a 2000-epoch run the
learning rate is still high, so the value there is WORSE than what a dedicated
600-epoch run would have reached by annealing into its own floor. The
improvement this script reports is thus the most that lengthening could
possibly have bought. That makes it decisive in one direction: if the upper
bound is already small, 600 was enough and future campaigns can go back to it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"

HALVES = ("bot", "top")
ARMS = ("c4", "c8", "c16", "c8kl")

# The stage-1 line written to log_file by _run_ae_stage_ddp. Note it carries a
# bare `Epoch 125` while the stdout copy of the same event carries
# `Epoch 125/2000`; the optional group accepts either, so this also parses a
# captured stdout log.
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)(?:/\d+)?\b.*?Valid\s+recon=([0-9.eE+-]+)")

# init_log_file embeds the whole config at the top of the log, so the arm's
# hyper-parameters can be read back from the run itself rather than trusted
# from the filename.
CFG_RE = re.compile(r"^(\w+)\s+(\S+)", re.MULTILINE)


def read_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    curve = [(int(e), float(v)) for e, v in EPOCH_RE.findall(text)]
    curve.sort()
    cfg = {}
    for key, value in CFG_RE.findall(text):
        cfg.setdefault(key, value.split("#", 1)[0].strip())
    return curve, cfg


def at_or_before(curve, epoch):
    """Last validation point recorded at or before `epoch`."""
    seen = [(e, v) for e, v in curve if e <= epoch]
    return seen[-1] if seen else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--half", choices=HALVES, action="append",
                    help="restrict to one half (repeatable)")
    ap.add_argument("--probe-epoch", type=int, default=600,
                    help="epoch to compare the final value against (default: "
                         "600, the length SAOI_run uses for stage 1)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    log_dir = out_root / "ae"
    halves = args.half or list(HALVES)

    print(f"Phase A -- stage-1 (compressor) convergence")
    print(f"  logs: {log_dir}")
    print(f"  probe epoch: {args.probe_epoch}\n")

    if not log_dir.is_dir():
        print(f"FAIL: {log_dir} does not exist. Has Phase A run?")
        return 1

    any_short = False
    found_any = False

    for half in halves:
        rows = []
        for arm in ARMS:
            path = log_dir / f"{half}.ae_{arm}.log"
            if not path.is_file():
                rows.append((arm, None, None, None, None, None, "no log"))
                continue
            curve, cfg = read_log(path)
            if not curve:
                rows.append((arm, cfg.get("latent_ch"), cfg.get("ae_kl_weight"),
                             None, None, None, "no epochs yet"))
                continue
            found_any = True
            best_epoch, best = min(curve, key=lambda ev: ev[1])
            final_epoch, final = curve[-1]
            probe = at_or_before(curve, args.probe_epoch)
            if probe is None:
                gain = None
                note = f"only reached epoch {final_epoch}"
            else:
                # Negative = the run kept improving after the probe.
                gain = (final - probe[1]) / probe[1] * 100.0
                note = ""
            rows.append((arm, cfg.get("latent_ch"), cfg.get("ae_kl_weight"),
                         (best, best_epoch), probe[1] if probe else None,
                         (final, final_epoch), note or f"{gain:+.1f}%"))
            if gain is not None and gain < -5.0:
                any_short = True

        print(f"  half {half}")
        print(f"    {'arm':<6} {'l_ch':>5} {'kl':>7} {'best recon':>12} "
              f"{'@ep':>6} {'@%d' % args.probe_epoch:>12} {'final':>12} {'gain':>9}")
        ranked = [r for r in rows if r[3] is not None]
        ranked.sort(key=lambda r: r[3][0])
        for arm, lch, kl, best, probe, final, note in rows:
            if best is None:
                print(f"    {arm:<6} {str(lch or '-'):>5} {str(kl or '-'):>7} "
                      f"{'-':>12} {'-':>6} {'-':>12} {'-':>12} {note:>9}")
                continue
            mark = " <-- lowest recon" if ranked and ranked[0][0] == arm else ""
            print(f"    {arm:<6} {str(lch):>5} {str(kl):>7} {best[0]:>12.4e} "
                  f"{best[1]:>6d} "
                  f"{(('%12.4e' % probe) if probe is not None else '%12s' % '-')} "
                  f"{final[0]:>12.4e} {note:>9}{mark}")
        if ranked:
            print(f"    NOTE: this is a report, not a promotion policy -- the lowest")
            print(f"    recon almost always belongs to the largest latent_ch, which")
            print(f"    is not automatically the right pick (see select_ae.py's")
            print(f"    docstring for the knee + ceiling-gate policy it applies).")
            print(f"    `run_sweep.sh` (no argument) selects and promotes this half")
            print(f"    automatically; to force one arm:")
            print(f"       PROMOTE_{half.upper()}={ranked[0][0]} bash "
                  f"configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh")
        print()

    if not found_any:
        print("No Phase-A epoch lines found yet.")
        return 1

    print("Reading the gain column")
    print("  It is (final - probe) / probe, so NEGATIVE means the run kept")
    print("  improving after the probe epoch. Because the learning rate at the")
    print("  probe is still high in a long run, this is an UPPER BOUND on what")
    print("  the extra length actually bought.")
    if any_short:
        print(f"\n  At least one arm improved by more than 5% after epoch "
              f"{args.probe_epoch}.")
        print("  That is an upper bound, so it does not prove the long run was")
        print("  needed -- but it does mean the question is open, and the")
        print("  cheap way to close it is one dedicated short run to compare")
        print("  its annealed floor against these finals.")
    else:
        print(f"\n  No arm improved by more than 5% after epoch "
              f"{args.probe_epoch}.")
        print("  Since that figure is an upper bound, stage 1 was converged by")
        print(f"  then: {args.probe_epoch} epochs is enough for this data and")
        print("  future campaigns can drop back to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
