#!/usr/bin/env python3
"""Pick one Phase-A compressor per half automatically, then promote it.

    python configs/HI_MGNFlow/hyperparameter_sweep/select_ae.py --half bot
    python configs/HI_MGNFlow/hyperparameter_sweep/select_ae.py --half bot --force-arm c16

This is the step `run_sweep.sh` (no argument) calls between Phase A and Phase
B so the whole campaign finishes unattended. It reuses `ae_report.py`'s log
reader and `ae_ceiling_check.py`'s ceiling-ratio computation, decides, patches
`config_train_prior_<half>_*.txt` / `config_infer_<half>_*.txt` if the winner's
`latent_ch`/`ae_kl_weight` differ from what they currently assume, and calls
`promote_ae.py` to do the actual copy under its strict verification.

THE POLICY, AND WHY IT IS NOT "TAKE THE BEST RECON". c4/c8/c16 are a capacity
ladder: more channels per coarse node can only help reconstruction, so the
largest arm wins a bare recon comparison almost by definition. Taking it
blindly would buy capacity the Phase-B prior then has to model for no reason.
Instead:

  1. Drop any arm whose ae_ceiling_check ratio is FATAL (>= --fatal-threshold,
     default 1.0, checked against every one of the three evaluation parts) --
     it cannot represent the spread being measured, and no prior setting
     downstream fixes that.
  2. Within the surviving c4/c8/c16 ladder, take the SMALLEST latent_ch whose
     best validation recon is within --tol (default 10%) of the ladder's best
     value. This is the knee of the capacity curve, not its minimum.
  3. c8kl shares latent_ch=8 with c8 and only raises ae_kl_weight. If it
     survived step 1, sits at the same latent_ch as the knee, and beats it by
     more than --tol, prefer c8kl -- same cost, better-regularized latent.
  4. If EVERY arm fails the ceiling gate, there is no good choice. The
     least-bad arm (by the same knee/kl logic, computed without the gate) is
     still selected and promoted -- so the pipeline can still finish and
     report a number -- but the half is flagged NEEDS_ATTENTION (exit 3) and
     `run_sweep.sh` skips Phase B/C for it rather than spend eight more
     GPU-hours training a prior against a proven-broken compressor.

Exit codes: 0 promoted cleanly; 3 promoted but NEEDS_ATTENTION (see above);
1 could not decide (no Phase-A logs yet) or promote_ae.py rejected the result.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import ae_report as aer          # noqa: E402  (local module, path just inserted)
import ae_ceiling_check as aec   # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "SAOI"

ARMS = ("c4", "c8", "c16", "c8kl")
LADDER = ("c4", "c8", "c16")               # ascending latent_ch, by construction
LATENT_CH = {"c4": 4, "c8": 8, "c16": 16, "c8kl": 8}
AE_KL = {"c4": "1e-6", "c8": "1e-6", "c16": "1e-6", "c8kl": "1e-4"}

DEFAULT_TOL = 0.10


def best_recon(half: str, arm: str, out_root: Path):
    path = out_root / "ae" / f"{half}.ae_{arm}.log"
    if not path.is_file():
        return None
    curve, _cfg = aer.read_log(path)
    if not curve:
        return None
    return min(v for _e, v in curve)


def ceiling_ratio(half: str, arm: str, out_root: Path, cfg_dir: Path, dataset_dir: Path):
    """Worst ae_ceiling_check ratio over the three eval parts, or (None, reason)."""
    cfg = cfg_dir / f"config_train_ae_{half}_{arm}.txt"
    if not cfg.is_file():
        return None, "no config"
    try:
        gpu_ids = aec.parse_gpu_ids(cfg)
        epoch_dir = aec.latest_epoch_dir(out_root / "ae" / "test" / gpu_ids)
    except SystemExit:
        return None, "no dumps yet"
    fractions, _abs, _truth, _n = aec.recon_fractions(epoch_dir)
    if fractions.size == 0:
        return None, "dumps had no usable samples"
    rel_recon = float(np.sqrt(np.mean(fractions ** 2)))

    worst = 0.0
    any_compare = False
    for stem in aec.STEMS:
        path = dataset_dir / f"test_{stem}_compare_{half}.h5"
        if not path.is_file():
            continue
        spreads = aec.compare_spreads(path)
        if spreads.size < 2 or spreads.mean() == 0.0:
            continue
        any_compare = True
        rel_spread = float(spreads.std() / abs(spreads.mean()))
        if rel_spread > 0:
            worst = max(worst, rel_recon / rel_spread)
    if not any_compare:
        return None, "no _compare_ files"
    return worst, None


def pick_knee(pool, recon, tol):
    """Smallest-latent_ch ladder arm within `tol` of the ladder's best recon."""
    ladder_pool = [a for a in LADDER if a in pool]
    if not ladder_pool:
        return None
    best_val = min(recon[a] for a in ladder_pool)
    for a in ladder_pool:  # LADDER is already ascending latent_ch
        if recon[a] <= best_val * (1.0 + tol):
            return a
    return None  # unreachable: the arm achieving best_val always qualifies


def decide(half, out_root, cfg_dir, dataset_dir, tol, fatal_threshold):
    """Returns (arm, flagged, recon, ceiling) or None if no arm has a recon yet."""
    recon = {arm: best_recon(half, arm, out_root) for arm in ARMS}
    ceiling = {arm: ceiling_ratio(half, arm, out_root, cfg_dir, dataset_dir) for arm in ARMS}

    scored = [a for a in ARMS if recon[a] is not None]
    if not scored:
        return None

    usable = [a for a in scored
              if ceiling[a][0] is None or ceiling[a][0] < fatal_threshold]
    flagged = not usable
    pool = usable if usable else scored

    candidate = pick_knee(pool, recon, tol)
    if "c8kl" in pool and candidate is not None \
            and LATENT_CH["c8kl"] == LATENT_CH[candidate]:
        if recon["c8kl"] <= recon[candidate] * (1.0 - tol):
            candidate = "c8kl"
    elif "c8kl" in pool and candidate is None:
        candidate = "c8kl"

    if candidate is None:
        candidate = min(pool, key=lambda a: recon[a])

    return candidate, flagged, recon, ceiling


def patch_key(path: Path, key: str, value: str) -> int:
    """Rewrite `key\tvalue...` at column 0. Comments and every other line are
    untouched; a `%`-commented-out key line does not start with `key` so it
    never matches."""
    pattern = re.compile(r"^(" + re.escape(key) + r")([ \t]+)(\S+)(.*)$")
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = 0
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m and m.group(3) != value:
            lines[i] = f"{m.group(1)}{m.group(2)}{value}{m.group(4)}"
            changed += 1
    if changed:
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return changed


def patch_configs(half: str, cfg_dir: Path, latent_ch: int, ae_kl: str) -> None:
    prior_files = sorted(cfg_dir.glob(f"config_train_prior_{half}_*.txt"))
    infer_files = sorted(cfg_dir.glob(f"config_infer_{half}_*.txt"))
    changed = 0
    for path in prior_files:
        changed += patch_key(path, "latent_ch", str(latent_ch))
        changed += patch_key(path, "ae_kl_weight", str(ae_kl))
    for path in infer_files:
        changed += patch_key(path, "latent_ch", str(latent_ch))
    if changed:
        print(f"  patched {changed} line(s) across "
              f"{len(prior_files) + len(infer_files)} config(s) "
              f"to latent_ch={latent_ch} ae_kl_weight={ae_kl}")


def print_table(half, recon, ceiling):
    print(f"  {'arm':<6} {'l_ch':>5} {'best recon':>12} {'ceiling':>9}")
    for arm in ARMS:
        r = recon[arm]
        ratio, reason = ceiling[arm]
        r_s = f"{r:.4e}" if r is not None else "-"
        c_s = f"{ratio:.3f}" if ratio is not None else f"({reason})"
        print(f"  {arm:<6} {LATENT_CH[arm]:>5} {r_s:>12} {c_s:>9}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", required=True, choices=("bot", "top"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--config-dir", default=str(SCRIPT_DIR))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="fractional tolerance for the knee/KL comparisons "
                         "(default: 0.10)")
    ap.add_argument("--fatal-threshold", type=float, default=aec.AMBER,
                    help=f"ceiling ratio at or above which an arm is dropped "
                         f"(default: {aec.AMBER}, ae_ceiling_check.py's own line)")
    ap.add_argument("--force-arm", choices=ARMS,
                    help="skip the policy and promote this arm directly "
                         "(still patches configs and runs promote_ae.py)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    cfg_dir = Path(args.config_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    print(f"select_ae: half {args.half}")

    if args.force_arm:
        candidate, flagged = args.force_arm, False
        recon = {arm: best_recon(args.half, arm, out_root) for arm in ARMS}
        ceiling = {arm: ceiling_ratio(args.half, arm, out_root, cfg_dir, dataset_dir)
                   for arm in ARMS}
        print(f"  --force-arm given: skipping the policy, using {candidate}")
    else:
        result = decide(args.half, out_root, cfg_dir, dataset_dir,
                        args.tol, args.fatal_threshold)
        if result is None:
            print(f"FAIL: no Phase-A logs for half {args.half} under "
                  f"{out_root / 'ae'} yet.", file=sys.stderr)
            return 1
        candidate, flagged, recon, ceiling = result

    print_table(args.half, recon, ceiling)
    print(f"  -> selected: {candidate} "
          f"(latent_ch={LATENT_CH[candidate]}, ae_kl_weight={AE_KL[candidate]})")

    patch_configs(args.half, cfg_dir, LATENT_CH[candidate], AE_KL[candidate])

    rc = subprocess.call([
        sys.executable, str(SCRIPT_DIR / "promote_ae.py"),
        "--half", args.half, "--arm", candidate,
        "--out-root", str(out_root), "--config-dir", str(cfg_dir),
    ])
    if rc != 0:
        print(f"FAIL: promote_ae.py rejected {candidate} even after patching "
              f"-- this should not happen; inspect the checkpoint by hand.",
              file=sys.stderr)
        return 1

    if flagged:
        print(f"\nNEEDS_ATTENTION: half {args.half} has no compressor under "
              f"the ceiling threshold ({args.fatal_threshold}); {candidate} "
              f"was promoted anyway as the least-bad option so it can be "
              f"inspected, but Phase B/C should not spend GPU-hours on it "
              f"automatically.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
