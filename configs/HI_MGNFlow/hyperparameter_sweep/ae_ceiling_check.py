#!/usr/bin/env python3
"""Ask whether the compressor can represent the warpage spread at all.

    python configs/HI_MGNFlow/hyperparameter_sweep/ae_ceiling_check.py --half bot --arm c8

THE QUESTION. Stage 2 is trained against a frozen stage 1 and never sees the
fine mesh except through its coarse latent, so the compressor sets a hard
ceiling on the whole model. The campaign measures one number -- the width of
the z-displacement peak-to-valley distribution across realizations of one part
-- and if the compressor's own reconstruction error is comparable to that
width, it is erasing exactly the signal being measured. No prior setting can
recover it, and every GPU-hour spent on Phase B is wasted.

This runs before Phase B and needs no GPU, no torch and no model: `mode
train_ae` already dumps its ENCODE/DECODE reconstruction against truth in
physical units every `test_interval` epochs (training_loop.py takes the
is_ae_only branch and writes predicted_denorm / target_denorm), and the
_compare_ files already hold the realizations.

WHAT IS COMPARED, AND WHY IT IS DIMENSIONLESS. The stage-1 dumps are samples of
the TRAINING split -- assorted parts -- while a _compare_ file is 125
realizations of ONE part, so the two have no common physical scale. Both sides
are therefore reduced to a dimensionless fraction of peak-to-valley before they
meet:

    relative recon error = RMS over dumped samples of
                           |p2v(reconstruction) - p2v(truth)| / p2v(truth)
    relative spread      = std(p2v) / mean(p2v) over the 125 realizations

    ceiling ratio        = relative recon error / relative spread

A ratio near or above 1 means the compressor distorts peak-to-valley by as much
as the physics varies it. This is an order-of-magnitude gate, not an exact
statistic -- it is answering "is stage 1 the bottleneck", which is a question
about orders of magnitude.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("FAIL: h5py is not importable. Run this with the HI_MGNFlow "
             "interpreter (see ai_cae4all.local.toml).")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"
DATASET_DIR = REPO_ROOT / "dataset" / "SAOI"

ARMS = ("c4", "c8", "c16", "c8kl")
STEMS = ("S26FE-MAIN", "S26FE-SEC", "SM-L345U-MAIN")

# Row 5 of nodal_data is the z displacement: rows 0:3 are reference coordinates
# and 3:6 the displacement state. Kept in step with Z_DISP_CHANNEL in
# methods/HI_MGNFlow/inference_profiles/rollout.py.
Z_DISP_ROW = 5

# Column 2 of a dump is the same quantity on the other side: predicted_denorm
# has output_var = 3 columns (dx, dy, dz), which is why the configs set
# plot_feature_idx 2.
Z_DISP_COL = 2

GREEN, AMBER = 0.30, 1.00


def parse_gpu_ids(config_path: Path) -> str:
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        line = line.split("#", 1)[0].strip()
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) == 2 and parts[0] == "gpu_ids":
            return parts[1].strip()
    raise SystemExit(f"FAIL: no gpu_ids in {config_path}")


def latest_epoch_dir(test_root: Path) -> Path:
    if not test_root.is_dir():
        raise SystemExit(
            f"FAIL: no dump directory at {test_root}.\n"
            f"      Stage 1 writes one every test_interval epochs. Either the "
            f"run has not reached the first interval, or display_testset was "
            f"turned off.")
    epochs = sorted((p for p in test_root.iterdir() if p.is_dir() and p.name.isdigit()),
                    key=lambda p: int(p.name))
    if not epochs:
        raise SystemExit(f"FAIL: {test_root} holds no epoch directories yet.")
    return epochs[-1]


def recon_fractions(epoch_dir: Path):
    """Per-sample |p2v(recon) - p2v(truth)| / p2v(truth) from the stage-1 dumps."""
    fractions, absolutes, truths = [], [], []
    files = sorted(epoch_dir.glob("*.h5"))
    for path in files:
        with h5py.File(path, "r") as fh:
            nodes = fh.get("nodes")
            if nodes is None or "predicted_denorm" not in nodes:
                continue
            pred = np.asarray(nodes["predicted_denorm"], dtype=np.float64)
            true = np.asarray(nodes["target_denorm"], dtype=np.float64)
        if pred.ndim != 2 or pred.shape[1] <= Z_DISP_COL:
            continue
        p2v_pred = float(pred[:, Z_DISP_COL].max() - pred[:, Z_DISP_COL].min())
        p2v_true = float(true[:, Z_DISP_COL].max() - true[:, Z_DISP_COL].min())
        if not np.isfinite(p2v_true) or p2v_true <= 0.0:
            continue
        absolutes.append(abs(p2v_pred - p2v_true))
        fractions.append(abs(p2v_pred - p2v_true) / p2v_true)
        truths.append(p2v_true)
    return np.asarray(fractions), np.asarray(absolutes), np.asarray(truths), len(files)


def compare_spreads(path: Path) -> np.ndarray:
    """Peak-to-valley of z displacement, one value per realization."""
    values = []
    with h5py.File(path, "r") as fh:
        group = fh.get("data")
        if group is None:
            raise SystemExit(f"FAIL: {path} has no 'data' group.")
        for key in group:
            nodal = group[key].get("nodal_data")
            if nodal is None or nodal.shape[0] <= Z_DISP_ROW:
                continue
            row = np.asarray(nodal[Z_DISP_ROW, 0, :], dtype=np.float64)
            values.append(float(row.max() - row.min()))
    return np.asarray(values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", required=True, choices=("bot", "top"))
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--config-dir", default=str(SCRIPT_DIR))
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--epoch", type=int, default=None,
                    help="dump epoch to read (default: the latest present)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    cfg = Path(args.config_dir).resolve() / f"config_train_ae_{args.half}_{args.arm}.txt"
    if not cfg.is_file():
        raise SystemExit(f"FAIL: no such config: {cfg}")
    gpu_ids = parse_gpu_ids(cfg)

    # log_dir is dirname(log_file_dir), and the dumps land at
    # <log_dir>/test/<gpu_ids>/<epoch>/. gpu_ids is what keeps the four arms of
    # a half from overwriting each other, and the per-phase log subdirectory is
    # what keeps Phase A from colliding with Phase B on the same card.
    test_root = out_root / "ae" / "test" / gpu_ids
    epoch_dir = (test_root / str(args.epoch)) if args.epoch is not None \
        else latest_epoch_dir(test_root)
    if not epoch_dir.is_dir():
        raise SystemExit(f"FAIL: no such dump directory: {epoch_dir}")

    print(f"AE ceiling check -- half {args.half}, arm {args.arm}")
    print(f"  dumps  : {epoch_dir}")

    fractions, absolutes, truths, n_files = recon_fractions(epoch_dir)
    if fractions.size == 0:
        raise SystemExit(
            f"FAIL: {epoch_dir} holds {n_files} file(s) but none carried a "
            f"usable nodes/predicted_denorm with a z column.")

    rel_recon = float(np.sqrt(np.mean(fractions ** 2)))
    print(f"  samples: {fractions.size}")
    print(f"  relative recon error (RMS) : {rel_recon:.4f}"
          f"   [{fractions.min():.4f} .. {fractions.max():.4f}]")
    print(f"  absolute p2v error   (RMS) : {np.sqrt(np.mean(absolutes ** 2)):.4e}"
          f"   over truths of mean {truths.mean():.4e}")
    print()

    worst = 0.0
    rows = []
    for stem in STEMS:
        path = Path(args.dataset_dir).resolve() / f"test_{stem}_compare_{args.half}.h5"
        if not path.is_file():
            rows.append((stem, None, None, None, "missing"))
            continue
        spreads = compare_spreads(path)
        if spreads.size < 2 or spreads.mean() == 0.0:
            rows.append((stem, spreads.size, None, None, "degenerate"))
            continue
        rel_spread = float(spreads.std() / abs(spreads.mean()))
        ratio = rel_recon / rel_spread if rel_spread > 0 else float("inf")
        worst = max(worst, ratio)
        rows.append((stem, spreads.size, rel_spread, ratio, ""))

    print(f"  {'eval part':<16} {'n':>5} {'rel spread':>12} {'ceiling ratio':>14}")
    for stem, n, rel_spread, ratio, note in rows:
        if rel_spread is None:
            print(f"  {stem:<16} {str(n or '-'):>5} {'-':>12} {'-':>14}  {note}")
            continue
        flag = "OK" if ratio < GREEN else ("MARGINAL" if ratio < AMBER else "FATAL")
        print(f"  {stem:<16} {n:>5} {rel_spread:>12.4f} {ratio:>14.3f}  {flag}")

    if all(r[3] is None for r in rows):
        print("\nNo _compare_ file was readable; nothing to compare against.")
        return 1

    print()
    if worst < GREEN:
        print(f"VERDICT: OK (worst ratio {worst:.3f} < {GREEN}).")
        print("  The compressor distorts peak-to-valley by well under the")
        print("  physical variation the histogram measures, so stage 1 is not")
        print("  the bottleneck. Promote this arm and run Phase B.")
        return 0
    if worst < AMBER:
        print(f"VERDICT: MARGINAL (worst ratio {worst:.3f}).")
        print("  The compressor is eating a visible fraction of the signal.")
        print("  Phase B can still run, but expect a spread ratio biased")
        print("  narrow no matter which prior wins, and prefer a higher")
        print("  latent_ch arm here even if its recon loss ranks second.")
        return 0
    print(f"VERDICT: FATAL (worst ratio {worst:.3f} >= {AMBER}).")
    print("  Reconstruction error alone is as large as the spread being")
    print("  measured. Stage 2 cannot recover it -- it never sees the fine")
    print("  mesh, and stage 1 is frozen while it trains. Do NOT start Phase")
    print("  B on this compressor: raise latent_ch (the ladder already goes to")
    print("  16) or widen the coarsest level via voronoi_clusters, and re-run")
    print("  Phase A. Every arm failing this way is the one outcome that")
    print("  invalidates the whole campaign rather than just one cell of it.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
