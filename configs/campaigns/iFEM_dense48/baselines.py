#!/usr/bin/env python3
"""Reference bars for the iFEM_dense48 held-out benchmark.

Run from the AI-CAE4ALL suite root (takes a couple of minutes, once):

    $PYBIN configs/campaigns/iFEM_dense48/baselines.py

A trained arm is only worth reporting if it beats every line printed here. The metric is
the one score_infer.py reports, so the numbers are directly comparable to scores.json:

    rel_L2(case) = ||u_pred - u_true||_F / ||u_true||_F

Two channel sets are printed because both conventions appear in this project:
    all4 = u_eps1_x, u_eps1_y, u_eps2_x, u_eps2_y   (what score_infer.py reports)
    ch01 = u_eps1_x, u_eps1_y only                  (displacement alone)

Baselines:
    zero        predict 0 everywhere -> exactly 1.0 per case, by construction
    train_mean  predict the per-node mean field of the training file
    knn_k       mean target of the k nearest training cases in condition space
                (the 13 conditioning rows flattened: 13 x 81 = 1053 features)

These are measured on the files in dataset/iFEM_dense48/ -- do not carry numbers over
from any earlier dataset or split; the bars depend on both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CAMPAIGN = "iFEM_dense48"


def load(path: Path):
    """Returns condition features [n, 1053] and targets [n, 4, 81]."""
    with h5py.File(path, "r") as f:
        ids = sorted(f["data"].keys(), key=int)
        cond = np.empty((len(ids), 13 * 81), dtype=np.float32)
        targ = np.empty((len(ids), 4, 81), dtype=np.float32)
        for i, k in enumerate(ids):
            nd = f[f"data/{k}/nodal_data"][:, 0, :]
            cond[i] = nd[7:20].reshape(-1)
            targ[i] = nd[3:7]
    return cond, targ


def rel_l2(pred: np.ndarray, true: np.ndarray, rows: slice) -> np.ndarray:
    p = pred[:, rows, :].reshape(pred.shape[0], -1).astype(np.float64)
    t = true[:, rows, :].reshape(true.shape[0], -1).astype(np.float64)
    den = np.linalg.norm(t, axis=1)
    num = np.linalg.norm(p - t, axis=1)
    ok = den > 0
    return num[ok] / den[ok]


def report(name: str, pred, true, out: dict) -> None:
    row = {}
    for label, rows in (("all4", slice(0, 4)), ("ch01", slice(0, 2))):
        r = rel_l2(pred, true, rows)
        row[label] = {"median": float(np.median(r)), "mean": float(r.mean()),
                      "p90": float(np.percentile(r, 90)),
                      "frac_below_1": float((r < 1).mean())}
    out[name] = row
    a, c = row["all4"], row["ch01"]
    print(f"{name:<14} {a['median']:>9.4f} {a['mean']:>9.4f} {a['frac_below_1']:>7.3f}   "
          f"{c['median']:>9.4f} {c['mean']:>9.4f} {c['frac_below_1']:>7.3f}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(here.parent.parent.parent))
    ap.add_argument("--k", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    suite = Path(args.suite)
    data = suite / "dataset" / CAMPAIGN
    print("loading train ...", flush=True)
    ctr, ttr = load(data / "train.h5")
    print(f"  {ctr.shape[0]} cases")
    print("loading infer ...", flush=True)
    cte, tte = load(data / "infer.h5")
    print(f"  {cte.shape[0]} cases\n")

    out: dict = {}
    print(f"{'baseline':<14} {'all4 med':>9} {'mean':>9} {'<1.0':>7}   "
          f"{'ch01 med':>9} {'mean':>9} {'<1.0':>7}")
    print("-" * 72)
    report("zero", np.zeros_like(tte), tte, out)
    report("train_mean", np.repeat(ttr.mean(axis=0)[None], tte.shape[0], axis=0), tte, out)

    if args.k:
        # |a-b|^2 = |a|^2 - 2ab + |b|^2; the |a|^2 term is constant per query row.
        tr_sq = (ctr.astype(np.float64) ** 2).sum(axis=1)
        kmax = max(args.k)
        idx = np.empty((cte.shape[0], kmax), dtype=np.int64)
        for s in range(0, cte.shape[0], args.chunk):
            q = cte[s:s + args.chunk].astype(np.float64)
            d = tr_sq[None, :] - 2.0 * (q @ ctr.T.astype(np.float64))
            idx[s:s + args.chunk] = np.argpartition(d, kmax - 1, axis=1)[:, :kmax]
        for k in args.k:
            report(f"knn_k{k}", ttr[idx[:, :k]].mean(axis=1), tte, out)

    dst = suite / "output" / CAMPAIGN / "baselines.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
