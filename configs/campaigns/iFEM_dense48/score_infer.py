#!/usr/bin/env python3
"""Score the held-out inference dumps of every arm in the iFEM_dense48 sweep.

Run from the AI-CAE4ALL suite root:

    $PYBIN configs/campaigns/iFEM_dense48/score_infer.py

Why not just read "Valid" from the train log: that number is an ABSOLUTE, amplitude-
weighted pooled squared error in z-scored space, computed on the loader's own random
re-split of the training file -- so every case in it shares its geometry with training
cases, and its value is dominated by the few highest-amplitude samples (per-case target
norms on this dataset span four orders of magnitude). It is a training curve, not a
generalization measure. The unseen-geometry number comes from here.

Metrics, all on the 8,482 held-out cases whose 177 geometries appear nowhere in training:

  rel_L2(case)    = ||u_pred - u_true||_F / ||u_true||_F  over the full [4, 81] block.
                    An all-zero prediction scores exactly 1.0 on every case.
  rel_L2(channel) = the same per channel, over the cases where that channel's truth is
                    nonzero. u_eps2 (channels 2 and 3) is identically zero in ~80% of
                    cases; those are EXCLUDED and counted, never scored as 0 or 1.
  pooled_rel_L2   = sqrt(sum ||err||^2 / sum ||true||^2), the energy-weighted figure.
  R2(channel)     = 1 - sum(err^2) / sum((true - mean_true)^2), pooled over all nodes.

gval / gtest: with 33 arms, ranking and reporting on the same cases is selection on the
test set. The held-out file is therefore split once more BY GEOMETRY into two halves.
Rank the arms on gval; quote the winner's gtest number. The split is deterministic
(geometries sorted by a stable phi digest, assigned alternately), so it needs no stored
file and reproduces exactly on any machine.

Prediction layout, verified by running all three models: the rollout writes
data/<id>/nodal_data with 8 rows [x, y, z, out_0..out_3, part], and the prediction is
always the LAST timestep -- Transolver emits (8, 1, 81) (_steps0.h5), MeshGraphNets and
DeepONet emit (8, 2, 81) (_steps1.h5) whose t=0 is the zero initial state.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

CAMPAIGN = "iFEM_dense48"
CHANNELS = ["u_eps1_x", "u_eps1_y", "u_eps2_x", "u_eps2_y"]
SAMPLE_RE = re.compile(r"rollout_sample(\d+)_steps\d+\.h5$")


def load_truth(infer_h5: Path):
    """Returns {sample_id: [4, 81] target} and {sample_id: 'gval'|'gtest'}."""
    truth, phi = {}, {}
    with h5py.File(infer_h5, "r") as f:
        for key in f["data"].keys():
            sid = int(key)
            nd = f[f"data/{key}/nodal_data"]
            truth[sid] = nd[3:7, 0, :].astype(np.float64)
            # hashlib, not hash(): the builtin hash of bytes is salted per process.
            phi[sid] = hashlib.blake2b(nd[7, 0, :].astype(np.float32).tobytes(),
                                       digest_size=16).hexdigest()
    geoms = sorted(set(phi.values()))
    half = {g: ("gval" if i % 2 == 0 else "gtest") for i, g in enumerate(geoms)}
    return truth, {sid: half[h] for sid, h in phi.items()}, len(geoms)


class Accum:
    def __init__(self) -> None:
        self.rel: list[float] = []
        self.per_ch: list[list[float]] = [[] for _ in CHANNELS]
        self.zero_true = [0] * len(CHANNELS)
        self.err_energy = 0.0
        self.true_energy = 0.0
        self.ch_err = np.zeros(4)
        self.ch_true_sq = np.zeros(4)
        self.ch_sum = np.zeros(4)
        self.ch_count = 0

    def add(self, pred: np.ndarray, true: np.ndarray) -> None:
        err = pred - true
        tn = np.linalg.norm(true)
        if tn > 0:
            self.rel.append(float(np.linalg.norm(err) / tn))
        self.err_energy += float((err ** 2).sum())
        self.true_energy += float((true ** 2).sum())
        for c in range(4):
            tc = np.linalg.norm(true[c])
            if tc > 0:
                self.per_ch[c].append(float(np.linalg.norm(err[c]) / tc))
            else:
                self.zero_true[c] += 1
            self.ch_err[c] += float((err[c] ** 2).sum())
            self.ch_true_sq[c] += float((true[c] ** 2).sum())
            self.ch_sum[c] += float(true[c].sum())
        self.ch_count += true.shape[1]

    def summary(self) -> dict | None:
        if not self.rel:
            return None
        rel = np.asarray(self.rel)
        mean = self.ch_sum / self.ch_count
        r2 = []
        for c in range(4):
            den = self.ch_true_sq[c] - self.ch_count * mean[c] ** 2
            r2.append(float(1.0 - self.ch_err[c] / den) if den > 0 else float("nan"))
        return {
            "samples_scored": int(rel.size),
            "rel_L2_median": float(np.median(rel)),
            "rel_L2_mean": float(rel.mean()),
            "rel_L2_p90": float(np.percentile(rel, 90)),
            "rel_L2_frac_below_1": float((rel < 1.0).mean()),
            "pooled_rel_L2": float(np.sqrt(self.err_energy / self.true_energy))
            if self.true_energy > 0 else float("nan"),
            "per_channel": {
                CHANNELS[c]: {
                    "rel_L2_median": float(np.median(self.per_ch[c])) if self.per_ch[c] else None,
                    "scored_cases": len(self.per_ch[c]),
                    "cases_with_zero_truth": self.zero_true[c],
                    "R2": r2[c],
                } for c in range(4)
            },
        }


def score_arm(arm_dir: Path, truth, fold) -> dict | None:
    files = sorted(glob.glob(str(arm_dir / "infer" / "rollout_sample*_steps*.h5")))
    if not files:
        return None
    acc = {"all": Accum(), "gval": Accum(), "gtest": Accum()}
    missing = 0
    for path in files:
        m = SAMPLE_RE.search(os.path.basename(path))
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in truth:
            missing += 1
            continue
        with h5py.File(path, "r") as h:
            pred = h[f"data/{sid}/nodal_data"][3:7, -1, :].astype(np.float64)
        acc["all"].add(pred, truth[sid])
        acc[fold[sid]].add(pred, truth[sid])
    out = {k: v.summary() for k, v in acc.items()}
    if out["all"] is None:
        return None
    out["cases_missing_truth"] = missing
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    suite = here.parent.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(suite), help="AI-CAE4ALL root")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    suite = Path(args.suite)
    infer_h5 = suite / "dataset" / CAMPAIGN / "infer.h5"
    outputs = suite / "output" / CAMPAIGN

    print(f"ground truth : {infer_h5}")
    truth, fold, n_geom = load_truth(infer_h5)
    n_val = sum(1 for v in fold.values() if v == "gval")
    print(f"               {len(truth)} held-out cases over {n_geom} geometries")
    print(f"               geometry-split into gval {n_val} / gtest {len(truth) - n_val}\n")

    arms = args.arms or sorted(d.name for d in outputs.iterdir() if d.is_dir() and d.name != "packed")
    results = {}
    head = (f"{'arm':<5} {'n':>6} | {'gval med':>9} {'gtest med':>9} | "
            f"{'all med':>8} {'mean':>8} {'p90':>8} {'<1.0':>7} {'pooled':>8}")
    print(head)
    print("-" * len(head))
    for arm in arms:
        res = score_arm(outputs / arm, truth, fold)
        if res is None:
            continue
        results[arm] = res
        a, gv, gt = res["all"], res["gval"], res["gtest"]
        print(f"{arm:<5} {a['samples_scored']:>6} | {gv['rel_L2_median']:>9.4f} "
              f"{gt['rel_L2_median']:>9.4f} | {a['rel_L2_median']:>8.4f} {a['rel_L2_mean']:>8.4f} "
              f"{a['rel_L2_p90']:>8.4f} {a['rel_L2_frac_below_1']:>7.3f} {a['pooled_rel_L2']:>8.4f}")

    if not results:
        print("(no inference dumps found -- run infer_all.sh first)")
        return

    print("\nper-channel median relative L2, all held-out cases")
    print("(scored only on cases whose truth for that channel is nonzero):")
    print(f"{'arm':<5} " + " ".join(f"{c:>14}" for c in CHANNELS))
    for arm, res in results.items():
        cells = []
        for c in CHANNELS:
            v = res["all"]["per_channel"][c]["rel_L2_median"]
            cells.append(f"{v:>14.4f}" if v is not None else f"{'n/a':>14}")
        print(f"{arm:<5} " + " ".join(cells))

    any_arm = next(iter(results.values()))["all"]
    print("\nchannel coverage (identical for every arm):")
    for c in CHANNELS:
        pc = any_arm["per_channel"][c]
        print(f"  {c:<12} scored {pc['scored_cases']:>5}   "
              f"zero-truth, excluded {pc['cases_with_zero_truth']:>5}")

    dst = outputs / "scores.json"
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")

    bl = outputs / "baselines.json"
    print("\nreference bars (an all-zero prediction is 1.0000 by construction):")
    if bl.exists():
        data = json.loads(bl.read_text(encoding="utf-8"))
        for name, row in data.items():
            print(f"  {name:<12} {row['all4']['median']:.4f}")
    else:
        print(f"  run baselines.py first to write {bl}")
    print("\nRank the arms on gval, then quote the winner's gtest number.")


if __name__ == "__main__":
    main()
