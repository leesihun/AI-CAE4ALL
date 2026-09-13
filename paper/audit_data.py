"""Freeze and recompute the measurements used by the manuscript.

The historical SAOI table is a recorded result, not a reproduction: its raw
predictions are absent from this checkout. Buckling measurements are recomputed
from hashed draw archives. No success is inferred from a filtered failure list.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.distance import cdist, pdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = HERE / "evidence"
SNAPSHOT = EVIDENCE / "production_snapshot.json"
NOTE = ROOT / "docs/research/SAOI_PROBABILISTIC_SWEEP_2026-09.md"
PROD = ROOT / "buckling/work/production"
sys.path.insert(0, str(ROOT / "buckling/src"))
from geometry import ShellMesh
from analyse import radial, circumferential_spectrum


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geom_key(g):
    return "%s_rt%d_lr%03d_a%02d_g%02d" % (
        g["shape"][:3], g["r_over_t"], round(g["l_over_r"] * 100),
        round(g["alpha_deg"] * 10), round(g["thickness_gamma"] * 100))


def saoi_rows():
    """Read the two named tables, avoiding a second hand transcription."""
    note = NOTE.read_text(encoding="utf-8")
    out = {"mgnv": [], "flow": []}
    family = None
    for line in note.splitlines():
        if line.startswith("| MGN-V arm |"):
            family = "mgnv"
        elif line.startswith("| flow arm |"):
            family = "flow"
        elif family and re.match(r"\| \d+ \|", line):
            cells = [v.strip().replace("**", "") for v in line.split("|")[1:-1]]
            if "missing" in cells[2]:
                continue
            out[family].append([int(cells[0]), cells[1], *map(float, cells[2:])])
        elif family and line.startswith("## "):
            family = None
    if [len(out[k]) for k in ("mgnv", "flow")] != [7, 8]:
        raise ValueError("The historical SAOI tables changed; inspect their provenance.")
    return out


def capture():
    if SNAPSHOT.exists():
        raise FileExistsError("Snapshot already exists; retain it for reproducibility.")
    raw = (PROD / "results.json").read_bytes()
    records = json.loads(raw)
    planpath = ROOT / "buckling/work/plan_full.json"
    plan = json.loads(planpath.read_text())
    archives = {}
    for r in records:
        if r.get("ok"):
            p = PROD / r["key"] / "draw.npz"
            archives[r["key"]] = sha(p)
    snapshot = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                    results_sha256=hashlib.sha256(raw).hexdigest(),
                    plan_sha256=sha(planpath), plan=plan, records=records,
                    archives_sha256=archives, saoi_note_sha256=sha(NOTE),
                    saoi_evidence="historical research-note table; raw predictions unavailable")
    EVIDENCE.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def measurements(verify_hashes=True):
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if sha(NOTE) != snap["saoi_note_sha256"]:
        raise ValueError("Historical SAOI source has changed since capture")
    records = snap["records"]
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate keys in production snapshot")
    planned = {geom_key(g): g for g in snap["plan"]}
    if len(planned) != len(snap["plan"]):
        raise ValueError("Geometry key collision in plan")
    groups, descriptors = defaultdict(list), defaultdict(list)
    excluded = []
    for record in records:
        if not record.get("ok"):
            continue
        r = dict(record)
        key = r["key"]
        gk = key.rsplit("_s", 1)[0]
        g = planned[gk]
        if not g["seed0"] <= r["seed"] < g["seed0"] + g["n_draws"]:
            raise ValueError("Seed outside declared geometry plan: " + key)
        p = PROD / key / "draw.npz"
        if verify_hashes and sha(p) != snap["archives_sha256"][key]:
            raise ValueError("Draw changed since snapshot: " + key)
        with np.load(p, allow_pickle=False) as z:
            # Legacy tapered draws were solved at constant thickness. They must
            # never be presented as structural-OOD evidence.
            if g["thickness_gamma"] and "solver_contract_json" not in z:
                excluded.append(dict(key=key, reason="legacy uniform-thickness solve"))
                continue
            m = ShellMesh(g["r_over_t"], g["l_over_r"], shape=g["shape"],
                          alpha_deg=g["alpha_deg"], thickness_gamma=g["thickness_gamma"],
                          elems_per_half_wave=3.0)
            for name in ("disp", "w", "spectrum", "drift", "imp_C_real", "imp_C_imag", "k1", "k2"):
                if not np.isfinite(z[name]).all():
                    raise ValueError("Nonfinite %s in %s" % (name, key))
            if z["disp"].shape != (m.n_nodes, 3) or z["w"].shape != (m.n_nodes,):
                raise ValueError("Field shape mismatch: " + key)
            w = radial(m, z["disp"]) / m.t
            if not np.allclose(w, z["w"], rtol=2e-6, atol=2e-6):
                raise ValueError("Radial/displacement inconsistency: " + key)
            spectrum, _ = circumferential_spectrum(m, z["w"])
            if not np.allclose(spectrum, z["spectrum"], rtol=3e-5, atol=2e-5):
                raise ValueError("Stored spectrum differs from field: " + key)
            n = int(np.argmax(spectrum[1:])) + 1
            if n != int(z["n_dominant"]) or n != r["n_dominant"]:
                raise ValueError("Mode label differs from field: " + key)
            if abs(float(z["drift"]) - r["drift"]) > 5.1e-6:
                raise ValueError("Drift JSON/archive mismatch: " + key)
            r["drift"] = float(z["drift"])
            r["n_nodes"] = int(m.n_nodes)
            r["Z"] = float(m.batdorf_Z)
            r["n_crit_pred"] = float(0.86 * np.sqrt(g["r_over_t"]))
            groups[gk].append(r)
            vec = spectrum[1:41]
            descriptors[gk].append(vec / np.linalg.norm(vec))
    rows = []
    for key, rs in sorted(groups.items()):
        g = planned[key]
        ns = [r["n_dominant"] for r in rs]
        drift = np.array([r["drift"] for r in rs])
        timing = [r["seconds"] for r in rs if not r.get("cached") and r["seconds"] > 0]
        rows.append(dict(key=key, tier=g["tier"], n=len(rs), planned=g["n_draws"],
                         r_over_t=g["r_over_t"], l_over_r=g["l_over_r"],
                         n_nodes=rs[0]["n_nodes"], Z=rs[0]["Z"],
                         modes=dict(sorted(Counter(ns).items())), mean_mode=float(np.mean(ns)),
                         top_share=max(Counter(ns).values()) / len(ns),
                         empirical_reference=rs[0]["n_crit_pred"],
                         drift_median=float(np.median(drift)), drift_p90=float(np.quantile(drift, .9)),
                         drift_over_015=int((drift > .15).sum()),
                         seconds_mean=float(np.mean(timing)) if timing else None,
                         seconds_min=min(timing) if timing else None,
                         seconds_max=max(timing) if timing else None))
    train = [r for r in rows if r["tier"] == "train"]
    if len(train) != 12 or any(r["n"] != r["planned"] for r in train):
        raise ValueError("The full training-tier characterization requires all planned draws")
    # Cross-geometry comparison uses a fixed 40-component, unit-norm, axially
    # averaged spectrum. It is NOT the variable-length per-station field score.
    pairs = []
    for i, a in enumerate(train):
        for b in train[i + 1:]:
            X, Y = np.array(descriptors[a["key"]]), np.array(descriptors[b["key"]])
            e = 2 * cdist(X, Y).mean() - pdist(X).mean() - pdist(Y).mean()
            # The unbiased statistic can be negative in finite samples.
            pairs.append(dict(a=a["key"], b=b["key"], energy_distance_unbiased=float(e)))
    cost_x = np.array([r["Z"] for r in train])
    cost_y = np.array([r["seconds_mean"] for r in train])
    return dict(snapshot_captured_utc=snap["captured_utc"], snapshot_sha256=sha(SNAPSHOT),
                records=len(records), solver_successes=sum(bool(r.get("ok")) for r in records),
                solver_failures=sum(not bool(r.get("ok")) for r in records),
                excluded_invalid_physics=excluded, characterized=sum(r["n"] for r in rows),
                planned_geometries=len(planned), planned_draws=sum(g["n_draws"] for g in planned.values()),
                groups=rows, train_draws=sum(r["n"] for r in train),
                train_over_drift=sum(r["drift_over_015"] for r in train),
                cost_exponent=float(np.polyfit(np.log(cost_x), np.log(cost_y), 1)[0]),
                spectrum_energy_distances=pairs, saoi=saoi_rows())


def write_report(report):
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / "measurements.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    train = [r for r in report["groups"] if r["tier"] == "train"]
    macros = {
        "SnapshotDraws": str(report["records"]), "SnapshotSuccesses": str(report["solver_successes"]),
        "SnapshotFailures": str(report["solver_failures"]), "TrainDraws": str(report["train_draws"]),
        "TrainDriftFailures": str(report["train_over_drift"]),
        "TrainDriftPercent": "%.1f" % (100 * report["train_over_drift"] / report["train_draws"]),
        "TrainTopMin": "%.1f" % (100 * min(r["top_share"] for r in train)),
        "TrainTopMax": "%.1f" % (100 * max(r["top_share"] for r in train)),
        "TrainModesMin": str(min(len(r["modes"]) for r in train)),
        "TrainModesMax": str(max(len(r["modes"]) for r in train)),
        "CostExponent": "%.2f" % report["cost_exponent"],
    }
    (EVIDENCE / "numbers.tex").write_text("% Generated by audit_data.py; do not transcribe.\n" +
        "".join("\\newcommand{\\%s}{%s}\n" % kv for kv in macros.items()), encoding="utf-8")
    lines = ["% Generated from the frozen, hashed training-tier draws."]
    for r in sorted(train, key=lambda r: r["Z"]):
        lines.append("$%g$, $%g$ & %d & %d & %d & %d--%d \\\\" %
                     (r["r_over_t"], r["l_over_r"], round(r["Z"]), r["n_nodes"],
                      round(r["seconds_mean"]), round(r["seconds_min"]), round(r["seconds_max"])))
    (EVIDENCE / "cost_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", action="store_true", help="Create the immutable source snapshot once")
    a = ap.parse_args()
    if a.capture:
        capture()
    report = measurements()
    write_report(report)
    print(json.dumps({k: report[k] for k in ("records", "solver_successes", "solver_failures", "characterized", "train_draws", "train_over_drift")}, indent=2))


if __name__ == "__main__":
    main()
