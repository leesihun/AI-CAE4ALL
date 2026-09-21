"""Production campaign driver: runs one run_batch.run_pilot() call per shape
in sampling.production_tiers()' fixed 54-shape roster (50 train + 4 OOD).

--start/--end select a disjoint index range so the SAME script drives both
machines against the SAME deterministic roster (production_tiers(seed=0) is
reproducible) -- e.g. local covers [0, 27), aarl covers [27, 54) -- with no
row list needing to be passed between them. Each shape's draws land under
their own run_root/<idx>_<tier>_rt..._lr.../, and a manifest.json in
run_root is rewritten after every shape so progress survives an interruption
partway through the roster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_batch
import sampling


def run_campaign(runs_root, start=0, end=None, max_workers=None):
    rows = sampling.production_tiers()
    if end is None:
        end = len(rows)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

    for idx in range(start, end):
        tier, r_over_t, l_over_r, draws = rows[idx]
        name = f"{idx:03d}_{tier}_rt{r_over_t:.1f}_lr{l_over_r:.3f}"
        run_root = runs_root / name
        print(f"=== shape {idx}/{len(rows) - 1}: {name} ({draws} draws) ===", flush=True)
        mesh, results = run_batch.run_pilot(
            run_root, r_over_t=r_over_t, l_over_r=l_over_r, n_draws=draws,
            max_workers=max_workers,
        )
        ok = sum(1 for r in results if r["ok"])
        manifest.append(dict(idx=idx, tier=tier, r_over_t=r_over_t,
                              l_over_r=l_over_r, draws=draws, ok=ok,
                              run_root=str(run_root)))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"    {ok}/{draws} ok", flush=True)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs_root")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()
    run_campaign(args.runs_root, start=args.start, end=args.end,
                 max_workers=args.max_workers)
