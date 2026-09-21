"""Is train_tight_2's (R/t=158.59, L/R=0.849) engine-divergence failure
really seed-independent (any imperfection realization at this geometry
fails), or were seeds 0,1 just unlucky? Test 6 more seeds."""
from __future__ import annotations
import run_batch

if __name__ == "__main__":
    mesh, results = run_batch.run_pilot(
        "seedsweep_train_tight_2", r_over_t=158.59, l_over_r=0.849,
        n_draws=6, seeds=[2, 3, 4, 5, 6, 7], max_workers=4)
    ok = sum(1 for r in results if r["ok"])
    print(f"\n{ok}/6 ok", flush=True)
    for r in results:
        print(" ", r, flush=True)
