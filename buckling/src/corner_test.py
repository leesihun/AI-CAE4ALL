"""One-off validation: run 2 draws at each of the parameter-space extremes
(the 4 OOD corners, post edge-margin-feasibility fix, + 2 tight train-range
points) to catch a mesh/RSA-placement edge case before firing the full
unattended production campaign, which so far has only ever exercised one
"nominal" geometry (R/t=150, L/R=2.0).

The original OOD grid's two low-L/R corners (95, 0.6) and (245, 0.6) were
found hard-infeasible for the imperfection law (edge margin > L, no draw at
any seed could place a dimple) and were replaced in sampling.py -- see its
OOD_SHAPES comment. This re-validates the REPLACED values end-to-end through
actual Starter/Engine execution, not just the imperfection law in isolation.
run_batch.run_one_draw now also retries internally on a soft RSA-placement
failure (run_batch.py's max_rsa_attempts loop), so this also exercises that
retry path on tight geometries rather than crashing the whole run_pilot call.
"""
from __future__ import annotations

import run_batch

CORNERS = [
    ("ood_lo_lo", 95.0, 1.4), ("ood_lo_hi", 95.0, 1.5),
    ("ood_hi_lo", 245.0, 0.7), ("ood_hi_hi", 245.0, 1.5),
    # feasible but tight (~20-30% single-attempt RSA success) -- exercises
    # run_batch.py's retry-on-soft-failure path, not just the happy path
    ("train_tight_1", 120.64, 0.959), ("train_tight_2", 158.59, 0.849),
]

if __name__ == "__main__":
    for name, rt, lr in CORNERS:
        print(f"--- {name}: R/t={rt} L/R={lr} ---", flush=True)
        try:
            mesh, results = run_batch.run_pilot(
                f"corner_{name}", r_over_t=rt, l_over_r=lr, n_draws=2, max_workers=2)
            ok = sum(1 for r in results if r["ok"])
            print(f"  {ok}/2 ok", flush=True)
            for r in results:
                print(" ", r, flush=True)
        except Exception as e:
            print(f"  EXCEPTION: {e!r}", flush=True)
