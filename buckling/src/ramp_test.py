"""Diagnostic: does a slower (more quasi-static) loading rate fix the
deterministic NaN-divergence failures seen at ood_lo_lo and train_tight_2
in corner_test.py? Same 2 seeds, same geometries, tend doubled from 1.0 to
2.0 (halves the imposed velocity v = 0.1*d_cr/tend for the same 10% target
end-shortening fraction)."""
from __future__ import annotations
import run_batch

CASES = [("ood_lo_lo", 95.0, 1.4), ("train_tight_2", 158.59, 0.849)]

if __name__ == "__main__":
    for name, rt, lr in CASES:
        print(f"--- {name}: R/t={rt} L/R={lr}, tend=2.0 ---", flush=True)
        try:
            mesh, results = run_batch.run_pilot(
                f"ramp_{name}", r_over_t=rt, l_over_r=lr, n_draws=2,
                max_workers=2, tend=2.0)
            ok = sum(1 for r in results if r["ok"])
            print(f"  {ok}/2 ok", flush=True)
            for r in results:
                print(" ", r, flush=True)
        except Exception as e:
            print(f"  EXCEPTION: {e!r}", flush=True)
