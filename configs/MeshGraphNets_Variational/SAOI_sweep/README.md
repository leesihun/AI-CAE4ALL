# MeshGraphNets-V — SAOI warpage sweep

Eight arms, one per card 0–7. **The axis is `beta_aux`.**

```bash
nohup bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run.out 2>&1 &
tail -f output/meshgraphnets-v/saoi_sweep/run.out
```

One command: preflight → train (8 parallel) → inference → ranking → document.

| arm | card | section | `beta_aux` |
| --- | --- | --- | --- |
| 1 | 0 | bot | **0** (control) |
| 2 | 1 | bot | 10 |
| 3 | 2 | bot | 100 |
| 4 | 3 | bot | 1000 |
| 5 | 4 | top | **0** (control) |
| 6 | 5 | top | 10 |
| 7 | 6 | top | 100 |
| 8 | 7 | top | 1000 |

1000 epochs, batch 16, `best_by crps`. At a measured ~346 s/epoch that is about
four days per arm, all eight in parallel.

## What is being asked

`beta_aux` weights an MSE on the per-graph **peak-to-valley of the decoded
field** — `max(z_disp) − min(z_disp)`, the statistic the warpage report scores.

It replaced a head that regressed per-graph `[mean, std]` **from z**. That head
could not move `sd_ratio` — 0.449–0.526 across the entire previous grid — for
three structural reasons, and the new term fixes each:

* it read `z`, and nothing tied a z-readout's accuracy to the field a rollout
  writes;
* it used a node-axis standard deviation, not the extreme-value statistic that
  is scored;
* at `beta_aux 1.0` against `alpha_recon 1000` it carried roughly 1% of the
  objective.

**Four weights over three decades**, because the natural scale is not knowable
in advance: with realistic recon/aux magnitudes, the weight putting the term at
~30% of the objective came out near 8. Two points cannot separate *the
mechanism fails* from *the weight crushed the reconstruction*; a curve can.

`beta_aux 0` is the control. The old head was **deleted**, so without it nothing
separates the new term's effect from that removal.

**Both sections carry the same ladder**, so the curve's shape is replicated
rather than fitted once — the previous sweep had no replication at all, which
is why its factor effects could support no claim either way. `top` is the weaker
half in every previous measurement, so a fix that works only on `bot` is not a
fix.

## Why nothing else is swept

The previous `2^(4-1)` grid moved `W1/sd` by 0.001–0.054 across all four of its
factors, against a 0.10 arm-to-arm range. `z_conditioning`,
`prior_grad_to_encoder`, capacity and `lambda_mmd` are not where this model's
deficit lives.

## Reading the result

`docs/research/SAOI_SWEEP_MGNV.md`, written automatically when the run ends.
The metric is **`sd_ratio`, whose target is 1** — each eval set is one geometry
with 125 realizations of it, so the truth *is* the conditional distribution.

Read the **ladder's shape**, not the single best value. A curve that rises and
turns over says the weight matters; one that is flat across three decades says
the mechanism does not. The document's objective-balance table shows what share
of the loss the term actually carried at each weight, which is how a null result
gets attributed to the mechanism rather than to the scaling.

## Files

| | |
| --- | --- |
| `gen_configs.py` | generates every config from `../SAOI_all_input/`. **Do not hand-edit the configs** — regenerate |
| `run_sweep.sh` | the one command above |
| `check_eval_inputs.py` | comprehensive validator for the `_infer_`/`_compare_` pairing; run it when an eval set looks wrong |

Shared tooling lives in `configs/campaigns/`: `sweep_common.py` (the renderer
both sweeps use), `rank_arms.py` (metrics from the spread dumps alone) and
`write_report.py` (the document).

## If an arm dies

A failing arm is **dropped, not fatal** — the others still run, and everything
dropped is reprinted at the end so a shortened sweep cannot be mistaken for a
complete one. Re-run just those: `ARMS="3 7" bash .../run_sweep.sh`.

There is no resume, and `cosine_T0 = epochs − warmup`, so a killed arm restarts
from zero.

When the sweep is done, delete the shared hierarchy cache:

```bash
rm dataset/SAOI/saoi_train_{bot,top}.mscache.*.h5
```
