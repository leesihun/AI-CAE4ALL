# cHI-MGNflow — SAOI warpage sweep

Eight arms, one per card 0–7. **A `2^3` full factorial: section × head ×
capacity.**

```bash
nohup bash configs/HI_MGNFlow/SAOI_sweep/run_sweep.sh \
    > output/chi-mgnflow/saoi_sweep/run.out 2>&1 &
tail -f output/chi-mgnflow/saoi_sweep/run.out
```

One command: preflight → train (8 parallel) → inference + deterministic
control → ranking → document.

| arm | card | section | head | capacity |
| --- | --- | --- | --- | --- |
| 1 | 0 | bot | v | k0 — 128 / mp 4,6,8,6,4 |
| 2 | 1 | bot | v | k1 — 192 / mp 6,8,12,8,6 |
| 3 | 2 | bot | **x** | k0 |
| 4 | 3 | bot | **x** | k1 |
| 5 | 4 | top | v | k0 |
| 6 | 5 | top | v | k1 |
| 7 | 6 | top | **x** | k0 |
| 8 | 7 | top | **x** | k1 |

2000 epochs, batch 16, `learningr 1e-4`, `flow_t_sampling uniform`,
`best_by crps`.

## The fix under test: what the network output means

flow's headline failure is a **bias on held-out geometry** — on `sm_l345u` the
error was ~85% pure mean offset, with a sign that flips between families. That
is a wrong *conditional mean*, not a wrong width, and it points at what the
network is asked to represent rather than at the sampler.

With a velocity head, at small `t` the output is dominated by the term that
cancels the initial noise field, node by node. A hierarchy built from pooled
features has to carry that per-node information down through every level and
back up to emit it again — capacity spent on plumbing, not on the field.

`flow_head x` makes the network predict the **clean field** `h` instead and
converts to velocity with the exact path identity inside the model:

```
v = (h − s·y_t) / (1 − s·t),    s = 1 − σ_min
```

so the loss, the integrator and the mean readout are untouched. The loss stays
in velocity space on purpose: that isolates the parameterization from the
objective weighting. The denominator is floored at `flow_head_eps 0.05` near
`t = 1` so the last ~5% of the path is regressed with bounded gain rather than
`1/σ_min`. The image-generation result behind this (Li & He, *Back to Basics*,
2025) is a hypothesis for a graph hierarchy — which is exactly why it is an arm
and not a default.

## Why capacity stays

In the previous grid capacity moved `W1/sd` by 0.074 — nothing — while
`batch_size`, i.e. the optimizer step count, moved it by 0.288. A dead capacity
axis beside a live step axis is the signature of a run that could not train the
model it had; the budget doubles here. And if the `x` head frees capacity from
plumbing, the bigger trunk is where that shows.

## What is held fixed

`learningr 1e-4` (the previous grid's marginal winner; the tie at 0.083 is not
worth a factor against a fix), `flow_t_sampling uniform` (beat logit-normal by
0.333) and `batch_size 16` (beat 32 by 0.288).

## Budget

At a **measured 113 s/epoch** on one card, 2000 epochs is ~63 h for a k0 arm.
The four k1 arms cost ~1.5× per epoch and land about a day later. Held identical
on purpose.

## Reading the result

`docs/research/SAOI_SWEEP_FLOW.md`, written automatically.

**Read the middle of the run.** `cosine_T0 = epochs − warmup` with
`eta_min 1e-8`, so every curve flattens at the end whether or not the model
converged. The report computes the fractional gain in `det(1fwd) mse` through
the middle half, where convergence is actually decided.

The **Deterministic control** section is the one to read first for this sweep:
`flow_predict mean` is the one-forward `E[y|g]`, and its `dmean/sd` against the
125 truths is the pure bias the `x` head is meant to reduce. `W1 ≥ |dmean|`
always, so the stochastic table beside it separates a *shifted* distribution
from a wrong-*shaped* one.

## Files

| | |
| --- | --- |
| `gen_configs.py` | generates every config from `../SAOI_all_input/`. **Do not hand-edit the configs** — regenerate |
| `run_sweep.sh` | the one command above |
| `check_eval_inputs.py` | comprehensive validator for the `_infer_`/`_compare_` pairing |

Shared tooling in `configs/campaigns/`: `sweep_common.py`, `rank_arms.py`,
`write_report.py`.

## Cost note

A draw is **not** one forward: it integrates the ODE, so it costs
`flow_steps × 2` network evaluations under Heun. Inference at 2000 draws per
scene is the most expensive stage after training.

## If an arm dies

A failing arm is **dropped, not fatal**; everything dropped is reprinted at the
end. Re-run just those: `ARMS="3 7" bash .../run_sweep.sh`. There is no resume,
so a killed arm restarts from zero.

When the sweep is done, delete the shared hierarchy cache:

```bash
rm dataset/SAOI/saoi_train_{bot,top}.mscache.*.h5
```
