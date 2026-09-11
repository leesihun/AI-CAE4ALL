# cHI-MGNflow — SAOI warpage sweep

Eight arms, one per card 0–7. **A `2^3` full factorial: section × learningr ×
capacity.**

```bash
nohup bash configs/HI_MGNFlow/SAOI_sweep/run_sweep.sh \
    > output/chi-mgnflow/saoi_sweep/run.out 2>&1 &
tail -f output/chi-mgnflow/saoi_sweep/run.out
```

One command: preflight → train (8 parallel) → inference → ranking → document.

| arm | card | section | `learningr` | capacity |
| --- | --- | --- | --- | --- |
| 1 | 0 | bot | 1e-4 | k0 — 128 / mp 4,6,8,6,4 |
| 2 | 1 | bot | 1e-4 | k1 — 192 / mp 6,8,12,8,6 |
| 3 | 2 | bot | 3e-4 | k0 |
| 4 | 3 | bot | 3e-4 | k1 |
| 5 | 4 | top | 1e-4 | k0 |
| 6 | 5 | top | 1e-4 | k1 |
| 7 | 6 | top | 3e-4 | k0 |
| 8 | 7 | top | 3e-4 | k1 |

2000 epochs, batch 16, `flow_t_sampling uniform`, `best_by crps`.

## Why a full factorial

Eight arms is exactly the full grid of three two-level factors, so **nothing is
confounded with anything**. The previous SAOI grid was a `2^(4-1)` half fraction
whose two-factor effects came in confounded pairs; at eight arms for three
factors that compromise buys nothing.

## Why these three factors

The earlier grid **resolved** its other two axes, and both are fixed here at the
winner: `flow_t_sampling` uniform beat logit-normal by its largest margin
(0.333), and batch 16 beat 32 (0.288).

It could **not** resolve `learningr` (0.083) or capacity (0.074) — and those two
ties are the interesting part. A dead capacity axis sitting next to a live
step-count axis is the signature of a run that could not train the model it
had: a bigger network cannot show its capacity until it can be trained. The
budget doubles here, which is the condition under which that ranking can invert.
`learningr` is the other factor coupled to the budget, since stretching a cosine
changes which starting rate is right.

## Budget

At a **measured 113 s/epoch** on one card — the old config's 500 s/epoch
estimate was 4.4× pessimistic — 2000 epochs is about 63 h for a k0 arm. The four
k1 arms cost roughly 1.5× per epoch and land about a day later. The epoch count
is held identical on purpose: a capacity comparison at a different budget
answers a different question.

## Reading the result

`docs/research/SAOI_SWEEP_FLOW.md`, written automatically when the run ends.

**Read the middle of the run.** `cosine_T0 = epochs − warmup` with
`eta_min 1e-8`, so the learning rate is essentially zero by the last epoch and
**every curve flattens there whether or not the model converged**. The report
computes the fractional gain in `det(1fwd) mse` through the middle half, which
is where convergence is actually decided, and prints the last quarter beside it
for contrast only.

On the distribution side the metric is `W1/sd` with `dmean/sd` next to it.
`W1 ≥ |dmean|` always, so the two columns together separate a distribution that
is *shifted* from one that is the wrong *shape* — which is how the previous
grid's `sm_l345u` failure was identified as ~85% pure bias.

## Files

| | |
| --- | --- |
| `gen_configs.py` | generates every config from `../SAOI_all_input/`. **Do not hand-edit the configs** — regenerate |
| `run_sweep.sh` | the one command above |
| `check_eval_inputs.py` | comprehensive validator for the `_infer_`/`_compare_` pairing |

Shared tooling lives in `configs/campaigns/`: `sweep_common.py` (the renderer
both sweeps use), `rank_arms.py` and `write_report.py`.

## Cost note

A draw here is **not** one forward: it integrates the ODE, so it costs
`flow_steps × 2` network evaluations under Heun. Inference at 2000 draws per
scene is 8 arms × 3 eval sets × 2000 × 60 evaluations before batching savings —
comfortably the most expensive stage after training.

## If an arm dies

A failing arm is **dropped, not fatal**; everything dropped is reprinted at the
end. Re-run just those: `ARMS="3 7" bash .../run_sweep.sh`. There is no resume,
so a killed arm restarts from zero.

When the sweep is done, delete the shared hierarchy cache:

```bash
rm dataset/SAOI/saoi_train_{bot,top}.mscache.*.h5
```
