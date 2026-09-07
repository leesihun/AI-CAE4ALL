# SAOI probabilistic warpage: MeshGraphNets-V vs cHI-MGNflow

> ## Note on the eval design, 2026-09
>
> Each eval set is **one geometry with 125 realizations of it**:
> `infer_dataset` holds that part, `test_<MODEL>_compare_<half>.h5` holds its
> 125 outcomes. So `gt` IS the true conditional p(spread | geometry) and `gen`
> is the model's conditional for the same part. The comparison is exact, `sd(gt)`
> is real, and **`sd_ratio`'s target is 1** — every finding below stands as
> written.
>
> Two things follow for the tooling:
>
> - The **between/within decomposition** added in this cycle does not apply
>   here. With a single condition there is no between-scene term, which is why
>   the per-scene table reads `scenes 1`. It becomes useful only on a
>   multi-geometry eval set. Same for the deterministic control: one geometry
>   gives one point.
>
> - The test this design *does* support is a **rank/PIT histogram of the 125
>   realizations inside the model's own ensemble** (`## Rank calibration`).
>   It needs no scene labels, so it runs on dumps already on disk. Validated on
>   synthetic reproductions of this exact design: a calibrated ensemble reports
>   PIT KS 0.12 / tails 0.016, one that is 2x too narrow reports tails 0.256
>   against `sd_ratio` 0.574, and a biased one reports PIT mean 0.023. `sd_ratio`
>   and `PIT tails` are independent views of the same defect and should agree.

**Run 2026-09.** Two probabilistic upsampling models, one 8-arm fractional
factorial each, trained on `dataset/SAOI/saoi_train_bot.h5` and scored on three
held-out part families. This is the first head-to-head between the two.

Configs: [configs/MeshGraphNets_Variational/SAOI_sweep3/](../../configs/MeshGraphNets_Variational/SAOI_sweep3/),
[configs/HI_MGNFlow/SAOI_sweepB/](../../configs/HI_MGNFlow/SAOI_sweepB/).

> Design context, not implementation truth. Where this and the code disagree,
> the code is authoritative.

## What was run

Both sweeps are `2^(4-1)` resolution IV, 8 arms, one arm per GPU, 1000 epochs,
arms numbered 1–8 (the number *is* the design point; `score_sweep.arm_tags()`
maps it back).

| | factor A | B | C | D (= A⊕B⊕C) |
| --- | --- | --- | --- | --- |
| MGN-V | `z_conditioning` cc/ad | `prior_grad_to_encoder` g0/g1 | capacity c0/c1 (depth only) | `lambda_mmd` r001/r100 |
| flow | `batch_size` b16/b32 | `flow_t_sampling` tu/tl | capacity k0/k1 (128/192) | `learningr` lr1/lr3 |

Inference draws 2000 samples per scene against three eval sets
(`s26fe_main`, `s26fe_sec`, `sm_l345u_main`), each paired with its
`test_<MODEL>_compare_<half>.h5` ground truth.

### Metric definitions

From `score_sweep.collect_warpage()`. The per-realization statistic is the
warpage **spread** = `max(z_disp) − min(z_disp)` over nodes at the last
timestep, i.e. peak-to-valley. Everything is normalized by the ground truth
spread's own standard deviation so the three families are comparable.

| column | meaning | ideal |
| --- | --- | --- |
| `W1/sd` | 1-Wasserstein(gt, gen) / sd(gt) | **0** |
| `dmean/sd` | (mean(gen) − mean(gt)) / sd(gt) — bias | **0** |
| `sd_ratio` | sd(gen) / sd(gt) — dispersion | **1** |

`sd_ratio < 1` means the generated ensemble is **too narrow**. That reading
is sound here: the 125 eval samples are realizations of the very geometry
being predicted, so sd(gt) is the part's true process variation.

## Result

Mean over the three eval sets:

| | best arm | W1/sd | sd_ratio |
| --- | --- | --- | --- |
| **MGN-V** | **7** (ad g1 c0 r001) | **0.410** | 0.475 |
| flow | 1 (b16 tu k0 lr1) | 0.896 | 0.475 |

**MGN-V wins by 2.2×, and the two do not overlap**: flow's worst arm (1.604) is
3× MGN-V's worst (0.510).

Per arm, sorted by mean `W1/sd`:

| MGN-V arm | design | W1/sd | \|dmean\|/sd | sd_ratio | l345u W1 |
| --- | --- | --- | --- | --- | --- |
| 7 | ad g1 c0 r001 | **0.410** | 0.218 | 0.475 | 0.255 |
| 1 | cc g0 c0 r001 | 0.430 | 0.157 | 0.455 | 0.267 |
| 5 | ad g0 c0 r100 | 0.431 | 0.178 | 0.496 | 0.301 |
| 3 | cc g1 c0 r100 | 0.434 | 0.176 | **0.526** | **0.224** |
| 8 | ad g1 c1 r100 | 0.443 | 0.268 | 0.524 | 0.267 |
| 6 | ad g0 c1 r001 | 0.488 | 0.461 | 0.496 | 0.248 |
| 4 | cc g1 c1 r001 | 0.510 | 0.306 | 0.449 | 0.230 |
| 2 | cc g0 c1 r100 | *(missing — died in training)* | | | |

| flow arm | design | W1/sd | \|dmean\|/sd | sd_ratio | l345u W1 |
| --- | --- | --- | --- | --- | --- |
| 1 | b16 tu k0 lr1 | **0.896** | 0.724 | 0.475 | 1.425 |
| 2 | b16 tu k1 lr3 | 0.991 | 0.609 | 0.470 | 1.676 |
| 6 | b32 tu k1 lr1 | 1.310 | 0.985 | 0.548 | 2.706 |
| 4 | b16 tl k1 lr1 | 1.365 | 1.206 | 0.769 | 3.114 |
| 5 | b32 tu k0 lr3 | 1.477 | 1.270 | 0.487 | 3.192 |
| 3 | b16 tl k0 lr3 | 1.512 | 1.339 | 0.718 | 3.448 |
| 8 | b32 tl k1 lr3 | 1.525 | 1.395 | 0.761 | 3.468 |
| 7 | b32 tl k0 lr1 | 1.604 | 1.396 | 0.716 | 3.659 |

## Finding 1 — flow's failure is one part family, and it is pure bias

flow is only slightly behind MGN-V on the first two eval sets (0.48–0.66 vs
0.42–0.66). It collapses on **SM-L345U-MAIN**, where `W1/sd` runs 1.4–3.7.

The cause is written into the numbers: on that set `W1/sd` and `dmean/sd` are
almost identical.

```
flow arm 1:  W1 1.425   dmean 1.408
flow arm 7:  W1 3.659   dmean 3.641
```

The entire error is a **mean offset** — flow over-predicts warpage spread on
that family by 1.4–3.6 ground-truth standard deviations. Not a calibration
problem; a failure to generalize to the geometry.

Decisively, **the bias flips sign between families**: −0.13…−0.44 on
`s26fe_sec`, +1.4…+3.6 on `sm_l345u_main`. A unit or scaling bug would be
one-directional. This is the per-geometry conditional being wrong.

MGN-V on the same set: `dmean` −0.21…−0.02, `W1` 0.22…0.30. Location is right.

The two failure modes are cleanly opposite:

- **MGN-V** — location right, distribution ~2× too narrow
- **flow on l345u** — width plausible (0.86–1.42), location off by 3σ

## Finding 2 — neither model is calibrated

`sd_ratio` should be 1. It is nowhere near it.

| | best `sd_ratio` | worst |
| --- | --- | --- |
| MGN-V | 0.526 (arm 3) | 0.449 |
| flow | 0.769 (arm 4) | 0.470 |

Per set it is worse still: flow's `s26fe_sec` runs **0.199–0.378** — the
generated ensemble spans 20–38% of the truth's width.

**Both models produce ensembles roughly 2× too narrow.** The stated goal —
matching the distribution across samples — is met by neither. MGN-V is merely
less wrong. This, not the ranking, is the blocker.

## Finding 3 — main effects

**flow.** Large relative to its 0.708 arm-to-arm range, so these are real. The
best direction (b16 + tu + lr1) *is* arm 1, which is also the best arm —
consistent, which supports the effects being genuine.

| factor | result | Δ |
| --- | --- | --- |
| `flow_t_sampling` | **tu (uniform) 1.169** < tl 1.502 | 0.333 |
| `batch_size` | **b16 1.191** < b32 1.479 | 0.288 |
| `learningr` | lr1 1.294 < lr3 1.376 | 0.083 |
| capacity | negligible | 0.074 |

**MGN-V.** Effects are *tiny* — largest Δ 0.054 against a 0.10 arm-to-arm
range, with no replication and arm 2 missing. Treat as provisional.

| factor | result | Δ |
| --- | --- | --- |
| capacity | **c0 0.426** < c1 0.480 | 0.054 |
| `lambda_mmd` | r100 0.436 < r001 0.459 | 0.023 |
| `z_conditioning` | ad 0.443 < cc 0.458 | 0.015 |
| `prior_grad_to_encoder` | g0 0.450 ≈ g1 0.449 | **0.001** |

## Finding 4 — two predictions were falsified

Recorded because both were argued from mechanism and both were wrong.

**`prior_grad_to_encoder` does nothing measurable.** Δ 0.001 on `W1/sd`;
`sd_ratio` g0 0.482 vs g1 0.494, i.e. if anything marginally *wider*. The
prediction was that opening the `fm_loss → encoder` gradient would let the
encoder make `z` trivially predictable from `g`, narrowing the conditional
distribution (better CRPS, worse `W1/sd`). See
[MGN_ENHANCEMENT_RESEARCH.md](meshgraphnets_variational/MGN_ENHANCEMENT_RESEARCH.md)
for the argument. `beta_aux 1.0` is on in every arm and may be holding the
degenerate solution off — that is untested, but the *net* effect of the flag at
this budget is nil, so it is not a lever.

**`flow_t_sampling`: uniform beats logitnormal**, by flow's largest margin
(Δ 0.333). The prediction was the reverse, on the SD3 rationale that
concentrating training near t≈0.5 spends the budget where velocity is hardest.
On this data the endpoints evidently matter more.

## Gaps — read before citing any of this

1. **MGN-V arm 2 (`cc g0 c1 r100`) is missing**, killed by a GPU error. The
   design is 7/8, so main effects are computed 4-vs-3 (the `n=3` groups above)
   and `2^(4-1)` resolution IV does not hold. Arm 2 sits in the losing `c1`
   group, so the capacity conclusion in particular could flip. Re-run:
   `ARMS="2" TRAIN=1 INFER=1 SCORE=0 bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep.sh`

2. **Only one geometry per eval set.** `infer_dataset` holds a single part, so
   the sweep measures each model's conditional for ONE geometry per family, not
   across a population. The 2.2x ranking and the under-dispersion both hold for
   those three parts; nothing here says how either model behaves on a fourth.
   Widening the infer sets is what would let the between/within decomposition
   and `corr` say anything.

3. **1000 epochs is a budget, not convergence.** Both sweeps anneal a single
   cosine to `eta_min` over the run, so each arm is a finished run at *this*
   budget; none is asymptotic.

4. **CRPS is not a usable primary metric here.** With one truth per geometry its
   accuracy term dominates the spread term, so a mean-collapsed model scores
   well. It ranks regressors, not generative models.

## Recommendation

**Use MGN-V arm 3 (`cc g1 c0 r100`)**, not the headline winner.

| arm | W1/sd | sd_ratio | l345u W1 |
| --- | --- | --- | --- |
| 7 | **0.410** | 0.475 | 0.255 |
| 3 | 0.434 | **0.526** | **0.224** |

The 0.024 `W1/sd` gap is noise against a 0.10 arm-to-arm spread, while arm 3
has the best dispersion and the best generalization to the third family. When
the goal is distribution matching and everything is under-dispersed,
`sd_ratio` is the more direct measure. Arm 7 wins the headline number.

## Next

1. Re-run MGN-V arm 2 — without it the factor conclusions cannot be cited.
2. **Diagnose flow's `sm_l345u_main` bias.** Check first whether that family is
   represented in the training set at all. A sign-flipping bias is a broken
   conditional, not a tuning problem.
3. **Attack the 2× under-dispersion** — the real blocker. Gap 2 above has to
   land first: it separates "not enough between-geometry variation" from "not
   enough within-geometry noise", and those have different fixes.

## Tooling note

The per-scene decomposition and the deterministic-control configs added in this
cycle were built on a wrong reading of the data — that each eval set held many
geometries. It holds one, with 125 realizations, so:

- `scenes 1` in the per-scene table is the data being reported faithfully. That
  table now says "not applicable" rather than printing degenerate rows.
- `## Rank calibration` is the test that fits: the 125 realizations ranked
  inside the 2000-draw ensemble. No scene labels needed, so it can be run on
  every dump already written.

With one condition there is nothing to decompose: `sd_ratio` alone already says
whether the ensemble is the right width, and Finding 2 is the answer.
