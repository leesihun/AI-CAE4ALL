# Build log

What has actually been run, and what it measured. Design intent lives in
[BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md); this file records reality, including
the places where reality corrected the design.

---

## 2026-09-10 — Environment

**CalculiX 2.22 running under WSL (Ubuntu 24.04), 32 cores visible.**

`apt` needs sudo, which is not available here, so the solver was installed
entirely in userspace:

- `http://www.dhondt.de/ccx_2.22.tar.bz2` ships a **prebuilt** ELF binary at
  `CalculiX/ccx_2.22/src/ccx_2.22` — no compilation needed.
- It links against `libgfortran.so.4`, which Ubuntu 24.04 dropped. Extracted
  from the Ubuntu 18.04 `libgfortran4` `.deb` with `dpkg-deb -x` into `~/lib`
  and picked up via `LD_LIBRARY_PATH`.

---

## 2026-09-10 — Mesh generator

`geometry.py` produces mapped S8R meshes analytically in float64. Counts land
where the design predicted:

| R/t | L/R | n_circ | n_axial | nodes | elems | Z | d_cr |
|---|---|---:|---:|---:|---:|---:|---:|
| 110 | 1.0 | 155 | 24 | 11,470 | 3,720 | 104.9 | 5.50e-3 |
| 110 | 1.6 | 155 | 39 | 18,445 | 6,045 | 268.6 | 8.80e-3 |
| 230 | 1.0 | 223 | 35 | 23,861 | 7,805 | 219.4 | 2.63e-3 |
| 230 | 1.6 | 223 | 56 | 37,910 | 12,488 | 561.7 | 4.21e-3 |

Design predicted 11.4k–37k nodes and Z ∈ [105, 562]. Matches.

`avoid_resonant_ncirc` nudges the circumferential division count away from
small multiples of plausible critical wavenumbers, because a structured mesh
retains a discrete C_N symmetry and is not neutral among modes.

---

## 2026-09-10 — BUG: CalculiX silently corrupts a `%.15e` deck

**The design said "write coordinates at ≥15 significant digits". Doing that
breaks the deck.**

CalculiX caps a free-format numeric field at **20 characters**. `%.15e` is 21
characters (22 with a sign). The failure is not a parse error at the offending
line — it surfaces as:

```
*ERROR reading *ELASTIC: Poisson coefficient should be less than 0.5
        1.000000000000000E+00,3.000000000000000E-01
```

…which has nothing to do with the Poisson ratio, plus a cascade of
`*WARNING reading *NSET: value N > nk`. Confirmed by bisection on a 1-element
model:

| format | field chars (pos / neg) | result |
|---|---|---|
| `%.15e` | 21 / 22 | `*ERROR` |
| `%.14e` | 20 / 21 | Job finished |
| `%.13e` | 19 / 20 | Job finished |

**Adopted `%.13e`** — 14 significant digits, 20 characters *including* the sign,
so negative coordinates are safe too. That is ~1e-14 relative precision against
a concern at 1e-6, so eight orders of margin remain.

---

## 2026-09-10 — Gate G1: PASSED

`python run_g1.py` — perfect-shell eigenvalue census, 60 eigenvalues per corner.
Total ~9 min.

### G1a — symmetry-breaking floor: **0.0, exactly**

Requirement was < 1e-4 relative splitting of nominally degenerate cos/sin pairs.

| corner | exact-zero gaps | fraction | λ₁ |
|---|---:|---:|---:|
| R/t 110, L/R 1.0 | 29 / 59 | 0.49 | 0.9866 |
| R/t 110, L/R 1.6 | 29 / 59 | 0.49 | 0.9806 |
| R/t 230, L/R 1.0 | 29 / 59 | 0.49 | 0.9888 |
| R/t 230, L/R 1.6 | 29 / 59 | 0.49 | 0.9949 |

30 machine-exact degenerate pairs out of 60 eigenvalues on every corner — 49 %
of consecutive gaps are **identically zero in float64**. The mapped mesh, the
analytic float64 coordinates and the tightened tolerances together put the
pipeline's total symmetry-breaking floor at machine precision, not at 1e-4.

λ₁ sits at 0.981–0.995 of the classical value, which is the expected small
elevation from clamped ends — an independent check that the mesh and BCs are
right.

> **Reporting bug, now fixed.** The first version of `run_g1.py` paired any two
> eigenvalues within 5e-3 and called the gap a "splitting", so it reported
> `max_split ≈ 2e-3` and a G1a failure. It was measuring the *spacing between
> distinct modes*, not symmetry breaking. The correct statistic is the largest
> gap among **exactly-degenerate** partners, which is 0.0.

### G1b — cluster size: depends on amplitude, and this settles σ̂

Modes within the knockdown window that a given imperfection amplitude opens:

| corner | Z | σ̂=1e-4 (0.4 %) | σ̂=1e-3 (1.56 %) | σ̂=1e-2 (14.6 %) |
|---|---:|---:|---:|---:|
| 110 / 1.0 | 105 | **4** | 10 | ≥60 |
| 110 / 1.6 | 269 | **4** | 20 | ≥60 |
| 230 / 1.0 | 219 | 8 | 20 | ≥60 |
| 230 / 1.6 | 562 | 22 | 60 | ≥60 |

**This is a direct empirical answer to the open amplitude question.** At
σ̂ = 1e-4 the low-Z corners have only 4 competing modes — far too few for the
multimodality the benchmark needs, and the dataset would have been close to
deterministic there. The original 1e-4 choice would have failed quietly.

At σ̂ = 1e-2 every computed eigenvalue is inside the window on every corner, so
the cluster is ≥60 modes everywhere — arguably *too* diffuse, since rare-branch
occupancy at K≈60 would push the required reference draws well past 500.

σ̂ = 1e-3 gives K = 10–60 across the box, closest to the K ≈ 12–21 the
allocation was sized for. **Provisional choice: σ̂ = 1e-3**, to be confirmed by
the realized branch count in G2 — an eigenvalue inside the window is only an
*upper bound* on branches that actually win.

---

## 2026-09-10 — Nonlinear continuation: static diverges, dynamics works

**Pure displacement control fails at the limit point.** `write_static_inp` on
the smallest corner at σ̂=1e-2 reached normalized time 0.528 of a 1.5·d_cr
target — i.e. 0.79·d_cr — then died with `*ERROR: too many cutbacks` after
increments shrank to 3.9e-4.

That is not a bug, and the number is reassuring: Koiter predicts γ = 0.855 at
σ̂ = 1e-2, so the shell is buckling essentially where theory says it should. The
solver simply cannot follow the snap, which is risk R3 in the design.

CalculiX 2.22 has **no** `*STATIC, STABILIZE` and **no** Riks. Confirmed by
scanning the binary's keyword table. Available instead: `*DYNAMIC` with HHT
`ALPHA` and `*DAMPING, ALPHA=/BETA=`.

**`write_dynamic_inp` — quasi-static implicit dynamics** is the adopted route:
inertia absorbs the snap, mass-proportional damping bleeds off the kinetic
energy so the terminal state is a genuine static equilibrium, and HHT
`ALPHA=-0.3` (near CalculiX's -1/3 limit) dissipates the high-frequency content
the snap excites.

First run marched **101 increments at 2 Newton iterations each with zero
cutbacks**, versus the static run's collapse. The method works.

### Three more bugs, each of which silently produced a non-buckling dataset

**(a) `*DYNAMIC` as step 2 aborts unless step 1 requests energy output.**
```
*ERROR in CalculiX: in nonlinear calculations
        energy output requests, if any, must be specified in the first step
```
A `*DYNAMIC` step implicitly requests energy output. Fix: an explicit
`*EL FILE / ENER` in step 1.

**(b) Shell results come back on CalculiX's expanded 3-D mesh.** The `.frd`
held 14,803 nodes for a 6,328-node shell. `*NODE FILE, OUTPUT=2D` returns
results in the original shell numbering.

**(c) `damp_alpha = 8.0` suppressed the instability entirely.** This was the
serious one, and it looked like success. Every damped run completed cleanly,
and every one of them was still on the linear pre-buckling path at 1.5x the
classical critical shortening:

| run | F(end)/F(start) | displacement ratio |
|---|---|---|
| two-step, damp 8.0 | 2.117 | 2.143 |

Force tracked displacement to within 1.2 % of linear. Worse, `sigma_hat=1e-3`
and `1e-2` gave **byte-identical** terminal fields, which is the signature of an
imperfection that is not doing anything. The apparent "mode" was n = 2, 3, 8 --
i.e. Component A's ovalization and weld-panel count -- not a buckle at all.

Isolated with undamped static runs, which do buckle:

| sigma_hat | d/d_cr | F/F_linear |
|---|---|---|
| 1e-3 | 0.874 -> 1.200 | 0.990 -> **0.461** |
| 1e-1 | 0.761 -> 0.824 | 0.821 -> **0.633** |

Load collapses to 43-46 % of linear: a genuine limit point. The `sigma_hat=1e-1`
onset at d/d_cr ~ 0.76 sits near Koiter's gamma = 0.611, and the `1e-2` static
run stalled at 0.876 against Koiter's 0.855.

Fix: `damp_alpha` 8.0 -> **0.3**. Mass-proportional damping is there to bleed
kinetic energy off *after* the snap, not to prevent it.

**(d) The terminal state was still ringing.** After the fix, step 2 ended at
F/F_linear = 1.086 -- an oscillation snapshot, not the static equilibrium the
QoI is defined as. Added **step 3**: hold the displacement, let damping settle.

---

## 2026-09-10 — End-to-end validation: PASSED

Three seeds, same geometry (R/t=110, L/R=1.0, eph=3), `sigma_hat=1e-3`,
Component A off, three-step solve:

| seed | F_end/F_lin | rms/t | max/t | dominant n | top-5 n |
|---|---|---|---|---|---|
| 1 | 0.566 | 0.450 | 0.968 | 10 | 10, 9, 11, 7, 8 |
| 2 | 0.565 | 0.448 | 0.967 | 10 | 10, 8, 9, 7, 11 |
| 3 | 0.565 | 0.452 | 1.020 | **9** | 9, 10, 11, 12, 7 |

- **Settled.** F_end/F_lin is 0.565-0.566 across all three, versus 1.086 before
  step 3. The terminal state is a static equilibrium.
- **Buckled.** Load carried is 57 % of linear-elastic.
- **Genuinely one-to-many.** Pairwise relative L2 between draws: **0.249, 0.349,
  0.386** (mean **0.33**), against the design's G2 threshold of 0.30. And the
  selected mode changes between draws -- seeds 1 and 2 pick n=10, seed 3 picks
  n=9 -- from an imperfection of RMS 1e-3 * t.

Runtime 368.7 / 369.0 / 368.8 s -- reproducible to 0.1 %.

### Component A must be reconsidered

At the designed RMS of 0.30*t, Component A **dominates the outcome**: the
terminal field was n = 2, 3, 8 (its own ovalization and weld wavenumbers) rather
than a buckling pattern, and it is *deterministic*, so it removes exactly the
stochasticity the benchmark exists to measure. The design's own warning applies
-- "past sigma ~ 0.5t the largest defect locks the response".

All validation above therefore ran with **Component A off**. Its role in
breaking the SO(2) symmetry (design section 3.3 F3) is still needed, so the open
question is the largest Component A amplitude that breaks symmetry *without*
dominating mode selection. Provisionally this is 0.01-0.05*t rather than 0.30*t,
and it needs its own sweep.

---

## Cost

Measured **369 s/draw** single-core at eph=3 on the smallest corner
(6,328 nodes), reproducible to 0.1 %.

| | |
|---|---|
| thread scaling | poor -- 1->4 threads = 1.18x, 4->8 = nothing. **Run 24 single-core jobs, not 4 threaded ones.** Confirms the design's lane model. |
| eph 4 -> 3 | 2.0x faster (44.6 s vs 22.0 s on the static phase); 11,470 -> 6,328 nodes |
| static pre-buckling | 7 increments, essentially free -- which is what makes the two-step worthwhile |

17,520 draws x 369 s / 24 lanes = **75 h = 3.1 days** *at the smallest geometry*.
The box-spanning average will be roughly 2-2.5x that, i.e. **6-8 days**, against
a 5-day target. The section-10 sensitivity plan (cut Train A geometries first,
then T1/T3 draws) covers the gap; no production launch until the box-spanning
cost is measured on all four corners.

---

## 2026-09-10 — Parallel runner: /mnt/c was the bottleneck, not the CPU

First 24-lane batch ran at **82 s per increment against ~4 s solo** — a 20x
slowdown that looked like memory contention and was not. CalculiX does a great
many small reads and writes, and the jobs were running on `/mnt/c`, WSL2's
Windows-filesystem bridge. Measured on a 200 MB bulk write:

| path | time |
|---|---|
| `/mnt/c/...` | 3.67 s |
| WSL-native `~/` | 0.63 s |

5.8x on bulk, far worse on CalculiX's small-I/O pattern. `run_ccx(scratch=True)`
now copies the deck into `~/ccxrun/<job>`, solves there, and copies back only
`.frd`/`.dat`/`.sta`.

Lane count set to **19** (60 % of 32 logical cores) at the user's request. The
machine is a Ryzen 9 3950X: 16 physical cores, 32 threads. SPOOLES thread
scaling is poor (1→4 threads = 1.18x, 4→8 = nothing), so many single-core jobs
is right regardless.

---

## 2026-09-10 — Increment schedule and the settle step

Every increment was converging in 2 Newton iterations at the maximum allowed
step, i.e. the schedule was far finer than needed. Relaxing `dt_max` cut
increments from 107 to 36 and CalculiX auto-cut-back to 24 at the snap, exactly
as intended.

But the terminal **force** does not converge in the increment count:

| n_inc_dyn | F_end/F_lin | rms/t | dominant n |
|---|---|---|---|
| 20 | 0.672 | 0.4616 | 10 |
| 40 | 0.822 | 0.4538 | 10 |
| 80 | 0.566 | 0.4495 | 10 |

Non-monotonic — the terminal state is still ringing, so `F_end` reports wherever
the oscillation happens to be. The **field** converges cleanly (rms within
2.7 % across a 4x increment change) and the **mode label is invariant** (n=10
throughout, identical top-5).

**A static settle step was tried and is wrong.** A deep post-buckled cylinder
state is not an isolated stable equilibrium, so the static solve wanders off:
rms/t of 8.0 and 12.7 against a physical 0.45. Damped dynamics is the only
thing that holds the branch. Reverted.

Consequence for the protocol: **F_end/F_lin is a buckled / not-buckled
indicator, not a QoI.** The QoI is the field.

---

## 2026-09-10 — Component A amplitude sweep: 0.30t is fatal, 0.01t is right

60 draws at R/t=110, L/R=1.0, sigma_hat=1e-3, 12 seeds per level.

| A / t | dominant-n distribution | mean pairwise spread | rms/t |
|---|---|---|---|
| 0.00 | 10(5), 9(4), 11(3) | 0.352 | 0.451 |
| **0.01** | **10(5), 9(3), 8(3), 11(1)** | **0.350** | 0.452 |
| 0.03 | 8(**7**), 10(3), 9(2) | 0.345 | 0.455 |
| 0.10 | 8(2) — pinned | 0.313 | 0.488 |

**Adopted A = 0.01 t.** It is the largest amplitude that breaks the SO(2)
symmetry while leaving the outcome distribution as diverse as A=0: four
distinct modes, spread 0.350 against 0.352, top-branch weight 0.42 (inside the
0.45 limit). At 0.03 Component A begins to pin the mode; by 0.10 it dictates
it. The design's 0.30 t is confirmed catastrophic — it produced n = 2, 3, 8,
its own ovalization and weld wavenumbers, with no buckle at all.

Note that even at A = 0 there are three distinct modes across 12 seeds, so
mode selection is genuinely stochastic before any symmetry breaking is added.

**Caveat:** "branch" here is the dominant circumferential wavenumber, which is
coarser than the SO(2)-invariant field clustering G2 actually prescribes, and
12 draws is thin for counting branches. On this coarse label the effective K is
3.55, below the G2 threshold of 4.0. The full 80-draw pilot decides.

---

## 2026-09-10 — The branch structure may be continuous, not discrete

The G2 clustering as first written was wrong and gave physically impossible
answers. At Component A = 0.30 t **every one of 12 draws pinned to n = 8**, yet
it reported 7 branches with effective K = 6.45; at A = 0.01, four distinct
dominant wavenumbers were reported as 2 branches. Causes: a 760-dimensional
descriptor PCA-ed to 32 dimensions from 12 samples (rank <= 11), and an
elbow-on-inertia rule that rewards splitting a single blob.

Replaced with silhouette-based k selection, a PCA rank cap of N-1, and
shape-normalised descriptors. k collapses to 1 when the best silhouette is
below 0.15, i.e. when there is no cluster structure to find.

The corrected result is more interesting than the broken one:

| A / t | dominant n (count) | spread | field branches | silhouette verdict |
|---|---|---|---|---|
| 0.00 | 10(5), 9(4), 11(3) | 0.352 | 1 | no separation |
| 0.01 | 10(5), 9(3), 8(3), 11(1) | 0.350 | 1 | no separation |
| 0.03 | 8(7), 10(3), 9(2) | 0.345 | 2 | weak |
| 0.10 | 8(12) | 0.313 | 1 | pinned |
| 0.30 | 8(12) | 0.191 | 1 | pinned |

**The dominant-wavenumber label is multi-valued while the fields form a
continuum.** Both statements are supported: 3-4 distinct n at low A, and no
well-separated clusters in field space. The near-critical modes are
near-degenerate and mix, so terminal states are mixtures rather than pure modes.

This is design risk R2 realised: *"the mode label may be continuous, not
discrete... if so the multinomial sizing behind R_ref = 500 is void and the
correct statistic is a circular density."* Consequences if it holds at 80 draws:

- G2's ">= 6 branches, effective K >= 4.0" is the wrong criterion. The right
  statistic is a density over a continuous descriptor, not a mixture weight.
- R_ref = 500 was sized by rare-branch occupancy under a multinomial model and
  must be re-derived.
- It is *not* bad news for the model: a continuous conditional is what flow
  matching handles natively, and it removes the concern that a discrete label
  could be predicted by a classifier.

12 draws is far too thin to settle this. The 80-draw pilot decides.

---

## 2026-09-10 — Gate G2 on 80 draws: essentially PASSED

Corner R/t=110, L/R=1.0, sigma_hat=1e-3, Component A = 0.01 t, 80 draws,
**80/80 converged**.

| criterion | measured | threshold | |
|---|---|---|---|
| mean pairwise relative L2 (spread) | **0.370** | > 0.30 | PASS |
| effective K on the mode label | **4.25** | >= 4.0 | PASS |
| top branch weight | **0.325** | <= 0.45 | PASS |
| distinct branches | 5 | >= 6 | marginal fail |
| non-convergence (G5) | **0 / 80** | < 5 % | PASS |

Dominant circumferential wavenumber over 80 draws:

```
n= 7    4   5.0%  ####
n= 8   21  26.2%  #####################
n= 9   26  32.5%  ##########################
n=10   21  26.2%  #####################
n=11    8  10.0%  ########
```

A broad, well-centred distribution over five adjacent modes, produced entirely
by an imperfection of RMS 1e-3 * t. This is the benchmark working as intended.

### The branch-count criterion is the wrong test

Field clustering finds 2 branches with a best silhouette of **0.165**, and every
k from 2 to 10 scores between 0.086 and 0.165 — i.e. there is no discrete
cluster structure to find. The distribution is a **continuum over adjacent
wavenumbers**, not a mixture of separated branches.

So G2's ">= 6 branches" should be retired and replaced by a criterion suited to
a continuous conditional. Effective K on the physical mode label (4.25) and the
pairwise spread (0.370) are the meaningful statistics, and both pass. This
resolves design risk R2 in the "continuous label" direction, which means:

- **R_ref = 500 must be re-derived.** It was sized by rare-branch occupancy
  under a multinomial model that does not apply. The right basis is the
  standard error of a density over n, which is a weaker requirement -- the
  reference tiers are likely over-sized rather than under-sized.
- Good for the method: a continuous conditional is what flow matching handles
  natively, and it removes the objection that a discrete branch label could be
  predicted by a plain classifier.

### The load is deterministic while the field is not

F_end/F_lin over the 80 draws spans **0.560 to 0.569** -- a range of 1.6 % --
while the field spread is 0.370. The knockdown scalar is essentially
deterministic; the pattern is not.

That is a useful property and a warning. Useful, because it is a sharp argument
for field-level distributional modelling: the engineering scalar everyone
reports is predictable, and all the uncertainty lives in the shape. A warning,
because the knockdown factor therefore **cannot** serve as the engineering-QoI
distribution the design asked for in section 8; a different functional is
needed.

---

## Status

| Gate | State |
|---|---|
| Environment | done -- CalculiX 2.22 in WSL, userspace install |
| Mesh generator | done -- matches design counts |
| Imperfection law | done -- spectral coefficients persisted, exact at any node position |
| Nonlinear continuation | done -- 3-step static / dynamic / settle |
| **G1a** symmetry floor | **PASSED** -- 0.0 vs 1e-4 required |
| **G1b** cluster size | **PASSED at sigma_hat >= 1e-3**; fails at 1e-4, retiring that amplitude |
| **G2** multimodal | **provisionally passed at n=3** -- spread 0.33 > 0.30, and the mode label changes between draws. Needs the full 80-draw run for effective-K and top-weight. |
| G3 geometry-varying | not yet run -- needs all four corners |
| G4 OOD is OOD | not yet run |
| G5 cost / failure honesty | cost measured; failure-rate statistics need the 80-draw run |

### Next

1. Component A amplitude sweep -- find the level that breaks symmetry without
   dominating (0.30*t is too high).
2. Box-spanning cost on all four corners, to firm up the schedule.
3. Full 80-draw pilot per corner -> G2, G3, G5 proper.
4. G4 equivalent-cylinder check before any cone is generated.

---

# Production rebuild, 2026-09-12

The pilot is retired. Three defects made it unfit to ship, all of them found by
verification rather than by the runs failing -- every one of the 320 draws
completed cleanly.

## Fix 1 -- Component A was selecting the outcome, not just breaking symmetry

Component A exists to break the SO(2) symmetry so the buckle *location* is
predictable from geometry. It was also choosing the *mode number*, which is
supposed to be the withheld variable's job.

The harmonic set was `l = 2, 3` (ovalization) plus `j*N_p` for `j = 1..4` with
`N_p = 8`. At R/t = 110 the critical circumferential wavenumber is
`n ~ 0.86*sqrt(R/t) = 9.0`, so the weld harmonic `l = 8` sits directly on it.

Dose-response, R/t = 110, L/R = 1.0:

| Component A RMS | draws | n = 8 |
|---|---|---|
| 0 (off) | 12 | 0 / 12 |
| 0.01*t | 6 | 4 / 6 |
| 0.10*t | 12 | 12 / 12 |

`process_signature` now takes `critical_n` and drops every harmonic within
`guard = 3.0` wavenumbers of it. `apply_imperfection` passes
`critical_wavenumber(R/t)`. Ovalization survives at every geometry in the box
and is what does the symmetry breaking; the colliding weld harmonic does not.

| R/t | n_crit | kept | dropped |
|---|---|---|---|
| 110 | 9.0 | 2, 3, 16, 24, 32 | 8 |
| 140-190 | 10.5-11.9 | 2, 3, 16, 24, 32 | 8 |
| 230 | 13.0 | 2, 3, 8, 24, 32 | 16 |

Verified with 16 fresh seeds at R/t = 110, converged settle: modes
{8: 8, 9: 7, 10: 1}, three distinct values, top mode share 50 % against the
100 % the biased component produced at 0.10*t.

## Fix 2 -- the settle window was 10x too short

`write_twostep_inp` defaulted to `n_inc_settle = 20, t_settle = 4.0`. The
converged value is `200 / 40.0`, which was measured earlier but never promoted
to the default, so the pilot ran at the short one and understated the
displacement amplitude by 2.64x. Defaults changed.

## Fix 3 -- convergence was never measured

Nothing in the pilot checked whether a draw had actually settled, which is why
320 non-equilibrium states shipped looking clean. `run_production.py` now
records `drift` per draw -- the fractional change in field RMS over the last
third of the settle window -- and `pack_hdf5.py` rejects draws above
`--max-drift`.

## New tooling

| File | Role |
|---|---|
| `src/run_production.py` | resumable family-aware generator; keeps full 3-D displacement + circumferential spectrum + drift |
| `src/make_plan.py` | emits the geometry plan: train / t1 param-extrapolation / t2 cones / t3 thickness taper |
| `src/pack_hdf5.py` | packs to the suite mesh contract; withheld Component B goes under `latent/`, never into `nodal_data` |
| `src/validate_dataset.py` | D1-D5 audit: constant channels, output-from-input leakage, within-geometry spread, withheld-latent integrity, contract sanity |

The argmax mode label is kept but is no longer the only target: the full
circumferential spectrum is stored per draw, because the label is not
damping-invariant on near-degenerate draws (1 of 6 flipped between alpha = 0.3
and 0.6, on the lowest-`n_share` sample).

## Fix 4 -- the slender corner is excluded, on a measurement

R/t = 230, L/R = 1.6 (Batdorf Z = 562) does not converge at the production
target and does not become tractable by backing the target off:

| target | outcome |
|---|---|
| 1.5 * d_cr | exceeded 2 h, no result |
| 1.2 * d_cr | `ok=False` at the 5400 s timeout |

Reducing the target does not help because the difficulty is not the depth of
the post-buckled state, it is that a shell this slender goes through a cascade
of mode jumps rather than settling into one equilibrium. Dropping it.

The production box is therefore capped at R/t <= 200, L/R <= 1.3, whose worst
corner is Z = 322 against the excluded 562:

| geometry | Z | vs fast corner |
|---|---|---|
| R/t 110, L/R 1.0 (fast, 26 min) | 105 | 1.0x |
| R/t 200, L/R 1.3 (plan worst train) | 322 | 3.1x |
| R/t 140, L/R 1.5 (plan worst t1) | 300 | 2.9x |
| R/t 230, L/R 1.6 (EXCLUDED) | 562 | 5.4x |

## Component A is inert at the amplitude the pilot used

With the colliding harmonic gone, Component A at 0.01*t changes nothing at all.
Sixteen paired seeds, A on versus A off, converged settle:

| quantity | A on (0.01*t) | A off |
|---|---|---|
| dominant mode | \multicolumn -- 16/16 identical | |
| mean rms/t | 1.3241 | 1.3232 |
| buckle phase | 0.0015 of a period apart | |
| location resultant R (n=8, 8 draws) | 0.470 | 0.467 |

So the withheld field is doing all of the work. That is correct for the mode
and wrong for the location: Component A's other job (design goal F3) is to make
the buckle LOCATION predictable from geometry, and it is not doing it. R = 0.47
on 8 draws is not evidence that it is -- the Rayleigh null gives
P(R >= 0.47 | 8 uniform draws) = 0.17, so the location is consistent with
uniform. That also clears a confound worth stating: the structured mesh's
residual C_N symmetry is NOT biasing where the buckle lands.

0.01*t was tuned against the buggy Component A, in which `l = 8` sat on the
critical wavenumber and dominated at any amplitude. That tuning does not carry
over. `src/analyse_compa.py` re-picks the amplitude against BOTH jobs:

  * DEFER -- mode histogram stays as wide as the A-off control
    (>= 3 distinct modes, top share <= 60 %)
  * PIN   -- buckle angle beats the Rayleigh uniform null (p < 0.05)

and takes the largest amplitude that still defers. `src/phase_metric.py`
carries the location measurement, with the Rayleigh null built in so that weak
evidence cannot be read as pinning -- the naive version of this metric
(complex-averaging the circumferential FFT across axial stations) cancels on
the axial sign flip and reports the argmax of noise.

## Component A amplitude re-picked: 0.15*t, and the location goal is retired

Sweep at R/t = 110, L/R = 1.0, ten paired seeds per amplitude, converged settle.

| Component A RMS | mode histogram | distinct | top share | mode flips vs A-off | relL2 vs A-off | Rayleigh p |
|---|---|---|---|---|---|---|
| off | {8:8, 9:7, 10:1} | 3 | 50 % | - | - | 0.177 |
| 0.01*t | {8:8, 9:7, 10:1} | 3 | 50 % | 0 / 10 | 0.011 | 0.173 |
| 0.05*t | {8:4, 9:5, 10:1} | 3 | 50 % | 0 / 10 | 0.039 | 0.235 |
| **0.15*t** | {8:4, 9:5, 10:1} | 3 | 50 % | **0 / 10** | **0.111** | 0.234 |
| 0.40*t | {8:3, 9:6, 10:1} | 3 | 60 % | 2 / 10 | 0.398 | 0.304 |

Between-seed relL2 at A-off, for scale: **1.220**.

**Chosen: 0.15*t.** Largest amplitude that leaves mode selection completely
untouched, while contributing a real geometry-dependent field component
(relL2 0.111) instead of sitting there inert. It stays 11x below the withheld
field's own spread, so the one-to-many property is untouched. 0.40*t is
rejected: it flips 2 of 10 modes, which is the defect this rebuild removes.

**The location goal (F3) is retired, and not because the tuning failed.**
No amplitude pins the buckle angle -- every one of them sits far from the
Rayleigh threshold. That is a physical result, not a knob left unturned:
imperfection sensitivity is mode-specific, so a long-wavelength ovalization
(l = 2, 3) cannot seed a short-wavelength n ~ 9 buckle at any sane amplitude,
and anything placed in the critical band selects the mode instead of merely
locating it. Location and mode are the same knob under this imperfection law,
and the benchmark needs the mode left free.

Consequence for evaluation: **the buckle location is uniform on the circle by
construction.** This is an exact known symmetry of p(field | geometry) rather
than a nuisance -- it gives a free correctness check on any learned
distribution -- but scoring must use SO(2)-invariant descriptors.
`src/gates.py::so2_invariant()` already implements the quotient.

Future work if a pinned location is ever wanted: the single-perturbation-load
approach (a localized dent at a fixed angle, broadband in n) is the established
experimental technique and is the right shape for the job, since it excites a
band of wavenumbers rather than one. It is a different imperfection law and
would need its own validation cycle.

## Production launch, 2026-09-12

Cost was calibrated on two measured single-lane draws rather than guessed:

| geometry | Batdorf Z | measured |
|---|---|---|
| R/t 110, L/R 0.8 | 67 | 18.7 min |
| R/t 200, L/R 1.3 | 322 | 48.2 min |

giving cost ~ Z^0.60. Mesh resolution holds across the whole box -- 5,311 to
15,190 nodes, 6.0 nodes per buckling half-wave at every corner -- so the large
geometries are resolved, not merely slower.

| tier | geometries | draws |
|---|---|---|
| train | 12 | 576 |
| t1 parameter extrapolation | 4 | 128 |
| t2 cones | 4 | 128 |
| t3 thickness taper | 2 | 64 |
| **total** | **22** | **896** |

454 core-hours, about 33 wall-clock hours at 19 lanes (60 % of 32 logical
cores, the standing constraint). The runner is resumable on `draw.npz`, so an
interruption costs at most the draws in flight.

`--max-drift` default raised 0.05 -> 0.15 on measurement: observed settle
residuals run 0.02-0.10, and the high-drift draws are the ones with low
`n_share` -- genuinely near-degenerate states that keep exchanging energy
between competing modes. Rejecting them would bias the dataset toward decisive
outcomes and *understate* the spread, which is the quantity being benchmarked.
They are kept and flagged.

Note for anyone resuming: the smoke and probe draws were generated at the old
`COMP_A = 0.01` and were deleted before launch, because `run_production.one()`
reuses any existing `draw.npz` and would have silently mixed two imperfection
laws in one dataset.

## G3 on live production data (208 draws, 4 complete geometries)

Does `p(outcome | geometry)` actually depend on geometry? If the between-geometry
difference is the size of the within-geometry sampling noise, the conditional
task is fake and the whole benchmark collapses to one global distribution.

**On the discrete mode label, only 3 of 6 pairs separate.**

| | rt110 lr0.8 | rt110 lr1.0 | rt110 lr1.3 | rt140 lr0.8 |
|---|---|---|---|---|
| rt110 lr0.8 | - | 0.062 | 0.125 | **0.729** |
| rt110 lr1.0 | | - | 0.188 | **0.688** |
| rt110 lr1.3 | | | - | **0.750** |

TV within geometry (pure sampling noise, 200 random half-splits): mean 0.143,
p95 0.292. So the three R/t pairs clear it easily and the three L/R pairs do not.

That is not a defect, and reading it as one would have been a mistake: the
critical wavenumber is `n ~ 0.86*sqrt(R/t)` and **does not depend on L/R at
all**. Changing L/R at fixed R/t is *supposed* to leave the circumferential
mode distribution alone. The measured means confirm the law holds:

| geometry | mean n | 0.86*sqrt(R/t) |
|---|---|---|
| R/t 110, L/R 0.8 | 8.77 | 9.02 |
| R/t 110, L/R 1.0 | 8.88 | 9.02 |
| R/t 110, L/R 1.3 | 8.58 | 9.02 |
| R/t 140, L/R 0.8 | 9.92 | 10.18 |

**On the field, all 12 ordered pairs separate.** Energy score of geometry A's
ensemble predicting geometry B, divided by B's own self floor (1.0 = as good as
the truth, higher = separated), on the SO(2)-invariant spectrum with amplitude
normalised out:

| A \ B | rt110 lr0.8 | rt110 lr1.0 | rt110 lr1.3 | rt140 lr0.8 |
|---|---|---|---|---|
| rt110 lr0.8 | 0.97 | 1.08 | 1.08 | 2.19 |
| rt110 lr1.0 | 1.10 | 0.96 | 1.28 | 2.11 |
| rt110 lr1.3 | 1.09 | 1.25 | 0.96 | 2.39 |
| rt140 lr0.8 | 2.33 | 2.16 | 2.59 | 0.94 |

Off-diagonal mean 1.72, minimum 1.08; 8 of 12 separate by more than 10 %, and
the 4 weakest are precisely the same-R/t L/R pairs.

**The conclusion that matters for the benchmark design: the argmax mode label is
a lossy target.** The field carries strictly more conditioning information than
the label does -- the L/R pairs that are indistinguishable by mode label
(TV 0.06-0.19, inside sampling noise) separate at field level (1.08-1.28x the
floor). This is the measured justification for storing the full 3-D
displacement and full circumferential spectrum, and for scoring on the
invariant field descriptor rather than on the discrete label.

## Train tier complete: a convergence defect the easy corners hid

12 geometries x 48 draws = 576, zero solver failures. Two findings and one
defect.

**The conditional law broadens as the shell thins.** Top-mode share falls from
65-67 % at R/t = 110 to 38-56 % at R/t = 200, correlation -0.68 against R/t.
Thinner shells have a denser cluster of near-degenerate modes, so the withheld
field has more nearly equivalent branches to pick between. That is a built-in
difficulty axis, not noise.

**The classical wavenumber over-predicts, systematically.** The mean realised
mode is below 0.86*sqrt(R/t) at every one of the twelve geometries, by 1.6 % to
13.0 %, with the deficit growing with R/t (correlation +0.59 on magnitude). The
direction is expected for imperfect shells; the magnitude at the thin end means
the closed form is a reference line, not a label to reproduce. An earlier
version of the paper claimed "within 3 %" from the first four geometries -- that
was wrong and was caught by re-checking against the fuller data.

**DEFECT: the settle window does not transfer across aspect ratio.**

| L/R | median drift | draws over the 0.15 threshold |
|---|---|---|
| 0.8 | 0.021-0.034 | **0 % at every R/t** |
| 1.0 | 0.020-0.116 | 0-21 % |
| 1.3 | 0.038-0.191 | **0-81 %** |

The driver is L/R, not R/t (exceedance correlates with R/t at only 0.28). The
window was calibrated on R/t = 110, L/R = 1.0 -- a short shell -- and the long
shells do not settle inside it.

This is the same class of defect as the original pilot's transient snapshots,
returning at an aspect ratio the calibration never visited. It also corrects an
earlier conclusion recorded in DATASET_CARD s7.5: the natural-frequency
extraction found omega_1 of the UNDEFORMED shell nearly constant across the box
(spread 1.014x), and that was read as ruling out a geometry-dependent settle
requirement. It does not. The undeformed fundamental frequency is not the
post-buckled relaxation timescale, and the present data shows they are not
interchangeable.

Consequences, in order:

1. The L/R = 1.3 rows are **under-converged** at the current settle length. The
   per-draw `drift` is stored, so the threshold can be re-applied downstream.
2. **t1 includes R/t = 140, L/R = 1.5**, which will be worse still. Expect that
   geometry to be largely over threshold.
3. `T_SETTLE` must scale with L/R in any regeneration. Do NOT apply that change
   while a run is live: `run_production.py` workers are spawned, so Windows
   re-imports the module and a mid-run edit would silently split the dataset
   across two settle lengths. Finish the run, then regenerate the affected rows.
4. Drift still tracks near-degeneracy across the full tier (correlation -0.54
   with top-mode share), so the draws that fail to settle are preferentially the
   marginal ones. They stay in, flagged -- dropping them would bias the dataset
   toward decisive outcomes and understate exactly the spread being measured.

## 2026-09-13 audit corrections to earlier interpretations

The observations above are historical snapshots. The current contract is
[DATASET_CARD.md section 9](DATASET_CARD.md#9-current-production-contract-audited-2026-09-13).
The following corrections supersede incompatible interpretations above:

- Raw NPZ outputs retain high-drift draws, but the HDF5 packer defaults to
  rejecting `drift > 0.15`. These are different populations. Export rejection
  counts must accompany results, especially when rejection varies by geometry.
- `drift` measures RMS range over the last third of saved usable displacement
  blocks, not a certified time window or full-field convergence. A steady RMS
  does not establish a steady shape, force equilibrium or negligible inertia.
- The production deck originally ignored thickness variation. Corrected decks
  assign the requested thickness to every node; unversioned taper caches are
  rejected. Their solver success alone did not validate the intended physics.
- A phase test that does not reject uniformity does not prove exact SO(2)
  invariance with a fixed non-axisymmetric visible signature.
- The reference half-split energy score estimates an intrinsic expected-score
  baseline, not a hard floor. A ratio above one is an observed difference;
  without uncertainty analysis it does not establish statistical separation.
  The normalized spectral descriptor also discards amplitude, phase and axial
  arrangement, so it does not establish equality or separation of full field laws.
- The coefficient `0.86` in the mode guide was fitted empirically to pilot
  outcomes. It is not an independently derived classical prediction.
- Row 9 of new exports contains actual `t(z)` in reference-radius units,
  replacing the former thickness ratio. All 11 rows retain a single timestep.

The regression suite in `tests/test_buckling_dataset_contract.py` checks nodal
thickness cards, cache rejection without overwriting original artifacts,
packed physical thickness and static layout, corrupt inputs/edges in the 61st
sample, latent and shape mismatches, fair score normalization, and randomized
rank ties. These checks validate the data/software contract; they do not prove
physical equilibrium or model performance.
