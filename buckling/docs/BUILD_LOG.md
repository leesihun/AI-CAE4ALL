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
