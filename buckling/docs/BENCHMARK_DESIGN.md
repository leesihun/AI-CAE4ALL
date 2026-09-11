# Shell Post-Buckling Benchmark — Design and Build Plan

**Status:** design frozen except where marked *pilot-decided*. Nothing generated yet.
**Date:** 2026-09-06
**Purpose:** a static one-to-many benchmark where an infinitesimal, portable imperfection produces a
broad, multimodal outcome distribution whose shape is a learnable function of geometry — so that a
distribution learned on family AA predicts the distribution on an **out-of-distribution** family BB.

---

## 0. Why this exists

Every learned mesh simulator is trained with a pointwise loss and therefore predicts a conditional
mean. Validating anything else needs data with **many samples from `p(field | geometry)`**. That data
does not exist: see [the prior-art audit](../../../paper/introduction.tex) and the memory note
`lino-dgn4cfd-prior-art-and-clean-negative-2026-09-06`. The literature splits into
*many geometries × one solve* and *many draws × one geometry*, and never crosses — because in every
many-draws dataset the community put the random realization **on the input side**, which makes the
map deterministic.

The distinguishing move here is to **withhold** the imperfection realization. The model sees the
nominal geometry and nothing else; the imperfection is drawn from a law we specify but never expose.

Positive evidence that the gap is real, not just unsearched: the Butterfly Defect study
(arXiv:2607.19245) ran 10 geometries × 200 stochastic imperfection realizations, >20,000 nonlinear
solves, >20,000 CPU-hours — and released **scalars only**. Structured Chaos II: 300 GMNIA per design
point → 19 MB of Excel. The fields were computed and deliberately discarded.

---

## 1. The physics, in one page

A thin cylinder under axial compression has its critical load attained *simultaneously* by a family
of doubly-periodic modes `(m, n)` lying on the **Koiter circle**, plus a continuous SO(2)
circumferential degeneracy. The cluster is narrow — with a moderate Batdorf parameter, K ≈ 12–21
modes sit within 0.4 % of λ₁.

A tiny geometric imperfection breaks the degeneracy and selects which mode actually grows. Koiter's
law relates amplitude to knockdown:

```
γ = 1 + αξ − sqrt(αξ(2 + αξ)),    α = (3/4)·sqrt(3(1−ν²)) = 1.2392 at ν = 0.3
```

The number of *competing* modes **grows** with amplitude — this is counter-intuitive and it corrects
an earlier assumption in this project that a smaller imperfection means more competition:

| σ/t | knockdown | competing modes (R/t = 200) |
|---|---|---|
| 1e-4 | 1.6 % | 26 |
| 1e-3 | 4.9 % | 57 |
| **1e-2** | **14.6 %** | **94** |
| 1e-1 | 38.9 % | ~200 |

Above σ ≈ 0.5 t the largest defect locks the response and the outcome distribution collapses to a
point mass. So the usable window is roughly **1e-3 ≤ σ/t ≤ 1e-1**.

---

## 2. The imperfection law

Two components. The first carries engineering realism and the mean knockdown; the second carries the
mode selection. **They are the same spectrum read at two places** — this is what dissolves the
"realistic vs. portable" tension.

### Component A — deterministic process signature (per family, geometry-derived)

```
w_A(s,z)/t = Σ_{l∈L} â_l · sin(πz/L) · cos(l·s/R + ψ_l),     L = {2,3} ∪ {j·N_p}_{j=1..4}
```

- `k = 1` axial half-wave only (NASA SP-8007 Rev 2 §4.3: components with k > 4 are typically small in
  cylinders without circumferential joints).
- `l = 2, 3` ovalization; `l = N_p, 2N_p, …` for an `N_p`-panel weld pattern. **`N_p` is a property of
  the geometry family**, which is what makes "geometry → amplification" learnable rather than noise.
- Amplitudes decaying monotonically in `l`, **total RMS = 0.30 t** — matches measured shells
  (ECSS-E-HB-32-24A; NASA C3 shell: ξ_asy = 0.658 t).
- Phases `ψ_l` fixed per family. **This component does not vary across draws.**

Component A also does load-bearing work beyond realism: **it breaks the SO(2) symmetry.** See §3.3.

### Component B — band-limited stochastic residual (the mode selector)

| Property | Value |
|---|---|
| Field | zero-mean Gaussian random field on the mid-surface, truncated ±3σ (report ~0.27 % rejection) |
| Kernel | **Matérn**, ν = 3/2 nominal; sweep ν ∈ {1/2, 3/2, 5/2} |
| Correlation length | `ā = ℓ/sqrt(Rt) = 0.70` nominal (= 0.405 λ_half); sweep {0.35, 0.70, 1.40, 2.80} |
| Amplitude | `σ̂ = σ/t = 1e-2` nominal; sweep {1e-3, 1e-2, 3e-2, 1e-1} — *pilot-decided*, see §6 |
| Application | **mid-surface normal offset**, `x → x + (w_A + w_B)·n̂`. Not thickness. |
| Generation | FFT spectral synthesis / circulant embedding, lattice `h_field ≤ ℓ/8`, padded axially to 2L, cropped, bicubic-interpolated to FE nodes |
| Seeding | store `(family_id, geometry_id, draw_index, master_seed)`; **persist the spectral coefficients `{A_pq, φ_pq}`** |

Spectral density (normalized so `∫S d²κ = σ²`):

```
S(κ) = σ² · [Γ(ν+1)(2ν)^ν / (Γ(ν) π ℓ^{2ν})] · (2ν/ℓ² + |κ|²)^{−(ν+1)}
```

**Use Matérn, not squared-exponential.** A sq-exp fit to the measured Delft data puts a power ratio of
~1e-45 between the critical wavenumber (n ≈ 54) and the dominant measured one (n ≈ 4); a power-law
(Donnell–Wan) fit to *the same data* puts ~0.5 % there. Twenty orders of magnitude of disagreement in
exactly the band that selects the mode. A Gaussian kernel asserts smoothness the measurements cannot
support; Matérn's algebraic tail makes the exponent an explicit swept parameter.

### Why the coefficients, not the nodal values, are the artifact

Storing nodal values makes a realization un-reusable on a different mesh. Storing `{A_pq, φ_pq}` lets
the **identical realization** be re-sampled onto family BB's mesh and onto a refined mesh. This is what
makes the perturbation law portable in practice, not just on paper.

### The mesh trap (this would have silently killed the dataset)

For i.i.d. nodal values, `std(modal amplitude) = σ/sqrt(N_nodes) ∝ h`. Measured: **exactly 2.00× per
mesh halving** over 16× refinement. Halving element size halves the effective imperfection.

Worse, the natural sanity check proves nothing: pointwise variance is *preserved* under gross
undersampling (std 0.9975 → 1.0000) while the modal amplitude is wrong by **+291 %**. Undersampling
aliases high-wavenumber power *into* the buckling mode.

> **Converge on modal amplitude, never on variance.**

Mesh rule: `h ≤ min(ℓ/4, λ_half/16)` where `λ_half = 1.728·sqrt(Rt)` at ν = 0.3.

### Thickness variation — second arm only

Rejected as primary for four reasons: (a) i.i.d. implementations hit the mesh trap; (b) thickness
drives *scatter* not mean knockdown, leaving a mode fight in a ~0 %-wide window where solver settings
pick the winner (documented: arc-length step ≥0.05 → pole mode, ≤0.02 → boundary mode, same geometry
same defect); (c) it couples to the mode only through the 2n harmonic — splitting is machine-zero for
every m ≠ 2n — making it ~34× weaker at the geometrically optimal correlation length; (d) a **uniform**
marginal truncates exactly the extreme tail that decides a weakest-link phenomenon (pointwise minimum
over 800 realizations: −3.71σ Gaussian vs −1.73σ uniform, a hard floor at −√3 hit essentially every
realization).

Kept as a declared ablation arm: same spectral spec applied to `t(s,z) = t₀[1 + ξ]`, **lognormal**
marginal, `ā_thick = ā_geom/2 = 0.35`. Report the `δt/t ↔ w₀/t` calibration curve.

---

## 3. Geometry families and the OOD split

**This is the most important section.** The evaluation set must be out-of-distribution *geometry*, and
there are three distinct ways a claimed-OOD split can be fake. All three have to be defeated at once,
and they pull against each other.

### 3.1 The three failure modes

**F1 — Coordinate-change equivalence.** A cone of semi-vertex angle α behaves like a cylinder of
radius `R_e = R/cos α`. If BB maps *inside* AA's box under that transformation, BB is AA in disguise
and a zero-shot success proves nothing. This is the same trap that turned a DeepJEB "extrapolation"
split into an interpolation split (memory: `deepjeb-lineage-leakage-and-deepjebpp`).
→ **Check the equivalent-cylinder map before generating a single BB solve.**

**F2 — Regime confound.** Shell knockdown collapses on the Batdorf parameter
`Z = (L²/Rt)·sqrt(1−ν²)`, not on R/t alone. If AA and BB occupy *disjoint* Z ranges, the experiment
tests Z-extrapolation rather than shape-family transfer, and a failure is uninterpretable.
→ **Require Z-range overlap by construction, and report both families' Z distributions.**

**F3 — Degenerate win by symmetry.** On a perfectly axisymmetric cylinder the buckle *location* is
uniform on SO(2) by symmetry. A model that predicts "uniform over the symmetry orbit" then scores
perfectly while having learned nothing about geometry — the same degenerate-win class as the ex9
`cond_var` bug (memory: `ex9-cond-var-degeneracy-2026-08-19`).
→ **The geometry must break SO(2) itself.** Component A does this: its `l = 2,3` ovalization and
`N_p`-panel weld signature are deterministic, per-family and geometry-derived, so the location
distribution becomes non-uniform *and predictable from the geometry*. Report its non-uniformity as a
validation metric, not an afterthought.

F1 pushes BB to be structurally different; F2 pushes it to stay in the same regime; F3 constrains AA
itself. The resolution is: **vary the shape family while holding Z overlapping, and let Component A
break the symmetry in both families.**

### 3.2 Family definitions

| | Family AA (train) | Family BB (transfer) |
|---|---|---|
| Shape | circular cylinder | truncated cone |
| R/t | [110, 230] log-uniform Sobol | same box |
| L/R | [1.0, 1.6] log-uniform Sobol | same box |
| Extra axis | — | semi-vertex angle α ∈ [6°, 12°] |
| Z | [105, 562] | **overlapping by construction** |
| ν | 0.3 | 0.3 |
| BCs | clamped both ends | clamped both ends |

### 3.3 The test tiers — the inference set is OOD by design

Four tiers of increasing distance from training. **The headline result is T2 and T3; T0 exists only as
a control.**

| Tier | What | Distance from AA | What a good score proves |
|---|---|---|---|
| **T0** | held-out AA cylinders **inside** the box | in-distribution | the model works at all; the calibration control and the finite-M debias anchor |
| **T1** | AA cylinders **outside** the (R/t, L/R) box, Z still overlapping | parameter extrapolation | the conditioning map extrapolates along its own axes |
| **T2** | **cones**, Z-matched, α large enough to fail the equivalent-cylinder test | shape-family transfer | **the headline claim** — a different shape family, same imperfection law |
| **T3** | **axial thickness variation** `t(x) = t₀(1 + γ·sin(πx/L))`, γ ∈ [0.05, 0.20] | structural OOD | strongest claim: *no* equivalent-cylinder map exists for this family at all |

T3 is the insurance policy. If the pilot's G4 gate shows the cones map back into AA under
`R_e = R/cos α`, T2 is compromised and T3 becomes the headline instead — which is why it must be
budgeted from the start, not added later.

**Split rule: by geometric parameter, never by RNG seed.** A seed split is an in-distribution split
wearing OOD clothes.

**Acceptance for "genuinely OOD"** (checked before generating BB):
- nearest AA training point ≥ **2.5 nearest-neighbour spacings** away in the log box, on at least one
  axis, after applying the equivalent-cylinder map;
- branch-support Jaccard against AA ∈ **[0.40, 0.80]** — below 0.40 the model is asked to generate
  patterns it has never seen (a different and much harder claim); above 0.80 BB is not OOD.

### 3.4 What we can say that the competitor cannot

Lino et al.'s WING test set has **no OOD component at all**: all 16 test wings sit strictly inside the
training ranges for thickness, taper, sweep and twist. Their entire generalization claim rests on a
±10–20 % scalar shift on a 2-D one-parameter ellipse. A tiered OOD protocol with a shape-family
transfer is a strictly stronger evaluation, and it is cheap to state.

---

## 4. Allocation

**Cost basis:** 8.1 core-min/solve at box median (9.0 for box-spanning tiers); **24 effective lanes**
(32 logical cores × 75 % scheduling efficiency); 15 % attempted-solve loss to non-convergence.

| Row | Split | Geoms | Draws/geom | Usable solves | Wall-clock |
|---|---|---:|---:|---:|---:|
| 0 | **Pilot** (4 AA corners + 1 BB probe) | 4 (+1) | 80 | 320 | 2.4 h |
| 1 | Train A — breadth | 1,680 | **2** | 3,360 | 22.2 h |
| 2 | Train B — spread | 420 | **8** | 3,360 | 22.2 h |
| 3 | Val (early stop / HP) | 100 | 2 | 200 | 1.3 h |
| 4 | **T0** in-dist reference | 6 | **500** | 3,000 | 22.1 h |
| 5 | **T1** parameter extrapolation | 4 | 250 | 1,000 | 7.4 h |
| 6 | **T2** cones — headline | 8 | **500** | 4,000 | 29.4 h |
| 7 | **T3** thickness-varying — structural OOD | 4 | 250 | 1,000 | 7.4 h |
| 8 | Deep anchor (1 from T0 + 1 from T2, 500 → 1,300) | 2 | +800 | 1,600 | 11.8 h |
| | **Total** | **≈2,230** | — | **17,520** usable (≈20,600 attempted) | **≈122 h ≈ 5.1 d** |

Storage ≈ 0.30 MB/draw + 2 MB/mesh ⇒ **~10 GB** against 1.7 TB free. **Keep every terminal field and
every RNG seed.** Storage is not a constraint here and discarding fields is exactly the mistake the
existing literature made.

### Where the numbers come from

**Draws per geometry in training: 2 and 8.**
`R = 2` is a *hard floor*, not a preference: at one draw per condition the MMD-InfoVAE objective has a
**global optimum with exactly zero conditional spread**. HI-MGN-V cannot be trained at D = 1.
`R = 8` in tier B is the branch-exhibition floor — `R ≥ ln(10)/(−ln(1−p))` gives 90 % probability that
a p = 0.25 branch appears at least once.
Tiering is free for cHI-MGNflow because **no term in the flow-matching loss couples two draws at the
same condition**; the VAE branch gets 420 conditions resolved to `1/sqrt(2·7) = 26.7 %` instead of
3,000 conditions resolved to 100 %.

**Reference draws: 500.**
Not set by CRPS — the fair estimator is unbiased at every ensemble size and, pooled over ~50
independent spatial patches × 8 geometries, reaches 0.25 % relative SE at R = 400 and **1.1 % even at
R = 20**, roughly 1,600× less demanding. Per-node spread is satisfied at 200
(`relSE(σ̂) = 1/sqrt(2(M−1))` = 5.0 %). The binding criterion is **rare-branch occupancy**:
P(rarest branch of weight p observed ≥ 30 times) ≥ 0.95.

**Training geometries: 2,100.**
Measured, not argued. Lino et al.'s Fig. 16 re-digitised pixel-by-pixel and refit `W2 = a + b·N^−p`
gives asymptote 0.2378 and crossings at N = 1,555 (20 %), 2,336 (10 %), 3,511 (5 %) — the commonly
quoted "saturates by ~2,000" is ~25 % optimistic. Critically, the *latent* variant saturates while the
**mesh-space DGN — the structural analogue of cHI-MGNflow — had not saturated at N = 5,000**, still
falling 26.5 % from N = 2,000 to 5,000.

**Deep anchor: 1,300 draws on two geometries.**
Its job is to **measure the finite-M bias curve** by subsampling at 50/100/200/400/800/1,300, so every
R = 500 number can be debiased. Bootstrapping the reference tier cannot substitute — it underestimates
the very bias it is meant to measure.

### Free experiment — design it in now

Order the 2,100 AA geometries by a **scrambled Sobol** sequence so every prefix
(400 / 700 / 1,100 / 1,600 / 2,100) is independently space-filling. The geometry-count ablation — our
own Fig. 16, on the *transfer* metric — then costs zero extra solves, only retraining. Do the same for
tier-B draw order so prefixes 2/4/8 are usable. **Retrofitting a space-filling prefix onto an
unordered set is impossible.**

---

## 5. Solver settings that are load-bearing, not defaults

Two plausible defaults silently produce a **deterministic dataset** — every draw returns the same
mode, no error is reported, and the one-to-many claim evaporates only when someone plots the
histogram.

| Setting | Value | Why |
|---|---|---|
| Mesh | **mapped/structured, quadratic (S8R)** | a free mesh with *linear* elements leaks 2.1e-4·t of spurious symmetry breaking — **110× above the physical signal**. Fatal and silent. |
| Coordinates | **analytic, float64, written at ≥15 significant digits** | a 6-digit `.inp` file injects ~3e-4·t of coordinate noise. Never round-trip through a low-precision text mesh. |
| Tolerances | `*CONTROLS, PARAMETERS=FIELD` with Rn=1e-6, Cn=1e-8, Rp=1e-6, Rl=1e-12 | CalculiX defaults (Rn=0.005, Cn=0.01) sit **~4 orders above the signal**. Near a bifurcation the softened tangent maps a 0.005 residual tolerance to ~50 % modal displacement error. |
| Continuation | **displacement control on end shortening** | **CalculiX 2.22 has no Riks/arc-length method** — `*STATIC` has five parameters and none is RIKS. Displacement control is not a fallback: with the imperfection present the bifurcation unfolds into a smooth limit-point path, `dP/du = 0` is a regular point of the displacement-controlled system, and it matches the QoI definition exactly. |
| Mesh density | 4 elements per `λ_half = 1.728·sqrt(Rt)` ⇒ `n_circ = 14.54·sqrt(R/t)`, `n_axial = 2.314·(L/R)·sqrt(R/t)` | FE dispersion error 5.1e-4, i.e. 8× below the 0.4 % physical branch window |
| Structured-mesh caution | pick circumferential division count `N` **away from** critical `n` (9–20 here) and its harmonics | a structured mesh retains a discrete C_N symmetry and is *not* neutral among modes |
| Imperfection | analytic Fourier synthesis, **not** CalculiX `*ROBUST DESIGN` | the KL basis on a cylinder *is* the Fourier basis; FFT is O(N log N) vs a dense O(N³) eigensolve, and it yields the per-mode projection `ξ̄_k` for free |

CalculiX expands S8R to C3D20R at 21 DOF/quad. Median mesh ≈ 20.3k shell nodes (≈142k DOF).

---

## 6. The pilot — runs first, gates everything

**320 nonlinear solves + ~20 linear eigenvalue runs ≈ 2.4 h.** No production solve runs until all
gates pass.

| Item | Spec |
|---|---|
| Geometries | 4 AA box corners: (R/t, L/R) = (110,1.0), (110,1.6), (230,1.0), (230,1.6); plus 1 BB cone at (170, 1.3, α=9°) |
| Draws | 80 per AA corner; BB probe reuses one AA seed set at 40 draws |
| Amplitude | **sweep σ̂ ∈ {1e-2, 1e-3, 1e-4, 1e-5}** on one fixed realization — this settles the open amplitude question (§9) |
| Free auxiliaries | perfect-shell `*BUCKLE` first 60 eigenvalues on each mesh; one mesh-refinement pair (4 → 6 elem/half-wave); one **negative control** — same geometry with a 0.15R cutout, 20 draws |

### Gates

**G1 — signal above noise.** On the perfect shell, nominally degenerate cos/sin eigenvalue pairs must
split by **< 1e-4 relative** (target < 1e-5), and **≥ 8 eigenvalues within 0.4 %** of λ₁. This single
number is the *total* symmetry-breaking floor of mesh + assembly + solver, measured before any
production solve. *Fail ⇒ the mesh is free/linear, or tolerances are at defaults, or Z is too low.*

**G2 — multimodal.** Quotient the SO(2) phase, PCA to d ≈ 32, cluster the 80 fields per corner.
Require **≥ 6 branches**, **effective K = exp(entropy) ≥ 4.0**, **top-branch weight ≤ 0.45**. The
cutout negative control must show **K ≤ 1.5** — if it does not, the clustering is finding noise.
Also verify the SO(2) phase histogram is uniform *within* each branch; if it is not, the ends are
pinning the phase and the degeneracy structure differs from what this design assumes.

**G3 — geometry-varying.** Pairwise total-variation distance between mode-weight vectors across the 4
corners: **TV ≥ 0.35 on ≥ 5 of 6 pairs**, **≥ 0.15 on all 6**. *Fail ⇒ geometry does not condition the
outcome and the benchmark is vacuous.*

**G4 — OOD is genuinely OOD.** Equivalent-cylinder check per §3.3. *Fail ⇒ raise α, or promote T3 to
the headline.*

**G5 — cost and failure honesty.** Non-convergence rate **< 5 %**, and a χ² test of failure rate
against realised branch with **p > 0.05**. Violent snap-through is exactly what fails to converge and
it correlates with branch, so discarding failures preferentially deletes the most unstable branch. A
"20 % contingency" that only replaces the count does **not** fix this — it re-draws from the same
biased conditional. Also require ≥ 95 % of draws to exhibit a genuine limit point (`dP/du < 0`
somewhere on the path) with terminal shortening past it.

---

## 7. Build order

1. **Install CalculiX**, confirm `ccx` runs and `*BUCKLE` works. Windows: bConverged/PrePoMax
   distribution, or WSL. *Unverified — see §9.*
2. **Analytic mesh generator** (float64, mapped, S8R, `%.15e` output). No gmsh — gmsh writes text
   meshes at limited precision, which is a §5 failure.
3. **Imperfection synthesizer** — FFT spectral, circulant embedding, Matérn ν; emits both the nodal
   offset field and the persisted `{A_pq, φ_pq}`.
4. **G1 gate** — perfect-shell `*BUCKLE` on all 4 corner meshes. Cheapest possible kill signal.
5. **Pilot** (320 solves, 2.4 h) → **G2–G5**.
6. **Freeze** amplitude, box, mesh law, tolerances. Record the frozen config in this doc.
7. **Production** rows 1–8 in the order: T0 → T2 → train A → train B → T1 → T3 → val → deep anchor,
   so the scoring sets exist early and a budget overrun degrades training breadth rather than
   evaluation quality.
8. **Convert to the repo mesh HDF5 contract**: `data/{id}/nodal_data` = `[num_features, 1, num_nodes]`,
   rows 0:3 reference coordinates, then the terminal displacement components as state rows; the
   imperfection is **not** written (that is the point). `data/{id}/mesh_edge` from the shell
   connectivity. `cond_var` carries only nominal geometry scalars if we choose to expose them (§9).

---

## 8. Scoring protocol

- **Model-side ensemble** `M_train = 4`; use the **almost-fair** CRPS/energy score at α = 0.95, not
  the fair α = 1, which has a documented degeneracy leaving one member unconstrained.
- **`M_score = R_ref = 500`, matched, frozen, identical for every model compared.** The two-sample W2
  null floor is hard-capped by the reference count (`1/M_eff = 1/M + 1/R_ref`), so M = 512 against
  R_ref = 500 buys nothing.
- Report **ΔW2 against a matched-M null** drawn by splitting the reference set in half. **Never an
  absolute W2.**
- **Lead with the mode-index histogram TV and the knockdown 1-D W2** (both converge as n^−1/2). Treat
  phase-aligned `W2_graph` as secondary (n^−1/4).
- **Quotient out SO(2) before clustering**, and **PCA to d_eff ≈ 32–64** — clustering in the raw
  ~20k-dim space is on the wrong side of the Kwon–Caramanis bound.
- **Mandatory baselines:** (i) predict the time/ensemble mean field; (ii) predict the *marginal* mode
  histogram ignoring geometry. Baseline (ii) is the honest floor the model must beat, and nobody
  publishes it because nobody publishes mode distributions at all.
- Score every model on the **same** reference draws and compare **paired** — the finite-sample metric
  floor and the per-geometry difficulty both cancel. This matters: at the measured per-geometry CV of
  0.45, the 95 % CI on the *mean* W2 over 8 geometries is ±31 %.

---

## 9. Open items and unverified facts

**Must be settled before or during the pilot:**

1. **Amplitude, 1e-2 vs 1e-4.** Two analyses disagree. One computes that at 1e-4·t the per-mode
   knockdown is 0.39 % against a 0.4 %-wide cluster — viable, with the noise floor 9 orders below
   *provided* coordinates are analytic float64. The other says 1e-4 is dead because a 6-digit mesh
   file injects 3e-4·t. **Both are right about different failure modes** and agree on the fix. The
   independent argument for 1e-2 (more competing modes, and it is where the literature lives) is
   what tips it. **The pilot amplitude sweep decides.**
2. **CalculiX on Windows** — no build has been verified on this machine. Fallback: WSL.
3. **Per-solve cost** — 8.1 core-min is a flop-model estimate, not measured. If the pilot median
   exceeds ~13 core-min, re-plan before launching (see §10).
4. **Is the mode label discrete or continuous?** ~95 % of near-critical `(m,n)` pairs carry a
   continuous SO(2) phase. If the label is continuous, the multinomial sizing behind R_ref = 500 is
   void and the correct statistic is a circular density with R_ref re-derived from the SE of a von
   Mises concentration parameter. **G2 settles this.**
5. **Should nominal geometry scalars (R/t, L/R, α) be exposed as `cond_var`?** Exposing them makes
   transfer easier but weakens the claim that the model reads geometry from the mesh. Recommendation:
   **do not expose**; run an ablation that does, and report both.

**Cited but not independently verified — do not put in a paper without institutional access:**
Imbert's fitted `X, r, s` (Caltech 1971, server unreachable); Schenk & Schüller correlation lengths
(paywalled); Koiter/Elishakoff thickness-variation figures (paywalled); Castro et al. 2014, the best
cross-form imperfection comparison in existence (paywalled).

**One open contradiction worth ~1 h:** `cmudrc/SFEM` (MIT, real per-node fields) has a HuggingFace
card claiming *50 realizations per geometry*, but its file counts (39,894 STEP / 48,803 H5) imply
~1.2. The file counts probably win — and it is linear elastic with a point load, so no bifurcation —
but "we downloaded it and counted geometry IDs" is a far stronger rebuttal than an inference.

---

## 10. Sensitivity — if per-solve cost is 3× the estimate

Cut in this order, and only in this order:

1. **Train A geometries** 1,680 → 900 (costs ~2 % of asymptotic distributional error).
2. **T1 and T3** 250 → 150 draws each (widens their CIs; document it).
3. **Deep anchor** 1,300 → 800 draws (still measures the bias curve, less far).
4. **Never cut** T0 or T2 draws below 500, and never cut the pilot. The reference tiers are the
   experiment; the training tiers are an input to it.

---

## 11. The three risks that would invalidate the design

**R1 — the imperfection does not break the degeneracy above numerical noise.** Two plausible defaults
are fatal and *silent* (free+linear mesh; CalculiX default tolerances). Caught by **G1**, which costs
20 linear eigenvalue runs.

**R2 — the distribution collapses, or the label space is wrong.** Either effective K < 3 (mode
selection is deterministic in practice — clamped-end boundary-layer splitting is O(1 %), *comparable
to the 0.39 % imperfection signal*), or the mode label is continuous rather than discrete. Caught by
**G2**. A naive field-level W2 between two phase-random ensembles measures phase mismatch, not
physics, and sits near saturation for every model including a good one — a metric bug that reads as a
modelling result.

**R3 — branch-correlated non-convergence biases every mode weight.** At R = 500 with p = 0.10, a 5 %
branch-correlated failure rate shifts p̂ by more than the 13.4 % relative SE the tier was sized to
deliver. Caught by **G5**'s χ² test. Log every failure with geometry, seed and last converged
increment.

**Honourable mention (cheap to detect, hence not top-three):** family BB may not be OOD at all — §3.3
F1. Check the equivalent-cylinder map before generating a single BB solve.
