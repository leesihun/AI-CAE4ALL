# Shell post-buckling one-to-many benchmark — OpenRadioss build plan

Status: **design draft, not yet implemented.** This restarts the benchmark
described in the (now-deleted) `buckling/` project on a different solver. It
is written to be read before any code is written, and it is not a finished
decision — see §8 for what still needs to be chosen with the user before or
during the pilot.

## 0. What this is, one paragraph

A thin shell under axial compression buckles into one of several
near-degenerate modes. Which one it picks is decided by an imperfection too
small to matter for the load-carrying behavior itself. We specify the
imperfection **law** and withhold the realization, so a surrogate (MGN-V) is
given only the nominal geometry and must learn a genuine distribution over
terminal fields — not reconstruct a deterministic map, which is what happens
in every public dataset that puts the random draw on the input side instead
(confirmed this session: neither the Butterfly Defect paper, nor
ConcreteShellFEA, nor the Bristol composite-cylinder set has this structure —
see chat history; also `[[lino-dgn4cfd-prior-art-and-clean-negative-2026-09-06]]`).

## 1. Prior art: what this inherits from `buckling/`

`buckling/` existed on this branch through commit `f58b169` (2026-09-13) and
was deleted in `07043c4` (2026-09-16) with no commit message. It is recovered
in full via `git show f58b169:buckling/...`. It is **not a false start** — it
already solved most of the hard design problems this benchmark has:

| Piece | Verdict | Carries forward as-is? |
|---|---|---|
| Analytic mapped-mesh geometry generator (cylinder / cone / axial thickness taper), float64 | Sound | Yes — port the geometry math, rewrite only the mesh *emission* for whatever element OpenRadioss needs (§4) |
| Two-component imperfection law (Component A deterministic pattern + Component B withheld Matérn field, spectral-coefficient storage) | Sound, audited | Yes, verbatim (§5) |
| Storage contract: coordinates always input, Component A input-only, displacement is the only output row block, spectral coefficients live outside the input rows | Sound | Yes (§6) |
| Argmax mode label is a lossy target; ship the full field/spectrum, not just a discrete label | Sound, empirically demonstrated (label failed to separate 2 geometry pairs the field separated; label flips 1/6 under a damping change) | Yes |
| OOD tiers (train / parameter-extrapolation / cone / thickness-taper) | Sound design, not yet re-validated on new solver | Yes, as a target shape — re-derive exact tier boundaries after the pilot |
| Gates G1–G5 (signal-above-noise, multimodality, geometry-sensitivity, OOD-is-genuinely-OOD, cost/failure honesty) | Sound framework | Yes, same five questions |
| **Solver: CalculiX 2.22, displacement-controlled quasi-static, fixed-duration "settle window," RMS-tail `drift` as a convergence proxy** | **This is what broke.** See §2. | **No — this is exactly what OpenRadioss replaces.** |

The full recovered documents (`BENCHMARK_DESIGN.md`, `BUILD_LOG.md`,
`DATASET_CARD.md`, `README.md`, `geometry.py`, `imperfection.py`, `ccx.py`)
are the working reference for this rebuild; they are not re-copied here in
full to keep this document buildable at a glance.

## 2. Why the solver is being replaced, specifically

CalculiX 2.22 has no Riks / arc-length continuation (confirmed independently
twice: by the old project's own build log, and by a live GitHub issue dated
2026-09-06 tracking it as an unimplemented enhancement). The old project's
workaround was **displacement-controlled quasi-static loading**, reading out
a "terminal" state after a **fixed-duration settle window** (a fixed
increment count, chosen once against one reference geometry) and accepting
the draw if a residual metric (`drift`, defined over the RMS of the last
third of the saved displacement history) fell under a threshold.

Three real defects fell out of this, all downstream of the same root cause
— **there was no physically grounded stopping criterion, only a fixed clock
and a post-hoc residual check**:

1. **Drift is not a QA metric — it is a predictor.** Within every one of 20
   geometries, `drift` is Spearman-correlated with the buckling mode label
   itself (17/20 negative, mean |ρ| = 0.31, pooled |z| = 8.2 —
   `[[buckling-drift-label-coupling]]`). A metric meant to flag "not yet
   settled" was actually partly measuring *which* near-degenerate branch a
   draw was in transit toward. Filtering on it doesn't remove noise, it
   biases the mode distribution (TV shift ≈0.176, mean mode number shift
   +0.31 at the threshold that looked safest).
2. **The settle window did not transfer across geometry.** Calibrated on
   `L/R = 1.0`, it left `L/R = 1.3` draws 0–81% over threshold — the
   deficiency tracked aspect ratio, not slenderness (`R/t`). An earlier
   argument that this couldn't happen (based on the *undefined* shell's
   modal frequency being roughly constant across the box) was wrong; it
   ignored that the imperfect, loaded shell's actual transient timescale
   is not the undeformed natural frequency.
3. **The mode label is not even damping-invariant.** 1 of 6 near-degenerate
   draws flipped its argmax mode when the numerical damping coefficient
   changed from 0.3 to 0.6. A displacement-controlled scheme with no real
   dynamics has no principled damping to begin with, so this ambiguity had
   no natural resolution.

**OpenRadioss changes the physics of the read-out, not just the label.**
`/DYREL` or `/KEREL` dynamic relaxation is real transient dynamics with an
actual, monitorable energy balance. "Has this draw reached quasi-static
equilibrium" becomes a real, checkable question — kinetic energy relative to
internal energy, tracked continuously — instead of a fixed wall-clock window
validated on one corner of the geometry box. This does not automatically fix
the underlying issue (see §7 — the KE/IE check is this project's next
analogous risk, and must be gated *before* a production run, not discovered
after one). It replaces a metadata proxy with a physical one.

## 3. Architecture: two solvers, one for each question

OpenRadioss is an explicit/dynamic-relaxation code. It has no linear
eigenvalue buckling extraction. The old G1 gate (perfect-shell eigenvalue
census — degenerate-pair detection, mode count near the critical load) is a
**linear** question and does not need arc-length or dynamics at all; the old
project already ran this cheaply and correctly in CalculiX. Reusing that is
strictly better than inventing a substitute:

```
G1  perfect-shell linear eigenvalue buckling    → CalculiX *BUCKLE (unchanged, keep this)
G2–G5  imperfect nonlinear quasi-static campaign → OpenRadioss /DYREL or /KEREL (new)
```

This is a deliberate hybrid, not indecision: G1 answers "how many
near-degenerate modes does this nominal geometry have and how close are
they," which is exactly what a linear solve is for. G2 onward answers "given
a withheld imperfection draw, which one does it actually pick and what does
the terminal field look like," which is exactly what CalculiX could not do
without Riks and what OpenRadioss's dynamics is being brought in for. Nothing
about the imperfection law, mesh, or storage contract needs to differ between
the two solves — G1 runs on the *perfect* nominal mesh with no imperfection
applied at all, same as before.

## 4. Geometry and mesh — what ports, what needs a fresh audit

The analytic geometry math (`geometry.py`: mapped cylinder / cone / axial
thickness-taper surfaces in float64) is solver-independent and ports
directly. What does **not** port unexamined is the old mesh-precision
analysis, because it was specific to CalculiX's quadratic S8R shell:

- The old rule "mapped quadratic mesh only — a free or linear mesh leaks
  ~110× the physical imperfection signal" was measured for **8-node
  quadratic** shells. OpenRadioss's standard shells are 4-node
  (Belytschko-Tsay, QEPH, or similar) — there is no drop-in quadratic shell
  in the explicit formulation. **This precision audit has to be redone from
  scratch for whichever 4-node formulation is chosen** — a 4-node mesh at a
  given element count is not the same discretization error as an 8-node
  mesh at the same node count, and the old numbers cannot be reused.
- The old rule "coordinates at `%.13e`, never `%.15e`" was specific to
  CalculiX's 20-character free-format field. OpenRadioss's fixed-format
  Radioss block deck has its own field widths (10-character fields in the
  classic format, wider in the free-format/HM Reader variant) — this must be
  re-derived against whatever format the deck writer targets, not assumed.
- The old rule "band-limit the stochastic field to a physical correlation
  length, persist spectral coefficients, never i.i.d. nodal values" is
  solver-independent and carries forward unchanged (§5).

## 5. Imperfection law — ported verbatim

No change from the audited v2 law in the old `DATASET_CARD.md` §9.2:

- **Component A** (known, deterministic, given to the model as input):
  ovalization plus a weld-like pattern at harmonics `l ∈ {2,3} ∪ {8,16,24,32}`,
  with any harmonic within 3 wavenumbers of the critical `n ≈ 0.86·√(R/t)`
  dropped from the set. RMS = 0.15·t. Identical across every draw of one
  geometry.
- **Component B** (withheld — the entire source of outcome spread): a
  random-phase field with Matérn-shaped spectral density, correlation length
  `0.70·√(R·t)`, `ν = 1.5`, rectangular wavenumber support. Persisted as
  spectral coefficients (amplitude + phase), not nodal values, so it can be
  evaluated in closed form at any mesh's node coordinates — this is what
  makes it portable across the CalculiX→OpenRadioss solver change with no
  re-derivation.

Two things earn a fresh look, not because the law is suspect but because the
solver changed:

- The A/B RMS ratio (≈150, chosen empirically against CalculiX's numerics)
  should be re-swept once real OpenRadioss draws exist — there is no reason
  to assume the same ratio keeps Component A "seeding the winning mode
  without freezing it" (the property it was tuned for) under different
  solver noise characteristics.
- The 0.86 critical-wavenumber coefficient used to build Component A's
  exclusion band was flagged in the old build log as empirically fit, not
  independently derived. Cheap to re-verify from the new G1 eigen-census
  before it is relied on again.

## 6. Storage contract — ported verbatim

Same shared-mesh HDF5 contract as the rest of this repo
(`docs/reference/DATASET_FORMAT.md`), same row layout as the old project's
audited v2:

| rows | content | role |
|---|---|---|
| 0:3 | reference (perfect-shell) coordinates | input, never a prediction target |
| 3:6 | solver displacement `(ux,uy,uz)` from the imperfect starting surface | **output** |
| 6:9 | Component A offset vector | input only |
| 9 | actual nodal thickness `t(z)` | input only |
| 10 | node type (interior / clamped base / loaded top) | input only |

`latent/{sample_id}/...` holds the withheld Component B spectral
coefficients — diagnostics for auditing the generator, never a model
feature. The full circumferential displacement spectrum ships alongside the
argmax mode label for the same reason as before: the label is lossy and not
even self-consistent under a damping change, so it cannot be the only
target.

## 7. The new convergence contract — this is the load-bearing decision

This is the one place a straight port is not safe, because it is precisely
where the old pipeline failed. The replacement principle:

> **Never accept a draw on a fixed clock. Accept it on a physical
> criterion, computed per-draw, and ship the criterion's full trace as
> metadata — never filter production data on it by default.**

Concretely for `/DYREL`/`/KEREL`:

1. Run dynamic relaxation to a **physical stopping test** — kinetic energy
   relative to internal energy (or to the external work done) below a
   threshold, checked at every write interval, not a fixed increment count.
   OpenRadioss reports both energies natively; this replaces the RMS-tail
   `drift` proxy with the actual quantity the old metric was trying to
   approximate.
2. Store the full KE/IE trace (not just the final ratio) per draw, exactly
   as `drift` was eventually stored as an HDF5 attribute rather than used as
   a silent filter. `pack_hdf5.py`-equivalent should default to no cut, same
   as the old fix, and warn loudly if a finite cut is requested.
3. **Before any production run**, repeat the exact audit that caught the
   old coupling: for every pilot geometry, Spearman-correlate the
   convergence metric against the argmax mode label, within-geometry. If
   the correlation reappears, it means near-degenerate branches settle at
   different rates under dynamic relaxation too — plausible, since it is a
   property of the energy landscape near a near-degenerate bifurcation, not
   an artifact specific to displacement control. If it does, the fix is the
   same as before (report honestly, describe the target as a finite-time
   response under a stated protocol, never as an equilibrium distribution)
   — but this time it should be a pilot-stage gate, not a post-production
   discovery.
4. Re-run the aspect-ratio settle-transfer check (old §7.2–7.3 of
   `DATASET_CARD.md`) on the new criterion before trusting it across the
   full geometry box: pick the box corner most different in `L/R` from
   whatever corner the relaxation parameters (mass scaling / damping) get
   tuned against, and confirm the KE/IE threshold is reached at all, and at
   a comparable normalized time, there too.

## 8. Open decisions — not mine to make unilaterally

| # | Decision | What's known | Recommendation |
|---|---|---|---|
| 1 | Shell element formulation in OpenRadioss (Belytschko-Tsay vs QEPH vs other) | QEPH has better in-plane/warping behavior at modest extra cost; Belytschko-Tsay is cheaper and more common in quasi-static drop/crash decks | Pilot with QEPH first (buckling is sensitive to exactly the membrane/bending coupling this trades off), fall back to Belytschko-Tsay only if it's too slow for the campaign size |
| 2 | Mass scaling / damping tuning strategy for `/DYREL` | Not yet swept on this geometry family | Tune on the *most flexible* box corner (largest `L/R`, largest `R/t`), not the reference corner as the old settle window was — that inverted ordering is exactly what caused the old transfer failure |
| 3 | Keep the exact Component A/B RMS ratio (≈150) or re-sweep | Untested on OpenRadioss noise floor | Re-sweep in the pilot; don't assume |
| 4 | Deck format: classic fixed-format Radioss block vs free-format | Affects the coordinate-precision rule in §4 | Needs one concrete answer before `geometry.py`'s mesh emitter is written — pick the format that supports the coordinate precision this benchmark needs, not the default |
| 5 | Whether to keep the OOD tier boundaries (train/t1/t2/t3) identical to the old plan | Old plan was sound but never solver-validated | Keep as a target shape; confirm tier sample counts survive a real G3 (geometry-sensitivity) re-run before treating them as final |

## 9. Build order

1. **G1, unchanged**: confirm the recovered CalculiX `*BUCKLE` eigen-census
   still runs on this machine/WSL setup; re-derive the 0.86 coefficient
   check against 2-3 box corners as a sanity check, not a fresh derivation.
2. **Mesh precision audit for the chosen OpenRadioss shell (decision #1,
   #4)**: repeat the old "mapped vs free mesh, quadratic vs linear" leakage
   measurement from scratch on a bare perfect cylinder, no imperfection, no
   dynamics — just confirm the discretization noise floor sits below
   Component B's amplitude before building anything else on top of it.
3. **Port `imperfection.py` verbatim** (Component A/B, spectral storage) —
   no solver dependency, should need no changes.
4. **New `radioss.py`**: starter + engine deck writer, `/DYREL` or `/KEREL`
   block, one nominal geometry, no imperfection, confirm a bare axial
   compression run reaches a sane pre-buckling stress state before adding
   any imperfection.
5. **Smoke pilot**: one geometry, R=5-10 draws, imperfection on, confirm
   (a) draws terminate under the KE/IE criterion, not a timeout, (b) the
   terminal fields are not bitwise-identical across draws (checks that
   Component B is actually reaching the solve), (c) run the drift-label
   coupling audit from §7.3 on this small batch before any scale-up.
6. Only after 5 passes cleanly: repeat the old G1-G5 gate structure at pilot
   scale (few geometries, tens of draws) before any production-sized run.

This mirrors the old project's own quick-start ordering
(`run_g1.py --quick` → `run_g1.py` → `run_pilot.py --smoke`) — the gate
discipline is being kept, only the solver underneath G2 onward changes.
