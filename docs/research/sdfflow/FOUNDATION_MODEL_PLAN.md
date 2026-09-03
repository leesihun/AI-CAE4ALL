# AI-CAE Geometry(+Physics) Foundation Model — Design Plan

**Status:** design plan (companion to `SOTA_CONDITIONAL_GEOMETRY_SURVEY_2026-07.md` and `GEOMETRY_GENERATION_RESEARCH.md`)
**Date:** 2026-07-23
**Premise:** grow SDFFlow from a single-family (DeepJEB bracket) generator into a *reusable geometry foundation model* — a pretrained shape encoder + latent generative prior + physics heads that transfers across part families and CAE tasks, in the spirit of PhysicsX LGM-Aero and PhysGen, but trainable in-house and coupled to the surrogates already in this repo (GINO, Transolver, MeshGraphNets, Neural Operators).

---

## 0. What "foundation model" means here (and what it does not)

It does **not** mean "train another LGM-Aero" (100M params, 25M meshes, 128×H100 for weeks). That is not reproducible in-house and its scale is not the transferable idea.

It **does** mean the four things every geometry foundation model in the survey actually shares:

1. **One pretrained geometry encoder** that maps *any* watertight surface to a compact latent (PhysicsX: a universal 512-D vector; TripoSG/Hunyuan3D/Dora: a VecSet). Reused as a frozen feature extractor for many downstream tasks.
2. **A latent generative prior** (rectified-flow / diffusion DiT) that samples new shapes in that latent, conditionally.
3. **Physics heads / regressors on the same latent** (PhysGen: SDF + pressure + drag; PhysicsX: GP/regressors on the 512-D code) so geometry and performance share a representation.
4. **Transfer**: pretrain on all available geometry, then fine-tune the prior/heads per domain with little data.

SDFFlow-v2 (just built: VecSet VAE + DiT + hybrid loss + logit-normal FM) is exactly component (1)+(2) at *proto* scale. This plan scales it and adds (3)+(4), reusing this repo's surrogates for the physics side.

---

## 1. Target architecture

```
                         ┌─────────────────────────────────────────────┐
                         │            GEOMETRY FOUNDATION CORE          │
  any watertight mesh    │                                             │
  ──► salient+uniform ──►│  Encoder (cross+self attn)                  │
      point sampling     │      │                                      │
      (Dora sharp edge)  │      ▼                                      │
                         │   VecSet latent  z ∈ (T tokens × C)   ◄── the universal code
                         │      │            (T≈256–512, C≈32–64)      │
                         │      ├──► SDF decoder  f(x,z)  ──► MC ──► STL │
                         │      ├──► normal/eikonal (metric SDF)        │
                         │      └──► (frozen for downstream reuse)      │
                         └─────────────────────────────────────────────┘
                                  │                         │
        conditions c ────────────┤                         │  z (frozen features)
   (descriptors, physics targets,│                         ▼
    image/sketch, partial geom)  ▼                 ┌───────────────────────────┐
                       ┌───────────────────┐       │  PHYSICS / SURROGATE HEADS │
                       │ RECTIFIED-FLOW DiT │       │  • scalar QoI heads (MLP)  │
                       │ over the token set │       │  • field heads (pressure/  │
                       │  (CFG + guidance)  │       │    stress) on z or surface │
                       └───────────────────┘       │  • OR route surface mesh to│
                                  │                 │    existing GINO/Transolver│
                                  ▼                 │    /MeshGraphNet surrogate │
                          sampled latents ──────────┴───────────────────────────┘
                                  │                         │
                                  ▼                         ▼
                          decode → STL           predicted performance
                                  │                         │
                                  └──────► validity + true-CAE gate ◄── active learning
```

Concretely, three components:

**A. Geometry core (scale-up of SDFFlow-v2).**
- Encoder: Dora-style salient (sharp-edge) + uniform point sampling → cross-attention to learned/FPS query tokens → self-attention among tokens. Hybrid loss (SDF + normal + eikonal) for a true metric field.
- Latent: **structured VecSet**, `T` grown with data (32 for one family → 256–512 for a foundation corpus), small channel `C≈32–64`, KL ≈ 1e-3.
- Decoder: VecSet cross-attention SDF decoder → Marching Cubes → STL.
- This core is trained once on *all* geometry and then **frozen** as the reusable encoder/decoder.

**B. Generative prior (scale-up of the DiT).**
- Rectified-flow DiT over the token set; logit-normal timestep schedule; classifier-free guidance.
- Grows to the standard DiT scaling knobs (hidden 768–1536, depth 12–28) only when the corpus supports it.
- Conditioning via AdaLN (numeric descriptors, physics targets) + cross-attention (image/sketch/partial-geometry tokens).

**C. Physics coupling (reuse this repo).** Two options, use both:
- *Shared-latent heads* (PhysGen pattern): lightweight scalar/field decoders off the frozen `z` for in-loop generation guidance.
- *External surrogate* (this repo's strength): decode candidate → surface mesh → score with the existing **GINO / Transolver / MeshGraphNet** models. These are already built, benchmarked, and STL/point/SDF-aware. This is the pragmatic, higher-accuracy path and is exactly the "generator + separate surrogate" pattern every vendor uses.

---

## 2. Representation & latent design decisions

| Decision | Proto (1 family) | Foundation (many families) | Rationale / source |
|---|---|---|---|
| Latent structure | VecSet 32×64 | VecSet 256–512 × 32–64 | Global token bottlenecks detail; every SOTA VAE uses a set (Michelangelo, 3DShape2VecSet, Dora) |
| Query tokens | learned params | **FPS-sampled from the point cloud** | Enables variable token count + multi-resolution training (Dora/Hunyuan); required for #7 |
| KL weight | 1e-4 → tune | ~1e-3 | 3DShape2VecSet; keeps latent FM-friendly |
| Surface sampling | uniform + sharp | sharp-edge dominant | Dora: 8× smaller latent at equal reconstruction |
| VAE loss | SDF+normal+eikonal | + multi-resolution | TripoSG hybrid supervision |
| Generator | rectified-flow DiT | same, scaled | TripoSG/Hunyuan3D/ShapeR all rectified flow |
| Output | SDF→MC→STL | + differentiable extraction (FlexiCubes) when guiding | MeshSDF/DMTet/FlexiCubes for gradient-in-the-loop |

**Key engineering change vs today:** move the encoder from *learned query parameters* to *FPS-sampled query tokens* drawn from the input point cloud. That single change unlocks (a) variable/multi-resolution token counts, (b) locality (tokens anchored to geometry), and (c) the scaling behaviour the foundation corpus needs. It is the natural next PR after v2.

---

## 3. Data strategy (the hard part — and the real differentiator)

Geometry foundation quality is data-bound, not architecture-bound. Plan the corpus in three tiers, counted separately for *geometry-only* vs *CAE-labeled*:

**Tier G — geometry-only pretraining (as much as possible).**
- In-house: every usable STL/STEP the org has, across families.
- Public, engineering: **ABC** (1M CAD), **DeepJEB/DeepJEB++** (brackets), **DrivAerNet++** (cars), plus **DeepWheel**.
- Public, general (for encoder robustness only): ShapeNet/Objaverse subsets — use for the *encoder*, not for engineering conditioning.
- Preprocess: normalize → watertight repair → salient+uniform SDF samples (the `build_dataset.py --sharp_edge_fraction` path is ready).

**Tier P — CAE-labeled fine-tuning (expensive, curated).**
- A *diverse subset* of Tier G run through the real solver (the org's CFD/FEA) for scalar QoIs + fields.
- DrivAerNet++ (CFD) and DeepJEB (FEA, 4 load cases + modal) are ready-made labeled anchors.

**Tier A — augmentation without new 3D training (DeepJEB++ recipe).**
- Fine-tune a 2D image diffusion model on multi-view renders of a target family → augment + filter in image space → lift validated images to 3D with a pretrained lifter (TRELLIS/TripoSG) → run the same validity + FEA gates.
- Buys web-scale 2D diversity without training a large 3D model in-house. 15,360 brackets from <400 seeds (DeepJEB++) is the proof point.

**Splits (report all; random splits overstate generalization):** interpolation-in-range, held-out parameter cells, held-out subfamilies/topologies, held-out operating conditions, and lineage-grouped duplicates.

---

## 4. Training curriculum (staged; each stage gated before the next)

1. **Geometry VAE pretrain (Tier G).** Train the core on *all* shapes with hybrid loss + salient sampling. Gate: Dora-bench-style reconstruction (surface + sharp-feature error) and watertight-STL/meshability pass rate on held-out subfamilies.
2. **Unconditional latent flow pretrain (Tier G).** Freeze VAE; train the rectified-flow DiT unconditionally to learn the shape manifold. Gate: prior-sampled (not reconstructed) shapes are valid and diverse.
3. **Conditional fine-tune (per domain, Tier G + descriptors).** Add descriptor/physics-target conditioning + CFG. Gate: condition accuracy on decoded meshes.
4. **Physics heads / surrogate coupling (Tier P).** Train shared-latent scalar/field heads *and* wire the existing GINO/Transolver/MeshGraphNet surrogates for candidate scoring. Gate: surrogate error vs true solver on held-out.
5. **Guided generation + active learning (Tier P + A).** Physics-guided sampling (surrogate gradient / PhysGen-style refinement / differentiable FlexiCubes), then true-CAE verification of uncertain/high-value/near-constraint candidates; append and retrain. Gate: validated design objective beats DOE + direct optimization at fixed solver budget.

The staging matters: **good reconstruction ≠ good prior sampling ≠ good conditional accuracy**, and each must be measured on its own held-out split. This is the discipline the vendors' "latent + surrogate loop" hides.

---

## 5. Compute & scaling tiers (be honest about the regime)

| Tier | Corpus | Latent | DiT | Hardware | Purpose |
|---|---|---|---|---|---|
| **T0 proto (now)** | 1 family (~2K) | 32×64 | 512w/8blk | 1 GPU | SDFFlow-v2; validate recipe |
| **T1 domain** | 3–5 families (10–50K) | 64–128 × 64 | 768w/12blk | 1–4 GPU | first reusable encoder + per-domain priors |
| **T2 multi-domain** | 100K–1M (Tier G+A) | 256–512 × 32–64 | 1024–1536w / 16–28blk | 8–32 GPU | genuine foundation encoder |
| **T3 reference** | (LGM-Aero regime) | — | — | 100+ GPU-weeks | out of scope; benchmark target only |

PhysGen (5,819 shapes, 4×H100, ~2 days) and DeepJEB (learned from 263 seeds) prove **T1 is enough for a credible domain foundation**. Aim T0→T1→T2; treat T3 as an external yardstick, not a build target. Scaling law (from DiT/image-3D work): more tokens + depth ⇒ better, but only with corpus growth — do not scale model ahead of data.

---

## 6. Conditioning & guidance surface (what makes it a *design* tool)

- **Numeric:** descriptors + engineering targets (drag, mass, max-stress, compliance, frequency) via AdaLN at every block. *(SDFFlow already conditions on descriptors; add physics targets.)*
- **Spatial / multimodal:** sketch, image, partial geometry, load/BC field → tokenized and cross-attended (ShapeR's tri-modal FLUX-DiT is the template; DINOv2 for images).
- **Guidance:** CFG (built), surrogate-gradient guidance, PhysGen-style alternating physics refinement, and — when mesh-level objectives are needed — differentiable extraction (FlexiCubes) so the physics loss reaches the latent.
- **Inverse / abstaining:** an invertible flow (Diagonal-FM) that both generates and predicts and refuses infeasible targets — a principled replacement for the current `max_condition_z` guard.

---

## 7. Evaluation & gates (reused from the research doc, made continuous)

1. **Reconstruction:** surface + normal + sharp-feature error (Dora-bench style), watertight/manifold/meshability rate — on held-out subfamilies.
2. **Generation:** prior-sampled validity + diversity + novelty vs nearest training shape.
3. **Conditional:** true-CAE target error per requested condition; % within tolerance; per-condition coverage.
4. **Physics head:** surrogate error vs true solver; uncertainty calibration.
5. **Usefulness:** validated objective / Pareto hypervolume vs DOE, direct optimization, nearest-neighbour, cVAE/flow baselines — at a fixed solver budget.

Store failures (non-watertight, non-converged) with their class — they train feasibility models and expose extrapolation.

---

## 8. Deployment shape (how it's actually used)

- **Encoder-as-a-service** (PhysicsX pattern): expose the frozen encoder; any shape → a fixed-length code that downstream regressors/GPs consume. Highest-ROI early product: a universal geometry feature for the org's existing ML.
- **Generator API:** conditional sampling with CFG + guidance + validity gate → ranked STL candidates + condition audit (the current `sample.py` metadata contract already does the audit — extend it).
- **Active-learning loop:** generator ↔ surrogate ↔ true solver, appending verified samples. This is the flywheel that compounds over time and is the moat the vendors are building.

---

## 9. Roadmap & milestones

- **M0 (done):** SDFFlow-v2 recipe — VecSet + DiT + hybrid loss + logit-normal + salient sampling. Validate on DeepJEB (ex2).
- **M1:** FPS-query encoder + multi-resolution token training (unlocks scaling); Dora-bench reconstruction harness.
- **M2 (T1):** pretrain encoder on 3–5 in-house families + DeepJEB/DrivAerNet++; per-domain conditional priors; freeze encoder.
- **M3:** physics heads + wire existing GINO/Transolver/MeshGraphNet surrogates; guided sampling + validity gate.
- **M4:** Tier-A 2D-augmentation/3D-lifting data expansion; active-learning loop with true CAE.
- **M5 (T2):** scale to multi-domain foundation encoder; multimodal (sketch/image) conditioning.
- **STEP branch (parallel, off critical path):** faceted STEP via BR-DF, or CAD-program/HoLa for editable B-rep — only when a clean-STEP deliverable is firm.

## 10. Top risks

- **Data, not model, is the ceiling.** Under-investing in Tier G/P/A caps everything. Budget data engineering first.
- **Reconstruction↔generation↔condition gap.** Measuring the wrong one (reconstruction) hides prior/conditional weakness. Enforce separate held-out splits.
- **Surrogate exploitation.** Guided generation games the surrogate; mandatory periodic true-CAE verification + calibrated uncertainty + latent trust region.
- **Scaling ahead of data.** A big DiT on a small corpus overfits. Grow model with corpus, per §5.
- **STEP scope creep.** Editable B-rep is its own product; keep it off the SDF critical path.

---

## 11. Immediate next PR (concrete)

1. Train `config_train_v2.txt` on DeepJEB → confirm the v2 recipe reconstructs and generates at least as well as ex1 (A/B on held-out).
2. Rebuild the dataset with `build_dataset.py --sharp_edge_fraction 0.3` → measure sharp-feature reconstruction delta.
3. Land the **FPS-query encoder** (replace the learned-query `nn.Parameter` with farthest-point-sampled query tokens) — the enabling change for multi-resolution and scale (M1).
4. Add a **scalar physics head** off the frozen latent and wire one existing surrogate (Transolver or GINO) for candidate scoring (M3 seed).

See `SOTA_CONDITIONAL_GEOMETRY_SURVEY_2026-07.md` §4b for the per-change evidence and `GEOMETRY_GENERATION_RESEARCH.md` §5–§14 for the full technique taxonomy and named-model audit.
