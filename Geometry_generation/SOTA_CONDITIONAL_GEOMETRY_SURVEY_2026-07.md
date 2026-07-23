# SOTA Conditional Geometry Generation — Survey Positioned Against SDFFlow

**Status:** fresh research pass, companion to `GEOMETRY_GENERATION_RESEARCH.md`
**Date:** 2026-07-23
**Scope:** what AI-CAE vendors (Ansys, Altair, Siemens, Narnia Labs, Neural Concept, PhysicsX) most likely do internally; and the freshest arxiv SOTA conditional geometry generators — SDF, mesh, point, voxel, CAD/B-rep — all read against what SDFFlow already does.

This document does **not** re-derive the taxonomy in `GEOMETRY_GENERATION_RESEARCH.md` (2026-07-17). It adds: (1) the two vendors that doc omitted (Altair, Siemens) plus the Siemens↔Altair consolidation; (2) the fact that the whole 3D-native generation field has now converged on SDFFlow's exact recipe; (3) the newest 2025-H2 / 2026 papers; (4) a concrete, prioritized upgrade path for SDFFlow. Company internals are marked **[confirmed]**, **[stated by vendor]**, or **[inference]**.

---

## 0. What SDFFlow actually is (the reference point)

From the checked-in code (`model/sdf_vae.py`, `model/velocity_net.py`, `training_profiles/train_pipeline.py`):

- **Stage 1 — SDF-VAE.** Encode geometry (surface points/normals + near-surface SDF samples) → a **single global latent token** (`latent_tokens × latent_dim`, currently 1 token) with KL regularization. SDF decoder `f(x, z)`; reconstruction on clamped SDF. Marching Cubes → STL.
- **Stage 2 — latent rectified flow.** `z_t = (1-t)·noise + t·data`, target velocity `v = data - noise`. Velocity net = **AdaLN-Zero residual MLP blocks** conditioned on `(t, cond)`. Condition = a handful of **geometric descriptors** (`bbox_x, bbox_z, volume, area`). **Classifier-free guidance** via condition dropout + a learned null token. Euler ODE sampler, `cfg_scale` sweep.
- **Conditioning today:** low-dimensional numeric descriptors, FiLM/AdaLN injection at every block. Extrapolation guard, candidate over-sampling + geometric-error ranking, latent interpolation.
- **Data:** DeepJEB brackets. **No physics heads. No structured/local latent. No image/sketch conditioning.**

**The one sentence that matters:** *SDFFlow is a scaled-down, descriptor-conditioned instance of the single architecture that both the SOTA 3D-generation literature and the CAE vendors have converged on — an SDF-family VAE followed by a rectified-flow / diffusion transformer over its latent.* Everything below is either (a) that same recipe at larger scale / richer conditioning, or (b) a deliberately different representation (mesh, CAD/B-rep) with its own trade-offs.

---

## 1. What the AI-CAE companies do

### 1.0 The single biggest structural fact: Siemens now owns Altair

**[confirmed]** Siemens closed its acquisition of **Altair** on **2025-03-26** (~US$10B, $113/share). So the "vendor landscape" is consolidating: Altair's **physicsAI** and Siemens **Simcenter / NX / HEEDS** are becoming one AI-CAE portfolio under Siemens Xcelerator. When you benchmark "Altair" and "Siemens" you are increasingly benchmarking one roadmap. Ansys, meanwhile, is inside **Synopsys** (Ansys 2026 R1 shipped as a Synopsys release). The independent pure-plays left are **Neural Concept**, **PhysicsX**, and **Narnia Labs**.

### 1.1 Summary table

| Vendor | Product (2026) | Geometry generator — most likely internals | Conditioning | How close to SDFFlow |
|---|---|---|---|---|
| **Ansys** (Synopsys) | **GeomAI** + SimAI + optiSLang | **[stated]** learns a **latent design space** from user reference geometries, samples/optimizes it to emit new concepts with **topology variation**; representation undisclosed. **[inference]** an implicit/point autoencoder + latent generative prior — the same VAE-then-latent-sampler pattern as SDFFlow | latent-space search driven by SimAI/solver metrics via optiSLang | **Very close in concept.** SDFFlow is essentially a self-hosted GeomAI core (latent + generator); GeomAI adds the surrogate-in-the-loop optimizer and topology-rich latent |
| **Siemens** | **Simcenter PhysicsAI Generate** (2026.1, in Simcenter HyperMesh) | **[stated]** an explicit **diffusion model**: user supplies a dataset of *design parameters + KPIs*, trains the generator, samples novel concepts "in seconds" | design parameters + KPI targets (conditional diffusion) | **Direct sibling.** Same "conditional generative model over an engineering design distribution." Diffusion vs SDFFlow's rectified flow is a minor axis; conditioning on KPIs is exactly the physics-target conditioning SDFFlow lacks |
| **Altair** (Siemens) | **physicsAI** in HyperWorks 2026 | **[stated]** geometric-deep-learning **surrogate** (physics from CAD/mesh, up to ~1000× faster) + **generative algorithms that "automatically create and refine geometry"** + GPU ROMs | performance targets; surrogate-guided | **Complementary.** Altair's public strength is the *surrogate*; the generator is less disclosed. This is the "generate-then-rank-with-surrogate" loop SDFFlow would bolt on |
| **Neural Concept** | Engineering Intelligence platform | **[confirmed link]** co-founder **Pascal Fua** (EPFL) — his lab authored **MeshSDF** and **PhysGen** (CVPR 2026). **[inference]** production representation is a shape latent with a physics-aware generative prior (PhysGen-lineage), plus GCNN/graph surrogates | performance targets + physics fields; "physics-grounded" generation | **This is SDFFlow's north star.** PhysGen = SDF-VAE + physics decoders (pressure/drag) + **physics-guided rectified flow**. SDFFlow is PhysGen minus the physics heads |
| **PhysicsX** | **LGM-Aero** foundation model + Ai.rplane | **[stated]** 100M-param foundation model, ~25M meshes, mesh encoder → **512-D latent** → implicit distance decoder; generative models over the embedding; physics regressors on the latent | latent sampling/optimization; downstream physics heads | **Same skeleton, foundation scale.** LGM-Aero ≈ SDFFlow's VAE scaled to a pretrained geometry-encoder + physics heads. Not reproducible in-house, but architecturally aligned |
| **Narnia Labs** | DeepJEB / DeepJEB++ / DeepWheel datasets & method | **[confirmed]** **DeepSDF auto-decoder → Marching Cubes → geometry filter → automated FEM** (DeepJEB); **[confirmed]** DeepJEB++ = 2D diffusion augmentation + TRELLIS lifting + FEA gates | descriptor/latent exploration coupled to performance | **Same data lineage as you.** You already train on DeepJEB. Their generator is DeepSDF+MC; SDFFlow's VAE+flow is a strictly more modern prior over the same shapes |

### 1.2 The six patterns that recur across all of them

1. **Latent generative geometry, not parametric CAD trees.** Ansys GeomAI, Siemens PhysicsAI Generate, PhysicsX LGM, Narnia DeepSDF — every one learns a *latent* over a shape distribution and samples it. This is exactly SDFFlow's design and validates the direction. Ansys explicitly markets "moving beyond CAD parameters."
2. **The generator and the surrogate are separate, then coupled.** Nobody does end-to-end physics-conditioned decoding as the product. They generate candidates, score them with a surrogate (SimAI / physicsAI / GCNN / GP-on-latent), optimize in latent space, and periodically verify with the real solver. SDFFlow has the generator half; the surrogate/coupling half is the missing product loop (and it's the higher-value half commercially).
3. **Diffusion / rectified flow is now the default generative engine.** Siemens says "diffusion — the same class used for images/video." SDFFlow's rectified flow is the same family; you are not behind on the generative-model axis.
4. **Topology variation is the headline selling point.** GeomAI's pitch is "structural changes that are cumbersome with traditional parameterization." SDF + Marching Cubes gives this for free — a genuine SDFFlow advantage over any mesh-morphing competitor.
5. **Validity gating is part of the architecture.** DeepJEB discarded ~43% of generated brackets before FEM. Nobody ships raw samples. SDFFlow's candidate over-sampling + geometric-error ranking is the seed of this; it should grow into an explicit watertightness/meshability/FEA gate.
6. **Foundation-model scale is arriving but is bought, not built.** LGM-Aero (25M meshes) and DeepJEB++ (TRELLIS lifting) show the frontier. The in-house-trainable move is *2D augmentation + pretrained-3D lifting to grow the dataset*, not training a billion-param 3D model.

---

## 2. Arxiv SOTA — grouped by paradigm, read against SDFFlow

### Group A — 3D-native latent generators = the SDFFlow family at scale (most relevant)

These are **the same recipe as SDFFlow** — an SDF / vector-set VAE, then a rectified-flow or diffusion transformer over the latent — differing mainly in latent *structure*, *scale*, and *conditioning modality*. This convergence is the strongest single signal that SDFFlow's architecture is correct.

| Model | Latent | Generator | Conditioning | What SDFFlow can borrow |
|---|---|---|---|---|
| **TripoSG** (2502.06608, 2025) | SDF-VAE with **hybrid SDF + surface-normal + eikonal** supervision | large **rectified-flow transformer** | image (2M curated image-SDF pairs) | The **hybrid VAE loss** (add normal + eikonal terms to your SDF recon loss) is a cheap, high-value upgrade for surface/normal fidelity |
| **Hunyuan3D 2.0 / 2.5** (2501.12202, 2025) | **ShapeVAE**: mesh → continuous **token sequence** | **flow-based DiT** (Hunyuan3D-DiT) | image | Move from 1 global token → a **token set**; DiT over tokens is the direct successor to your MLP velocity net "behind the same interface" (your code comment already anticipates this) |
| **TRELLIS** (2412.01506, CVPR 2025) | **SLat** — structured latents on active voxels; decodes to mesh / 3DGS / radiance field | rectified-flow transformer, up to 2B params | text, image | **Structured/local latent** is the fix for a global token's detail bottleneck; one latent, multiple decoders |
| **Direct3D-S2 / sparse-voxel line** | sparse-voxel SDF latent | diffusion | image | Sparse voxels scale resolution without dense 3D grids — relevant if bracket thin-walls/holes vanish at your MC resolution |
| **3DShape2VecSet** (2301.11445, 2023) | **set of latent vectors** (VecSet) | transformer diffusion | class/image/partial | The canonical VecSet formulation nearly every model above builds on; the reference for your token-set upgrade |
| **ShapeR** (2601.11514, Meta, **Jan 2026**) | VecSet latents | **rectified-flow transformer on FLUX DiT** | **tri-modal**: sparse SLAM points + posed images (DINOv2) + text captions; 2.7× Chamfer over SOTA | The state of the art in *multimodal conditioning* — cross-attention fusion of point/image/text is the template if you ever add sketch/scan/partial-geometry conditioning |
| **UltraShape 1.0** (2512.21185, 2025) | SDF latent + **scalable geometric refinement** | flow | image | Coarse-then-refine is how to get CAE-grade surface detail out of a compact latent |
| **PartCrafter** (2506.05573, 2025) | **compositional per-part** latents | latent diffusion transformer | image | Part-structured generation → maps cleanly to engineering assemblies / named boundary regions |
| **LTM3D** (2505.24245, 2025) | 3DShape2VecSet SDF-VAE | **AR + diffusion hybrid** | multimodal | Shows AR/diffusion hybrids over the same SDF-VAE substrate |

**Takeaway for SDFFlow:** you are on the main line. The three cheapest, highest-return moves are (1) **hybrid VAE loss** (normal + eikonal, à la TripoSG), (2) **global token → VecSet/token-set + DiT velocity net** (your own code comment says this is the planned upgrade path), (3) **structured/sparse latent** only if MC resolution is losing thin features.

### Group B — engineering-specific conditional & inverse-design generators (closest to your *use case*)

These target CAE/design directly and are the papers to imitate for conditioning and validity — most are self-trainable at your data scale.

- **PhysGen** (CVPR 2026; 2512.00422) — **the blueprint.** SP-VAE with a shared latent feeding an **SDF decoder + surface-pressure decoder + scalar-drag decoder**; **physics-guided rectified flow** that alternates a learned velocity step with a differentiable physics-refinement step; image or unconditional. Trained on 5,819 DrivAerNet++ shapes (4×H100, ~2 days) — reproducible. **SDFFlow → PhysGen is a matter of adding physics heads on the existing latent and a physics-refinement step in the sampler.**
- **AirfoilGen** (2605.20303, 2026) — **valid-by-construction + performance-aware latent diffusion.** Bakes geometric validity into the representation and conditions on aero performance. The "valid-by-construction" idea is the principled version of your candidate-ranking gate.
- **HeatGen** (2511.09578, 2025) — **guided diffusion for multiphysics heat-sink design.** Classifier/surrogate-guided sampling toward a multiphysics objective; a template for guiding SDFFlow toward a target QoI rather than only descriptors.
- **Dflow-SUR** (2512.08336, 2025) — **differentiation *throughout* flow matching** for aerodynamic inverse design: backprop a surrogate objective through the whole flow ODE, not just the endpoint. Directly upgrades your CFG sampler into a gradient-guided inverse-design sampler.
- **Diagonal Flow Matching / generative inverse design with abstention** (2603.15925, 2026) — **invertible conditional flow matching** that runs *bidirectionally*: design→performance prediction and performance→design generation in one model, with an **abstention** mechanism when a target is infeasible. This is a genuinely different and useful idea for you: one model that both generates and predicts, and *knows when to refuse* out-of-distribution targets — a principled replacement for your `max_condition_z` guard.
- **Inverse Design for Conditional Distribution Matching** (2605.09439, 2026) — formalizes matching an induced performance distribution to a target; theory for "generate shapes whose *distribution* of outcomes hits a spec."
- **CFG-guided DDPM for aerodynamic optimization** (2503.07056, 2026) — confirms classifier-free guidance (which SDFFlow already implements) as a standard inverse-design lever; validates your `cfg_scale` sweep.
- **BladeSDF** (2601.13445, 2026, in prior doc) — DeepSDF latent + compact descriptor→latent conditioner for turbine blades; the closest published twin of SDFFlow's descriptor conditioning, with ~1%-of-max-dimension surface error.
- **TopoDiff / TopologyGAN** — conditional density/topology generation under loads/BCs; the route if you ever want load-conditioned structural layout rather than SDF shapes.

**Takeaway:** the engineering frontier is (a) **conditioning on physics targets, not just geometry descriptors**, (b) **guidance through the sampler** (surrogate gradients / physics refinement — HeatGen, Dflow-SUR, PhysGen), and (c) **bidirectional / abstaining inverse design** (Diagonal FM). All three sit directly on top of your existing rectified-flow sampler.

### Group C — mesh-native and CAD/B-rep generators (deliberately *different* from SDFFlow)

Covered in depth in the prior doc; the fresh-pass deltas:

- **Mesh-native:** MeshGPT / MeshAnything-style autoregressive triangle generation and **PartCrafter** give sharp features and part structure but no watertightness/CAE guarantee. Useful as *low-face proposal* or *assembly-structured* generation, not as your primary CAE route.
- **CAD/B-rep-native:** **HoLa** (SIGGRAPH 2025, ~82–84% valid unconditional B-rep vs BrepGen's ~50%) and **BR-DF** (global SDF + per-face UDFs → extended MC → 100% *faceted* B-rep) remain the cleanest research bridges to STEP. Still the separate, data-hungry "clean editable STEP" track — keep it off the SDF critical path exactly as the prior doc concluded.
- **CAD-program:** Text2CAD, CAD-Recode, Zero-to-CAD (million-scale synthetic programs) — the editable-STEP route when a bounded family exists.

**Takeaway:** nothing here changes the prior recommendation — SDF+MC is your primary free-form/STL route; STEP is a second branch.

---

## 3. Gap analysis + prioritized upgrade path for SDFFlow

Ordered by value/effort. Each maps to a concrete SOTA source above.

**Tier 1 — cheap, high return, no architecture change**
1. **Hybrid VAE loss** — add surface-normal alignment + eikonal (`‖∇f‖→1`) terms to the SDF reconstruction (TripoSG). Better surfaces/normals, better meshability; directly helps CAE mesh quality.
2. **Validity gate as a first-class stage** — turn candidate ranking into an explicit watertight / manifold / min-thickness / meshability filter with recorded failure classes (DeepJEB's ~43% reject rate is the norm, not a bug).
3. **Guidance-through-sampler** — extend the Euler ODE sampler to accept a surrogate/physics gradient at each step (Dflow-SUR, HeatGen). Reuses the exact loop in `velocity_net.sample_latents`.

**Tier 2 — moderate, unlocks the vendor "product loop"**
4. **Physics heads on the shared latent** — add pressure/stress/field + scalar-QoI decoders off the VAE latent (PhysGen SP-VAE, PhysicsX LGM). Turns SDFFlow from "geometry generator" into "generate + predict," which is what Ansys/Siemens/Neural Concept actually sell.
5. **Condition on physics targets, not only descriptors** — feed KPI/target-QoI (drag, max-stress, mass) into the FM condition vector alongside bbox/volume/area (Siemens PhysicsAI Generate, BladeSDF, AirfoilGen).
6. **Global token → VecSet + DiT velocity net** — your own code comment marks this path ("a DiT over token sets can be added later behind the same interface"). Fixes the global-latent detail bottleneck (3DShape2VecSet, Hunyuan3D, ShapeR).

**Tier 3 — research-grade, matches the frontier**
7. **Bidirectional / abstaining inverse design** — one invertible flow that generates *and* predicts and refuses infeasible targets (Diagonal FM) — a principled replacement for `max_condition_z`.
8. **Structured/sparse latent** — SLat / sparse-voxel (TRELLIS, Direct3D-S2) if thin walls/holes are lost at MC resolution or you need broader topology.
9. **Multimodal conditioning** — sketch/image/partial-geometry via cross-attention (ShapeR's tri-modal FLUX-DiT is the template) if the product needs designer-in-the-loop input.
10. **Dataset scale-up via 2D augmentation + pretrained-3D lifting** (DeepJEB++) — grow beyond DeepJEB without training a foundation model in-house.

**Do NOT** put clean editable STEP on the SDF path (unchanged from prior doc); keep it as a HoLa/BR-DF/CAD-program branch.

---

## 4. Bottom line

- **You are not behind on architecture.** SDFFlow (SDF-VAE + conditional rectified flow + CFG) is the *same* recipe that TripoSG, Hunyuan3D, TRELLIS, ShapeR, PhysGen, and (as far as can be inferred) Ansys GeomAI, Siemens PhysicsAI Generate, and PhysicsX LGM all use. The convergence is the validation.
- **The gap is conditioning + coupling, not the generator.** The vendors' and the engineering papers' edge is (1) physics-target conditioning, (2) surrogate/physics guidance through the sampler, (3) a shared geometry↔physics latent, and (4) an explicit validity/inverse-design loop. Every one of these is an additive change on top of your existing `VelocityNet` and VAE.
- **The clearest single target is PhysGen** (Neural Concept lineage): add physics heads to the SDF-VAE and a physics-refinement step to the flow sampler, and SDFFlow becomes a self-hosted, engineering-conditioned equivalent of what the pure-play vendors are commercializing.

---

## 4b. Technical deep-dive — how SOTA actually gets better results (VAE, generator, architecture)

This section answers the concrete questions: is diffusion better than your rectified flow, is the VAE the lever, does a bigger/deeper network help, and is moving off the global token worth it. Numbers are from the papers' own configs/ablations.

### The order of impact (most systems find this ranking)

**VAE quality ≫ generator recipe > network depth.** The VAE is a hard ceiling: the flow/diffusion model can only ever sample latents the decoder can turn back into good geometry. Every SOTA report says the same thing — reconstruction quality of the autoencoder caps final generation quality. So spend effort here first. SDFFlow's current VAE is the weakest link, not its rectified flow.

### (A) Diffusion vs. flow matching — you already have the better one; don't switch

- **They are the same framework.** Gaussian diffusion and flow matching are mathematically interchangeable (Google's "Diffusion Meets Flow Matching"); the difference is the network parameterization and the noise/time schedule, not a different model class.
- **Rectified flow (what SDFFlow uses) is the modern winner:** straight probability paths → fewer sampling steps, more stable training. TripoSG, Hunyuan3D, ShapeR, Stable Diffusion 3/3.5 all use rectified flow, not DDPM. Switching SDFFlow to classical diffusion would be a step *backward*.
- **The real gain is the SD3 recipe *on top of* rectified flow**, and it's ~10 lines:
  - **Logit-normal timestep sampling.** SD3's headline ablation: sample the training time `t` from a logit-normal (emphasize middle timesteps) instead of uniform. SDFFlow currently does `t = torch.rand(...)` (uniform) in `flow_matching_loss` — this is the single cheapest generator upgrade.
  - **Timestep "shift"** toward noisier steps as latent size grows (matters once you move to token sets).
  - Optionally an EMA-of-velocity / v-parameterization sanity check (you already keep EMA).
- **Verdict:** keep rectified flow. Change the *time-sampling distribution*, not the paradigm.

### (B) VAE — the highest-leverage work, with concrete recipes

Your VAE today (`model/sdf_vae.py`): **1 global token × 256-dim**, a **2-block / 256-wide** cross-attention encoder, DeepSDF MLP decoder, and **L1 on truncated SDF only**. Four evidence-backed upgrades, in order:

1. **Sharp-edge / salient point sampling (Dora, CVPR 2025) — biggest detail win, cheap.** Uniform surface sampling systematically loses edges and corners. Dora samples points by *dihedral-angle saliency* and adds dual cross-attention; it matches the dense XCube-VAE's reconstruction with an **8× smaller latent (1,280 vs >10,000 codes)**. Brackets are full of sharp fillets/edges — this is directly applicable. You already have `surface_points`/`surface_normals`; add a saliency-weighted sampler in `sdf_sampling.py`.
2. **Hybrid VAE loss (TripoSG).** Replace L1-only with **SDF + surface-normal alignment + eikonal (`‖∇_x f‖ → 1`)**. Normal loss sharpens surfaces; eikonal makes the field a true metric SDF (better Marching Cubes, better GINO-style downstream use). Needs grad-enabled query points (autograd through the decoder) — modest cost.
3. **KL weight + latent shape.** 3DShape2VecSet uses **KL weight ≈ 1e-3** and a **small per-token channel C₀ = 32**, but *many* tokens. Your 1×256 is the opposite: one fat token. Check your KL/β warmup — too high → mean/blurry shapes (posterior collapse), too low → a latent the FM can't model. Target the ~1e-3 regime and normalize per-channel before the FM (you already feed normalized encoder means, which is correct — same as SD scaling the latent).
4. **Multi-resolution token training (Hunyuan3D).** Randomly vary the token-sequence length during VAE training; speeds convergence and makes the decoder robust to token count.

### (C) Global token vs. token set — yes, this is the single biggest architectural lever

Your literal question: *is moving off the global token much better?* **Yes, substantially — but pair it with a DiT flow model and respect your data scale.**

- **No SOTA shape VAE uses a single global token.** Michelangelo, 3DShape2VecSet, Hunyuan3D-ShapeVAE, Dora, TripoSG all use a **set of latent vectors**. A single 256-D vector is a severe information bottleneck — it's exactly why global DeepSDF "struggles with detailed, spatially heterogeneous geometry" (your own prior doc §3.6/3.8). Detail and topological variety live in the token *set*.
- **How many tokens?** Reconstruction VAEs go big (Hunyuan3D up to **3072**; Dora ~1,280). But for *generation* you can be far smaller: "Representing 3D Shapes with 64 Latent Vectors" (ICCV 2025) shows **64 vectors** is enough for strong 3D diffusion. For a narrow bracket family, start modest: **16–64 tokens × 32–64 dims**, not thousands.
- **Data-scale caveat (important for you).** DeepJEB is ~2K brackets. A 3072-token latent + an 80-block DiT will overfit badly. More tokens ⇒ more data needed. The right move is a *modest* set (16–64 tokens), which your code already supports (`latent_tokens > 1` + `decoder_type attention` — the attention decoder is written and unused).
- **The catch:** your FM velocity net is an **MLP over the *flattened* latent**. Flattening a token set throws away its set structure and won't scale. Moving to a token set **requires** upgrading the velocity net to a **DiT** (self-attention among the latent tokens + AdaLN-Zero for `(t, cond)`, cross-attention for richer conditions). Your `velocity_net.py` docstring already flags this ("a DiT over token sets can be added later behind the same interface"). This is the concrete "better FM model."

### (D) Deeper / wider network — helps, but only after (B) and (C), and only to a point

- Large image/3D DiTs scale to **hidden 1024–3072, depth 28–80 blocks** — that's the *foundation-model* regime (millions of shapes). It is not your regime.
- Your current FM net (hidden 512, **6 MLP blocks**) is small, but making *that MLP* deeper won't help much because the bottleneck is upstream (single-token latent). Depth pays off once you have (i) a token-set VAE and (ii) a DiT that can use self-attention. For ~2K brackets, a sensible target is **hidden 512–768, 8–12 DiT blocks, 6–8 heads** — deeper than now, far shallower than TripoSG/Hunyuan3D.
- The encoder is also thin (2 blocks / 256). Bumping to **4–6 blocks / 512** with self-attention among the latent tokens (Dora's dual cross-attention pattern) is worth more than a deeper decoder.

### Concrete, ranked action list for SDFFlow

**Implementation status (2026-07-23): #1–#6 landed and smoke-tested; enabled together in `configs/Geometry_generation/config_train_v2.txt` (trains to `ex2/`, leaving `ex1/` intact). All changes are config-gated and backward-compatible — the existing `config_train.txt` and checkpoints are unaffected. #2 (sharp-edge) is a `build_dataset.py --sharp_edge_fraction` option needing a dataset rebuild. #7 (multi-resolution tokens) is deferred to the FPS-query encoder change tracked in `FOUNDATION_MODEL_PLAN.md` M1.**

| # | Change | File | Effort | Why |
|---|---|---|---|---|
| 1 | Logit-normal `t` sampling (not uniform) | `velocity_net.flow_matching_loss` | ~5 lines | SD3's biggest generator win; free |
| 2 | Sharp-edge / salient surface sampling | `general_modules/sdf_sampling.py` | small | Dora: recover edges lost to uniform sampling |
| 3 | Hybrid VAE loss (SDF + normal + eikonal) | `model/sdf_vae.py` | medium | TripoSG: true metric SDF, sharper surfaces |
| 4 | Global token → 16–64-token set (attention decoder already exists) | config + `train_fm` | medium | Removes the core detail bottleneck |
| 5 | MLP velocity net → DiT over tokens | `model/velocity_net.py` | medium-large | Required to exploit the token set; the "better FM" |
| 6 | Encoder 2→4-6 blocks + latent self-attention; check KL≈1e-3 | `model/sdf_vae.py` | small-medium | Better encode; avoid collapse |
| 7 | Multi-resolution token training | `train_vae` | small | Faster convergence, robust decoder |

**Do #1–#3 first** (cheap, no architecture change, and #3 has the best quality/effort ratio for CAE meshing). **#4–#5 together** are the real step-change but need the DiT and more careful validation. Don't jump to giant token sets / 80-block DiTs — you don't have the data for it.

---

## 5. Sources (2026-07-23 pass)

**Vendors**
- Ansys GeomAI + SimAI, 2026 R1: https://www.ansys.com/blog/introducing-ansys-geomai-software ; https://www.ansys.com/products/ai/geomai ; "Moving Beyond CAD Parameters: Latent Space in Engineering Design": https://www.ansys.com/blog/latent-space-engineering-design
- Siemens Simcenter PhysicsAI Generate (diffusion, 2026.1): https://blogs.sw.siemens.com/simcenter/generative-ai-engineering-design/
- Siemens ↔ Altair close (2025-03-26, ~$10B): https://press.siemens.com/global/en/pressrelease/siemens-acquires-altair-create-most-complete-ai-powered-portfolio-industrial-software ; https://investor.altair.com/news-releases/news-release-details/altair-signs-definitive-agreement-siemens-be-acquired-106
- Altair physicsAI / HyperWorks 2026: https://altair.com/newsroom/news-releases/altair-hyperworks-2026 ; https://altair.com/blog/executive-insights/geometric-deep-learning-ai-engineering-altair-physicsai
- Neural Concept (physics-grounded generative; PhysGen at CVPR 2026): https://www.tipranks.com/news/private-companies/neural-concept-emphasizes-physics-grounded-generative-ai-for-performance-driven-design
- PhysicsX LGM-Aero (512-D geometry latent, foundation model): https://www.physicsx.ai/newsroom/building-beyond-human-imagination-with-foundation-models-for-geometry-and-physics
- Narnia Labs DeepJEB / DeepJEB++ / DeepWheel: arXiv 2406.09047 ; 2606.12994 ; 2504.11347

**3D-native latent generators (SDFFlow family)**
- TripoSG (rectified flow, SDF-VAE hybrid loss): https://arxiv.org/abs/2502.06608
- Hunyuan3D 2.0 (ShapeVAE + flow DiT): https://arxiv.org/html/2501.12202v5
- TRELLIS / Structured 3D Latents (SLat): https://arxiv.org/html/2412.01506
- ShapeR (rectified flow transformer, tri-modal, FLUX DiT, Jan 2026): https://arxiv.org/pdf/2601.11514 ; https://facebookresearch.github.io/ShapeR/
- 3DShape2VecSet: https://arxiv.org/abs/2301.11445
- UltraShape 1.0: https://arxiv.org/html/2512.21185v2
- PartCrafter (compositional part latents): https://arxiv.org/pdf/2506.05573
- LTM3D (AR+diffusion over SDF-VAE): https://arxiv.org/html/2505.24245v1
- Locally Attentional SDF Diffusion: https://arxiv.org/pdf/2305.04461 ; WaLa (billion-param wavelet SDF): https://arxiv.org/pdf/2411.08017

**Engineering conditional / inverse design**
- PhysGen (CVPR 2026): https://arxiv.org/html/2512.00422v2 ; https://kasvii.github.io/PhysGen/
- AirfoilGen (valid-by-construction + performance-aware latent diffusion): https://arxiv.org/pdf/2605.20303
- HeatGen (guided diffusion, heat-sink multiphysics): https://arxiv.org/pdf/2511.09578
- Dflow-SUR (differentiation throughout flow matching, aero inverse design): https://arxiv.org/pdf/2512.08336
- Generative Inverse Design with Abstention via Diagonal Flow Matching: https://arxiv.org/html/2603.15925
- Inverse Design for Conditional Distribution Matching: https://arxiv.org/abs/2605.09439
- CFG-guided DDPM aerodynamic optimization: https://arxiv.org/pdf/2503.07056
- BladeSDF: https://arxiv.org/abs/2601.13445

**VAE / generator technical recipe (§4b)**
- Dora — Sharp Edge Sampling for 3D shape VAEs (CVPR 2025): https://aruichen.github.io/Dora/ ; https://arxiv.org/html/2412.17808v1 ; code https://github.com/Seed3D/Dora
- 3DShape2VecSet (KL≈1e-3, C₀=32, VecSet): https://arxiv.org/pdf/2301.11445
- Hunyuan3D-ShapeVAE (up to 3072 tokens, multi-resolution training): https://arxiv.org/html/2501.12202v1
- Michelangelo (multiple 1D latent vectors, aligned VAE): referenced via Hunyuan3D/Direct3D above
- Direct3D — D3D-DiT scalable 3D latent diffusion transformer: https://arxiv.org/html/2405.14832v2
- "Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion" (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/papers/Cho_Representing_3D_Shapes_with_64_Latent_Vectors_for_3D_Diffusion_paper.pdf
- Diffusion ≡ Flow Matching (Google): https://diffusionflow.github.io/
- Stable Diffusion 3 (rectified flow + logit-normal timestep sampling): https://learnopencv.com/stable-diffusion-3/
- DiT scaling (depth/width vs FID; Peebles & Xie): https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf

**CAD / B-rep (STEP branch)** — see `GEOMETRY_GENERATION_RESEARCH.md` §3.11–3.12 for HoLa, BrepGen, BR-DF, DeepCAD, Text2CAD, CAD-Recode, Zero-to-CAD.
