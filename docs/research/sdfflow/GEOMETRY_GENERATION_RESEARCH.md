# Conditional Geometry Generation for AI-CAE

**Status:** research and architecture recommendation  
**Date:** 2026-07-17  
**Required outputs:** realistic, condition-controlled geometries; `.stp`/`.step`; `.stl`; downstream AI-CAE field and scalar prediction  
**Training constraint:** the geometry generator must be trainable on an in-house geometry/CAE dataset, rather than depending on a closed or billion-parameter foundation model

## Executive decision

Yes: a signed-distance field followed by iso-surface extraction is a complete, trainable geometry-generation route. For the stated objective, the **primary learned geometry path should be a conditional SDF**, with Marching Cubes as the first surface extractor and MeshSDF, DMTet, or FlexiCubes added when gradients through the extracted surface are required. CAD-native generation remains valuable, but mainly as a structured baseline and as the route to clean editable STEP rather than as a prerequisite for learned STL generation.

```text
 conditions c + stochastic latent z
 (loads, BCs, materials, envelope, target performance, sketch/image)
                         |
       conditional flow / cVAE / diffusion / flow matching
                         |
                    shape latent
             +-----------+-----------+
             |                       |
   conditional SDF decoder      AI-CAE decoders
       f(x, z, c) -> distance   pressure/stress/flow + scalars
             |
  adaptive SDF sampling (grid, octree, or tetrahedral grid)
             |
  Marching Cubes baseline; MeshSDF/DMTet/FlexiCubes if differentiable
             |
  repair + remesh + boundary-label transfer -> watertight STL
             |
      Gmsh/solver mesh -> high-fidelity CAE validation
             |
             +---- optional CAD bridge ----+
                                             |
                 faceted STEP or fitted/dual-decoded B-rep STEP
```

The recommended first implementation is therefore:

1. Preprocess the in-house CAD/STL collection into normalized watertight surfaces, near-surface SDF samples, surface points/normals, boundary labels, and CAE targets.
2. Train a **DeepSDF-style auto-decoder or SDF autoencoder** and compare ReLU, Fourier-feature, and SIREN decoders under identical losses.
3. Sample the zero level set with Marching Cubes, repair/remesh it, export STL, and require surface, watertightness, meshing, and solver checks.
4. Learn `p(z|c)` with a cVAE or conditional normalizing flow; advance to conditional diffusion or flow matching when the baseline is established.
5. Couple the same geometry latent to pressure/stress/flow-field and scalar-performance decoders, then use high-fidelity CAE for validation and active learning.
6. Implement STEP separately: faceted STEP if exchange alone is sufficient; surface fitting, a dual CAD/SDF decoder, or CAD-program/B-rep generation if editable engineering STEP is required.
7. Keep a domain-specific CadQuery/OpenCASCADE parameter generator as a high-validity baseline whenever the design family admits a useful template.

This recommendation is intentionally different from starting with a large text/image-to-3D model. TRELLIS, TripoSG, and similar systems are useful architectural references, but they are optimized largely for visual assets and were trained with hundreds of thousands to millions of 3D examples. They do not solve the engineering STEP, boundary-label, manufacturability, or CAE-validity contract by themselves.

## Evidence from industry: Narnia Labs, Neural Concept, Ansys, PhysicsX, Autodesk, and implicit-modeling vendors

Public evidence shows that engineering groups use several geometry routes rather than one universal representation.

| Organization/system | Publicly documented geometry route | What is confirmed | Relevance here |
|---|---|---|---|
| Neural Concept/EPFL, **MeshSDF** | Deep implicit SDF; evaluate the field on a grid; Marching Cubes in the forward pass; analytic implicit-surface gradients in the backward pass | The paper is co-authored by Neural Concept researchers and explicitly demonstrates physically driven shape optimization with changing topology | Direct confirmation that SDF plus Marching Cubes can sit inside a differentiable engineering design loop |
| Neural Concept, early shape optimization | Geodesic CNN surrogate operating on surface meshes, followed by gradient-based movement of shape vertices/parameters | Public ICML work and company explanation | A second valid route when topology is fixed and an existing mesh is deformed rather than generated from a distribution |
| Neural Concept, current commercial platform | Markets CAD-ready, editable downstream outcomes | The public product material does not disclose whether the production representation is SDF, mesh, parametric CAD, B-rep, or a hybrid | Do not infer that SDF-to-Marching-Cubes alone creates clean editable STEP |
| Narnia Labs/KAIST, **DeepJEB** | DeepSDF-style auto-decoder trained from 263 selected seed brackets -> SDF -> Marching Cubes -> minimum-Jacobian abnormality filtering -> automated FEM labeling under four load cases plus modal analysis | Explicitly described in the public paper; 2,096 of 4,833 generated brackets were discarded by the geometric filter before simulation | The clearest published industry instance of exactly the SDF-plus-Marching-Cubes generation route recommended here, including the mandatory generate-then-filter gate |
| Narnia Labs, **DeepWheel** | Generated image -> depth -> point cloud -> regular grid -> Marching Cubes -> smoothing/decimation/artifact removal -> watertight wheel mesh | Explicitly described in the public paper | Direct industry example of Marching Cubes as the geometry-construction stage, although the gridded field is not described as a learned metric SDF |
| Narnia Labs/KAIST/Hanyang, **DeepJEB++** (2026) | Fine-tuned Stable Diffusion multi-view image augmentation -> VLM quality filtering -> TRELLIS 2D-to-3D lifting -> watertight mesh -> automated boundary-condition recognition and FEA gates | Explicitly described; 22,495 validated 2D designs yielded 15,360 final simulation-labeled 3D brackets, a 40x expansion of the 381-sample SimJEB base | Shows the newer foundation-model route: augment in the data-rich 2D image space and lift to 3D with a pretrained model, replacing the domain-specific implicit auto-decoder used for DeepJEB |
| Narnia Labs patent | Implicit neural representation/auto-decoder; preprocessing may use SDF, mesh, voxel, or point cloud; names DeepSDF, DualSDF, and Deep Implicit Templates; latent exploration is coupled to performance objectives | The patent discloses/claims these options; it does not prove which exact option is deployed in each current product | Strong evidence that conditional implicit fields and physics-aware latent search are part of its technical design space |
| Narnia Labs earlier CAD/CAE workflow | 2D generative design -> latent DOE -> CAD automation -> CAE automation -> transfer learning | Public paper and product material | Demonstrates why an engineering product may combine learned generation, CAD automation, and simulation instead of requiring one representation to do everything |
| nTop | Commercial implicit modeling based on signed-distance fields | Explicitly documented by nTop | Confirms that SDF-native geometry is industrially useful even outside neural generation; CAD interoperability still needs an explicit exchange route |
| Leap71, **PicoGK** | Open-source computational-engineering kernel whose single source of truth is a narrow-band voxel signed-distance field | Explicitly documented and open source | Second industrial confirmation that SDF/voxel fields are production engineering representations, here procedural rather than learned |
| Ansys, **GeomAI** (2026 R1) | Trains a generative model on user-supplied reference geometries, builds a parametric latent design space, generates new concepts from latent values, and closes the loop with SimAI surrogate predictions and latent-space optimization | Product capability and workflow are documented; the underlying representation is not publicly disclosed | A major CAE vendor now offers the latent-generative-geometry-plus-AI-CAE-surrogate loop this document recommends |
| PhysicsX, **LGM-Aero** | 100M-parameter geometry-plus-physics foundation model pretrained on 25M+ meshes (10B+ vertices); generates aero shapes and infers performance, stability, and stress zero-shot; showcased in the Ai.rplane application | Publicly announced with model scale and training-corpus figures | Demonstrates the shared geometry/physics latent direction at foundation-model scale; far beyond in-house training budgets but architecturally aligned with the advanced track here |
| Autodesk Research, **WaLa / Project Bernini** | Wavelet-tree latent of a 256^3 signed-distance grid (about 2,427x compression) with a roughly 1B-parameter diffusion generator; conditions on text, sketch, image, voxels, point clouds, and depth maps | Published paper and released code | SDF grids remain the substrate even at billion-parameter scale; wavelet latents are an alternative to triplanes/latent sets for detail-preserving compression |
| Backflip AI | Purpose-built 3D AI for scan/mesh-to-CAD; outputs editable STEP and native Onshape feature trees | Product capability is documented; its training data and internal geometry representation are not publicly specified on the product page | Commercial evidence that mesh-to-editable-B-rep reverse engineering is viable but is its own product-scale problem, consistent with keeping clean STEP off the implicit critical path |
| Luminary Cloud / NVIDIA | Physics AI surrogates (PhysicsNeMo, DoMINO, SHIFT models) over conventionally parameterized geometry variants | Publicly documented | Not a geometry generator: reinforces that the surrogate and the generator are separable investments |

Six patterns recur across these systems:

1. **SDF plus iso-surfacing is a dominant learned free-form engineering route.** Narnia Labs' DeepJEB, Autodesk's WaLa, and BladeSDF (2026) all learn signed-distance geometry, while MeshSDF and PhysGen explicitly use Marching Cubes-family extraction.
2. **Every serious pipeline gates generation with filtering.** DeepJEB discarded about 43% of its SDF-generated brackets. DeepJEB++ applies filters in 2D, after 3D lifting, during boundary recognition, and before/after FEA. Validity filtering is part of the architecture, not a temporary patch.
3. **Conditioning is usually latent-space search against a surrogate,** not end-to-end physics-conditioned decoding: GeomAI plus SimAI, Neural Concept's latent morphing plus GCNN surrogate, and Narnia's performance-coupled latent exploration all follow this pattern.
4. **Editable STEP is a separate product problem.** Backflip sells reverse engineering as its own product; no vendor claims native clean B-rep output from an implicit generator.
5. **Geometry foundation models are arriving** (LGM-Aero, WaLa, TRELLIS-based lifting in DeepJEB++). The practical way to exploit them with in-house data is 2D-space augmentation or pretrained 3D lifting, not training billion-parameter 3D models from scratch.
6. **SDF-to-B-rep is emerging, but currently faceted.** BR-DF jointly generates a global SDF and per-face UDFs and converts them with extended Marching Cubes into a watertight faceted B-rep, which is directly relevant to STEP transport but not equivalent to analytic/NURBS CAD.

The closest public research blueprint to the requested combined generator and AI-CAE system is **PhysGen (CVPR 2026)**. Its shared latent representation has a shape decoder that predicts an SDF, a surface-pressure decoder, and a global-drag decoder; the final mesh is reconstructed with Marching Cubes. Physics-guided flow matching generates shapes conditioned on an image/sketch and steers them toward target drag. This is almost exactly the desired decomposition: a conditional distribution over geometry latents, an implicit geometry decoder, an STL-capable extraction step, and physics heads trained around the same shape representation. It is an EPFL research system, not evidence of Neural Concept's current proprietary product internals.

### Important Marching Cubes distinction

Classic Marching Cubes is perfectly suitable for producing the first STL baseline. Its discrete case selection is not, by itself, a convenient differentiable layer. Use one of three modes:

1. **Generation only:** SDF -> dense/adaptive grid -> Marching Cubes -> mesh repair. Train through sampled SDF losses; no gradient through extraction is needed.
2. **Differentiable mesh objective:** use MeshSDF's implicit-surface gradient or a differentiable tetrahedral/cubic extractor such as DMTet or FlexiCubes.
3. **Solver-in-the-loop optimization:** use a differentiable AI-CAE surrogate for most updates, but periodically regenerate the mesh and validate candidates with the actual CAE solver.

## 1. What the generator must actually learn

The desired conditional distribution is

\[
p(G \mid c),
\]

where `G` is geometry and `c` may contain:

- operating conditions: velocity, Reynolds number, temperature, pressure, frequency;
- loads and boundary conditions;
- materials and manufacturing process;
- dimensional limits and packaging envelopes;
- target integral performance: drag, lift, compliance, mass, pressure drop;
- target field summaries or a full desired field;
- part/category/topology identifiers;
- partial geometry, a sketch, control points, point cloud, or existing CAD body.

This is usually **one-to-many**. Multiple valid shapes can meet the same physical target. A deterministic regressor `G = f(c)` is a useful baseline, but it will tend to average valid alternatives and cannot represent conditional diversity. A proper generator must therefore also receive stochastic information `z`:

\[
G = D(z,c), \qquad z \sim p(z\mid c) \text{ or } z \sim \mathcal{N}(0,I).
\]

“Realistic” must be evaluated at four levels:

1. **Geometric:** watertight, non-self-intersecting, correct normals, no zero-thickness features.
2. **CAD:** valid B-rep, successful STEP round trip, editable analytic or spline surfaces, meaningful parts/faces.
3. **CAE:** successful surface/volume meshing, acceptable cell quality, preserved boundary groups, converged solver.
4. **Physical/conditional:** true CAE results—not only a neural surrogate—meet the requested condition within tolerance.

## 2. The essential STEP versus STL distinction

### STL

STL is a tessellated surface. DeepSDF, SIREN-based SDFs, occupancy networks, point clouds, and mesh generators can all eventually produce STL after surface reconstruction or triangulation. It does not retain analytic NURBS surfaces, construction history, semantic face names, material regions, or robust CAD editability.

### STEP (`.stp`, `.step`)

STEP is an ISO 10303 product-data exchange format used between CAD, CAM, CAE, and inspection systems. Engineering STEP commonly carries a boundary-representation solid made from faces, edges, vertices, and their topology. A useful STEP result is not merely a triangle soup stored in a STEP container.

### Consequence

| Generation route | STL | clean STEP B-rep | Assessment |
|---|---:|---:|---|
| Parametric CadQuery/OpenCASCADE model | Native export | Native export | Best structured baseline and clean-STEP path |
| Generated CAD operation program | Native export after execution | Native export after execution | Best when topology/operation sequence varies |
| Direct B-rep generation | Tessellate B-rep | Native in principle | Advanced and data-hungry |
| Conditional SDF / DeepSDF / SIREN | Natural after iso-surfacing | Not natural | Primary learned free-form and STL path; requires a separate clean-STEP strategy |
| BR-DF: global SDF + per-face UDFs | Natural after iso-surfacing | Watertight faceted B-rep, not analytic CAD | Most direct learned SDF-to-STEP research bridge |
| Direct triangle-mesh model | Native | No clean conversion | Good for visual meshes, weaker for engineering CAD |
| Point-cloud generator | Requires reconstruction | Requires difficult reverse engineering | Intermediate representation only |

Converting an SDF-derived mesh into CAD requires surface segmentation, primitive or NURBS fitting, intersection, trimming, and topology recovery. ParSeNet, ComplexGen, and Point2CAD show that this is an entire research pipeline. A faceted STEP file can instead be made by wrapping/sewing triangles as planar B-rep faces, but it is usually large, brittle, and not meaningfully editable. Therefore the exact STEP contract must be stated: exchange-only faceted STEP is feasible after Marching Cubes; clean editable engineering STEP needs reconstruction or a second CAD-native decoder.

### Recommended format hierarchy

Use one authoritative model per generation track and record the conversion lineage. For the implicit track, the SDF is authoritative and the STL/CAE meshes are derived. For the CAD track, the B-rep is authoritative and both STEP and STL are derived.

| Role | Preferred format | Notes |
|---|---|---|
| Implicit geometry source | decoder checkpoint, shape latent, conditions, normalization, and extraction settings | Reproduces the continuous SDF and its derived surfaces |
| CAD source during development | CadQuery program plus OCCT `.brep` | Retains exact kernel topology and is quick to rebuild |
| Neutral CAD exchange | `.step` / `.stp` | Record whether it is faceted exchange geometry or a clean fitted/native B-rep |
| Surface geometry | binary `.stl`, `.ply`, or `.obj` | Derive from SDF extraction or B-rep tessellation with recorded tolerances |
| Surface/volume CAE mesh | `.msh`, `.vtu`, `.inp`, CGNS, or solver-native | Preserve named boundary/material groups |
| ML training fields | HDF5/Zarr plus explicit mesh connectivity | Avoid making a CAD exchange file carry tensor training data |

IGES can be supported for legacy surface exchange, but STEP should be preferred for solid/product exchange. OBJ/glTF are useful for visualization, not as engineering sources of truth.

## 3. Representation families

### 3.1 Parametric CAD templates and classical geometry bases

Examples include dimensions, NACA/CST airfoil coefficients, B-spline or NURBS control points, Bézier patches, free-form deformation lattices, superquadrics, CSG primitives, and a domain-specific CadQuery program.

**Conditional generator:** MLP, cVAE, normalizing flow, diffusion, or flow matching over the parameter vector.

**Strengths**

- STEP and STL come from the same valid CAD solid.
- Hard constraints can be encoded in the parameterization.
- Low-dimensional learning works with the smallest in-house datasets.
- Every sample is interpretable and easy to optimize.
- Stable face/part semantics can be designed for boundary conditions.

**Weaknesses**

- The template bounds novelty and topology.
- A poor parameterization can make the design manifold unnecessarily difficult.
- Topology changes require discrete template or operation choices.

**Best use:** the first production-capable system for a known component family such as an airfoil, blade, hull, heat sink, bracket, duct, or die.

### 3.2 Fixed-connectivity deformation models

The generator predicts vertex offsets from a common template mesh, optionally using a graph network or spectral mesh autoencoder.

**Strengths**

- The same connectivity and node correspondence simplify AI-CAE training.
- Surface labels and boundary correspondence remain stable.
- Laplacian, edge-length, normal, and strain regularizers are easy to add.

**Weaknesses**

- No arbitrary topology changes.
- Large deformation can invert or tangle elements.
- Direct STEP export still needs a corresponding CAD parameterization or surface fit.

**Best use:** advanced baseline where all parts share topology and the existing CAE mesh can be morphed safely.

### 3.3 Point clouds

PointFlow models shape and point distributions using continuous normalizing flows; point-cloud diffusion directly denoises sets of points. Point clouds are easy to obtain from CAD and are permutation invariant, but they carry neither topology nor a watertight solid.

**Best use:** encoders, completion conditions, comparison metrics, or an intermediate latent—not the final engineering artifact.

### 3.4 Direct mesh generation

- **AtlasNet** decodes a shape as a set of learned parametric surface patches.
- **PolyGen** autoregressively predicts vertices and faces and supports conditioning.
- **MeshGPT** learns discrete local mesh tokens and then autoregressively generates triangle meshes.

These models avoid iso-surfacing and can preserve sharp features better than a dense SDF extraction. However, visual mesh validity does not imply watertightness, manifoldness, consistent CAD topology, or a mesh appropriate for a numerical solver. Autoregressive face generation also becomes expensive for high-resolution CAE surfaces.

**Best use:** compact visual/graphics meshes or low-face-count shape proposals. It is not the recommended primary route to STEP or a high-quality CAE volume mesh.

### 3.5 Occupancy networks

An occupancy network predicts whether a query point is inside a solid:

\[
o_\theta(x,z,c) \in [0,1].
\]

It naturally supports arbitrary topology and continuous queries, and it avoids needing metric SDF targets. The disadvantages for engineering are weaker surface-normal and distance information, threshold sensitivity, and the need for iso-surface extraction.

**Best use:** a simpler implicit baseline when inside/outside labels are much easier to obtain than reliable signed distances.

### 3.6 SDF neural fields: DeepSDF family

A conditional SDF decoder represents

\[
s = f_\theta(x,z,c),
\]

where the surface is `f = 0`, the sign denotes inside/outside, and the magnitude approximates distance to the surface.

DeepSDF is an **auto-decoder**: it learns a shared decoder and a separate latent code for every training shape. It is a strong continuous shape representation, but it is not by itself a complete random or conditional generator. To generate new engineering designs reliably, it needs a learned conditional prior over shape codes:

\[
z \sim p_\phi(z\mid c).
\]

Suitable choices for `p(z|c)` are a cVAE prior, conditional normalizing flow, diffusion model, or flow-matching model. Sampling a naive Gaussian around optimized DeepSDF codes is not a sufficient conditional-generation design.

**Strengths**

- continuous resolution and arbitrary topology;
- normals from `∇f` and surface distance for geometry-aware AI-CAE;
- smooth latent interpolation;
- compact global shape code for moderate shape families;
- direct compatibility with GINO-like SDF geometry inputs.

**Weaknesses**

- expensive query sampling and surface extraction;
- sign computation assumes a closed/oriented solid or robust winding-number preprocessing;
- thin gaps and small holes can disappear;
- global latent codes struggle with detailed, spatially heterogeneous geometry;
- output is a field/mesh, not a native CAD B-rep.

**Training losses**

- clamped or robust SDF regression;
- surface loss `|f(x_surface)|`;
- Eikonal loss `(||∇_x f|| - 1)^2`;
- normal alignment between normalized `∇f` and CAD normals;
- off-surface sign loss;
- optional curvature, thickness, or topology penalties.

### 3.7 SIREN

SIREN is a coordinate-network architecture using sinusoidal activations. It can represent high-frequency signals and their derivatives and has a carefully designed initialization. In this project it can serve as:

- the activation backbone of `f(x,z,c)`;
- a per-shape neural field whose weights are produced by a hypernetwork;
- a high-frequency residual on top of a smoother base SDF;
- a differentiable geometry or physics field.

SIREN is **not a generative distribution**. It still needs a conditional latent model or hypernetwork. It also introduces a frequency/initialization tradeoff: too little frequency smooths fine features; too much can create oscillation and spurious zero-level surfaces.

The correct experiment is an ablation with the same decoder width, depth, latent size, query samples, and losses:

1. ReLU MLP;
2. ReLU MLP with Fourier-feature coordinates;
3. SIREN with its prescribed initialization;
4. optional multiresolution/local feature decoder.

Judge them by surface error, normal error, small-feature recall, meshability, and true CAE sensitivity—not reconstruction Chamfer alone.

### 3.8 Local and structured neural fields

A single global code can bottleneck fine geometry. Convolutional Occupancy Networks attach local features to planes or volumes. 3DShape2VecSet represents a neural field using a set of latent vectors suitable for transformer diffusion. Modern systems also use triplanes, sparse voxel grids, octrees, or latent token sets.

**Strengths**

- much better local detail and scaling than one global DeepSDF vector;
- naturally compatible with diffusion or flow transformers;
- local condition injection and editing;
- high-resolution sparse representations can avoid dense 3D grids.

**Weaknesses**

- higher memory and data requirements;
- latent ordering, locality, and equivariance need careful design;
- still requires surface extraction and a separate STEP strategy.

**Best use:** the advanced free-form model after a global conditional SDF baseline is validated.

### 3.9 Hybrid implicit-explicit extraction

- **Marching Cubes** is the correct first extractor: evaluate the learned scalar field on a regular or adaptive grid and triangulate its zero level set. It is mature, easy to validate, and directly produces an STL-capable surface.
- **MeshSDF** uses an ordinary iso-surface extractor in the forward pass but supplies an implicit-surface gradient from mesh vertices back to the deep SDF. It is the most direct upgrade when an existing SDF decoder needs mesh-level physics or rendering gradients.
- **DMTet** stores SDF values on a deformable tetrahedral grid and uses differentiable marching tetrahedra to produce a surface mesh.
- **FlexiCubes** adds degrees of freedom designed for gradient-based, feature-preserving mesh optimization and can be used as a differentiable extraction layer.

Use plain Marching Cubes for reconstruction, generation benchmarks, STL export, and solver preprocessing. Use MeshSDF when the SDF itself should receive gradients from an extracted mesh. Use DMTet/FlexiCubes when the extraction lattice should also be optimized. These differentiable routes are particularly useful with a physics surrogate: losses can be applied to the extracted mesh while the representation retains topology-changing behavior.

Extraction resolution fixes the smallest recoverable feature, so thin walls and gaps need adaptive refinement, conservative bounds, and minimum-thickness constraints. Every extracted mesh still needs connected-component, watertightness, manifoldness, orientation, self-intersection, feature-retention, and remeshing checks. None of these extractors automatically makes a clean analytic CAD-quality STEP solid.

### 3.10 Density/voxel fields and topology optimization

For structural topology generation, a conditional image/voxel model can generate material density under loads, supports, volume fraction, and manufacturing conditions. TopologyGAN conditions on physical fields; TopoDiff adds conditional diffusion and surrogate guidance and reported substantially fewer infeasible samples than its GAN comparison.

**Best use:** topology optimization or material layout. A subsequent boundary extraction, smoothing, CAD reconstruction, and validation pipeline remains necessary for a manufactured STEP part.

### 3.11 CAD program generation

Instead of predicting the final surface, the model generates an executable sequence such as sketch, dimension, extrude, revolve, sweep, fillet, chamfer, Boolean, and pattern.

- **DeepCAD** uses a transformer over CAD operation sequences and introduced 178,238 construction sequences.
- **Fusion 360 Gallery** supplies 8,625 human sketch-and-extrude sequences.
- **CAD-Recode** shows a newer code-generation direction, although its 1.5B pretrained decoder and one-million-program dataset are much larger than the recommended initial system.

**Strengths**

- exact STEP/STL through a CAD kernel;
- editable, interpretable, manufacturable representation;
- hard validity checks during execution;
- construction parameters can become optimization variables.

**Weaknesses**

- sequences are discrete, ordered, and error-propagating;
- many programs can describe the same solid;
- one bad command can invalidate the rest of a sample;
- operation vocabulary limits expressiveness;
- realistic training sequences are harder to collect than final meshes.

**Best use:** the advanced STEP-native path when a fixed template is too restrictive.

### 3.12 Direct B-rep generation

- **SolidGen** autoregressively generates vertices, edges, and faces using an indexed B-rep representation and can be conditioned on class, images, or voxels.
- **BrepGen** uses hierarchical structured latent geometry and transformer diffusion to directly generate B-rep topology and geometry, including curved surfaces.
- **HoLa** compresses B-rep surface, curve, and topology information into one holistic latent and generates with a single VAE plus a single diffusion model, reporting roughly 82-84% valid unconditional B-reps versus about 50% for BrepGen, with point-cloud, image, sketch, and text conditioning.
- **BR-DF** takes a hybrid route: one global SDF encodes the solid surface, per-face unsigned distance fields encode B-rep faces/edges/vertices and topology, and an extended Marching Cubes algorithm reconstructs a watertight faceted B-rep. Its multi-branch 3D latent diffusion reportedly achieved a 100% faceted-B-rep production success rate. This is highly relevant to mandatory STEP output, but its output is explicitly faceted rather than clean analytic/NURBS CAD.

This is the most direct research route to STEP, but it is one of the hardest models to train. Valid topology, face-edge incidence, trims, parameter-space curves, tolerances, and CAD-kernel sewing must all agree.

**Best use:** a later research track after conditional template/program generation works and enough B-rep training data have been accumulated.

## 4. Conditional generative model choices

The geometry representation and generative distribution are separate design axes. DeepSDF/SIREN answer “how is one shape represented?”; cVAE/flow/diffusion answer “how are diverse shapes sampled for a condition?”

| Conditional model | Diversity | Training/data cost | Sampling | Main strength | Main risk | Recommendation |
|---|---:|---:|---:|---|---|---|
| Deterministic MLP | None | Lowest | One pass | Establishes achievable condition accuracy | Averages one-to-many designs | Required baseline only |
| cVAE | Moderate | Low | One pass | Simple, stable, self-trainable | KL collapse, blurry/mean shapes, prior-posterior gap | First probabilistic baseline |
| Conditional GAN/WGAN | Potentially high | Medium | One pass | Fast sampling and sharp outputs | Mode collapse, unstable training, no likelihood | Secondary comparison, not default |
| Conditional normalizing flow | High on compact latents | Low-medium | One/few passes | Explicit conditional likelihood, invertible latent map | Architectural constraints; costly in very high dimensions | Strong first choice for CAD parameters or SDF latents |
| Conditional diffusion | High | Medium-high | Many denoising steps | Robust multimodal modeling and flexible guidance | More data/compute; slower sampling | Advanced default |
| Flow matching / rectified flow | High | Medium-high | ODE steps | Modern scalable continuous generator; potentially straighter paths | Still sensitive to latent quality and solver steps | SOTA-oriented successor to latent diffusion |
| Autoregressive transformer | High | High | Sequential | Natural for CAD commands/topology tokens | Slow, invalid prefixes, ordering dependence | CAD-program/B-rep research path |

### Conditioning mechanisms

Use the mechanism that matches the condition type:

- **small numeric vector:** normalized MLP condition encoder plus FiLM/AdaLN modulation in every block;
- **spatial mask/load field:** convolutional or neural-operator encoder plus local concatenation;
- **point cloud/partial surface:** PointNet/GNN/set-transformer encoder plus cross-attention;
- **mesh/graph boundary conditions:** graph encoder and cross-attention or pooled condition tokens;
- **mixed conditions:** convert each modality to tokens with type embeddings and use cross-attention.

Concatenating `c` only once at the network input is a useful baseline, but deep decoders often learn to ignore it. Repeated FiLM/AdaLN or cross-attention makes conditioning harder to discard.

For diffusion/flow models, randomly drop conditions during training to support classifier-free guidance. At inference, condition strength must be swept: aggressive guidance can improve target alignment while reducing diversity and moving samples off the learned geometry manifold.

### Physics and constraint guidance

Four levels are possible, in increasing order of integration:

1. **Generate then filter:** validate CAD, mesh, surrogate, and selected candidates with CAE.
2. **Auxiliary consistency loss:** a frozen surrogate predicts performance from generated training shapes; penalize target mismatch during generator training.
3. **Guided sampling:** use gradients from a feasibility/performance surrogate during latent diffusion or flow sampling.
4. **Differentiable end-to-end:** propagate physics objectives through DMTet/FlexiCubes and a differentiable solver or neural operator.

The surrogate can be exploited by the generator. Every guided system therefore needs periodic high-fidelity CAE verification and active learning around generated candidates.

## 5. Basic, advanced, and SOTA-oriented techniques

### Basic: fastest route to a working engineering pipeline

#### B0. Design-of-experiments plus optimizer baseline

Use bounded CAD parameters, Latin hypercube/Sobol sampling, a CAE surrogate, and Bayesian/evolutionary/gradient optimization. This is not a learned generator, but it establishes what a generative model must beat in conditional accuracy, diversity, and sample efficiency.

#### B1. Global SDF reconstruction plus Marching Cubes

Train `f(x,z)` as a DeepSDF auto-decoder or use an encoder to infer `z`. Extract the zero level set on a regular grid with Marching Cubes, then repair, orient, remesh, export STL, and run the actual CAE mesher. Compare a ReLU MLP, Fourier features, and SIREN at equal capacity.

This isolates the first essential question: can the learned representation reproduce engineering geometry, including thin and sharp features, well enough to create valid STL and CAE meshes?

#### B2. Conditional distribution over SDF latents

Fit `p(z|c)` using a cVAE and conditional normalizing-flow baseline. Conditions can contain loads, BCs, material, envelope, target performance, category, and partial geometry. This is what turns an SDF representation into a one-to-many conditional generator.

BladeSDF (2026) is a published validation of exactly this recipe on turbine blades: a DeepSDF-style latent with near-Gaussian structure, unconditional generation by latent sampling and interpolation, and conditional generation by a compact network that maps engineering descriptors (for example, maximum directional strains) to latent codes, with reported surface errors within about 1% of the maximum blade dimension.

#### B3. Parametric CAD control baseline

When a useful component template exists, train a conditional cVAE/flow over bounded CAD parameters and export through CadQuery/OpenCASCADE. This supplies a high-validity clean-STEP baseline against which the more expressive SDF generator must be compared.

#### B4. AI-CAE surrogate and true-solver validation

Train field and scalar prediction from the SDF, extracted mesh, or a shared latent. Use the surrogate for candidate ranking, but periodically remesh and run the real solver; never equate surrogate agreement with validated physics.

### Advanced: recommended research progression

#### A1. Differentiable SDF-to-mesh extraction

Replace the final-only Marching Cubes step with MeshSDF-style implicit-surface gradients, DMTet, or FlexiCubes when losses must propagate from mesh geometry or an AI-CAE surface model back into the generator.

#### A2. Local SDF autoencoder plus conditional flow matching

Encode CAD-derived surfaces into a triplane, sparse grid, octree, or latent set. Freeze a validated decoder. Train conditional diffusion or flow matching in latent space, injecting numerical conditions through FiLM/AdaLN and multimodal/set conditions through cross-attention.

#### A3. Shared geometry-and-physics latent

Following the PhysGen pattern, use one encoded shape latent with:

- an SDF decoder for shape;
- a surface/volume-field decoder for pressure, stress, velocity, temperature, or warpage;
- scalar heads for drag, mass, maximum stress, compliance, and other QoIs.

The generative prior can then be guided toward target physics in the same latent space, subject to real-CAE validation.

#### A4. Dual SDF/CAD decoder

Use a shared stochastic latent with an SDF decoder for free-form geometry and a CAD-parameter/program or structured B-rep decoder for clean STEP. Add surface, normal, and topology-consistency losses between the decoded representations.

#### A5. Surrogate-guided latent sampling and active learning

Use the AI-CAE model to rank or guide large candidate sets. Run true CAE on high-value, uncertain, novel, and near-constraint candidates. Append verified samples and retrain both models.

### SOTA-oriented: valuable, but not the first self-trained deliverable

#### S1. Structured latent flow transformer

3DShape2VecSet, TRELLIS, CraftsMan3D, and TripoSG show the modern pattern:

1. learn a compressed structured 3D representation;
2. train a conditional diffusion/rectified-flow transformer in that latent space;
3. decode high-resolution geometry;
4. optionally refine the extracted surface.

For context, TRELLIS reports models up to 2B parameters trained on 500K assets, while TripoSG describes 2M processed shapes. The reusable idea is **structured latent generation**, not their full scale or their text/image conditioning stack.

#### S2. Direct conditional B-rep diffusion

Adapt BrepGen- or HoLa-style structured B-rep diffusion to numerical engineering conditions, add CAD-kernel validity losses, and preserve semantic faces for CAE. HoLa's holistic latent roughly doubled unconditional B-rep validity over BrepGen (about 82-84% versus 50%), which makes conditional engineering B-rep diffusion more plausible than it was in 2024. This is still the cleanest research path to native STEP, and it still carries the highest data-engineering and topology burden.

#### S3. Conditional CAD-code model with execution feedback

Generate a restricted CadQuery DSL rather than arbitrary Python. Use grammar masks, parameter bounds, kernel execution feedback, and CAE-aware reranking. A small domain-specific transformer trained from scratch is much more plausible than reproducing CAD-Recode's pretrained 1.5B language-model route. Zero-to-CAD (2026) shows the same route scaled by agentic synthesis: a million-scale corpus of interpretable CAD programs generated without real data, useful as pretraining or as a template for building an in-house program dataset.

#### S4. Physics-guided structured-latent flow

Combine a local/structured implicit latent, a flow-matching transformer, shared field/scalar physics decoders, feasibility/manufacturability heads, and uncertainty-aware high-fidelity CAE acquisition. This is the most direct SOTA-oriented extension of PhysGen for detailed 3D engineering parts.

#### S5. Foundation-model 2D-to-3D dataset augmentation

DeepJEB++ demonstrates a data-scaling pattern orthogonal to the generator itself: fine-tune a 2D image diffusion model on multi-view renders of the engineering family, augment and filter in image space, lift the validated images to 3D meshes with a pretrained 3D foundation model such as TRELLIS, and then apply the same geometric and FEA validation gates. This buys dataset diversity from web-scale 2D pretraining without training any large 3D model in-house, and the resulting meshes can feed the SDF pipeline above as additional training shapes.

#### S6. BR-DF latent diffusion for faceted B-rep/STEP

Generate the global solid SDF and per-face UDF channels jointly, then use BR-DF's extended Marching Cubes to recover a watertight faceted B-rep. This is the most direct SDF-family response to a mandatory `.stp` deliverable. It preserves face-level structure better than wrapping an arbitrary triangle soup in STEP, but it remains a faceted B-rep; analytic planes, cylinders, cones, and NURBS patches still require fitting or a native CAD decoder.

## 6. Recommended model architecture

### 6.1 Phase-one learned geometry system

```text
geometry -> SDF/surface encoder or optimized code -> shape latent z
                                                    |
condition c -> conditional prior p(z|c) ------------+
                                                    v
                                SDF decoder f(x,z,c)
                                                    |
                              Marching Cubes + repair
                                                    |
                              watertight STL surface
                                                    |
                     Gmsh/solver mesh + boundary map
                                                    |
                      AI-CAE fields and scalar QoIs
```

This path can be trained entirely in-house. The first milestone is reconstruction and STL/meshing reliability. Conditional generation is then learned over the validated shape latents; it must be evaluated by prior sampling rather than only by encoding held-out target geometries.

### 6.2 Structured CAD and STEP branch

```text
generated SDF -> Marching Cubes mesh -> triangle-face sewing -> faceted STEP

generated global SDF + per-face UDFs -> BR-DF extraction -> faceted B-rep STEP

condition c + noise -> conditional flow/cVAE -> bounded CAD parameters
                                                |
                                         CadQuery/OCCT model
                                                |
                                  native analytic B-rep -> STEP + STL
```

The first route is the quickest exchange-only proof of concept but creates one planar CAD face per retained triangle. BR-DF is the structured SDF-family alternative and can preserve learned face partitions/topology, but still produces a faceted B-rep. Use the CAD branch when clean editable STEP is required or when a bounded component family has a good engineering parameterization. A later hybrid can share a latent between the SDF decoder and CAD decoder, or can fit analytic/free-form surfaces to selected SDF meshes.

Keep these evaluation modes separate:

- **reconstruction:** encode a known test shape and decode it;
- **conditional prior generation:** sample `z ~ p(z|c)` without access to a target shape;
- **interpolation:** vary `z` or `c` continuously;
- **out-of-distribution condition:** report separately and do not infer reliable extrapolation from interpolation results.

Good reconstruction is not evidence that prior-sampled geometry is diverse, valid, or condition-accurate.

### 6.3 AI-CAE choices

| Geometry/mesh contract | Suitable AI-CAE family | Why |
|---|---|---|
| Fixed node correspondence/connectivity | Mesh morphing plus MLP/GNN/MeshGraphNet | Simplest labels and batching |
| Variable-size irregular surface/volume meshes | MeshGraphNet or Transolver | Operates directly on mesh/point nodes; preserve per-graph segmentation |
| SDF plus point-cloud surface queries | GINO | Explicitly designed for varying 3D geometry with SDF/point representations |
| Geometry mapped to regular latent domain | Geo-FNO | Fast spectral operator after learned deformation |
| Small scalar outputs only | Point/GNN/transformer encoder plus MLP | Cheaper than full field prediction |

For variable-size batched meshes, attention and message passing must never mix nodes from different geometries. Segment by graph membership/`ptr` and retain surface/volume/BC node types.

## 7. End-to-end dataset and file pipeline

Each sample should be one synchronized record rather than disconnected STEP, STL, and CAE files.

```text
sample_id/
  design.json             # source parameters/program, latent seed, units
  conditions.json         # loads, BCs, materials, operating point, targets
  geometry.step           # optional native, fitted, or explicitly marked faceted STEP
  geometry.stl            # controlled SDF extraction or CAD tessellation
  surface.msh             # optional surface mesh
  volume.msh              # optional volume mesh
  boundary_map.json       # semantic regions to CAD faces/mesh entity groups
  sdf_samples.npz         # x, signed distance, normal, sampling class
  cae_fields.h5           # mesh coordinates/connectivity and physical fields
  metrics.json            # geometry, mesh, solver, and physics checks
  provenance.json         # generator version, kernel version, solver setup
```

### Generation sequence

1. Import each source CAD solid or watertight surface and normalize units, orientation, scale, and domain coordinates without losing the inverse transform.
2. Assign semantic boundary/material regions and retain their mapping independently of STL.
3. Sample near-surface, on-surface, and uniform/far-field SDF points, surface normals, and optional salient-feature points.
4. Train and validate the SDF representation before fitting a conditional latent prior.
5. Generate the zero level set on a recorded grid/octree/tetrahedral resolution, then repair, orient, simplify/remesh, and export STL.
6. Check closedness, manifoldness, self-intersection, volume, minimum thickness, feature retention, and domain-specific constraints.
7. Generate the surface/volume CAE mesh; transfer boundary groups; run CAE; retain successful and failed/convergence metadata.
8. Train the AI-CAE model with consistent geometry, condition, mesh, and field normalization.
9. If STEP is required, create and validate the selected STEP tier: native/dual-decoded B-rep, reconstructed fitted B-rep, or clearly identified faceted STEP.
10. Retain a CAD-template generation branch where it provides useful high-validity synthetic data or a production-safe fallback.

Do not silently delete every invalid or non-converged generated design. Store the failure class. It can train feasibility models and reveal where the generator or mesher is extrapolating.

### Boundary-condition preservation

STL normally discards CAD face semantics. Preserve them separately or use STEP face/part information as the source of truth. Gmsh physical groups are the appropriate bridge to mathematical boundaries, materials, and CAE regions. Because Boolean CAD operations may reorder face identifiers, tag faces by geometric predicates, construction provenance, or persistent naming when available—not by `Face1`, `Face2`, and so on.

## 8. Data sources and self-training scale

### Useful public data

| Dataset | Scale/content | Best use | Limitation for this project |
|---|---|---|---|
| ABC | 1M CAD models with explicit parametric curves/surfaces | B-rep, surface, SDF, feature pretraining | Not paired to the user's CAE task |
| DeepCAD | 178,238 CAD construction sequences | CAD-program generation | Restricted operation language and dataset distribution |
| Fusion 360 Gallery reconstruction | 8,625 human CAD sequences | Small program-synthesis experiments | Primarily sketch/extrude and not physics-paired |
| DeepJEB | 2,138 brackets with STL, STEP, tetrahedral meshes, four static load cases, modes, and field/scalar labels | Direct SDF-to-MC-to-FEA reference and structural surrogate benchmark | One bracket family; many generated candidates failed filtering |
| DeepJEB++ | 15,360 simulation-labeled brackets generated from fewer than 400 seeds | Foundation-assisted data augmentation and BC-aware FEA automation | Uses pretrained 2D/3D foundation models rather than fully self-trained geometry generation |
| DrivAerNet++ | 8,000 parametric cars with high-fidelity CFD and multiple representations | Geometry + CFD + surrogate benchmark | Automotive-specific and very large storage |
| AirfRANS | 1,000 NACA-family RANS simulations | 2D airfoil AI-CAE baseline | Narrow parameter family and 2D |

### Indicative in-house scale

These are planning ranges, not universal thresholds. Geometry-only samples and expensive CAE-labeled samples should be counted separately:

- **200-2K:** narrow-family proof of concept for global DeepSDF, fixed-topology deformation, or descriptor-to-latent regression; expect limited extrapolation and aggressive validity filtering;
- **2K-20K:** credible domain-specific global SDF autoencoder plus cVAE/flow, bounded parametric generation, and shared-latent AI-CAE;
- **20K-200K:** local/structured latent diffusion with meaningful diversity;
- **50K-200K+:** constrained CAD-operation transformer;
- **hundreds of thousands+:** broad B-rep or large structured-latent generation.

DeepJEB is important evidence for self-training: it learned its implicit generator from only 263 selected bracket seeds, generated 4,833 candidates, rejected 2,096 with a geometry-quality filter, and then built a 2,138-sample released CAE dataset after downstream processing. PhysGen used 5,819 DrivAerNet++ training shapes for its SDF/physics latent. These examples show that a domain-specific global SDF model is feasible at thousands, or even hundreds, of shapes; they also show that high rejection rates, restricted domains, or multi-GPU training may be the price.

The strongest in-house strategy is staged: pretrain geometry on every usable STL/STEP shape, label only a diverse subset with expensive CAE, freeze the validated geometry decoder, train the conditional latent prior and physics heads, and use active learning to choose new true-solver cases. A procedural CAD generator is useful for initial valid geometry, but it is not required for the learned SDF branch.

### Dataset splits

Randomly splitting near-duplicate parameter samples exaggerates generalization. Report:

- interpolation split inside the covered parameter/condition range;
- held-out parameter cells or Latin-hypercube regions;
- held-out geometry subfamilies/topologies;
- held-out operating conditions;
- combined geometry-and-condition shift;
- duplicates grouped by source CAD or construction lineage.

## 9. Losses and constraints

### Geometry reconstruction

- CAD parameter L1/L2 with per-parameter scaling;
- surface Chamfer and point-to-surface distance;
- bidirectional Hausdorff or high-percentile surface error;
- normal consistency;
- SDF regression and Eikonal regularization;
- mesh edge, Laplacian, dihedral, aspect, and inversion penalties;
- B-rep incidence/topology validity for CAD models.

### Conditional generation

- cVAE evidence lower bound or flow likelihood;
- diffusion/flow-matching denoising/vector-field objective;
- condition-consistency loss from true labels or a frozen surrogate;
- diversity regularization within the same condition;
- validity and manufacturability penalties;
- optional performance/diversity objective such as PaDGAN-style augmentation.

### Physics

- normalized field loss with per-variable weighting;
- gradient/flux or conservation residuals where meaningful;
- surface-integrated force/moment loss;
- scalar QoI loss;
- boundary-condition satisfaction;
- uncertainty calibration for candidate selection.

Avoid optimizing only a surrogate objective. Use an ensemble or calibrated uncertainty estimate, impose a latent/design-space trust region, and periodically replace surrogate scores with high-fidelity CAE labels.

## 10. Evaluation gates

### Gate 1: CAD and file validity

- CAD-kernel solid-validity pass rate;
- STEP export and independent re-import pass rate;
- volume/area/bounds agreement after STEP round trip;
- STL watertight/manifold/orientation checks;
- no self-intersection or minimum-thickness violation;
- correct semantic boundary groups.

### Gate 2: meshing and solver validity

- surface/volume mesh success rate;
- min/max cell-quality statistics and negative-volume count;
- stable boundary groups after meshing;
- solver convergence rate and residual criteria;
- runtime and memory distribution, not only averages.

### Gate 3: geometry fidelity

- Chamfer plus Hausdorff/high-percentile distance;
- normal and curvature error;
- topology statistics: components, holes/genus, feature counts;
- sharp-feature and thin-feature preservation;
- reconstruction and prior generation reported separately.

### Gate 4: conditional quality

- true CAE target error for each requested condition;
- percentage within engineering tolerance;
- per-condition diversity and coverage;
- novelty versus nearest training shape;
- Pareto hypervolume/coverage for multiobjective generation;
- calibration: does uncertainty predict physical error or invalidity?

### Gate 5: usefulness against baselines

Compare the generator against:

- random/DOE sampling;
- direct optimization over CAD parameters;
- deterministic inverse regressor;
- nearest-neighbor retrieval;
- cVAE and conditional flow;
- diffusion only after the lower-cost baselines are established.

## 11. Experiment roadmap

### Concrete starter configurations

These are deliberately modest benchmark configurations, not fixed hyperparameter prescriptions.

**CAD-parameter conditional flow**

- standardize continuous conditions and represent categorical topology/family choices with embeddings;
- condition encoder: 3-4 layer MLP, width 128-256;
- latent/design vector: the native CAD parameter dimension, or a 16-64 dimensional auxiliary latent;
- 8-12 affine or spline coupling transforms with condition modulation;
- bounded output transforms for dimensions, angles, radii, thicknesses, and clearances;
- compare likelihood, conditional target accuracy, validity, and diversity against a similarly sized cVAE.

**Global conditional SDF**

- geometry latent: 64-256 dimensions;
- decoder: 6-8 layers, width 256-512, with a middle skip connection;
- compare raw coordinates, 6-10 Fourier-frequency bands, and SIREN initialization under the same parameter budget;
- draw most samples near the surface, while retaining uniform interior/exterior and far-field samples;
- begin with a frozen geometry decoder and train `p(z|c)` separately before any joint fine-tuning.

**Conditional latent diffusion/flow matching**

- use it only after the autoencoder's held-out reconstruction passes meshability gates;
- diffuse compact global latents first, then move to latent sets/triplanes only if detail requires it;
- inject numeric conditions with FiLM/AdaLN at every block and reserve cross-attention for set, field, graph, or multimodal conditions;
- use condition dropout for guidance and report a sweep of guidance strength versus diversity;
- keep an unconditional latent-prior baseline to quantify the actual value of conditioning.

### Experiment 0: establish the artifact and CAE contract

- Select one geometry family.
- Define the geometry normalization, STL quality, STEP tier, boundary-label, and solver contracts.
- Convert representative training geometry to reliable signed-distance samples.
- Build the SDF-to-Marching-Cubes-to-repair-to-CAE-mesh pipeline.
- Run the CAE and AI-CAE preprocessing end to end.

**Exit criterion:** reference geometries round-trip through SDF sampling and surface extraction within agreed geometric tolerances; their derived meshes solve successfully; all failure categories are recorded.

### Experiment 1: DeepSDF representation and explicit extraction

- global latent auto-decoder;
- balanced SDF sampling from exact CAD solids;
- ReLU, Fourier-feature, and SIREN decoder ablation;
- dense-grid Marching Cubes baseline with repair and remeshing.

**Exit criterion:** selected decoder meets surface, normal, small-feature, and meshing tolerances on held-out CAD.

### Experiment 2: conditional latent generation

- fit `p(z|c)` with a Gaussian cVAE prior and conditional normalizing flow;
- only then add latent diffusion or flow matching;
- measure prior-sampled outputs, not posterior reconstructions.

**Exit criterion:** conditional prior samples are diverse, valid, and CAE-accurate under true simulations.

### Experiment 3: shared-latent AI-CAE coupling

- train the irregular-mesh or SDF-aware AI-CAE surrogate;
- compare geometry encodings: CAD parameters, mesh/point graph, and SDF/local features;
- compare separate geometry/physics encoders against a PhysGen-style shared geometry latent with field and scalar heads;
- add surrogate reranking, then mild gradient guidance;
- run high-fidelity CAE on uncertain and high-value candidates.

**Exit criterion:** generated candidates improve the validated design objective or dataset coverage versus DOE plus direct optimization at a fixed CAE budget.

### Experiment 4: differentiable extraction and physics-guided generation

- replace ordinary extraction with MeshSDF-style gradients, DMTet, or FlexiCubes only if mesh-level objectives require it;
- add physics-guided flow matching or surrogate gradients inside a trust region;
- periodically remesh and validate with the real solver to detect surrogate exploitation.

**Exit criterion:** differentiable/physics guidance improves true-CAE performance at a fixed validation budget without reducing geometric or meshing validity.

### Experiment 5: STEP and structured-CAD options

Choose one based on domain need:

- faceted STEP export if the requirement is only exchange/import;
- parametric CadQuery/OCCT generation if the geometry family is bounded and editability is required;
- constrained CAD-program transformer if construction logic matters;
- dual CAD/SDF decoder if free-form detail and STEP are both required;
- direct B-rep diffusion only if data scale and research time justify it.

## 12. What should and should not be built first

### Build first

1. SDF dataset preprocessing plus Marching Cubes, STL repair, meshing, and validation pipeline.
2. Global DeepSDF/autoencoder baseline with ReLU, Fourier-feature, and SIREN ablation.
3. Conditional cVAE and normalizing-flow baselines over the validated geometry latent.
4. AI-CAE field/scalar heads plus high-fidelity active-validation loop.
5. Explicit STEP route selected by whether faceted exchange or editable B-rep is actually required.
6. Parametric CadQuery/OpenCASCADE baseline when the component family admits one.

### Build after evidence supports it

1. Local latent field plus conditional diffusion/flow matching.
2. MeshSDF, DMTet, or FlexiCubes differentiable extraction.
3. PhysGen-style shared latent and physics-guided generation.
4. Shared CAD/SDF latent, mesh-to-B-rep reconstruction, or conditional program/B-rep generation.

### Do not make these initial assumptions

- DeepSDF alone is a random or conditional generator.
- SIREN alone supplies a distribution over geometries.
- Low Chamfer distance guarantees a valid STEP solid or CAE mesh.
- Saving triangles in a STEP-compatible container creates editable CAD.
- A surrogate-optimized geometry will retain its predicted performance under true CAE.
- Posterior reconstruction quality demonstrates conditional prior-sampling quality.
- A visually realistic foundation-model mesh is engineering-realistic.

## 13. Final ranking for the stated goal

| Rank | Technique | STEP | STL | Conditional | Self-trainability | AI-CAE fit | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Conditional global SDF + cVAE/flow + Marching Cubes | Faceted STEP possible; clean STEP needs a branch | Excellent | Excellent | Excellent | Excellent | Start here for learned free-form geometry |
| 2 | Shared SDF/physics latent + conditional flow matching | Faceted STEP possible; clean STEP needs a branch | Excellent | Excellent | Good-medium | Excellent | Best advanced blueprint; PhysGen-like |
| 3 | Conditional flow/cVAE over parametric CAD | Excellent | Excellent | Excellent | Excellent | Excellent | Best structured baseline and clean-STEP route |
| 4 | BR-DF SDF/UDF latent diffusion | Excellent faceted B-rep; not analytic CAD | Excellent | Extendable | Medium-difficult | Good | SOTA SDF-to-STEP bridge |
| 5 | Local/structured SDF latent + diffusion/flow matching | Faceted STEP possible; clean STEP needs a branch | Excellent | Excellent | Medium | Excellent | SOTA-track for fine detail and broader topology |
| 6 | Fixed-template mesh deformation | Separate CAD path | Excellent | Excellent | Excellent | Excellent | Strong if topology is fixed |
| 7 | Conditional CAD-operation transformer | Excellent | Excellent | Excellent | Medium | Good | Advanced STEP-native track |
| 8 | Direct analytic B-rep diffusion | Potentially excellent | Excellent after tessellation | Good/extendable | Difficult | Good | Long-term clean-STEP research |
| 9 | TopoDiff-style density generation | Requires reconstruction | Good after extraction | Excellent | Good in 2D; harder in 3D | Excellent for TO | Use for topology/material layout |
| 10 | Point/mesh autoregressive or visual asset generation | Poor natively | Good visually | Good-excellent | Medium to poor | Fair-weak | Reference or special-purpose route |

## 14. Named-model audit: LGM-Aero, MeshSDF, PhysGen, and the strongest academic alternatives

### 14.1 The short answer

These three names describe different layers of a system and should not be placed on one undifferentiated leaderboard:

| Name | What it actually is | Generates a distribution of new shapes? | Physics-aware? | Publicly trainable? | Native output |
|---|---|---:|---:|---:|---|
| **LGM-Aero** | A proprietary 100M-parameter geometry foundation VAE-like model with a mesh encoder, 512-dimensional latent, implicit distance decoder, and downstream physics models | Yes | Yes | No public code, weights, data, paper, or benchmark | Implicit surface/mesh; no public native STEP claim |
| **MeshSDF** | A differentiable SDF-to-mesh extraction/back-propagation method | No; it needs a separate SDF generator | It lets mesh/rendering/physics losses update an SDF | Yes, code is public | Triangle mesh from an SDF |
| **PhysGen** | An open shape-and-physics VAE plus physics-guided rectified-flow generator | Yes | Yes: SDF shape, pressure field, and drag share a latent | Yes; code and weights are public, although full retraining is compute-heavy | Marching-Cubes triangle mesh |

**Best published academic match to this project:** **PhysGen**. It is the only one of the three that is simultaneously a conditional generative model, explicitly coupled to learned physics, documented in a peer-reviewed paper, and released for reproduction.

**Strongest closed industrial system by publicly claimed scale:** **LGM-Aero**. This is not the correct implementation starting point because its claims are not independently benchmarked and reproducing its pretraining would require an industrial-scale corpus and compute budget.

**Best role for MeshSDF:** an optional differentiable bridge inside an SDF generator/AI-CAE loop. MeshSDF cannot replace DeepSDF, a VAE, a flow model, or PhysGen because it does not learn the probability distribution over designs.

### 14.2 What is LGM-Aero?

PhysicsX describes LGM-Aero as a **100M-parameter geometry and physics foundation model** trained on more than **25 million watertight surface meshes**. The public architecture is:

1. A diffusion-based neural-operator mesh encoder maps a triangle mesh to a distribution over a **512-dimensional latent**.
2. A modulated residual decoder receives the latent and a 3D query point and predicts distance to the surface plus inside/outside information. Its zero level set defines the geometry. Functionally, this is a large amortized **implicit-SDF VAE-like model**.
3. The latent can feed Gaussian processes or other regressors for scalar engineering quantities. The coordinate decoder can also be fine-tuned to predict continuous fields such as surface pressure or surrounding flow.
4. Novel geometry is produced by sampling or optimizing the latent and extracting its implicit surface.

The training scale explains why it should be treated as a foundation-model reference rather than an in-house baseline. PhysicsX reports a maximum of 100,000 vertices per preprocessed mesh and training for four weeks on 128 H100 GPUs followed by four weeks on 64 A100 GPUs. Its launch material also reports tens of thousands of CFD and FEA simulations and predictions of lift, drag, stability, and structural stress.

Important qualifications:

- These figures and performance statements are **company-reported**. The December 2024 technical post promised a paper, but no public LGM-Aero paper, code, weights, training set, or standardized independent benchmark was found as of July 17, 2026.
- Its implicit decoder is highly relevant to this project, but it does not publicly establish clean analytic B-rep or parametric STEP generation.
- The transferable idea is not “train another LGM-Aero.” It is: pretrain an SDF geometry latent, attach physics heads, use the latent for regression/active learning, and later scale across domains.

### 14.3 What is MeshSDF?

MeshSDF, published at NeurIPS 2020, solves the gradient problem between an implicit field and an explicit triangle mesh:

- **Forward:** sample a deep SDF and extract its zero isosurface with Marching Cubes.
- **Backward:** use the implicit-surface derivative to relate movement of extracted surface vertices to perturbations of the underlying SDF.
- **Result:** a loss computed by a renderer, mesh model, or physical objective can update the SDF network or its latent while permitting topology changes.

This is why the paper could demonstrate both single-view reconstruction and physically driven shape optimization. It is especially relevant when an AI-CAE model consumes surface vertices or a mesh but the geometry generator lives in SDF space.

MeshSDF is **not**:

- a geometry autoencoder;
- a random or conditional generator;
- a physics surrogate;
- a CAD/STEP model;
- an improved SDF activation function.

For a new implementation, compare three extraction options:

| Extractor | Select it when | Main tradeoff |
|---|---|---|
| Marching Cubes | Reconstruction, STL export, and solver preprocessing do not require gradients through extraction | Simplest and most mature; grid resolution limits feature fidelity |
| MeshSDF | An existing continuous SDF decoder must receive gradients from mesh-level losses | Direct fit to DeepSDF-like networks; extraction lattice itself is not learned |
| DMTet or FlexiCubes | End-to-end surface fitting/optimization should also adjust the extraction representation | More modern optimization degrees of freedom; more implementation and memory complexity |

Thus, ordinary Marching Cubes should remain the baseline. Add MeshSDF or FlexiCubes only when an experiment genuinely requires end-to-end mesh-to-SDF gradients.

### 14.4 What is PhysGen?

PhysGen, published at CVPR 2026, is currently the closest open academic blueprint for the requested system. It has two major parts.

**Shape-and-Physics VAE (SP-VAE):**

- The geometry encoder receives both uniformly sampled surface points and salient/high-curvature points.
- A shared latent supports an SDF shape decoder, a continuous surface-pressure decoder, and a scalar drag decoder.
- The SDF zero level set is extracted with Marching Cubes.

**Physics-guided rectified flow:**

- A Diffusion Transformer learns a velocity field that transports noise to valid shape latents.
- It can generate unconditionally or condition on an image through DINOv2 features.
- Sampling alternates learned prior/velocity updates with differentiable physics refinement; a physics regularizer can steer the result toward a target or lower drag.

The released supplement makes the self-training requirement concrete. Its DrivAerNet++ experiment used **5,819 training** and **1,147 test** vehicle designs. The shape VAE trained for roughly two days on four H100 GPUs; the pressure and joint-training stages also used four H100s. This is reproducible research, but not a lightweight single-consumer-GPU job in its published configuration.

PhysGen's limits are equally important:

- It proves the concept mostly on automotive aerodynamics, with an additional structural-optimization demonstration; it is not yet a universal industrial geometry model.
- It outputs an SDF-derived mesh, not clean CAD features or analytic STEP.
- Its shared physics decoders are ideal for generation guidance, but a dedicated high-resolution Transolver, GINO, DoMINO, or MeshGraphNet should still be benchmarked for production CAE prediction.
- Its published image condition should be replaced or extended with the conditions engineering actually uses: dimensions, packaging constraints, material, boundary conditions, loads, inlet state, target quantities, and design-family label.

### 14.5 “Best” depends on the required artifact

There is no honest universal winner, but there are clear winners for individual engineering objectives:

| Objective | Best model/family | Why | Warning |
|---|---|---|---|
| **Conditional geometry + learned CAE, end to end** | **PhysGen** | Published shared geometry/pressure/drag latent plus physics-guided flow; open code/weights | Mesh output; demanding retraining |
| **Self-trained first free-form generator** | **Conditional DeepSDF/SP-VAE + latent flow** | Can train on hundreds to thousands of a domain's shapes; easy STL path; topology can change | This is a composed architecture, not DeepSDF alone |
| **Strongest closed industrial reference** | **LGM-Aero** | Largest public engineering geometry/physics scale claim among the audited systems | Proprietary and not independently reproducible |
| **Clean editable STEP, simple mechanical parts** | **DeepCAD/Text2CAD-style CAD program model** | Executes sketches/extrusions in a CAD kernel, preserving semantic operations and parameters | Restricted operation vocabulary and sequence validity |
| **Direct B-rep, free-form CAD research** | **HoLa**, with BrepGen as an important predecessor | Direct parametric-surface/curve/topology representation; HoLa reports 82.68% unconditional validity versus 47.74% for its filtered BrepGen comparison | Far harder to train and validate than SDF; no guarantee of modeling-feature history |
| **Reliable faceted B-rep/STEP from learned fields** | **BR-DF** | Global SDF plus per-face UDFs; extended Marching Cubes reports 100% faceted B-rep conversion | Faceted B-rep is not analytic CAD/NURBS |
| **Differentiable SDF-to-mesh bridge** | **MeshSDF** | Applies mesh-level loss to a continuous SDF while permitting topology change | Not a generator |
| **Differentiable mesh extraction/optimization** | **FlexiCubes** or **DMTet** | Learnable extraction degrees of freedom improve mesh fidelity and optimization | More complex than ordinary MC |
| **Visually rich general-purpose mesh generation** | **TRELLIS, TripoSG, WaLa, CraftsMan3D** | Strong modern structured/wavelet latents and diffusion/rectified-flow scaling | Visual realism is not CAE validity, watertightness, STEP quality, or manufacturability |
| **Domain-specific performance-to-shape SDF** | **BladeSDF** | Direct recent example of DeepSDF plus engineering-descriptor conditioning for turbine blades | 2026 preprint; narrow domain; much less established than DeepSDF/PhysGen |
| **Data-free constrained neural-field design** | **GINNs** | Learns a family of neural shapes from objectives, geometric/topological constraints, and diversity without a shape dataset | Research-stage optimization, not a general learned catalog prior |
| **Structural topology/material layout** | **TopoDiff** | Conditional diffusion with surrogate performance/manufacturability guidance | Published primary case is 2D density topology, not production 3D CAD |
| **AI-CFD directly from STL** | **DoMINO** | Explicitly accepts STL and predicts surface/volume aerodynamics with local multiscale geometry encoding | Predictor, not generator; currently domain-specific |
| **General irregular-geometry AI-CAE baseline** | **Transolver** | Strong general-geometry PDE transformer with linear-complexity physics attention | Must be trained and validated for the target solver/data |
| **Small-data varying-geometry neural operator** | **GINO** | Uses point clouds and SDF; published vehicle-pressure result trained with 500 samples | Regular latent grid and operator setup add implementation complexity |
| **Dynamic mesh simulation** | **MeshGraphNets** | Seminal, widely reused mesh-based learned simulator | Message passing can be costly on very large meshes |

### 14.6 The most famous and influential academic models

“Famous” here means historically influential, widely recognized in the research lineage, and important to understand. It is deliberately not presented as a volatile citation-count table.

#### Foundational must-know models

1. **DeepSDF (CVPR 2019)** — the seminal continuous signed-distance auto-decoder for a class of shapes. It remains the most directly relevant foundation for this project.
2. **Occupancy Networks (CVPR 2019)** — the parallel foundational continuous inside/outside representation. Easier labels, but less metric geometric information than a true SDF.
3. **SIREN (NeurIPS 2020)** — the famous periodic-activation coordinate network. It can represent SDFs and their derivatives accurately, but is a backbone/activation choice, not a shape distribution or conditional generator.
4. **MeshSDF (NeurIPS 2020)** — the influential differentiable link between deep SDFs and meshes.
5. **DMTet (NeurIPS 2021)** — the influential hybrid implicit/explicit representation with deformable tetrahedral grid and differentiable extraction.
6. **DeepCAD (ICCV 2021)** — the foundational learned CAD-operation-sequence generator and source of a 178,238-model public CAD-sequence dataset.
7. **GET3D (NeurIPS 2022)** — a landmark direct textured-mesh generator; famous for visual asset generation, less appropriate for CAE geometry.
8. **3DShape2VecSet (SIGGRAPH 2023)** — an influential set-structured neural-field latent designed for transformer diffusion and multiple conditioning modalities.
9. **MeshGraphNets (ICLR 2021 Outstanding Paper)** — the must-know learned mesh simulator on the AI-CAE side.
10. **GINO (NeurIPS 2023) and Transolver (ICML 2024)** — major varying-geometry neural-operator/transformer baselines for modern AI-CAE.

#### Recent SOTA-oriented models to study

- **PhysGen (CVPR 2026):** most relevant recent physics-aware 3D generator.
- **HoLa (SIGGRAPH 2025):** strongest audited direct B-rep generation direction by reported validity.
- **BR-DF (2025):** strongest audited SDF-to-faceted-B-rep bridge by conversion reliability.
- **TRELLIS (CVPR 2025):** major recent scalable structured-3D-latent foundation model, up to 2B parameters.
- **TripoSG (2025):** large-scale SDF-VAE plus rectified-flow mesh generation, trained on a reported two million shapes.
- **WaLa (2024):** one-billion-parameter wavelet-latent SDF generation with strong compression and open code.
- **CraftsMan3D (CVPR 2025):** coarse 3D-native diffusion followed by interactive/automatic geometry refinement.
- **PhysiOpt (SIGGRAPH Asia 2025):** physics optimization of pretrained implicit generative-model latents under material, load, and boundary-condition inputs.
- **BladeSDF (2026):** unusually direct domain-specific conditional DeepSDF example for engineering blades.
- **GINNs (2024/2025):** important alternative when constraints are available but a geometry corpus is not.

Large visual 3D foundation models are useful sources of representation ideas and possibly pretrained encoders, but they should not outrank domain SDF/CAD models for this project. Most optimize visual or geometric similarity, not boundary-condition preservation, meshability, minimum thickness, curvature limits, solver convergence, manufacturability, or verified CAE performance.

### 14.7 Final verdict for this project

If one existing published system must be named, **PhysGen is the best overall academic model for the stated target**. If one foundational representation must be chosen, **DeepSDF/SDF is the best starting point**. If one closed industrial reference must be named, **LGM-Aero is the most ambitious**, but it is not a reproducible choice. **MeshSDF is a useful component, not a contender for best generator.**

The recommended implementation is a smaller, engineering-conditioned PhysGen rather than a literal reproduction:

```text
watertight surface meshes
  -> uniform + sharp-feature point encoder
  -> 256-512D shape latent z
  -> SDF decoder D_sdf(z, x)
  -> adaptive Marching Cubes -> watertight STL

[dimensions, family, material, loads/BCs, target QoIs] + noise
  -> conditional rectified flow in z
  -> diverse candidate latents

z + operating condition
  -> lightweight scalar/field physics heads for generation guidance

extracted surface + operating condition
  -> Transolver/GINO/DoMINO/MeshGraphNet production surrogate
  -> uncertainty gate -> true CFD/FEA -> active-learning update

z or extracted surface
  -> BR-DF faceted B-rep OR CAD-program/HoLa branch
  -> STEP/STP
```

Recommended order:

1. Train a global DeepSDF-style auto-decoder or point-encoder SDF VAE and prove reconstruction, watertight STL export, and CAE meshing.
2. Add a conditional normalizing flow or rectified flow over the learned latent; condition on engineering variables rather than text/image first.
3. Train a separate trusted AI-CAE surrogate and scalar feasibility heads; use them for ranking and guidance, but periodically verify with the real solver.
4. Add PhysGen-style shared pressure/stress and scalar heads after the generator is stable.
5. Add MeshSDF or FlexiCubes only if gradients through surface extraction measurably improve the design loop.
6. Treat STEP as a second output branch: BR-DF for robust faceted STEP, or CAD-program/HoLa-style generation for clean editable CAD.

This hybrid is more trainable than LGM-Aero, more complete than MeshSDF, more engineering-conditionable than visual 3D foundation models, and more realistic about the STL-versus-STEP divide than a single-representation promise.

## 15. Primary sources and implementations

### Neural implicit representations

1. Park et al., [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html), CVPR 2019; [official code](https://github.com/facebookresearch/DeepSDF).
2. Sitzmann et al., [Implicit Neural Representations with Periodic Activation Functions (SIREN)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html), NeurIPS 2020; [official code](https://github.com/vsitzmann/siren).
3. Mescheder et al., [Occupancy Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.html), CVPR 2019.
4. Tancik et al., [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739), NeurIPS 2020.
5. Gropp et al., [Implicit Geometric Regularization for Learning Shapes](https://proceedings.mlr.press/v119/gropp20a.html), ICML 2020.
6. Peng et al., [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618), ECCV 2020.

### Conditional and generative shape models

7. Sohn et al., [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html), NeurIPS 2015.
8. Yang et al., [PointFlow](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_PointFlow_3D_Point_Cloud_Generation_With_Continuous_Normalizing_Flows_ICCV_2019_paper.html), ICCV 2019.
9. Luo and Hu, [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.pdf), CVPR 2021.
10. Zhang et al., [3DShape2VecSet](https://arxiv.org/abs/2301.11445), SIGGRAPH 2023.
11. Rombach et al., [Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html), CVPR 2022.
12. Lipman et al., [Flow Matching for Generative Modeling](https://openreview.net/pdf?id=PqvMRDCJT9t), ICLR 2023.
13. Ho and Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), 2022.

### Mesh and differentiable extraction

14. Groueix et al., [AtlasNet](https://arxiv.org/abs/1802.05384), CVPR 2018.
15. Nash et al., [PolyGen](https://proceedings.mlr.press/v119/nash20a.html), ICML 2020.
16. Siddiqui et al., [MeshGPT](https://openaccess.thecvf.com/content/CVPR2024/html/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.html), CVPR 2024.
17. Shen et al., [Deep Marching Tetrahedra](https://proceedings.neurips.cc/paper/2021/hash/30a237d18c50f563cba4531f1db44acf-Abstract.html), NeurIPS 2021.
18. Shen et al., [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/), SIGGRAPH 2023.
19. Remelli et al., [MeshSDF: Differentiable Iso-Surface Extraction](https://proceedings.neurips.cc/paper/2020/hash/fe40fb944ee700392ed51bfe84dd4e3d-Abstract.html), NeurIPS 2020; [official code](https://github.com/cvlab-epfl/MeshSDF).

### CAD-native generation and reconstruction

20. Wu et al., [DeepCAD](https://www.cs.columbia.edu/cg/deepcad/), ICCV 2021.
21. Jayaraman et al., [SolidGen](https://arxiv.org/abs/2203.13944), 2022.
22. Xu et al., [BrepGen](https://brepgen.github.io/), SIGGRAPH 2024; [official code](https://github.com/samxuxiang/BrepGen).
23. Liu et al., [HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257), ACM Transactions on Graphics (SIGGRAPH) 2025.
24. Sharma et al., [ParSeNet](https://arxiv.org/abs/2003.12181), ECCV 2020.
25. Guo et al., [ComplexGen](https://doi.org/10.1145/3528223.3530078), SIGGRAPH 2022.
26. Liu et al., [Point2CAD](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Point2CAD_Reverse_Engineering_CAD_Models_from_3D_Point_Clouds_CVPR_2024_paper.pdf), CVPR 2024.
27. Rukhovich et al., [CAD-Recode](https://openaccess.thecvf.com/content/ICCV2025/papers/Rukhovich_CAD-Recode_Reverse_Engineering_CAD_Code_from_Point_Clouds_ICCV_2025_paper.pdf), ICCV 2025.
28. [Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at Million-Scale Without Real Data](https://arxiv.org/abs/2604.24479), 2026.
29. Zhang et al., [B-Rep Distance Functions (BR-DF): How to Represent a B-Rep Model by Volumetric Distance Functions?](https://arxiv.org/abs/2511.14870), 2025; [project page](https://zhangfuyang.github.io/brdf/).

### Engineering conditional generation

30. Chen and Ahmed, [PaDGAN](https://decode.mit.edu/assets/papers/2020_chen_padgan.pdf), 2020.
31. Nie et al., [TopologyGAN](https://arxiv.org/abs/2003.04685), 2020.
32. Mazé and Ahmed, [Diffusion Models Beat GANs on Topology Optimization (TopoDiff)](https://arxiv.org/abs/2208.09591), 2022.
33. Graves and Barati Farimani, [Airfoil Diffusion](https://arxiv.org/abs/2408.15898), 2024.
34. Chen et al., [Inverse Design of 2D Airfoils using Conditional Generative Models and Surrogate Log-Likelihoods](https://ideal.umd.edu/papers/paper/jmd-qiuyi-inverse-airfoil), Journal of Mechanical Design 2022.
35. You et al., [PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](https://openaccess.thecvf.com/content/CVPR2026/papers/You_PhysGen_Physically_Grounded_3D_Shape_Generation_for_Industrial_Design_CVPR_2026_paper.pdf), CVPR 2026; [project page and code](https://kasvii.github.io/PhysGen/).
36. Baque et al., [Geodesic Convolutional Shape Optimization](https://proceedings.mlr.press/v80/baque18a.html), ICML 2018.
37. Nair et al., [BladeSDF: Unconditional and Conditional Generative Modeling of Representative Blade Geometries Using Signed Distance Functions](https://arxiv.org/abs/2601.13445), 2026.

### Public industrial and company evidence

38. Hong et al., [DeepJEB: 3D Deep Learning-Based Synthetic Jet Engine Bracket Dataset](https://arxiv.org/abs/2406.09047), 2024; [dataset page](https://dataset.narnia.ai/deepjeb.html). DeepSDF-style auto-decoder plus Marching Cubes plus automated FEM labeling.
39. Yoo and Kang, [DeepWheel: Generating a 3D Synthetic Wheel Dataset for Design and Performance Evaluation](https://arxiv.org/abs/2504.11347), 2025.
40. Yoo et al., [DeepJEB++: Foundation Model-Driven Large-Scale 3D Engineering Dataset via 2D Latent Space Augmentation](https://arxiv.org/abs/2606.12994), 2026; [released dataset](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB-PP).
41. Narnia Labs, [Artificial intelligence-based generative design method and device](https://patents.google.com/patent/EP4524807A1/en), European patent application EP4524807A1.
42. Yoo et al., [Integrating Deep Learning into CAD/CAE System](https://arxiv.org/abs/2006.02138), 2020; [Narnia Labs publication page](https://www.narnia.ai/ko/publications/publications-integrating-deep-learning).
43. Neural Concept, [AI Design Copilot press release](https://www.neuralconcept.com/press-release/neural-concept-ai-design-copilot), 2026; public evidence of claimed CAD-ready/editable workflows, not of the undisclosed internal geometry representation.
44. Ansys, [2026 R1 GeomAI and SimAI launch](https://www.ansys.com/blog/introducing-ansys-geomai-software), documenting reference-geometry training, a latent design space, and closed-loop evaluation with SimAI or solvers.
45. PhysicsX, [Introducing LGM-Aero and the Ai.rplane showcase application](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures), 2024.
46. Backflip AI, [mesh-to-CAD product page](https://www.backflip.ai/mesh-to-cad), documenting editable Onshape feature trees and STEP output; the internal representation is undisclosed.
47. nTop, [How nTop works](https://learn.ntop.com/courses/ntop-foundations/lessons/how-does-ntop-work/), [implicit modeling capabilities](https://www.ntop.com/software/capabilities/modeling/), and [B-reps versus implicits](https://www.ntop.com/resources/blog/understanding-the-basics-of-b-reps-and-implicits/).
48. LEAP 71, [PicoGK implicit/SDF explanation](https://picogk.org/coding-for-engineers/19-computational-geometry-part7.html) and [open-source geometry kernel](https://github.com/leap71/PicoGK).
49. NVIDIA, [DoMINO in PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html), an STL-, point-cloud-, and SDF-aware AI-CAE surrogate rather than a geometry generator.

### Modern large-scale reference systems

50. Xiang et al., [Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS)](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiang_Structured_3D_Latents_for_Scalable_and_Versatile_3D_Generation_CVPR_2025_paper.pdf), CVPR 2025.
51. Li et al., [CraftsMan3D](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_CraftsMan3D_High-fidelity_Mesh_Generation_with_3D_Native_Diffusion_and_Interactive_CVPR_2025_paper.pdf), CVPR 2025.
52. Li et al., [TripoSG](https://arxiv.org/abs/2502.06608), 2025.
53. Sanghi et al., [Wavelet Latent Diffusion (WaLa): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings](https://arxiv.org/abs/2411.08017), 2024; [official code](https://github.com/AutodeskAILab/WaLa); [Autodesk publication page](https://www.research.autodesk.com/publications/wala-billion-parameter-3d-generative-model-compact-wavelet-encodings/).

### AI-CAE and neural operators

54. Pfaff et al., [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409), ICLR 2021.
55. Li et al., [Fourier Neural Operator with Learned Deformations for PDEs on General Geometries (Geo-FNO)](https://arxiv.org/abs/2207.05209), JMLR 2023.
56. Li et al., [Geometry-Informed Neural Operator for Large-Scale 3D PDEs (GINO)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html), NeurIPS 2023.
57. Wu et al., [Transolver](https://icml.cc/virtual/2024/poster/33751), ICML 2024.

### Datasets and engineering tooling

58. Koch et al., [ABC: A Big CAD Model Dataset](https://openaccess.thecvf.com/content_CVPR_2019/html/Koch_ABC_A_Big_CAD_Model_Dataset_for_Geometric_Deep_Learning_CVPR_2019_paper.html), CVPR 2019.
59. Willis et al., [Fusion 360 Gallery](https://www.research.autodesk.com/publications/fusion-360-gallery/), SIGGRAPH 2021.
60. Elrefaie et al., [DrivAerNet++](https://papers.nips.cc/paper_files/paper/2024/hash/013cf29a9e68e4411d0593040a8a1eb3-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024.
61. Bonnet et al., [AirfRANS](https://airfrans.readthedocs.io/en/latest/index.html), NeurIPS 2022.
62. [CadQuery documentation](https://cadquery.readthedocs.io/en/stable/) for parametric Python CAD and STEP/STL export.
63. [OpenCASCADE STEPControl_Writer](https://dev.opencascade.org/doc/refman/html/class_s_t_e_p_control___writer.html) for STEP export.
64. [Gmsh documentation](https://gmsh.info/doc/texinfo/) for CAD import, surface/volume meshing, and physical boundary groups.
65. [NIST STEP File Analyzer and Viewer](https://www.nist.gov/services-resources/software/step-file-analyzer-and-viewer) for the ISO 10303/STEP exchange context and file inspection.

### Additional sources for the named-model audit

66. PhysicsX, [Building Beyond Human Imagination with Foundation Models for Geometry and Physics](https://www.physicsx.ai/newsroom/building-beyond-human-imagination-with-foundation-models-for-geometry-and-physics), 2024. Public LGM-Aero architecture, preprocessing, scale, compute, and loss description; it is a company technical post, not a peer-reviewed benchmark.
67. You et al., [PhysGen supplementary material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/You_PhysGen_Physically_Grounded_CVPR_2026_supplemental.pdf), CVPR 2026. Dataset split, training compute, SDF decoder, Marching Cubes, rectified-flow, and OpenFOAM verification details.
68. Gao et al., [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cebbd24f1e50bcb63d015611fe0fe767-Abstract-Conference.html), NeurIPS 2022.
69. Zhan et al., [PhysiOpt: Physics-Driven Shape Optimization for 3D Generative Models](https://research.ibm.com/publications/physiopt-physics-driven-shape-optimization-for-3d-generative-models), SIGGRAPH Asia 2025.
70. Berzins et al., [Geometry-Informed Neural Networks](https://arxiv.org/abs/2402.14009), 2024; [peer-reviewed OpenReview version](https://openreview.net/pdf?id=o4KpjiCdrk).
71. Khan et al., [Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts](https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf), NeurIPS 2024; [official code](https://github.com/SadilKhan/Text2CAD).
72. Shen et al., [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization](https://research.nvidia.com/labs/toronto-ai/flexicubes/), ACM TOG/SIGGRAPH 2023.
