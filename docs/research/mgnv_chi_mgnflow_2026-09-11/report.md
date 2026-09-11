# Enhancing MeshGraphNets-V and cHI-MGNflow

**The strongest strategy is to improve the two methods at their respective bottlenecks: conditional distribution transfer for MeshGraphNets-V, and representation, geometry conditioning, and sampling cost for cHI-MGNflow.** Preserve the existing HI-MGN decoder while diagnosing where MeshGraphNets-V loses variability. For cHI-MGNflow, prioritize clean-field prediction or analytic noise cancellation, mesh-consistent noise, and stronger reusable geometry features. A coarse spatial generative model with conditional fine-scale refinement is the most promising larger research direction.

This report evaluates the working tree at commit 1c2beabcf1df9212d814576dded096a48ff565d7, including uncommitted code present on September 11, 2026. It distinguishes current implementation, previously recorded results, mathematical deductions, and proposals requiring experiments. The scope emphasizes static conditional warpage generation because that is the concrete application in the production configs; temporal simulation is addressed separately. The target is the distribution of physically valid fields for a specified geometry and its available conditions, including multiple valid realizations of that same condition.

The SAOI datasets, trained checkpoints, and generated field/spread dumps are absent from this checkout. The existing SAOI research note is available, while the September 10 peak-to-valley report contains “no dump” entries and no training measurements. Consequently, this report does not establish a new performance ranking, reproduce the reported underdispersion, or validate an improvement on SAOI. The exact inspected files and artifact availability are recorded in [audit_snapshot.json](audit_snapshot.json). Mathematical checks are reproducible with [check_reasoning.py](check_reasoning.py), and their output is in [reasoning_checks.json](reasoning_checks.json).

| Priority | Recommendation | Why it matters | Evidence and effort |
|---|---|---|---|
| First, both | Evaluate the generated distribution per geometry, including field dependence and peak-to-valley | Separates mean error, lost variability, incorrect spatial structure, and numerical error | High confidence in diagnostic value; low implementation effort |
| First, MeshGraphNets-V | Compare posterior-decoded and prior-decoded distributions; fit a prior to a frozen encoder and decoder | Locates whether compression/decoding or conditional prior fitting causes the deficit | Strong causal experiment; moderate training effort |
| Next, MeshGraphNets-V | Calibrate generated fields with proper scores, retaining a posterior reconstruction anchor | Trains the deployment path to reproduce distributions and extremes | Well-founded objective; task improvement remains untested |
| First, cHI-MGNflow | Compare a clean-field output head and an analytically preconditioned velocity head | Reduces the burden of carrying arbitrary fine-scale noise through a hierarchy | Relevant image-model evidence plus code-specific reasoning; moderate effort |
| Next, cHI-MGNflow | Test Matérn source noise with area/volume-aware losses | Makes stochastic forcing and error measurement less dependent on mesh density | Direct 2026 mesh-generation evidence; moderate engineering effort |
| Next, both | Improve geometry information at coarse levels; cache invariant computations | Addresses long-range response and repeated inference cost | Mesh-surrogate evidence; moderate effort |
| Larger research | Generate a coarse spatial field, then sample conditioned refinements | Combines efficient global variability with local detail | Closely related 2025–2026 research; substantial implementation effort |
| After quality is established | Distill expensive field sampling or test MeanFlow | Can reduce full-mesh evaluations per draw | Established on other domains; higher risk for conditional calibration |

---

**The current implementations already contain several enhancements that should remain the starting point.** Both methods use a two-level HI-MGN hierarchy in the SAOI production configs, with 1,000 and 100 coarse nodes and a 4,6,8,6,4 message-passing schedule. Both have AMP, EMA, gradient checkpointing, and four cached hierarchy variants. Proposing these as new ideas would duplicate existing work.

MeshGraphNets-V currently uses three global latent slots of width 16 in the inspected SAOI production config, for 48 latent coordinates in total. The older 3 × 128 configuration belongs to a different configuration/history. Its posterior encoder runs a graph network over target and context, pools to one graph vector, and emits all slots from that vector. These are graph-level slots injected at different hierarchy levels; they are not a latent vector at every coarse node. The conditional FM prior already flattens all slots jointly, so it already learns cross-slot dependence.

Its training objective combines posterior reconstruction MSE, per-slot aggregate MMD, an auxiliary loss on the decoded field’s peak-to-valley, and conditional-prior FM loss. The prior_grad_to_encoder setting controls whether the prior objective also differentiates through the posterior target. AdaLN conditioning is implemented. The FM prior supports Euler and Heun; production requests Heun. Latent inflation is also implemented. The old auxiliary head predicting field summary statistics from z has been replaced by direct peak-to-valley loss on the reconstructed field.

cHI-MGNflow predicts a velocity on an N × output_var field. It receives geometry/context and the interpolated noisy field, with flow time driving AdaLN. It currently offers uniform and logit-normal time sampling, optional extra training at exactly zero flow time, velocity- or clean-field-equivalent loss weighting, Euler/Heun integration, and single-evaluation mean readout. Its default source is independent Gaussian noise at each node and output channel. Its standard training step uses one network evaluation, without ODE integration.

| Implementation evidence | Current behavior |
|---|---|
| [MGN-V model](../../../methods/MeshGraphNets_Variational/model/MeshGraphNets.py), _build_vae_components, _encode_vae, _pv_loss | Three default multiscale slots; posterior/global sampling; decoded peak-to-valley objective |
| [Posterior encoder](../../../methods/MeshGraphNets_Variational/model/vae.py), GNNVariationalEncoder | Attention pooling; global Gaussian posterior; MMD averaged over slots |
| [Conditional prior](../../../methods/MeshGraphNets_Variational/model/conditional_prior.py), ConditionalFMPrior | Joint flattened slots; conditional FM; Heun/Euler; inflation around the ODE image of zero |
| [MGN-V objective](../../../methods/MeshGraphNets_Variational/training_profiles/training_loop.py), _compose_loss | Fresh posterior samples; configurable encoder gradients; no generated-field proper score in the current objective |
| [Field-flow construction](../../../methods/HI_MGNFlow/model/flow.py) | IID source; exact interpolation targets; time sampling/weighting; sampling and mean modes |
| [Field-flow model](../../../methods/HI_MGNFlow/model/CHiMGNFlow.py), _forward_multiscale | Full hierarchy traversed each velocity evaluation; noisy field and conditions enter together |
| [Field-flow objective and metrics](../../../methods/HI_MGNFlow/training_profiles/training_loop.py) | Node-averaged velocity loss; fair marginal CRPS; spread diagnostic using spatial target standard deviation |
| [Dataset splitting](../../../methods/HI_MGNFlow/general_modules/mesh_dataset.py), _resolve_split_ids | Random split of sample IDs, rather than explicit grouping by geometry |
| [MGN-V rollout](../../../methods/MeshGraphNets_Variational/inference_profiles/rollout.py) and [field-flow rollout](../../../methods/HI_MGNFlow/inference_profiles/rollout.py), _SampleContext and _run_batch | Optional hierarchy rotation across draw batches; MGN-V currently re-encodes replicated identical graphs for prior sampling |
| [Production configs](../../../configs/HI_MGNFlow/SAOI_all_input/config_train_bot.txt) and [MGN-V counterpart](../../../configs/MeshGraphNets_Variational/SAOI_all_input/config_train_bot.txt) | Static displacement output, three input-only condition channels, augmentation, hierarchy, and training settings |

The code has also repaired the earlier unconditional final-checkpoint overwrite in both single-worker and distributed training paths. It should not be described as an outstanding defect. Some adjacent comments and research notes still describe older behavior; callable code and effective configs must take precedence.

**The recorded SAOI comparison provides useful hypotheses, with narrow evidential limits.** The [September SAOI note](../SAOI_PROBABILISTIC_SWEEP_2026-09.md) describes three evaluation geometries, each with 125 realizations, and reports the following averages:

| Previously reported model/arm | Peak-to-valley W1 / reference SD | Generated/reference SD |
|---|---:|---:|
| MeshGraphNets-V arm 7 | 0.410 | 0.475 |
| MeshGraphNets-V arm 3 | 0.434 | 0.526 |
| cHI-MGNflow arm 1 | 0.896 | 0.475 |

These are measurements transcribed from that note, not recomputed results. They concern the distribution of one scalar, peak-to-valley displacement. They do not establish fidelity of the full displacement-field distribution. The 125 reference realizations estimate a conditional law; they are not the exact population distribution.

The note suggests that MeshGraphNets-V mainly loses dispersion, whereas cHI-MGNflow additionally overpredicts peak-to-valley on SM-L345U-MAIN. That motivates different interventions. However, a sign change in bias across geometries does not rule out preprocessing or feature-semantic errors: geometry-dependent normalization, part encoding, alignment, or extrapolation can produce different signs. Similarly, consistent underdispersion across a small hyperparameter sweep does not prove a structural latent bottleneck.

The earlier sweep had one missing MeshGraphNets-V arm and no seed replication. Its factor effects cannot support statistical claims of “no effect” or “real effect” without uncertainty estimates and the missing design point. Changing batch size while holding epochs fixed also changes optimizer-update count, so the reported advantage of batch 16 may partly be an update-budget effect. Preserve the observed uniform-time baseline for field flow; the recorded logit-normal result does not justify switching to an image-derived default.

---

**Several mathematical distinctions change which improvements are worth testing.** They prevent expensive experiments from answering the wrong question.

1. **Good posterior reconstruction does not prove good generation.** Training reconstructs from q(z | y,g); inference draws from p(z | g). Evaluate both distributions using the same decoder, geometry, normalization, hierarchy, and output statistics. A decoder can represent valid fields while the prior places too much probability elsewhere. Staged latent modeling is an established strategy, but does not guarantee that the remaining prior can learn every conditional distribution. [2: Diagnosing and Enhancing VAE Models](https://arxiv.org/abs/1903.05789).

2. **Differentiating FM loss through the encoder does not make that loss KL(q || p).** The FM theorem equates conditional and marginal velocity-regression gradients with respect to the velocity-model parameters for a fixed target probability path. It does not identify FM MSE with a VAE rate term when the encoder changes that target distribution. A VAE ELBO contains a KL term, including posterior entropy, which is a different construction. [1: Flow Matching, Theorem 2](https://arxiv.org/html/2210.02747v2), [3: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

   An independent analytic example makes the distinction concrete. Let Z0 ~ N(0,1), Z1 ~ N(0,a²), Zt=(1-t)Z0+tZ1, and U=Z1-Z0. Even the optimal CFM regressor has conditional variance a² / ((1-t)²+a²t²). Its integrated loss is πa/2. A prior exactly matching the target has KL zero for every a. Thus the optimized CFM objective can decrease when the trainable target shrinks, despite perfect prior matching at every scale. The numerical checks reproduce the integral to better than 1e-8 relative tolerance. This demonstrates a possible optimization pressure, not that it caused the observed SAOI deficit.

3. **MMD and the peak-to-valley auxiliary loss do not guarantee conditional mutual information.** The present MMD sees aggregate slot distributions across graphs. A mapping encoding geometry while discarding realization-specific variation can still satisfy an aggregate constraint. Peak-to-valley reconstruction can encourage useful z information, but neither it nor the old summary head establishes a formal positive information floor. Reconstruction, latent sensitivity, posterior variance, and within-geometry covariance must be measured. InfoVAE supplies useful objective theory; it does not validate every weighted combination labeled InfoVAE. [4: InfoVAE](https://arxiv.org/abs/1706.02262).

4. **CRPS remains a proper probabilistic score with one truth per verification case.** A finite sample creates uncertainty in the estimated ranking. It does not convert CRPS into a deterministic regression metric. For truth N(0,1), expected CRPS is approximately 0.5642 for the correct forecast, 0.6100 for a forecast with SD 0.5, and 0.7979 for a point mass at zero. The included calculation verifies this. The actual limitation here is that averaging nodewise CRPS cannot determine the joint spatial distribution. [5: Strictly Proper Scoring Rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

5. **The existing fair CRPS correction is useful and should be retained.** Both loops use the off-diagonal S(S-1) ensemble correction. Its interpretation assumes independently sampled members. At S=1 the implementation falls back to absolute error; that is not an unbiased distributional CRPS estimate. Fixed independent noise banks across checkpoint comparisons are appropriate. Introducing dependent antithetic members would require reconsidering the pairwise estimator. [6: Fair Scores for Ensemble Forecasts](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf).

6. **The training log’s field spread ratio is not conditional calibration.** The cHI-MGNflow code divides average nodewise ensemble standard deviation by target.std(), computed over a target batch’s nodes/channels. This compares stochastic variation against spatial field variation, potentially across graphs. It is different from the SD ratio across repeated peak-to-valley realizations of one geometry. Neither a value near one nor a value near zero alone establishes calibration or collapse.

7. **Latent inflation does not preserve decoded location or shape.** The code centers inflation on the prior ODE image of zero noise, which is not generally the prior mean. Even inflation around the actual latent mean can shift a nonlinear decoder’s output mean. For Z~N(0,1) and D(z)=z², doubling z increases the decoded mean from 1 to 4. Tune inflation against bias, tails, and joint field behavior; matching one SD ratio is insufficient.

8. **The clean-field loss option is not a clean-field network architecture.** With s=1-sigma_min and a(t)=1-st, the exact identity is y = s*y_t + a(t)*u. Therefore a(t)² times velocity error is the clean-field error of that induced readout. The current network still directly predicts velocity. A network whose output is the clean field has a different inductive bias. Image experiments explicitly separate network output space from loss space and find materially different outcomes. [12: Back to Basics](https://arxiv.org/html/2511.13720v1).

   There is also a reduction detail: the present weighted loss divides by the sum of sampled time weights in each batch. With one graph, the time weight cancels entirely. With several graphs, it is a self-normalized estimator rather than an exactly fixed-normalization expectation. This matters when comparing small-batch experiments or claiming mathematical equivalence to a chosen population loss. The included check verifies the single-graph cancellation.

---

**For MeshGraphNets-V, the first experiment should identify where conditional variability is lost.** Use held-out repeated realizations of each geometry and form three ensembles:

| Ensemble | Construction | Question answered |
|---|---|---|
| True fields | The available realizations for geometry g | What variation must the model reproduce? |
| Posterior-decoded fields | Encode each true field, then decode its posterior mean and posterior draws | Does the existing encoder/decoder retain variation and extremes? |
| Prior-decoded fields | Draw from the conditional prior without accessing a target, then decode | Does deployment reproduce the posterior-supported field distribution? |

Posterior-mean decoding and posterior-sample decoding should be reported separately. A variance floor can add noise to reconstructions, and noise around an encoded realization is not interchangeable with variability between distinct realizations. Include target-independent deterministic prediction as an additional reference.

Measure peak-to-valley bias/dispersion, spatial covariance, low-frequency bending modes, gradients, and physically defined failure rates. Report posterior reconstruction error per realization, but do not use paired single-draw MSE as the primary generative score. If posterior-decoded distributions are already narrow, prior training alone is unlikely to recover details the decoder does not reproduce.

Poor posterior reconstruction is not itself proof of insufficient latent dimension. Optimize z for individual validation fields while freezing the decoder as a diagnostic. If that improves reconstruction substantially, the encoder has an inference/amortization problem. If neither posterior encoding nor latent refinement recovers the missing patterns, a representation/decoder change becomes more credible. This diagnostic uses targets and is not a deployable generation result.

**Fit a frozen-target conditional prior before expanding the latent architecture.** This is the smallest interpretable intervention that preserves the current decoder.

1. Train or select the posterior encoder and HI-MGN decoder using reconstruction and the existing auxiliary objective. Keep MMD, posterior width, and architecture fixed for this first experiment; remove the prior’s influence on encoder training. Select this stage on posterior reconstruction and reconstruction of the field statistics that matter.
2. Freeze the posterior encoder and decoder, including normalization and the chosen EMA snapshot. Train only the conditional prior trunk and velocity network against fresh samples from that frozen posterior. All 48 coordinates should stay joint, as they already are.
3. Keep geometry augmentation consistent. If encoding augmented targets, pair them with the correspondingly transformed condition. A cache of latent targets must record encoder identity, geometry transform, and normalization; frozen posterior means plus resampled posterior noise are a possible space-saving cache.
4. Compare generated-field scores and latent coverage on validation geometries. Do not select this stage solely on latent FM MSE, since latent fit and decoded field error need not rank checkpoints alike.

The main benefit is stable targets and clear attribution, not a promised improvement. If posterior reconstruction is strong but the frozen prior still misses modes or covariance, test a richer geometry conditioner and latent scaling before enlarging the entire HI-MGN decoder. A train-only affine standardization of latent coordinates is a controlled alternative to assuming MMD has made every conditional direction unit scale; its inverse must be restored before decoding.

Measure objective contributions and gradient norms by parameter group before adjusting weights. An alpha_recon value of 1000 and prior_nll_weight of 1 do not establish that the prior itself is undertrained: reconstruction does not update the separate prior parameters. Their competition matters where both losses reach shared parameters, especially the posterior encoder. Record unweighted losses, weighted contributions, and gradient alignment at those shared parameters; avoid choosing regularization strength from coefficient ratios alone.

A conditional neural spline flow is a useful alternative prior for this diagnostic. It gives an explicit invertible density model in the small latent space, allowing held-out likelihood and sampling without numerical ODE integration. Its suitability depends on multimodality and conditioning complexity; retain FM as the reference. This changes the prior family only and should not be combined with a decoder redesign in the same comparison. [18: Neural Spline Flows](https://arxiv.org/abs/1906.04032).

**If the decoder reconstructs well but deployment remains narrow, train a generated-field calibration stage.** Freeze the posterior encoder initially. Draw z from the prior, decode several samples for each g, and optimize a proper score of the generated fields plus their task statistics. Maintain a posterior reconstruction anchor so that calibration does not destroy the existing accurate reconstruction path.

A suitable proposed objective is:

\[
L_{\mathrm{cal}} =
 \lambda_{\mathrm{field}}\, ES(\{\hat y_s\},y)
 + \lambda_{\mathrm{PV}}\, CRPS(\{A(\hat y_s)\},A(y))
 + \lambda_{\mathrm{recon}}\,\|D(E(y,g),g)-y\|_M^2,
\qquad A(y)=\max_i y_{i,z}-\min_i y_{i,z}.
\]

Here ES uses a mass/area-aware field norm and independent draws, and all weights/scales are fixed using training data. The PV term directly assesses the distribution of the application statistic. It supplements a field score because PV alone allows entirely incorrect field shapes. Proper-score calibration is justified by forecast theory; the choice of weights and its benefit here require measurement. [5: Proper Scoring Rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

Run two distinct versions if needed. In the cheaper version, keep prior samples detached and tune only decoder modulation/output layers. In the fuller version, allow gradients through a dedicated differentiable prior sampler into the prior. The current public sampling functions are decorated with no_grad; simply adding a generated-field score around them will not train the prior through sampling. Keep the ordinary inference sampler untouched and build a separate training path with explicit gradient checks.

Start with a small training ensemble, such as 4–8 independently drawn members, and increase validation ensemble size to check score stability. Those are experiment settings, not established optima. Prior integration is cheap relative to repeatedly decoding the full mesh; memory and wall time must be measured. Unconditional MSE between each generated sample and an unrelated realization would instead punish valid diversity.

**Treat decoded peak-to-valley reconstruction as an existing ablation whose results remain unknown.** The current beta_aux loss already supervises the reconstructed field’s extrema. Do not reintroduce the retired z-readout head as though it were a new fix. First recover the results of the current beta sweep. If it improves posterior extrema but not generated extrema, this supports investigating the prior/deployment path.

A high weight on exact max-minus-min gives gradients concentrated at a few extreme nodes and can overfit outliers or create isolated spikes. Retain the exact engineering metric in evaluation, while optionally testing smoother auxiliary statistics, such as high/low quantiles or temperature-controlled smooth extrema. These are supporting losses; replacing actual peak-to-valley with a different statistic would change the task. Evaluate curvature and extreme-node neighborhoods to distinguish a realistic bend from one spurious vertex.

**Add spatial latents only when reconstruction evidence calls for them.** Forty-eight coordinates can describe complex local fields if those fields lie on a sufficiently low-dimensional conditional manifold. The current global slots do not prove that local variation is impossible. Conversely, good average MSE can conceal systematic loss of high-frequency detail or rare modes.

Estimate conditional residual PCA spectra within repeated geometries and compare posterior reconstruction of those modes. Diagnose decoder sensitivity using Jacobian-vector products or finite differences in z, including the rank and amplitude of induced field covariance. With 125 realizations, empirical covariance rank is at most 124 before further grouping, so apparent rank saturation is not proof of true intrinsic dimension.

If compression is limiting, retain a global z for board-wide modes and add a small spatial latent field on the coarsest graph. Model global and spatial latents jointly or conditionally; an independent latent at every node could destroy desired correlation structure. Compare equal decoder capacity and report added prior cost. Lino, Pfaff, and Thuerey’s graph latent models offer direct precedent for retaining spatial latent structure, and their VAE baseline illustrates that compression and condition dependence can affect diversity. Their experiments are fluid distributions rather than manufacturing warpage. [8: Learning Distributions of Complex Fluid Simulations](https://arxiv.org/html/2504.02843v1).

---

**For cHI-MGNflow, start by changing what the network must carry through the mesh hierarchy.** At low flow time, the velocity contains a large component that cancels the initial noise. A hierarchy built from pooled features must retain that node-specific information while learning global geometry response. Skip paths help, but the current architecture has no explicit analytic output path for this cancellation.

A first ablation should compare the existing velocity head against a direct clean-field predictor h_theta(y_t,t,g), converted to velocity through:

\[
v_\theta(y_t,t,g)=
\frac{h_\theta(y_t,t,g)-s\,y_t}{1-st}.
\]

Keep the loss space fixed initially: using velocity loss with this head isolates representation from objective weighting. The denominator reaches sigma_min at the endpoint, making numerical treatment essential. An endpoint policy, FP32 conversion, time grid, and sampling stability checks belong to this experiment. The formula alone does not guarantee good integration.

The strongest recent motivation is the distinction between direct data prediction and weighted velocity prediction in high-dimensional generative modeling. Transfer to HI-MGN is a hypothesis because its graph skips and bottlenecks differ from a patch transformer. [12: Back to Basics, output-space ablations](https://arxiv.org/html/2511.13720v1).

An alternative with a bounded coefficient structure is **analytic preconditioning of the velocity head**. For a centered independent Gaussian reference with source variance one and target variance sigma_d², define:

\[
a=1-st,\quad V=a^2+t^2\sigma_d^2,\quad
c_{\mathrm{skip}}=\frac{t\sigma_d^2-sa}{V},\quad
c_{\mathrm{in}}=V^{-1/2},\quad
c_{\mathrm{out}}=\frac{\sigma_d}{\sqrt V},
\]
\[
v_\theta=c_{\mathrm{skip}}y_t+
c_{\mathrm{out}}\,F_\theta(c_{\mathrm{in}}y_t,t,g).
\]

The skip coefficient is the optimal linear Gaussian predictor of velocity from the noisy field. The residual variance is sigma_d²/V; the supporting calculation checks this identity. This is a proposed adaptation of the preconditioning principle, not a formula copied from an existing HI-MGN model. The physical target need not be Gaussian for the parameterization to be used, but its optimality claim is restricted to the stated Gaussian reference. Use fixed train-derived per-channel scales and retain the original velocity objective when isolating this change. [13: Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

Compare direct clean-field prediction and preconditioning as alternatives. Their advantages depend on the architecture; stacking both immediately would obscure attribution. A separate mean predictor plus a conditional residual flow is another option if the mean is demonstrably poor, but should remain a controlled alternative rather than replacing the existing methods by assumption.

**Make source noise and loss correspond to a physical field rather than a list of vertices.** IID nodal Gaussian noise changes its spatial character when mesh density changes. Mean-pooling m independent unit-variance values gives variance 1/m; thus a pooled noise component’s scale varies with cluster population. The real network pools learned nonlinear features, so this calculation is a mechanism to investigate, not a diagnosis of current collapse.

A particularly relevant paper introduces Matérn noise for mesh-based FM and demonstrates generated elastic equilibrium deformations. On a triangle surface, its sampling step solves:

\[
(L+\tau M)z_0=M^{1/2}\epsilon,\qquad \epsilon\sim N(0,I),
\]

where L is the cotangent stiffness/Laplacian matrix, M is the lumped area matrix, and tau controls screening. The sparse matrix can be prefactored per geometry. The authors distinguish consistency under retriangulation from consistency across different shapes or physical scales. [9: Matérn Noise for Triangulation-Agnostic Flow Matching](https://arxiv.org/html/2605.19305v1).

For this repository, introduce a source-sampler interface used identically by training, validation, and inference. Change both initial sampling and the training interpolation; replacing noise only at inference creates a mismatch. For volume meshes or nontriangle elements, use an appropriate finite-element operator and measure rather than silently applying a triangle-surface formula. Cache factorization only for the fixed reference domain. Preserve physical dimensions when choosing correlation length, and do not normalize away geometry-scale information required by the response.

Run a 2 × 2 comparison of IID versus Matérn source and node-uniform versus area/volume-weighted loss. This separates source effects from quadrature effects. Functional Flow Matching provides a broader basis for stochastic-process rather than discretization-specific generative modeling. It does not make arbitrary graph implementations resolution invariant. [10: Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html).

Use a geometry-balanced field objective such as:

\[
L=\frac{1}{B}\sum_g
 \frac{\sum_{i\in g} m_i\sum_f w_f e_{if}^2}
      {\sum_{i\in g}m_i},
\qquad \sum_f w_f=1.
\]

This gives each geometry equal weight and integrates error over its domain. If the application instead weights large parts more heavily, define that separately. Update normalization, metrics, and hierarchy restriction consistently; changing loss weighting alone does not remove discretization dependence.

Measure generated low-mode covariance, spatial correlation length, extrema, and behavior under local remeshing of the same physical part. Matérn noise is a source distribution, not an assumption that final physical variability must be Matérn. An overly smooth source can nevertheless make fine-scale variation harder to learn within finite capacity and budget. Add a resolution-consistent high-frequency component only if residual evidence requires it.

**Improve and reuse geometry conditioning before making every flow evaluation much larger.** The current network merges geometry features with the changing noisy state and traverses the V-cycle at every ODE evaluation. A separate condition encoder can compute nodal, coarse, and global context once per geometry, then provide it to each velocity evaluation through concatenation, modulation, or cross-attention.

A compact first version adds global attention only on the existing approximately 100-node coarsest graph. This permits long-range interaction without full fine-mesh quadratic attention. Compare against an equally costly increase in coarse message passing, and measure whether it specifically reduces held-out-geometry bias. Transolver’s learned-slice attention, DiffusionNet’s spatial diffusion layers, and static-PDE GNN work offer alternative communication mechanisms. None establishes that the current multiscale backbone lacks global information or that a transformer swap will improve SAOI. [14: Transolver](https://proceedings.mlr.press/v235/wu24r.html), [15: DiffusionNet](https://arxiv.org/abs/2012.00888), [17: Mesh-Based GNN Surrogates for Time-Independent PDEs](https://www.nature.com/articles/s41598-024-53185-y).

Inspect how coarsening treats disconnected components, interfaces, thin structures, and gaps. Proximity-based grouping can put physically separated regions in the same aggregate. Pooling respecting connectivity or interfaces is a reasonable ablation if that happens. Bi-stride multiscale GNNs specifically motivate topology-aware alternatives to proximity-derived hierarchies. The current coarsening file has an existing uncommitted change, so later implementation must be reviewed against that work. [16: Bi-Stride Multi-Scale GNN](https://proceedings.mlr.press/v202/cao23a.html).

Conditions should represent quantities with their correct semantics. The configs declare three condition channels, while several comments list only two names. Verify the stored contract before assuming what the third means. If a part/material identifier is categorical, treating its numeric value as a continuous physical magnitude may hurt unseen-part behavior. Prefer physical material/interface features when available; an embedding of a training-only identity does not solve generalization to new identities.

For caching, begin with computations whose inputs truly stay fixed: hierarchy metadata, reference geometry, initial edge embeddings, and coarse edge embeddings. Stateful processor features cannot be cached across ODE steps. Learned caches must be invalidated when parameters or the selected EMA snapshot change. The current code already caches hierarchy metadata, so that is not a new saving.

**Treat time sampling and solver settings as measured numerical choices.** Both FM implementations use Fourier features spanning up to 16,384 cycles over the unit flow interval. The network may learn to ignore high frequencies, but those features permit temporal variation much faster than a 12- or 30-step solver can resolve. Test a lower-frequency embedding with the same embedding dimension, alongside a smoothness diagnostic on the learned velocity. This is a code-specific hypothesis; no observed stiffness has been demonstrated.

Preserve uniform time sampling as the reference. Try stratified or loss-informed allocation only after measuring velocity/readout error in time bins on held-out data. Reweighting time with positive support keeps the unrestricted pointwise optimum, but changes finite-capacity optimization. An importance-sampled estimate of the original uniform-time loss requires the appropriate density correction. Stable Diffusion 3’s schedule evidence comes from image synthesis and did not predict the recorded SAOI ordering. [19: Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206).

Use the same checkpoint and initial noise for solver sweeps. Compare equal function-evaluation budgets: K Heun steps normally use 2K calls, whereas K Euler steps use K. More accurate integration of an inaccurate learned vector field need not improve agreement with data. Establish numerical convergence against a finer solve, and assess distributional quality separately. A claim that 30 Heun steps always outperform 100 Euler steps needs checkpoint-specific evidence.

The exact zero-time clean readout is s*z0 + v(z0,0,g) at the ideal regression optimum. The current z0+v expression leaves sigma_min*z0, in addition to network error. More importantly, a continuously sampled training time almost never equals zero. The existing flow_det_prob setting can explicitly train that slice if conditional mean prediction matters. This readout is not a one-step stochastic generator, and it is unrelated to the average-velocity target learned by MeanFlow.

---

**A larger architectural program should distribute randomness across spatial scales and allocate computation accordingly.** MeshGraphNets-V makes the entire random state small; cHI-MGNflow updates the entire fine field at every step. A useful middle ground factorizes the field into a global/coarse sample and fine residuals:

\[
p(y\mid g)
 =p(y_{\mathrm{coarse}}\mid g)
  \prod_{\ell}p(r_\ell\mid y_{\mathrm{coarser}},g),
\qquad y_\ell=U_\ell y_{\mathrm{coarser}}+r_\ell.
\]

A possible implementation uses the existing HI-MGN hierarchy, a generative model over the 100-node field, and progressively lighter conditioned refinement at the 1,000-node and fine levels. Start with more integration work at the coarsest level and fewer fine-level evaluations, choosing counts from quality/cost measurements.

The factorization is schematic: define restriction and upsampling operators, residual constraints, and the corresponding representation measure consistently. Otherwise coarse content can be represented twice in unconstrained residuals, and the displayed product need not be a density for the original field without further accounting.

The April 2026 scale-autoregressive paper directly studies this approach on unstructured fluid meshes. It reports concentrating denoising at coarse levels, separate geometry conditioning, and robustness training for errors in coarse inputs. Its speed advantage is relative to its particular baselines and tasks, not a forecast for this repository. [11: One Scale at a Time](https://arxiv.org/html/2604.11403v1).

~~~mermaid
flowchart LR
    G["Geometry and available conditions"] --> C["Reusable condition encoder"]
    E["Coarse random source"] --> P["Coarse field generator"]
    C --> P
    P --> R["Conditional spatial refinements"]
    C --> R
    N["Fine-scale random sources"] --> R
    R --> Y["Displacement field ensemble"]
    Y --> Q["Field, extrema, calibration and remeshing evaluation"]
~~~

This diagram describes a proposed model, not the present code. It also makes a training hazard visible: if refinements see only perfect coarse truth during training, they face generated coarse inputs at deployment. Train with controlled coarse perturbations and evaluate the actual generated cascade. Arbitrarily adding an independently generated MeshGraphNets-V field and an independently sampled residual generally gives the wrong joint distribution. Refinement must condition on the sampled coarse content.

A frozen deterministic mean plus a generative residual is a simpler competing factorization. The August 2026 MeshPriorDiT preprint combines a mesh GNN trajectory with a conditional generative residual for cloth dynamics, making it relevant to monitor. Its task has actions and temporal history, and its reported point errors do not demonstrate calibrated manufacturing variability. [23: MeshPriorDiT](https://arxiv.org/abs/2608.26766).

**Accelerate sampling after identifying a sufficiently accurate teacher.** Distillation should preserve the mapping from a particular source-noise draw and condition to the teacher’s output. In that setting, paired MSE is legitimate because the teacher target is paired with the same noise; this differs from comparing an arbitrary generated sample to an unrelated physical realization.

Consistency models and MeanFlow offer one/few-evaluation generation routes, with different training requirements. MeanFlow learns average rather than instantaneous velocity, so setting the existing field-flow integrator to one Euler step is not an implementation of MeanFlow. Validate teacher/student agreement in tails and covariance, not only average reconstruction of teacher samples. [20: Mean Flows](https://arxiv.org/abs/2505.13447), [21: Consistency Models](https://arxiv.org/abs/2303.01469).

A diffusion alternative should remain an evidence-driven comparison. Lino et al. report FM variants doing better at low denoising-step counts, while diffusion variants improve relative performance at larger counts in their experiments. This argues for a quality-versus-cost comparison rather than declaring either objective universally superior. [8: Graph Generative Models, Appendix D.7](https://arxiv.org/html/2504.02843v1).

Minibatch optimal transport is another optional way to change the coupling used in FM training. Apply it only where both conditional marginals remain correct. Unrestricted assignment across geometries can correlate source noise with geometry or mismatch field layouts. Within repeated conditions, or in a suitably defined latent space with valid conditioning, it is a testable refinement. It is not the first intervention for missing geometry information or a decoder bottleneck. [24: Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482).

**Use physics only where the observations and problem definition support it.** Flow time is an artificial sampling coordinate, not physical time. A displacement-flow velocity is not mechanical velocity. A physical residual should act on a clean-field estimate or final field under the correct material, load, and boundary model, rather than imposing a physical PDE on an arbitrary noisy interpolation.

For known linear boundary constraints A y=b, one can parameterize y=y_bc+P r with A P=0, and ensure noise/updates remain in the allowed subspace. This can enforce the actual constraint exactly. However, constraints, material laws, residual loads, and rigid-body alignment must match the data. The inspected three-channel conditioning config does not establish that a full mechanics residual is available.

The 2025 PCFM paper provides a relevant inference-time constraint framework. For this application, constraint satisfaction and conditional distribution fidelity must both be checked: projecting samples can alter variability or shift the law even when the constraint is satisfied. Do not use clipping or generic smoothing as a substitute for a physically justified model. [22: Physics-Constrained Flow Matching](https://proceedings.nips.cc/paper_files/paper/2025/hash/eacce1f7d11e2c3a7568468e0eff5d33-Abstract-Conference.html).

Augmentation also requires the right symmetry. The code rotates/reflects coordinates and displacement vectors while treating condition channels as scalars. If a future condition contains load direction, anisotropy orientation, or a vector-valued boundary input, it must transform consistently. Pure displacement differences do not automatically make a generic MLP rotation equivariant. Before introducing full equivariant networks, establish which rotations/reflections preserve the physical problem and whether a reliable part frame is available.

For temporal applications, validate the entire trajectory law. MeshGraphNets-V’s reused trajectory latent can represent persistent uncertainty. Fresh field-flow noise at each physical step can represent Markov transition uncertainty when the state is sufficient, but may miss persistent unresolved effects. A shared trajectory latent, history conditioning, or jointly generated time blocks should be considered only if the data require those dependencies. Static independent realizations must remain static samples, without an invented time axis.

---

**A credible comparison needs field structure, conditional calibration, and cost on the same protocol.** Freeze training/validation/test groups before tuning. The current sample-ID split is appropriate for some questions, but repeated realizations of the same geometry can be distributed across splits. That measures new-realization performance on familiar geometry; an explicit geometry-family split is needed to claim unseen-part generalization.

Use one split for calibration and model selection and a separate final test split. The three already examined parts are useful diagnostic cases but cannot remain an untouched test set after repeated tuning. Include interpolation among known geometry types and a clearly labeled extrapolation set. If only three parts are available, report per-part behavior and finite-data uncertainty without treating them as a broad population.

| Evaluation layer | Measures | What it can reveal |
|---|---|---|
| Mean field | Mass-weighted mean-field error, channelwise bias, low-order deformation modes | Geometry-conditioned location error |
| Marginals | Fair CRPS, interval coverage and width, bias at selected physical locations | Nodewise calibration |
| Joint field | Energy score, variogram score over physical-distance bins, cross-channel covariance | Coherent versus independently plausible fields |
| Engineering statistics | PV CRPS and W1, PV quantiles, peak location, curvature/strain diagnostics when meaningful | Wrong extrema or plausible scalar histograms with wrong shape |
| Conditional ensembles | PIT/rank diagnostics over repeated realizations for each geometry | Underdispersion, bias, missing modes |
| Geometry transfer | Same physical domain under multiple meshes; unseen shapes and interfaces | Dependence on tessellation or part identity |
| Numerical behavior | Paired-noise solver convergence, FP32/AMP comparison on selected cases | Integration or precision artifacts |
| Cost | End-to-end latency, throughput, NFE, peak memory, training GPU-hours | Actual benefit under a usable budget |

Energy score is proper for the multivariate law, but its sensitivity to correlation misspecification can be weak in high dimensions. Variogram scores offer a complementary dependence diagnostic and are not strictly proper by themselves. Keep both marginal/location and dependence assessments. For large meshes, use a fixed set of physical-distance-stratified pairs rather than all node pairs. [7: Variogram-Based Proper Scoring Rules](https://repository.library.noaa.gov/view/noaa/22327).

Use the same independent initial noise bank when comparing checkpoints, solvers, or ablations, with hierarchy fixed per geometry. Report sensitivity to several fixed hierarchies separately; variation caused by choosing a partition should not silently become physical uncertainty. Use a second unseen noise bank for a final robustness check.

This needs an explicit setting in the current code. Both rollout modules accept a list of hierarchy seeds and rotate the partition across draw batches. The inspected production inference configs actually request seeds 0, 1, 2, 3: see [MGN-V inference](../../../configs/MeshGraphNets_Variational/SAOI_all_input/config_infer_s26fe_main_bot.txt) and [field-flow inference](../../../configs/HI_MGNFlow/SAOI_all_input/config_infer_s26fe_main_bot.txt). A single seed stays fixed, and each field-flow ODE still uses a fixed partition internally. The multi-seed option therefore defines a valid mixture over partition-specific generators, but that mixture includes numerical/architectural variation. For a field statistic A, decompose its variance into average within-hierarchy variance plus variance of the hierarchy-specific mean. Compare that second term against physical realization variance. Batch-size changes can also alter mixture proportions, and cycling both hierarchy and inflation lists can couple the two factors; evaluate their combinations deliberately. The mixture could mask underdispersion; it has not been established as the cause of any recorded error.

Reference uncertainty matters as much as generated sample count. Two thousand generated draws cannot compensate for 125 reference realizations or three geometries. With 125 independent truths and a nominal 4% two-sided tail rate, the expected number of tail events is only five, with binomial SD about 2.19. Treat tail estimates accordingly. If realizations are correlated by production batch or simulation trajectory, resample those groups for uncertainty intervals.

Estimate uncertainty in differences using paired evaluation cases and paired geometry-level resampling where enough geometries exist. Reusing one generated ensemble for many truths also contributes Monte Carlo uncertainty, which can be assessed with independent ensemble banks. A ratio close to one or a visually flat rank histogram is not proof that all aspects of the joint law are correct.

**The first campaign should be sequential and small enough to interpret.** Do not immediately combine all proposals in another large fractional factorial.

| Stage | Concrete work | Decision criterion |
|---|---|---|
| 0: recover evidence | Recover original datasets/checkpoints/dumps; verify channels, units, geometry identity, preprocessing, and deployed checkpoints | Reproduce reported PV behavior and determine whether field statistics support it |
| 1: locate failure | Posterior mean/draw versus prior ensembles for MGN-V; time-binned errors, conditional mean, and solver sweeps for field flow | Identify representation loss, prior mismatch, condition bias, or numerical sensitivity |
| 2V: preserve architecture | Current joint reference versus frozen encoder/decoder with retrained prior | Better held-out generated-field scores without losing posterior reconstruction |
| 3V: calibrate deployment | Generated-field/PV proper score with reconstruction anchor; decoder-only or differentiable-prior version | Better dispersion and tails with stable field structure and bias |
| 2F: representation | Existing head versus clean-field head versus analytic preconditioning, with objective/budget controlled | Better held-out mean and sampled fields at comparable NFE |
| 3F: discretization | IID/Matérn × uniform/mass-weighted loss on the best simple representation | Better calibration and remeshing consistency without lost local variation |
| 4: spatial modeling | Coarse attention/reusable condition encoder; spatial latents if reconstruction evidence supports them | Improvement attributable to geometry communication or representation |
| 5: speed | Coarse-to-fine generation or student distillation | Better quality/latency frontier, including tails and covariance |

For screening, equal optimizer updates help isolate batch and objective changes, while equal GPU-hours answer the practical budget question. Report both instead of treating them as interchangeable. Finalists should use at least three independent training seeds if resources permit; a single run can reject a clearly broken method but cannot establish a small gain. Keep learning-rate schedule, EMA behavior, total updates, and actual learning rate used by the optimizer explicit.

Select winners using predeclared primary metrics, such as PV CRPS and a field dependence metric, with no material deterioration in physically defined failure rates. Choose absolute engineering tolerances with the application owner before final testing. An arbitrary universal target such as SD ratio within 10% is not warranted by the current data. Estimate the reference’s own finite-sample discrepancy by comparing independent or resampled reference subsets.

The cost distinction should guide resource allocation. A MeshGraphNets-V draw requires a conditional graph encoding, a small latent ODE, and one full decoder. Its prior API can amortize graph encoding across draws, but the current rollout constructs a batch of identical step-zero graphs and calls sample on that batch, repeating the condition trunk. A concrete optimization is to encode the single graph once and generate a latent bank through sample_n or the pooled-condition API, while preserving temperature/inflation semantics. A field-flow draw costs approximately 2K full-mesh evaluations under Heun. With 200 draws and K=30, that is 12,000 full-mesh evaluations per geometry before batching/caching savings. Reducing latent-prior steps and reducing field-flow steps have very different practical value.

**The strongest publication direction is a generative mesh model that preserves conditional spatial uncertainty and engineering extremes under geometry and mesh changes.** Adding an FM prior, AdaLN, a hierarchy, or Matérn noise individually is already established or implemented. A defensible contribution would connect latent/source representation to measured conditional covariance, demonstrate stable behavior under remeshing, and allocate expensive stochastic refinement where it materially improves fidelity.

A convincing study would compare the current global-latent model, the current full-field model, a spatial latent/coarse-to-fine model, and simpler conditional density/mean-residual references. It would publish the paired geometry/realization protocol, measure posterior-to-prior loss of variability, and show improvements across several unseen geometries under equal-update and equal-cost budgets. A new name for a combination of known components would be much weaker evidence.

The practical implementation order is: establish generated-field evaluation, run the frozen-prior diagnostic for MeshGraphNets-V, test direct clean-field prediction/preconditioning for cHI-MGNflow, then investigate Matérn noise and spatial generative refinement. These steps preserve useful existing behavior while exposing whether a larger architecture is necessary.

---

The following primary sources underpin the analysis. Publication status is stated where it affects confidence; domain differences are part of the recommendation rather than evidence of automatic transfer.

| Ref. | Source | Evidence used and limits |
|---|---|---|
| 1 | Lipman, Chen, Ben-Hamu, Nickel, Le. [Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2). ICLR 2023. | Fixed-path velocity regression; does not identify encoder-through-FM gradients with KL |
| 2 | Dai, Wipf. [Diagnosing and Enhancing VAE Models](https://arxiv.org/abs/1903.05789). ICLR 2019 / extended paper. | Separate latent density modeling; not a guarantee for conditional mechanics |
| 3 | Kingma, Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). ICLR 2014. | ELBO and posterior/prior distinction |
| 4 | Zhao, Song, Ermon. [InfoVAE: Information Maximizing Variational Autoencoders](https://arxiv.org/abs/1706.02262). 2017 preprint; AAAI 2019. | Aggregate matching and information objectives; custom losses require their own analysis |
| 5 | Gneiting, Raftery. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). JASA, 2007. | CRPS and energy-score foundation |
| 6 | Ferro. [Fair Scores for Ensemble Forecasts](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf). Online 2013; QJRMS 2014. | Finite-ensemble correction and dependence assumptions |
| 7 | Scheuerer, Hamill. [Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities](https://repository.library.noaa.gov/view/noaa/22327). Monthly Weather Review, 2015. | Dependence sensitivity and limits of energy/variogram scores |
| 8 | Lino, Pfaff, Thuerey. [Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks](https://arxiv.org/html/2504.02843v1). ICLR 2025. | Spatial latent generation; compression; FM/diffusion cost-quality comparison |
| 9 | Kuai, Maesumi, Ritchie, Aigerman. [Matérn Noise for Triangulation-Agnostic Flow Matching on Meshes](https://arxiv.org/html/2605.19305v1). TOG / SIGGRAPH 2026, per [author project](https://matern-fm.github.io/). | Mesh-consistent source, elastic deformations; surface and scale limitations |
| 10 | Kerrigan, Migliorini, Smyth. [Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html). AISTATS 2024. | Function-space generative formulation |
| 11 | Lino, Thuerey. [One Scale at a Time: Scale-Autoregressive Modeling for Fluid Flow Distributions](https://arxiv.org/html/2604.11403v1). April 2026 preprint. | Coarse-to-fine sampling and geometry conditioning; fluid benchmarks |
| 12 | Li, He. [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/html/2511.13720v1). November 2025 preprint version reviewed. | Output parameterization versus loss weighting; image-domain evidence |
| 13 | Karras, Aittala, Aila, Laine. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364). NeurIPS 2022. | Preconditioning principle; proposed FM coefficients here are independently derived |
| 14 | Wu, Luo, Wang, Wang, Long. [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://proceedings.mlr.press/v235/wu24r.html). ICML 2024. | Efficient global communication through learned slices |
| 15 | Sharp, Attaiki, Crane, Ovsjanikov. [DiffusionNet: Discretization Agnostic Learning on Surfaces](https://arxiv.org/abs/2012.00888). TOG 2022. | Discretization-robust spatial operators; not a stochastic generator by itself |
| 16 | Cao, Chai, Li, Jiang. [Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network](https://proceedings.mlr.press/v202/cao23a.html). ICML 2023. | Topology-aware hierarchy and long-range communication |
| 17 | Gladstone et al. [Mesh-Based GNN Surrogates for Time-Independent PDEs](https://www.nature.com/articles/s41598-024-53185-y). Scientific Reports, 2024. | Static mechanics, graph connectivity, and geometry representation |
| 18 | Durkan, Bekasov, Murray, Papamakarios. [Neural Spline Flows](https://arxiv.org/abs/1906.04032). NeurIPS 2019. | Explicit invertible-density alternative for a small conditional latent space |
| 19 | Esser et al. [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). ICML 2024. | Time-distribution design; image results do not override the SAOI comparison |
| 20 | Geng, Deng, Bai, Kolter, He. [Mean Flows for One-Step Generative Modeling](https://arxiv.org/abs/2505.13447). May 2025 paper. | Average-velocity target for one-evaluation generation |
| 21 | Song, Dhariwal, Chen, Sutskever. [Consistency Models](https://arxiv.org/abs/2303.01469). ICML 2023. | Distilled or independently trained one/few-step generators |
| 22 | Utkarsh, Cai, Edelman, Gomez-Bombarelli, Rackauckas. [Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](https://proceedings.nips.cc/paper_files/paper/2025/hash/eacce1f7d11e2c3a7568468e0eff5d33-Abstract-Conference.html). NeurIPS 2025. | Constraint enforcement; distributional effects require validation |
| 23 | Wang et al. [MeshPriorDiT: Hierarchical Modeling for Action-Conditioned Cloth Dynamics](https://arxiv.org/abs/2608.26766). August 27, 2026 preprint. | Mesh-prior/residual composition; temporal cloth rather than static calibration |
| 24 | Tong et al. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482). TMLR 2024. | Alternative FM coupling; conditional marginals must be preserved |

The local evidence consists of the linked source files, production configurations, the September SAOI research note, and the incomplete September 10 sweep report. The mathematical checks establish the stated identities and counterexamples only. Every proposed model improvement remains to be evaluated on the actual datasets and trained models.
