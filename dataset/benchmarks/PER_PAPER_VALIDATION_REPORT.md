# Per-paper implementation validation report

**Snapshot:** 2026-07-20 06:20 KST (UTC+09:00)  
**Scope:** the four configurable methods in `Neural_Operator` plus `Transolver`  
**Policy:** minimum opt-in paper-validation changes are authorized; defaults
must remain unchanged and must not gain runtime work; no commit

## What this report considers a valid implementation check

A direct check must use a benchmark from that implementation's own paper, the
paper's output quantity, and the paper's evaluation metric. A run on another
method's benchmark is supplemental evidence only. If the existing suite cannot
represent the paper dataset without loader or model work, this report says
**not applicable as-is** and does not substitute a different dataset.

The first completed FNO run used the unchanged suite plus benchmark data,
configs, and a post-inference evaluator. It exposed material paper-protocol
differences. The user subsequently authorized minimum validation-only changes
for all affected methods. Those paths are being made opt-in: normal training
and inference defaults must remain unchanged and must not perform paper-metric
or paper-data work.

An independent default-path audit confirms that constraint. Seeded default FNO
weights, forward output, and exported historical config are bit-exact to Git
HEAD; paper methods are instance-bound only in `paper_darcy` mode. Default GINO
continues to resolve `MeshGINO` without importing the Car-CFD paper module, and
the exact-paper Transolver module is unimported by normal runtime. Default FNO,
GINO, and Transolver preflights pass with zero errors/warnings. No paper branch
runs inside the normal forward/batch hot loops; the only residual gating cost
is one config-string check per epoch and construction.

## Current status and paper results

| Suite implementation | Own-paper benchmark selected | Applicable without production changes? | Result reported by its paper | Current suite result/status |
|---|---|---:|---:|---|
| FNO | 2D Darcy flow, 85x85 | **Yes; opt-in paper path complete** | mean relative L2 **0.0108** | Final exact-200 **0.0099181**; 0.918x paper and **passes** |
| Transolver | Elasticity, 972 irregular points | **Yes; complete** | mean relative L2 **0.0064** | Suite final exact-200 **0.0064211**, only 0.33% above paper and **passes**; redundant second runner stopped at epoch 343 after the user confirmed it was unnecessary |
| DeepONet | **2D fractional Laplacian on the unit disk** | **Yes through an isolated opt-in paper path; complete and independently audited** | paper plots normalized MSE at approximately **1.2e-3** after 5,000 epochs; exact curve value not tabulated | Released-style best MSE **0.00148703** (1.239x the plot approximation; relative L2 only 11.3% high) and **paper-similar** |
| Point-DeepONet | Non-parametric 3D structures with variable loads | **Yes through isolated paper path; complete** | smallest 1,000-case setting average R-squared **0.897** | Final 40,000-iteration average R-squared **0.892832**; 0.46% low and **paper-similar** |
| GINO | ShapeNet Car pressure prediction | **Requires opt-in paper model/data path** | decoder-only de-normalized test relative L2 **7.12%**; encoder-decoder **9.47%** | First exact-111 hybrid reconstruction **0.090992** (27.80% high); paper-era v2 correction run active because the decoder-only target was not met |

The FNO result is no longer pending. It is the arithmetic mean of 200
de-normalized per-case relative-L2 values from the final epoch-499 checkpoint.
An independent second calculation reproduced the JSON and CSV mean exactly,
and direct checks against the original MAT source had zero target conversion
error. The one-epoch/two-case smoke output remains wiring evidence only and is
excluded from the accuracy table.

## FNO: direct own-paper check complete

The original FNO paper's smaller 1D Burgers problem would use less memory, but
it is not applicable to this checkout: the live validation/runtime contract
accepts only operator dimension 2 or 3. The next-smallest directly supported
paper task is 2D Darcy at resolution 85.

Paper protocol and target:

- 1,000 training cases and 200 test cases
- resolution 85x85, Fourier modes 12, width 32
- batch size 20, learning rate 0.001, 500 epochs
- reported mean per-sample relative L2: **0.0108**

Prepared evidence:

- exact original `piececonst_r421_N1024_smooth1.mat` and
  `piececonst_r421_N1024_smooth2.mat` files, SHA-256 verified
- paper-style stride-five downsampling from 421x421 to 85x85
- 1,250-case non-test pool, for the unchanged seeded loader to produce
  1,000/125/125 optimization/validation/internal-test cases
- physically isolated `smooth2[0:200]` evaluation file in source order
- all 1,450 converted cases checked against the source MAT arrays with maximum
  absolute conversion error exactly `0.0`; no train/evaluation overlap
- real config preflight passed with zero errors, warnings, or notices
- one-epoch train/infer/evaluate smoke passed on two cases

The full run completed epoch 499 with normalized train/validation MSE
`2.3694561e-8`/`2.8838386e-8`, then produced one rollout for every isolated
test case. External physical-space evaluation returned:

| Quantity | Suite FNO | Paper FNO |
|---|---:|---:|
| Evaluated cases | 200 | 200 stated in paper |
| Mean relative L2 | **0.1103645243** | **0.0108** |
| Standard deviation | 0.0311230512 | not reported |
| Median | 0.1056127467 | not reported |
| Minimum / maximum | 0.0605456561 / 0.2387737661 | not reported |
| Suite minus paper | +0.0995645243 | - |
| Suite / paper | **10.2189x** | 1.0x reference |

**Verdict:** the current suite FNO does not reproduce paper-level Darcy
accuracy and therefore fails this paper-accuracy validation. The evaluator and
dataset contracts are verified; the likely causes are the material
architecture, target/loss, optimizer/schedule, and training-composition
differences documented below. The result must not be interpreted as merely a
small shuffle discrepancy.

This remains a same-task, same-resolution, same-metric implementation check,
not a paper-architecture reproduction or a stand-alone correctness
certificate. Seed 42 selects exactly 801 of the original first 1,000
`smooth1` cases plus 199 extra non-test `smooth2[200:450]` cases for
optimization; the remaining 199 original training cases land in suite
validation/internal test. The published 200 test cases remain fully isolated.

The baseline suite model is explicitly mesh-adapted FNO: splat/sample adapters
and five inputs (coefficient,
occupancy, density, x, y) replace the paper's three inputs (coefficient, x,
y); projection is `32 -> 32 -> 1` rather than the released `32 -> 128 -> 1`;
and GELU is applied after every block rather than the released ReLU pattern.
It predicts normalized `solution - coefficient` under MSE, then adds the
de-normalized residual back during rollout; the released model predicts the
solution directly under relative-L2 loss. The suite also keeps AdamW,
gradient clipping, three warmup epochs, and cosine decay instead of the
paper-era Adam/StepLR recipe. These material deviations must accompany the
baseline number. A minimum opt-in correction is now being implemented and will
be rerun; ordinary FNO defaults remain unchanged.

The strongest quantitative diagnosis is target conditioning, not an evaluator
bug. Across all 200 cases, the solution error is exactly the residual error
multiplied by `||u-a||/||u||`: mean residual relative error is `8.58534e-5`,
while the mean conditioning factor is `1309.37`. The 12-mode truncation error
is `0.9475%` for direct solution `u` but `13.210%` for residual `u-a`, because
the residual inherits the discontinuous coefficient. All 200 outputs are
present, coordinates and targets match the MAT source, adapter round-trip
error is `2.52e-7`, and 18 targeted spectral/grid/FNO tests pass. The correction
therefore targets direct physical `u`, the released architecture/activation,
relative-L2 optimization, paper optimizer/schedule, and exact first-1,000
training composition.

That opt-in correction is now implemented and regression-tested. It is gated
by `fno_variant=paper_darcy`; normal FNO configs retain the existing model and
hot loops. The generated paper files map the suite's seed-42 optimization IDs
exactly to `smooth1[0:1000]`, keep `smooth2[200:450]` outside optimization,
and isolate `smooth2[0:200]` for evaluation. The validation core has the
released three inputs, ReLU placement, `32 -> 128 -> 1` projection, decoded
relative-L2 objective, Adam, and `StepLR(100, 0.5)`. Twenty-eight targeted
regressions and the full suite config preflight pass. Its 500-epoch GPU rerun
completed with zero preflight errors, warnings, or notices. At epoch 499,
train/validation mean relative L2 was `0.00406`/`0.00914`; the final checkpoint
was then evaluated independently over every isolated test case.

An interim inference from the preserved epoch-100 checkpoint already improves
the exact isolated 200-case mean from the baseline's `0.1103645` to
**`0.0125933`** (standard deviation `0.0062803`, median `0.0108780`, range
`0.0061707` to `0.0513011`). This is 1.166 times the paper value, only 16.6%
high, and is not the final reportable checkpoint. The epoch-100 checkpoint,
JSON, and per-case CSV are preserved separately while training continues.

The preserved epoch-200 checkpoint improves the same exact-200 mean to
**`0.0112883`** (standard deviation `0.0057455`, median `0.0098024`, range
`0.0045495` to `0.0462399`). It is only 4.52% above the paper value and is
already paper-similar within ordinary seed/run variation, although the final
comparison remains the completed 500-epoch checkpoint.

The completed epoch-499 checkpoint obtains **`0.00991810`** mean relative L2
over the exact 200 de-normalized test cases (standard deviation `0.00489744`,
median `0.00864921`, range `0.00399643` to `0.03759923`). This is `0.91834`
times the paper's `0.0108`, or 8.17% lower, and therefore passes the direct
paper comparison. The improvement from baseline `0.1103645` verifies that the
original discrepancy came from the residual target, architecture, and
optimization protocol rather than the evaluator or the underlying FNO core.
An independent PowerShell arithmetic mean over the preserved 200-row CSV
reproduces `0.00991809802327584`.

Artifacts:

- [`fno_darcy/README.md`](fno_darcy/README.md)
- [`fno_darcy/prepare_darcy.py`](fno_darcy/prepare_darcy.py)
- [`fno_darcy/evaluate_relative_l2.py`](fno_darcy/evaluate_relative_l2.py)
- [`../../configs/benchmarks/fno_darcy/config_train_fno_paper_validation.txt`](../../configs/benchmarks/fno_darcy/config_train_fno_paper_validation.txt)
- [`../../configs/benchmarks/fno_darcy/config_infer_fno_paper_validation.txt`](../../configs/benchmarks/fno_darcy/config_infer_fno_paper_validation.txt)
- [`../../output/benchmarks/fno_darcy/paper_validation/inference/relative_l2_epoch499.json`](../../output/benchmarks/fno_darcy/paper_validation/inference/relative_l2_epoch499.json)
- [`../../output/benchmarks/fno_darcy/paper_validation/inference/relative_l2_epoch499.csv`](../../output/benchmarks/fno_darcy/paper_validation/inference/relative_l2_epoch499.csv)
- [`../../output/benchmarks/fno_darcy/fno/inference/relative_l2.json`](../../output/benchmarks/fno_darcy/fno/inference/relative_l2.json)
- [`../../output/benchmarks/fno_darcy/fno/inference/relative_l2.csv`](../../output/benchmarks/fno_darcy/fno/inference/relative_l2.csv)

Primary sources: [FNO paper](https://arxiv.org/abs/2010.08895),
[historical released Darcy recipe](https://github.com/li-Pingan/fourier-neural-operator/blob/main/FNO-torch.1.6/fourier_2d.py),
and [public Darcy data record](https://zenodo.org/records/12784353).

## Transolver: direct own-paper check complete

The public Elasticity benchmark is the Transolver paper's own task: 1,000
training cases, 200 test cases, 972 irregular 2D points, and scalar stress
prediction. The exact 200 published test cases are already isolated and the
post-inference evaluator computes de-normalized per-case relative L2 before
taking the arithmetic mean.

The benchmark config matches the paper/released architecture and major
hyperparameters: hidden width 128, 8 heads, 8 layers, 64 slices, batch size 1,
learning rate 0.001, gradient clipping 0.1, weight decay 1e-5, and 500 epochs.
The paper reports **0.0064** mean relative L2. The suite result will be qualified
for the same exact seed-42 composition of 801 original training cases plus 199
extra non-test cases. Its model embeds centered/RMS-scaled XYZ plus an
always-zero state channel instead of raw XY; it minimizes normalized stress
MSE rather than decoded relative L2; it adds a three-epoch 1%-LR warmup before
cosine decay; and it clamps learned attention temperature to `[0.1, 5.0]`.
The paper reports three repeated experiments, while this resource-constrained
workflow is one run. The core Physics-Attention scale and calculation are
aligned, making this the stronger of the two direct comparisons, but it is
still protocol-qualified. The scheduled suite run reached epoch 499 and saved
its final checkpoint. Inference over every isolated published test case
followed immediately. The exact arithmetic mean of 200 de-normalized per-case
relative-L2 values is **`0.0064211428`**, with standard deviation `0.00252772`,
median `0.00584239`, and range `0.00280445` to `0.01572988`. This is only
`0.00002114` above the paper's `0.0064` (ratio `1.00330`, or 0.33% high) and
therefore passes the direct implementation check despite the disclosed
suite-protocol differences. An independent CSV arithmetic mean reproduces
`0.0064211428` exactly.

An isolated exact-paper runner also passed a direct pinned-upstream parity
audit before launch: initialization, full 972-point forward output, decoded
relative-L2 loss, gradients, clipped AdamW update, and first cosine learning
rate all have maximum difference exactly `0.0`; both models contain 713,665
parameters. It uses raw XY, exact source indices `0:1000` and `1800:2000`, the
official float32 normalizer (`187.7387390`, unbiased std plus epsilon
`127.0828781`), decoded relative-L2 optimization, and unclamped attention
temperature. It reached epoch 343/499 before the user confirmed that this
second run was unnecessary because the completed suite implementation already
matches the paper. Its best intermediate exact-200 mean was `0.00775941` at
epoch 342, down from epoch 0's `0.498194`. The process was then deliberately
stopped to return GPU capacity to GINO; it is parity/health evidence only and
is not substituted for the completed suite comparison. The upstream run
exposed no seed, so fixed local seed 42 was its remaining reproducibility
qualification.

Artifacts:

- [`elasticity/README.md`](elasticity/README.md)
- [`elasticity/prepare_elasticity.py`](elasticity/prepare_elasticity.py)
- [`elasticity/evaluate_relative_l2.py`](elasticity/evaluate_relative_l2.py)
- [`../../configs/benchmarks/elasticity/config_train_transolver_paper.txt`](../../configs/benchmarks/elasticity/config_train_transolver_paper.txt)
- [`../../configs/benchmarks/elasticity/config_infer_transolver.txt`](../../configs/benchmarks/elasticity/config_infer_transolver.txt)
- [`../../output/benchmarks/elasticity/transolver/inference/relative_l2.json`](../../output/benchmarks/elasticity/transolver/inference/relative_l2.json)
- [`../../output/benchmarks/elasticity/transolver/inference/relative_l2.csv`](../../output/benchmarks/elasticity/transolver/inference/relative_l2.csv)
- [`../../configs/benchmarks/elasticity/config_train_transolver_paper_validation.txt`](../../configs/benchmarks/elasticity/config_train_transolver_paper_validation.txt)
- [`../../output/benchmarks/elasticity/transolver_paper_validation/train.jsonl`](../../output/benchmarks/elasticity/transolver_paper_validation/train.jsonl)

Primary source: [Transolver paper and Elasticity table](https://arxiv.org/html/2402.02366).

## Own-paper paths requiring validation-only adapters

### DeepONet

Per the user's explicit direction, the 1D antiderivative benchmark is not used.
The selected own-paper task is the **2D fractional Laplacian on the unit disk**.
The authors sample input functions from 15 Zernike basis functions with
coefficients in `[-2,2]`; use `15x15 = 225` branch sensors; evaluate at another
`15x15 = 225` spatial grid and 10 fractional orders; and use 5,000 input
functions for each released train/test split. This produces 11.25 million
branch/query/target triples per split. The released 2D MATLAB train and test
functions actually instantiate the same Sobol stream, so those two splits are
identical; the direct comparison preserves and discloses that behavior. The
released network uses a 225-wide branch input and 3D trunk query
`(x,y,alpha)`: its three branch affine layers have `tanh, tanh, linear`, while
all three trunk affine layers end in `tanh`. Width is 60, optimization is Adam
at `1e-3`, and the objective is global normalized MSE.

The paper reports this result as a plotted curve rather than a numeric table;
the DeepONet test curve finishes at approximately **1.2e-3 normalized MSE**
after 5,000 epochs. That plot-derived value will be treated as a tolerance band, not
as a falsely exact scalar. The official repository commit
`8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf` and its MATLAB generator are now
saved locally. MATLAB and Octave are not installed. The completed Python
translation follows the released 15 Zernike formulas, MATLAB sensor/query
ordering, 16-point angular quadrature, and vector Grunwald-Letnikov operator.
Its released alpha-1.5 manufactured-solution audit obtains `0.00485871`
relative projection error and `0.04608533` relative operator error. The
resulting compact HDF5 is `40,229,643` bytes and SHA-256
`d6bd0d8af94352b27ba177c334e8f4d4057b23fd98704ff899a0c843d2ccf600`.
It uses MATLAB standard point order with the official Joe-Kuo 2003
`joe-kuo-old.1111` direction table (SHA-256
`864a98b3af71806c1922feed53b9f77da29189f67a52bba0a88f7503d332e949`):
the first retained logical index is 28 and first coefficient is `-1.125`.

The isolated model/trainer is implemented without registering a new normal
factory mode. It indexes the compact arrays with the same expanded
alpha/function/query row order, so it avoids materializing approximately 9.6
GiB of repeated branch inputs. Two CPU smoke profiles pass and twelve targeted
architecture/data/resume regressions pass. Direct inspection of Figure 2e caught that its x-axis is
5,000 **epochs**, not 5,000 optimizer steps; with 11.25 million triples and a
100,000 batch this means 112 full updates per epoch, about 560,000 total. The
corrected loop performed the exact 560,000 updates, dropped the final 50,000
expanded rows per epoch like the release, and recorded both the released-style
pre-update best checkpoint and the final checkpoint. The CPU run completed all
5,000 epochs at below-normal priority with two math threads and no GPU
allocation. The released-style best checkpoint obtains global normalized MSE
**`0.0014870282`** and global relative L2 `0.0385620`; the final checkpoint
obtains `0.0018050402` and `0.0424858`. The primary value is 1.239 times the
approximately `0.0012` value read from Figure 2e, or 23.9% higher in MSE and
11.3% higher in relative L2. That reference is not tabulated, so the completed
run is classified as paper-similar rather than an exact scalar reproduction.
An independent direct evaluation of all 5,000 functions, 10 fractional orders,
and 225 query locations reproduced every best/final metric with numerical
difference exactly `0.0`. The log contains exactly 5,000 consecutive epochs,
112 batches per epoch, and 560,000 optimizer steps; both checkpoints preserve
the pinned config and dataset identity. No implementation defect or rerun
justification was found. No 1D data is substituted.

Completed artifacts:

- [`../../output/benchmarks/deeponet_fractional2d/paper_validation/result.json`](../../output/benchmarks/deeponet_fractional2d/paper_validation/result.json)
- [`../../output/benchmarks/deeponet_fractional2d/paper_validation/train.jsonl`](../../output/benchmarks/deeponet_fractional2d/paper_validation/train.jsonl)
- [`../../configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt`](../../configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt)

Primary sources: [DeepONet paper](https://doi.org/10.1038/s42256-021-00302-5),
[official supplement containing Tables S2, S3, and S6](https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-021-00302-5/MediaObjects/42256_2021_302_MOESM1_ESM.pdf),
and [authors' repository](https://github.com/lululxvi/deeponet), plus
[MATLAB's Sobol definition](https://www.mathworks.com/help/stats/sobolset.html)
and the [authors' Joe-Kuo 2003 table](https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111).

### Point-DeepONet

The official 3D structural dataset is publicly available, but the suite's
paper-profile model explicitly requires signed-distance features and declared
global load conditions. Live validation rejects declared global-condition
features because the existing dataset loader does not attach
`graph.global_conditions`. Running the shared scalar-stress Elasticity data
through the baseline profile does not test the paper architecture or its
variable-load input contract.

The paper uses 3,000 load-case samples over 1,785 unique geometries, split
2,400/600, with 5,000 sampled nodes for training. It reports average R-squared
`0.934` at the full 3,000-case/5,000-point setting; the smallest explicitly
quantified 1,000-case subset reports `0.897`. Its headline horizontal values
come from different evaluation resolutions: displacement `ux` is `0.987` on
the full mesh, while von Mises stress is `0.923` on the sampled 5,000 points
(`0.916` for full-mesh stress).

The paper dataset was not applicable to the shared loader as-is, so the direct
check uses an isolated validation-only adapter. The official archive is
approximately 99.15 GiB (`106,460,727,975` bytes), but the audited selective
range path transferred only the 1,000-case experiment's required members:
approximately 3.14 GiB of targets and 7.06 GiB of inputs, **10.197 GiB total**.
All 1,000 compact cases, the exact 800/200 manifests, CRCs, shapes, dtypes,
metadata, and 5,000,000 sampled rows pass the strict validator. The official
point-sampling script advances one RNG while iterating a Python set, whose
order is not reproducible across environments; the validation adapter uses
stable per-case samples and discloses that unavoidable qualification.

The isolated validation runner is now complete. It matches the released
251,936-parameter model bit-for-bit under the same state and input, the
DeepXDE NumPy batch sampler, batch size 16, AdamW/inverse-time schedule, 40,000
iterations, all-1,000-case scaling before the 800/200 split, and the pooled
sampled-data mean of 12 direction/component R-squared terms. Reproducing the
release also reproduces its validation-set scaling leakage, which is disclosed.
Eleven focused CPU tests pass. A real CUDA forward/backward smoke with the
unchanged paper batch shape `16 x 5000` also passes while the other GPU jobs
are active: output shape `16 x 5000 x 4`, 251,936 parameters, 1,032 MiB peak
allocated, 1,070 MiB peak reserved, and 0.47 seconds. The faithful 40,000-
iteration GPU run completed in 3,415.3 seconds. Standalone evaluation of both
best and last checkpoints gives the same paper-comparable average R-squared
**`0.8928321`** over all 200 cases and all 12 pooled direction/component terms,
versus `0.897`. The absolute difference is `-0.0041679` (0.46% low), which is
paper-similar and does not justify changing or rerunning the implementation.
Per-term R-squared spans `0.7427401` (horizontal `uy`) to `0.9691274`
(vertical `ux`).

Result artifact: [`point_deeponet/outputs/paper_n1000_p5000/evaluation_metrics.json`](point_deeponet/outputs/paper_n1000_p5000/evaluation_metrics.json).

Primary sources: [Point-DeepONet paper](https://arxiv.org/abs/2412.18362) and
[authors' repository](https://github.com/jangseop-park/Point-DeepONet), plus
[official Kaggle dataset metadata](https://www.kaggle.com/api/v1/datasets/view/jangseop/point-deeponet-dataset).

### GINO

The paper's ShapeNet Car Table 2 reports two distinct rows. Encoder-decoder
GINO obtains **9.47%** de-normalized test relative L2 (7.95% train), while the
stronger decoder-only GINO obtains **7.12%** (6.37% train). The direct
validation target is the paper's decoder-only row and therefore **7.12%**;
9.47% remains the secondary comparator if the local encoder-decoder topology is
validated later. The old shared-Elasticity GINO run is not comparable to
either row.

The own-paper dataset is now locally complete. The official archive is
`198,216,827` bytes and its published MD5
`24a46fe791085201d48ee5db7b6cfc86` matches exactly. The official manifests
contain 500 training and 111 test cases with zero overlap. Every manifest mesh
has 3,586 vertices and 7,168 faces; each stored pressure vector has 3,682
values, and the released crop `concat(pressure[:16], pressure[112:])` produces
the required 3,586 surface targets. The released normalizer fits the uncropped
training pressure and has mean `-37.1148433464` and population standard
deviation `48.1155421225`.

The normal local `MeshGINO` cannot certify the decoder-only paper row: it
always adds an input GNO, expects node features/SDF in different locations,
uses different integral-transform semantics, and its dense spectral weights
cannot reproduce the paper-scale factorized core within 8 GiB. An isolated
opt-in `paper_decoder` implementation is now complete. The first hybrid audit
corrected SDF-plus-XYZ FNO lifting, NeRF output-kernel embeddings, paper-era
per-query neighbor means, two InstanceNorm stages, gated channel-MLP skips,
final-layer activation placement, learned physical-space spectral biases,
pressure epsilon `1e-7`, and strict Open3D SDF generation. The corrected
two-train/two-test 32-cubed diagnostic obtains `79.6793%`; that remains wiring
evidence only and is explicitly not compared to the paper. All 35 focused
tests pass. Default `MeshGINO` behavior remains unchanged.

### What a GINO validation path requires

The official 198.2 MB archive is usable and is not missing quadrature data.
The direct paper-decoder path requires the following:

1. **Exact benchmark conversion:** preserve the 500/111 manifests, released
   pressure crop, train-global geometry scaling to `[1e-6,1]`, uncropped
   pressure normalization, and signed-distance convention.
2. **Paper latent grid:** the paper comparison uses a 64-cubed latent grid.
   The maintained post-paper recipe's 32-cubed grid is useful only as a smoke
   profile and cannot be reported as the paper-parity result.
3. **Decoder-only topology:** SDF is evaluated on the latent grid; there is no
   input GNO. The output GNO applies the paper integral kernel from latent
   features to surface queries using the paper-era per-query neighbor mean.
4. **Memory-compatible spectral core:** width 64, four FNO blocks, 16 centered
   modes, and Tucker rank 0.4 require a real factorized implementation.
   Instantiating the local dense per-corner spectral tensors would exceed the
   8 GiB GPU before useful activations. Batch size 1, activation checkpointing,
   and chunked surface queries are required.
5. **Paper optimization/evaluation:** 100 epochs with Adam and stepwise learning
   rate, then the de-normalized arithmetic mean of 111 per-case relative-L2
   values compared with `7.12%`.

Some exact hyperparameters are not fully tabulated in the paper. Where the
maintained official recipe is used to fill those gaps, the report labels that
as an inference rather than paper-stated fact. In particular, the strict local
64-cubed/100-epoch/radius-0.055/16-mode profile is a source-pinned hybrid
reconstruction: the current maintained recipe instead uses 32-cubed, 16 modes,
radius 0.033, and 301 epochs. Sixteen is the only mode count pinned by both the
paper-era defaults and maintained family; the earlier unproven 24-mode choice
was removed. Only the corrected
64-cubed, 100-epoch run can populate the direct comparison row. Production
defaults will not select or pay for this path.

The exact 500/111 Open3D artifact is complete at 531,190,763 bytes. The first
100-epoch run completed and its standalone evaluator covered all 111 official
test IDs. The independently reproduced de-normalized mean relative L2 is
**`0.0909919601`**, 27.80% above the decoder-only paper result. It is only 3.91%
below the paper's encoder-decoder value, but the local topology is decoder-only
and the result is not relabeled or accepted as a pass.

The accuracy miss triggered a direct paper-source audit. The first run mixed
the maintained NeRF coordinate embedding with the public 2023 decoder and also
used lifting width 128, no domain padding, no weight decay, and seed 0. The
paper-era/public CarCFD path instead supports the legacy logarithmic
coordinate embedding, lifting width 256, one-sided domain padding `0.125`,
weight decay `1e-4`, and seed 666. An isolated v2 retains the paper's 64-cubed,
100-epoch, radius-0.055, Adam/StepLR setting while correcting those five
differences. Its final corrected real-resolution forward/backward/Adam gate
passed at 2,028.99 MiB peak allocated and 2,812 MiB peak reserved with zero
empty queries. The v2 run is active as PID 47848 and, at the 06:20 KST
snapshot, is using about 95--96% GPU with 4,553 MiB reported device memory.
The process and its automatic exact-111 evaluator watcher are healthy; epoch 1
has not yet been emitted because the padded direct-Tucker epoch is slower than
the first reconstruction. A final independent read-only audit found no
mathematical or checkpoint-reconstruction issue requiring a restart. The first
run and all of its exact artifacts remain preserved.

Primary sources: [GINO paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf),
[official ShapeNet Car data record](https://zenodo.org/records/13936501),
[maintained CarCFD recipe](https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_gino_carcfd.py),
and [official integral-transform behavior](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/integral_transform.py).

## Supplemental shared-Elasticity runs (not paper validation)

Earlier shared-Elasticity training produced final checkpoints for three Neural
Operator methods and a stopped periodic GINO checkpoint. These numbers are
normalized validation losses, not de-normalized relative L2, and none is a
direct own-paper result except that the dataset itself belongs to Transolver.

| Method | Checkpoint epoch | Train normalized MSE | Validation normalized MSE |
|---|---:|---:|---:|
| Point-DeepONet | 499 | 0.0036094970 | 0.0082491396 |
| DeepONet | 499 | 0.0265681289 | 0.0375064024 |
| FNO | 499 | 0.0012305783 | 0.0062580291 |
| GINO | 50 | 0.0094984498 | 0.0139381740 |

They are preserved as supplemental implementation/wiring evidence only. The
common run was stopped when the scope was corrected to direct own-paper
validation. Transolver's own-paper Elasticity run and exact-200 evaluation are
now complete and are reported above.

## Remaining work

1. Preserve Transolver's completed suite exact-200 result and the deliberately
   stopped redundant runner; no further Transolver own-paper run is justified.
2. Preserve FNO's completed `0.0099181` exact-200 result and final artifacts;
   no further FNO rerun is currently justified.
3. Finish the active paper-era-corrected GINO v2 run, evaluate all 111 cases,
   and iterate against `7.12%` if the evidence still shows a material miss.
4. Preserve the completed 2D fractional-Laplacian DeepONet result and its
   independent full-tensor audit; no rerun is currently justified for the
   plot-derived comparison gap.
5. Preserve the completed Point-DeepONet result and its independent audit; no
   rerun is justified for the 0.46% paper gap.
6. Re-run final dataset/result audits, confirm normal defaults incur no new work,
   update this report with final distributions and explanations, and do not
   commit.
