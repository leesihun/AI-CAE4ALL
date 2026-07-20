# GINO CarCFD paper validation

This folder is an opt-in, decoder-only validation of GINO against its own
NeurIPS 2023 paper. It does not change the suite's normal GINO/FNO training or
inference path.

## Paper target

The paper's ShapeNet Car pressure table reports two different GINO tasks:

| Paper model | Train normalized relative L2 | Test de-normalized relative L2 |
|---|---:|---:|
| Encoder-decoder GINO | 7.95% | 9.47% |
| **Decoder-only GINO** | **6.37%** | **7.12%** |

The implementation here targets the **decoder-only 7.12% result**: a signed
distance field on a fixed latent grid is fed directly into the FNO, followed
by only the output GNO on the car surface. Consequently, it does not need mesh
input quadrature weights.

Primary references:

- [Geometry-Informed Neural Operator, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)
- [Official CarCFD archive, Zenodo record 13936501](https://zenodo.org/records/13936501)
- [Maintained NeuralOperator CarCFD training recipe](https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_gino_carcfd.py)

The official NeuralOperator source is pinned locally under
`source/neuraloperator_official`. The public paper-era decoder architecture is
commit `957f0b0fe540bf167f6138494297073d8aa97d98` (August 2023), while the
maintained recipe audited here is commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066` (July 2026).

## Verified source contract

The downloaded `processed-car-pressure-data.tar.gz` has MD5
`24a46fe791085201d48ee5db7b6cfc86`. Its authoritative manifests contain
exactly 500 training cases and 111 test cases with no overlap. Each manifest
mesh has 3,586 vertices and 7,168 faces. Each raw pressure vector has 3,682
values; the official crop is:

```python
pressure = np.concatenate([pressure[:16], pressure[112:]])
```

The released-loader pressure statistics over all 500 full raw vectors are:

```text
mean = -37.11484334643704
std  =  48.115568258070894
```

The archive contains a duplicate nested `data/data` tree and AppleDouble
files. The converter intentionally reads only the canonical outer `data`
directory and only IDs named by the manifests.

## Prepare data

The strict artifact is expensive because it evaluates an SDF for all 611
watertight cars on a 64 cubed grid:

```powershell
python dataset/benchmarks/gino_carcfd/prepare_carcfd.py --resolution 64
```

This writes `carcfd_paper_r64.h5`. Coordinates are mapped from the official
global raw bounds to `[-1,1]^3`. SDFs use the released Open3D
`RaycastingScene` sign convention (negative inside) and are scaled using
training-global extrema to
`[1e-6,1]`. Pressure is stored in physical units and normalized lazily by the
dataset.

Trimesh is available only through the explicit diagnostic backend. Its signed
distance is negated to use the same negative-inside sign, but numerical
equivalence to Open3D is not assumed and it cannot create a full validation
artifact.

For a diagnostic CPU pipeline check only:

```powershell
python dataset/benchmarks/gino_carcfd/prepare_carcfd.py `
  --resolution 32 --limit-train 2 --limit-test 2
```

That output is stamped `diagnostic_only=True`; it is not comparable to 7.12%.

## Train and evaluate

The benchmark-local trainer keeps relative-L2 training out of the suite's
default normalized-MSE loop:

```powershell
python dataset/benchmarks/gino_carcfd/train_carcfd.py `
  --config configs/benchmarks/gino_carcfd/config_train_gino_paper_r64.txt

python dataset/benchmarks/gino_carcfd/evaluate_carcfd.py `
  --config configs/benchmarks/gino_carcfd/config_eval_gino_paper_r64.txt
```

The evaluator refuses a non-64-cubed or non-111-case artifact unless
`--allow-diagnostic` is explicit. It writes per-case CSV, raw prediction HDF5,
and JSON containing the exact mean de-normalized relative L2, its difference
from 7.12%, and its ratio to 7.12%.

Before occupying the GPU for 100 epochs, run the real-resolution one-case
memory gate. It performs one forward, backward, and Adam update and reports
peak allocated/reserved CUDA memory without writing a checkpoint:

```powershell
python dataset/benchmarks/gino_carcfd/prepare_carcfd.py `
  --resolution 64 --limit-train 1 --limit-test 1 `
  --output dataset/benchmarks/gino_carcfd/carcfd_diagnostic_r64.h5

python dataset/benchmarks/gino_carcfd/preflight_carcfd.py `
  --config configs/benchmarks/gino_carcfd/config_train_gino_paper_r64.txt `
  --dataset dataset/benchmarks/gino_carcfd/carcfd_diagnostic_r64.h5 `
  --allow-diagnostic
```

## Architecture provenance and limitations

The paper fixes a 64-cubed latent grid, 100 Adam epochs, and an epoch-50
learning-rate halving, but it does not publish every architecture choice needed
for byte-for-byte reconstruction. The strict config therefore uses values
shared by the public paper-era architecture and/or maintained GINO family where
the paper is silent: four FNO blocks, 64 hidden channels, 16 total centered
modes, Tucker rank 0.4, output-kernel widths 512/256, and projection width 256.
The SDF is concatenated with the three latent-grid coordinates before FNO
lifting, as in the paper-era decoder. The output integral uses the paper-era
neighbor mean. The first completed run used the maintained NeRF coordinate
embedding, lifting width 128, no domain padding, no weight decay, and seed 0.
Its exact-111 result missed the decoder-only target, so the evidence-backed v2
uses the public August-2023 logarithmic positional embedding, lifting width
256, one-sided domain padding 0.125, weight decay `1e-4`, and seed 666. Both
source commits and the resolved choices are saved in the checkpoints.

This is consequently a **source-pinned hybrid reconstruction**, not a claim of
bit-identical reproduction: no public experiment file records every setting of
the exact run that produced 7.12%. In particular, the 2026 maintained default
recipe is now a different experiment (32 cubed, 16 modes, radius 0.033, 301
epochs, and different optimizer settings), whereas this validation
intentionally retains the paper's 64-cubed/100-epoch task and radius 0.055 from
its reported ablation range.

The native Tucker convolution is local to
`Neural_Operator/model/gino_carcfd.py`; it avoids the suite's per-corner dense
spectral parameterization, which would be unnecessarily expensive for this
3-D validation profile on an 8 GB GPU. Activation checkpointing and query
chunks are enabled without
changing the mathematical operator. No model-split implementation is claimed
for this isolated variant.

## Current verification status

The CPU-only real-data diagnostic path has completed for cases 001/002 and
658/659 at 32 cubed using Open3D. The generated HDF5 has the expected
`(2,32,32,32,1)` batched SDF contract and training SDF range `[1e-6,1]`. One
diagnostic epoch and the standalone evaluator both completed after the parity
corrections; the evaluator reported 79.68% over only two test cases, which is a
pipeline smoke value and **not** a model result or paper comparison.

A one-train/one-test Open3D diagnostic artifact at the real 64-cubed
resolution is also ready. Its CPU coverage preflight reports 16--32 latent
neighbors per surface query (median 22) and zero empty queries. The strict
16-mode model has 26,965,041 parameters (102.86 MiB of FP32 weights; about
411.45 MiB for weights, gradients, and two Adam moments before activations).
The first reconstruction's real 64-cubed CUDA forward/backward/Adam gate
measured 1,499.18 MiB peak
allocated and 1,796 MiB peak reserved on the 8 GB GPU. That measurement was
taken immediately before adding the four 64-element spectral bias vectors;
they add only 1 KiB of weights and no activation tensor, but the exact final
model gate should still be repeated mechanically before the full run.

The first 100-epoch run and independent evaluator are complete. All 111
official test IDs were evaluated, producing mean de-normalized relative L2
`0.0909919601`, or 27.80% above the decoder-only paper result `0.0712`. The
CSV statistics reproduce the JSON to floating-point roundoff. Because that is
a material miss, it is not marked as a pass.

The paper-era-corrected v2 is isolated under
`output/benchmarks/gino_carcfd/paper_r64_paper2023`. Its focused unit tests pass
and its final corrected real 64-cubed CUDA forward/backward/Adam gate measured
2,028.99 MiB peak allocated and 2,812 MiB peak reserved, with 16/22/32
minimum/median/maximum neighbors and zero empty queries. Its full 100-epoch run
is active as PID 47848; a separate watcher will run the exact-111 evaluator
after training finishes. At 06:20 KST on 2026-07-20 it was using 95--96% GPU
and 4,553 MiB device memory. The first epoch had not yet been emitted, and a
final read-only architecture/checkpoint audit found no issue requiring a
restart.
