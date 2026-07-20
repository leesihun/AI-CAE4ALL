# Elasticity accuracy benchmark

This is a benchmark-only validation workflow for all live methods in
`Neural_Operator` plus Transolver. It does not change the suite's production
training, inference, splitting, or loss code.

The current in-progress training state and paper-reference comparison are in
[`PRELIMINARY_REPORT.md`](PRELIMINARY_REPORT.md).

The public Geo-FNO/Transolver Elasticity benchmark contains 2,000 cases with
972 irregular 2D points and one scalar stress target. The two source arrays are
about 47 MB, making it substantially smaller than the 3D datasets used in the
Point-DeepONet and GINO papers and suitable for an 8 GB GPU.

## Prepare the data

Download the checksum-identical public mirror and convert it:

```powershell
hf download asatheesh/PICore original_data/Random_UnitCell_XY_10.npy original_data/Random_UnitCell_sigma_10.npy --repo-type dataset --local-dir dataset/benchmarks/elasticity/source
python dataset/benchmarks/elasticity/prepare_elasticity.py
```

The converter verifies both official Git-LFS SHA256 values before writing:

- `elasticity_train.h5`: source cases 0-1249. The unchanged seeded 80/10/10
  suite split produces 1,000 optimization, 125 validation, and 125 internal
  test cases.
- `elasticity_test.h5`: source cases 1800-1999, exactly the published 200-case
  test set. It is physically isolated from the training dataset.

The 1,000 optimization cases are therefore not exactly the paper's first
1,000 cases; they are the suite's seeded selection from 1,250 non-test cases.
The published test cases and their ordering are exact. This is the split
qualification; training-protocol qualifications are documented below.

## Train and infer as-is

The configs in `configs/benchmarks/elasticity` use the existing random split
and normalized MSE training objective. Run them through the normal suite
launcher. Each matching inference config reads only `elasticity_test.h5` and
writes de-normalized rollout HDF5 files.

The Neural Operator training batch sizes are benchmark-specific and sized for
the available 8 GB GPU (Point-DeepONet 8, DeepONet 32, FNO 16, GINO 4). Transolver
retains the paper configuration's batch size of 1. These values change no
runtime behavior outside these configs.

Example:

```powershell
python CAE_ML_Suite_main.py --config configs/benchmarks/elasticity/config_train_transolver_paper.txt
python CAE_ML_Suite_main.py --config configs/benchmarks/elasticity/config_infer_transolver.txt
```

## Post-inference relative L2

Calculate the paper metric only after inference:

```powershell
python dataset/benchmarks/elasticity/evaluate_relative_l2.py `
  --model transolver `
  --ground-truth dataset/benchmarks/elasticity/elasticity_test.h5 `
  --predictions output/benchmarks/elasticity/transolver/inference
```

The evaluator reads the final de-normalized stress field from every rollout and
computes the mean per-sample
`||prediction - target||_2 / ||target||_2`. It rejects missing samples,
duplicate outputs, geometry mismatches, and non-finite predictions, then writes
a JSON summary and per-sample CSV beside the predictions.

To exercise all five methods without touching their normal paths, use the
external validation runner. It runs sequentially to keep peak VRAM low:

```powershell
# One epoch on 20 training-pool cases and two published-test cases.
python dataset/benchmarks/elasticity/validate_all.py all --smoke

# Full configs: preflight, train, infer, evaluate, then comparison.md/csv/json.
python dataset/benchmarks/elasticity/validate_all.py all
```

The smoke mode writes runtime config copies and outputs under
`output/benchmarks/elasticity/smoke/<timestamp>`. The full mode uses the
checked-in benchmark configs and writes `output/benchmarks/elasticity/comparison.md`.

The Transolver paper reports `0.0064` on this benchmark. The benchmark config
matches the authors' released Elasticity recipe for hidden width 128, 8 heads,
8 layers, 64 slices, batch size 1, learning rate `1e-3`, gradient clipping
`0.1`, weight decay `1e-5`, and 500 epochs. It is still not a strict
reproduction: as requested, the suite retains normalized-MSE training and its
native warmup/cosine scheduler, and its seeded optimization-set composition
differs from the authors' first 1,000 cases. The official 200 test cases are
exact. The paper's `0.0229` Geo-FNO result is context only for the suite's
mesh-adapted FNO. Point-DeepONet and GINO were published on different 3D
datasets and metrics, so their Elasticity runs validate this implementation on
a common small dataset but are not direct paper reproductions.

## Opt-in exact Transolver-v1 protocol

The as-is run above remains unchanged. If its exact-200 result misses the paper,
the isolated paper-validation runner removes the known protocol differences
without touching the default Transolver wrapper or training hot path:

```powershell
python dataset/benchmarks/elasticity/train_transolver_paper.py `
  --config configs/benchmarks/elasticity/config_train_transolver_paper_validation.txt

python dataset/benchmarks/elasticity/train_transolver_paper.py `
  --config configs/benchmarks/elasticity/config_train_transolver_paper_validation.txt `
  --eval-only
```

This runner reads the checksum-verified original NPY arrays directly and uses
the released split: cases `0..999` for optimization and `1800..1999` for the
200-case paper test. It exercises this repository's Physics-Attention core with
the released raw-XY input, unclamped irregular-mesh temperature, truncated-
normal slice-projector initialization, decoded per-sample relative-L2 loss,
AdamW, and `CosineAnnealingLR(T_max=500)`. Architecture and optimization remain
width 128, 8 heads, 8 layers, 64 slices, batch 1, learning rate `1e-3`, weight
decay `1e-5`, gradient clipping `0.1`, and 500 epochs. The official source is
pinned locally at commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`.

Before a GPU run, the isolated path can be checked against that pinned source
entirely on CPU:

```powershell
python dataset/benchmarks/elasticity/verify_transolver_paper.py
```

The check imports the upstream model itself and compares initialization,
forward values with an explicitly out-of-range (therefore demonstrably
unclamped) slice temperature, gradients, decoded relative-L2, one clipped
AdamW step, the first cosine-scheduler step, checksums, and exact split
endpoints. The validation runner fixes seed `42` for reproducibility; the
released script did not set or expose a random seed, so no historical paper-run
shuffle/initialization stream can be recovered from the release.

The first as-is run is allowed to finish before this fallback consumes the same
8 GiB GPU. Its result is preserved separately; it is not overwritten.

## Primary paper references

- Transolver and the reported Elasticity comparison table:
  <https://arxiv.org/abs/2402.02366>
- Geo-FNO, the point-cloud Elasticity reference used only as FNO context:
  <https://arxiv.org/abs/2207.05209>
- Original FNO method (Burgers, Darcy, and Navier-Stokes rather than this
  point-cloud Elasticity adapter): <https://arxiv.org/abs/2010.08895>
- Original DeepONet method (different operator tasks):
  <https://arxiv.org/abs/1910.03193>
- Point-DeepONet (DeepJEB and R-squared metrics):
  <https://arxiv.org/abs/2412.18362>
- GINO (3D vehicle surface pressure):
  <https://arxiv.org/abs/2309.00583>
