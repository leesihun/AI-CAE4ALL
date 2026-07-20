# FNO own-paper Darcy validation

This benchmark validates the suite's FNO against the FNO paper's smallest
directly applicable 2D task. The paper's smaller 1D Burgers task is not
applicable because the existing suite implementation accepts only 2D or 3D
operators.

The target is the paper's Darcy-flow result at resolution 85: mean per-sample
relative L2 `0.0108`, using 1,000 training and 200 test cases over 500 epochs.

## Measured result

The final epoch-499 suite checkpoint was evaluated on all 200 isolated cases:

- mean relative L2: `0.1103645243`
- standard deviation: `0.0311230512`
- median: `0.1056127467`
- range: `0.0605456561` to `0.2387737661`
- paper result: `0.0108`
- ratio to paper: `10.2189x`

This fails the paper-accuracy check for the current mesh-adapted FNO. The
dataset and metric were independently audited; the model/protocol differences
described below are material and the result is not attributable only to the
seeded training shuffle.

## Data

Download the original 421x421 MAT files from the public benchmark mirror:

```powershell
hf download kmario23/standard-pde-benchmark `
  darcy/piececonst_r421_N1024_smooth1.mat `
  darcy/piececonst_r421_N1024_smooth2.mat `
  --repo-type dataset `
  --local-dir dataset/benchmarks/fno_darcy/source

python dataset/benchmarks/fno_darcy/prepare_darcy.py
```

The converter checks both LFS SHA-256 values, downsamples 421x421 to 85x85 by
the paper's stride of five, and writes:

- `darcy_train.h5`: 1,250 non-test cases. The unchanged seeded 80/10/10 loader
  split yields 1,000 optimization, 125 validation, and 125 internal-test cases.
- `darcy_test.h5`: the first 200 cases of `smooth2`, isolated from training and
  retained in source order.

Because the production loader is unchanged, the 1,000 optimization cases are
a seeded selection from the 1,250-case pool rather than exactly the first 1,000
`smooth1` cases. The additional pool cases are `smooth2` indices 200-449, so
none overlaps the official 200-case evaluation partition.

The suite's temporal contract represents each operator pair as one transition:
the Darcy coefficient is field row 3 at timestep 0 and the solution is row 3 at
timestep 1. Inference therefore saves the predicted physical solution at
timestep 1 without changing runtime code.

## Train, infer, and evaluate

```powershell
python CAE_ML_Suite_main.py --config configs/benchmarks/fno_darcy/config_train_fno_paper.txt
python CAE_ML_Suite_main.py --config configs/benchmarks/fno_darcy/config_infer_fno_paper.txt
python dataset/benchmarks/fno_darcy/evaluate_relative_l2.py `
  --predictions output/benchmarks/fno_darcy/fno/inference
```

The external evaluator reads the de-normalized saved solution and computes the
paper metric over all 200 cases. It rejects missing cases, duplicate rollouts,
geometry mismatches, and non-finite output.

This is a same-task, same-resolution, same-metric implementation check, but not
a strict reproduction. The suite retains its normalized residual-MSE training,
native warmup/cosine schedule, shuffled training composition, mesh-to-grid
adapter, and current FNO block details.

## Opt-in paper-validation mode

The original as-is workflow above remains unchanged. For a closer paper check,
prepare separate benchmark files and use the explicit `paper_darcy` configs:

```powershell
python dataset/benchmarks/fno_darcy/prepare_darcy.py --paper-protocol
python CAE_ML_Suite_main.py --config configs/benchmarks/fno_darcy/config_train_fno_paper_validation.txt
python CAE_ML_Suite_main.py --config configs/benchmarks/fno_darcy/config_infer_fno_paper_validation.txt
python dataset/benchmarks/fno_darcy/evaluate_relative_l2.py `
  --ground-truth dataset/benchmarks/fno_darcy/darcy_paper_test.h5 `
  --predictions output/benchmarks/fno_darcy/paper_validation/inference
```

This mode is opt-in and leaves all default model/training paths unchanged. It:

- maps the suite's seed-42 optimization IDs to exactly `smooth1[0:1000]` and
  keeps `smooth2[0:200]` as the isolated paper test;
- makes the unchanged temporal residual contract target the direct solution by
  storing `coefficient + solution` at timestep 1;
- uses the three-channel `[coefficient, x, y]` paper FNO core, four layers with
  ReLU after only the first three, and the `32 -> 128 -> 1` projection;
- optimizes decoded per-sample relative L2 with paper-era Adam and
  `StepLR(step_size=100, gamma=0.5)`, without warmup or gradient clipping.

Remaining qualifications are deliberately small and explicit: the exact grid
still passes through the deterministic splat/sample wrapper, and the suite fits
one scalar z-score per field whereas the released paper-era script fits
grid-point-wise normalizers. On this regular 85x85 dataset the adapter is an
identity data-layout operation, but the normalization distinction means this is
still a close validation mode rather than a byte-for-byte reproduction script.

Primary sources:

- FNO paper and table: <https://arxiv.org/abs/2010.08895>
- Public NeuralOperator Darcy record: <https://zenodo.org/records/12784353>
- Original released 2D recipe mirror:
  <https://github.com/li-Pingan/fourier-neural-operator/blob/main/FNO-torch.1.6/fourier_2d.py>
