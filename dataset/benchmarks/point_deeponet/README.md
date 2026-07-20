# Point-DeepONet selective benchmark preparation

This directory prepares the authors' smallest explicitly quantified experiment:
1,000 load cases, 5,000 sampled nodes per case, and the released 800/200 split.
It is isolated from normal `Neural_Operator` training and does not alter a
production loader or model path.

Validation anchors are the accepted paper
[`arXiv:2412.18362v2`](https://arxiv.org/abs/2412.18362v2), the authors'
official repository pinned locally at commit
`a22c34719dd629f48c099589f172451d5224a072`, and Kaggle dataset version 2
(`jangseop/point-deeponet-dataset`).

## Why range extraction is needed

The Kaggle release is about 99.15 GiB in full. Kaggle's combined
`archive.zip` recompresses `targets.npz` and `xyzdmlc.npz`, so it cannot be
randomly accessed inside those files. The preparer instead calls Kaggle's
official individual-file API for each NPZ. The returned GCS objects support
byte ranges, and the NPZ members themselves are stored rather than deflated.
Only the two selected `.npy` members for each case are downloaded.

For the exact 1,000-case selection, the current remote plan is:

- 800 training and 200 validation cases
- 837 unique bracket geometries
- 3.138 GiB of target members
- 7.060 GiB of `xyzdmlc` members
- 10.197 GiB total selected payload

## Safe commands

Inspect one real case and its exact remote ranges without writing data:

```powershell
python dataset\benchmarks\point_deeponet\prepare_dataset.py --dry-run --limit-cases 1
```

Create only the full selection, split, download-plan, and provenance manifests:

```powershell
python dataset\benchmarks\point_deeponet\prepare_dataset.py
```

Download and prepare one real case as a smoke test:

```powershell
python dataset\benchmarks\point_deeponet\prepare_dataset.py --download --limit-cases 1
```

Start the explicit 10.197 GiB selected transfer only after the smoke test:

```powershell
python dataset\benchmarks\point_deeponet\prepare_dataset.py --download
```

Each completed case is an atomic compressed NPZ under
`prepared/n1000_p5000/cases/{train,valid}/`. Interrupted raw-member downloads
remain under `.staging/` and resume from their current byte count. Completed
case files are validated and skipped on later runs. Every remote `.npy` member
is checked against its ZIP CRC before it is loaded.

## Reproduced author protocol

`prepare_dataset.py` reproduces the released preprocessing notebooks' 33-item
exclusion list, sorted item names, `ver`/`hor`/`dia` prefix order, legacy NumPy
seed-42 sampling without replacement, stable mass sort, fresh seed-42 shuffle,
and 80/20 split.

The released node-sampling notebook is not exactly replayable per case: it
iterates `list(set(...))`, whose order varies with Python's hash seed, while a
single process-global RNG advances across that unstable order. This preparer
keeps the authors' replacement rule but derives an order-independent seed from
SHA-256 of the base seed and case name. It stores both the resulting indices
and seed in every case file and records the qualification in `provenance.json`.

## Output schema

Each case file contains:

- `xyzdmlc`: float32 `[5000, 9]`
- `targets`: float32 `[5000, 4]`
- `sample_indices`: int32 `[5000]`
- case/split/mass, original-node-count, and sampling-seed metadata

No shared production training or inference path reads these files. The
benchmark-only adapter/trainer is an explicit opt-in path in
`paper_benchmark.py`.

## Faithful paper benchmark

The isolated benchmark follows the authors' released executable model:

- PointNet input: per-node `xyz`
- global branch input: one per-case `m/l/cx/cy/cz` vector
- trunk input: per-node `xyzd`, where `d` is signed distance (not a case constant)
- outputs: direct `ux/uy/uz/von Mises`, with the released direction-specific clipping
- released SIREN encoder, PointNet, multiplication/mean fusion, output refiner,
  and final `tanh` topology (strictly 251,936 trainable parameters)
- MSE, AdamW (`lr=1e-3`, `weight_decay=1e-5`), the released DeepXDE inverse-time
  learning-rate schedule and NumPy batch sampler, batch size 16, seed 2024,
  and 40,000 iterations

For direct paper parity, the paper profile fits its min/max ranges on all 1,000
selected cases before applying the 800/200 split, exactly as the release does.
This is validation leakage and is reported as such. The guarded CPU smoke keeps
train-only scaling because it is a wiring test, not a paper result.

The paper's 1,000-case Point-DeepONet comparator is average R2 = 0.897. Figure
20 and the released evaluator define that value as the arithmetic mean of 12
sampled-data values: for each
of three load directions and four output fields, R2 is first computed after
pooling all validation cases and sampled points for that direction/field. The
evaluator emits all 12 terms and only labels a result paper-comparable when all
12 are present. Full-mesh inference is a separate paper evaluation and is not
the source of the 0.897 dataset-size comparator.

The 1,000-case direction counts (336 vertical, 325 horizontal, 339 diagonal)
match Figure 20/Table 6. One source limitation remains: the released sampling
notebook iterates a Python `set` after seeding a process-global NumPy RNG, so its
exact 5,000 node indices cannot be reconstructed from the repository. The
preparer therefore uses a stable per-case seed while retaining the same sampling
rule. This changes the sampled nodes, not the cases, split sizes, topology, loss,
or reported R2 definition.

Strictly validate the full prepared dataset before training:

```powershell
python dataset\benchmarks\point_deeponet\paper_benchmark.py validate `
  --config dataset\benchmarks\point_deeponet\configs\paper_n1000_p5000.json `
  --deep
```

Start the isolated paper run only after that command succeeds:

```powershell
python dataset\benchmarks\point_deeponet\paper_benchmark.py train `
  --config dataset\benchmarks\point_deeponet\configs\paper_n1000_p5000.json
```

Re-evaluate a saved checkpoint in physical output units:

```powershell
python dataset\benchmarks\point_deeponet\paper_benchmark.py evaluate `
  --config dataset\benchmarks\point_deeponet\configs\paper_n1000_p5000.json `
  --checkpoint dataset\benchmarks\point_deeponet\outputs\paper_n1000_p5000\checkpoint_last.pt
```

The `configs/cpu_smoke.json` profile is guarded to CPU, at most two cases,
batch size at most two, and at most five iterations. Its partial-direction R2
is a wiring test only and is explicitly marked non-comparable to 0.897.
