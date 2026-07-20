# Transient Plasticity Dataset

`plasticity.h5` is the suite-native conversion of the Geo-FNO/Transolver
plastic-forging benchmark. It contains 987 cases, a logical `101 x 31` quad
mesh (3,131 nodes), and 20 stored time states.

## HDF5 layout

Each case uses the shared mesh-state contract:

```text
data/{sample_id}/nodal_data   [8, 20, 3131]
data/{sample_id}/mesh_edge    [2, 6130]
data/{sample_id}/die_profile  [101]
```

The `nodal_data` feature order is:

```text
0  x_ref_mm
1  y_ref_mm
2  z_ref_mm               (zero; 2-D plane-strain padding)
3  u_x_mm
4  u_y_mm
5  u_z_mm                 (zero; 2-D plane-strain padding)
6  die_profile_mm         (static conditioning, broadcast over y and time)
7  node_type              (zero; no released boundary labels)
```

For the unchanged temporal loaders, use `input_var 4`, `output_var 4`, and
`use_node_types False`. The state is `[u_x, u_y, u_z, die_profile]`. The die
profile has an exact zero target delta and should be excluded from physical
discrepancy metrics; score displacement channels only.

Because this is a 2-D problem and the die is static, preprocessing will report
near-zero delta variance for `u_z` and `die_profile`. This is expected. The
loader clamps their standard deviations to `1e-8`, which keeps both channels
finite and makes autoregressive drift in the static condition negligible.

The raw source stores current coordinates and displacement. Conversion uses

```text
reference_xy = mean_t(current_xy - displacement_xy)
```

so the normal suite geometry rule `current_xyz = reference_xyz + displacement`
remains valid without double-counting deformation.

## Splits and provenance

The file preserves the released split metadata:

- `splits/train`: source cases 0..899
- `splits/test`: source cases 907..986
- `splits/unused`: source cases 900..906
- `splits/val`: empty (the release did not define a validation partition)

The current model loaders still create their own seeded 80/10/10 split and do
not consume HDF5 split metadata. Benchmark tooling must select the stored split
explicitly when exact released-test isolation matters.

The paired benchmark configs use the suite-native split instead: training reads
`plasticity.h5` with `split_seed 42`, producing 789/98/100 train/validation/test
cases. `plasticity_seed42_test.h5` contains exactly those 100 held-out test IDs
and is the only file used for rollout inference. Rebuild it without touching
model code using:

```powershell
python dataset\benchmarks\plasticity\prepare_suite_test_split.py
```

All benchmark models learn one-step state deltas. Inference starts only from
time state 0 and performs 19 single-step autoregressive updates. For the final
time-averaged discrepancy, compare predicted and true displacement over time;
do not score the static die-profile channel.

The benchmark config set contains both graph baselines. `meshgraphnets` is the
standard single-scale processor. `hi_meshgraphnets` uses a two-level HI-MGN
V-cycle with `voronoi_seedmean` and 3,131 -> 500 -> 100 nodes. Their
checkpoints, rollout directories, and writable normalization copies are
separate.

MeshGraphNets writes train-fit normalization metadata into its configured
dataset. The two graph variants therefore point to independent, initially
bit-identical working copies so that `plasticity.h5` retains its recorded
provenance hash and the variants cannot race while writing HDF5 metadata.
Recreate both copies before a fresh graph campaign with:

```powershell
Copy-Item `
  dataset\benchmarks\plasticity\plasticity.h5 `
  dataset\benchmarks\plasticity\plasticity_meshgraphnets_runtime.h5
Copy-Item `
  dataset\benchmarks\plasticity\plasticity.h5 `
  dataset\benchmarks\plasticity\plasticity_hi_meshgraphnets_runtime.h5
```

`plasticity.provenance.json` records the deleted MAT artifact's SHA-256, source
shape, field mapping, split, HDF5 checksum, and exhaustive conversion errors.

The reproducible conversion command is:

```powershell
python dataset\benchmarks\plasticity\prepare_plasticity.py --delete-source
```

The converter deletes the MAT source only after checksum verification and a
full 987-case source-to-HDF5 audit.

## Strict post-inference evaluation

[`evaluate_rollouts.py`](evaluate_rollouts.py) is an isolated, read-only
postprocessor for the seven suite models. It requires the exact seed-42 held-out
IDs and exactly 100 `rollout_sample<ID>_steps19.h5` files. Every rollout must
contain an `[8, 20, 3131]` finite nodal trajectory with geometry and the seed
state channels `[u_x, u_y, u_z, die_profile]` matching the held-out truth. The
production ground-truth SHA-256 is pinned to
`5970cdcd362e94f5a54e0f7d18893b11c51f5e1ab345712bddfbbe8d130ad8be`;
the production CLI provides no override. Mesh edges are required and verified
for every model except Transolver, whose rollout writer does not include them.
Each rollout must also identify its exact sample ID and provide one consistent,
model-appropriate checkpoint path and config path across the complete set.

The primary metric is the arithmetic mean over cases of full-trajectory
relative L2 on de-normalized `u_x`, `u_y`, and `u_z` over time indices 1..19:

```text
mean_case ||predicted_displacement - true_displacement||_2
          / ||true_displacement||_2
```

Time index 0 is only the exact rollout seed. Saved channel 6 is the static die
profile and is never scored. The JSON also reports global trajectory relative
L2, mean per-case time-averaged per-timestep relative L2, final-time mean
relative L2, per-component RMSE/MAE, validation tolerances, input hashes, and
rollout timing when `total_rollout_time_s` is present. The CSV contains one row
per case and evaluated time, or 1,900 rows for a complete model result. Its
byte count, row count, and SHA-256 are recorded in the JSON. The comparison
tool reopens and hashes that CSV and independently reconstructs the reported
metrics before it accepts a model result.

Evaluate each completed inference directory from the suite root:

```powershell
$models = @('meshgraphnets', 'hi_meshgraphnets', 'deeponet', 'point_deeponet', 'fno', 'gino', 'transolver')
foreach ($model in $models) {
  python dataset/benchmarks/plasticity/evaluate_rollouts.py `
    --model $model `
    --predictions "output/benchmarks/plasticity/$model/inference"
  if ($LASTEXITCODE -ne 0) { throw "Plasticity evaluation failed: $model" }
}
```

Each inference directory receives `plasticity_metrics.json` and
`plasticity_per_case_time.csv`. Combine them only after all seven pass:

```powershell
python dataset/benchmarks/plasticity/compare_results.py `
  --results-root output/benchmarks/plasticity
```

The comparison writes `comparison.json`, `comparison.csv`, and
`comparison.md`, ranked by the primary metric. If any model is missing or
invalid, it writes an explicit `complete: false` diagnostic comparison and
exits with status 2; absent models are never presented as successful results.

Run the synthetic evaluator and comparison regressions with:

```powershell
python -m pytest -q dataset/benchmarks/plasticity/tests
```

## Resource-gated seven-model campaign

[`run_campaign.py`](run_campaign.py) is the isolated campaign entrypoint. It
does not import or modify model runtime code. A real run fails closed before
preflight or runtime-artifact creation unless
`output/benchmarks/paper_validation_completion_gate.json` passes the exact
five-validation, report, artifact-path, and fresh SHA-256 contract. Dry-run is
read-only and does not require that gate.

Every fresh model follows one private DAG:

```text
train -> final checkpoint identity/hash gate -> infer -> strict evaluate
```

The coordinator owns every child process tree, writes per-job attempt logs and
PID/state records atomically, and invokes `compare_results.py` only after all
seven results pass. The internal model key remains `transolver`; its display
name is **Transolver-3 (direct associative/tiled path)**.

Concurrency is evidence-gated. Without a valid resource-probe index, the safe
fallback is one GPU model at a time. A second GPU model may overlap only for an
exact pair whose rehashed probe record certifies at most 6,656 MiB peak use and
at least 10% aggregate-throughput improvement. DeepONet alone may occupy one
CPU lane concurrently with GPU work, and only when a rehashed matching CPU
record marks it eligible with a finite conservative projection. Its generated
runtime config uses `gpu_ids -1`, `use_amp False`, and bounded CPU-thread
environment variables. Missing, stale, aliased, or malformed evidence grants
no extra lane.

Generated runtime profiles preserve effective batch four without changing the
checked-in configs:

| Model | Safe profile order |
|---|---|
| MeshGraphNets | `4x1 -> 2x2 -> 1x4` |
| HI-MeshGraphNets | `4x1 -> 2x2 -> 1x4` |
| Point-DeepONet | fixed `2x2` because of BatchNorm |
| DeepONet | `4x1 -> 2x2 -> 1x4`; CPU only with strict evidence |
| FNO | `4x1 -> 2x2 -> 1x4` |
| GINO | fixed `1x4` |
| Transolver-3 | fixed `1x4` |

Here `BxA` means physical batch `B` and gradient accumulation `A`. There is no
automatic OOM retry into the same output directory: a failed selected profile
stops the campaign and must be resolved explicitly.

The common training budget is aligned across all seven models:

| Quantity | Common value |
|---|---:|
| Train transition pairs per epoch | `789 * 19 = 14,991` |
| Epochs | `500` |
| Effective batch | `4` |
| Optimizer updates per epoch | `3,748` |
| Total optimizer updates | `1,874,000` |
| Total pair exposures | `7,495,500` |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `1e-4` |
| Shared controls | 3 warmup epochs, EMA `0.99`, gradient clip `3`, epoch-level schedule |

The last optimizer update has three examples in every profile. The one exact
weighting exception is Point-DeepONet `2x2`: its final accumulation window has
physical batches of two and one, and the training loop averages those two batch
means equally. The singleton therefore has 50% rather than 33.3% weight in
only 1 of 3,748 updates per epoch. No sample is dropped, and update/exposure
counts remain aligned. The hot training loop is intentionally unchanged. The
upstream DrivAer Transolver-3 schedule is task-specific (800 surface / 600
volume epochs), so those epoch counts are not copied into Plasticity.

Inspect every command and run all train-config preflights without starting a
model or writing status, logs, model outputs, metrics, runtime configs, or a
Transolver runtime HDF5:

```powershell
python dataset/benchmarks/plasticity/run_campaign.py --dry-run
```

Start the full campaign only after reviewing that output. When the user has
explicitly redirected priority to Plasticity before the paper-validation gate
is complete, use the opt-in waiver; it is recorded as `waived_by_user` and does
not relabel any paper result as passed:

```powershell
python dataset/benchmarks/plasticity/run_campaign.py `
  --allow-incomplete-paper-validation
```

For a runtime-limited preliminary comparison, keep the canonical files at 500
epochs and make the reduced budget explicit in runtime-only configs:

```powershell
python dataset/benchmarks/plasticity/run_campaign.py `
  --allow-incomplete-paper-validation `
  --epoch-budget 5
```

The status and checkpoint gate record the requested epoch budget. Budgets below
500 use one validation per epoch and a shortened warmup; the default 500-epoch
campaign is unchanged.

The atomic state file is
`output/benchmarks/plasticity/campaign_status.json`. Separate stdout and stderr
logs are stored under `output/benchmarks/plasticity/campaign_logs/<model>/`,
using names such as `train.attempt01.stdout.log`. Campaign completion is true
only when every model is complete and the strict seven-model comparison
succeeds.

Every real invocation first acquires the kernel-owned exclusive lock at
`output/benchmarks/plasticity/.campaign.lock`; its PID JSON is diagnostic and
never substitutes for the OS lock. The file persists after release so stale
metadata cannot create a delete/recreate race. On Windows each concurrent
launcher is assigned its own kill-on-close Job Object. Interruption or an
orchestration exception terminates every active owned launcher/trainer
descendant tree, preventing detached trainers from surviving the wrapper.

The campaign pins the SHA-256 of all fourteen train/infer configs and the held-out
truth HDF5, and checks critical semantics including GPU 0, seed 42, 19 rollout
steps, four input/output channels, and 500 train epochs. A checkpoint is
accepted only when its backend schema and model/data identity are valid and its
epoch is exactly 499. Its SHA-256 is checked before inference and again after
evaluation. The strict comparison validator reconstructs the JSON, CSV,
rollout manifest, hashes, truth identity, and metrics before completion.

Recovery is artifact-based. A preexisting model is skipped only when its
nonempty checkpoint, exact 100-file rollout set, existing metrics JSON/CSV,
and a fresh strict evaluator run all pass. Any partial or unexpected model
output is refused instead of overwritten. A subset run never produces a
comparison or marks the campaign complete while another model result is absent
or invalid.

A fresh baseline or HI-MeshGraphNets run requires its own runtime HDF5 to be
bit-identical to the provenance-pinned source. Recovery may reuse a changed
working HDF5 only after the completed result passes and an audit confines all
differences to expected normalization metadata.

A fresh Transolver-3 run atomically creates or verifies a pristine
`plasticity_transolver_runtime.h5`, points generated train/infer configs to it,
and enables preprocessing writes only for training. Completion requires a
logical HDF5 audit proving that only the expected Transolver normalization
namespace changed; canonical `plasticity.h5` must remain byte-identical.

## Deferred, isolated resource probes

[`resource_probe.py`](resource_probe.py) profiles the seven Plasticity models
without touching the canonical campaign configs, datasets, checkpoints, or
rollouts. **Do not execute a real probe while any own-paper validation is
running.** The utility enforces that policy: real mode requires the explicit
`--execute` switch and refuses before creating outputs unless
`output/benchmarks/paper_validation_completion_gate.json` has this strict
machine-readable contract:

```json
{
  "schema_version": "paper_validation_completion_gate_v1",
  "complete": true,
  "status": "passed",
  "completed_at": "<ISO-8601 timestamp>",
  "report": {
    "path": "<repo-relative report path>",
    "sha256": "<64 lowercase hex>"
  },
  "validations": {
    "fno": {
      "complete": true,
      "status": "passed",
      "benchmark": "<paper benchmark>",
      "metric": "<paper metric>",
      "paper_value": 0.0,
      "measured_value": 0.0,
      "primary_artifact": {
        "path": "<unique repo-relative artifact path>",
        "sha256": "<64 lowercase hex>"
      }
    },
    "transolver": {"...": "same exact fields; optional supporting_artifacts"},
    "deeponet": {"...": "same exact fields"},
    "point_deeponet": {"...": "same exact fields"},
    "gino": {"...": "same exact fields"}
  }
}
```

The campaign requires exactly those five validation keys, re-hashes every
referenced regular file inside the repository, and rejects duplicate primary
artifacts or aliases of the report.

Read-only planning does not require or validate the gate and writes nothing:

```powershell
python dataset/benchmarks/plasticity/resource_probe.py `
  --dry-run --model hi_meshgraphnets --device gpu

python dataset/benchmarks/plasticity/resource_probe.py `
  --dry-run --pair deeponet fno `
  --profile deeponet=fallback_1x4 `
  --profile fno=fallback_1x4 --device gpu
```

When invoked after the gate passes, the tool deterministically selects 32
cases from the seed-42 suite training partition and creates a schema-preserving
probe HDF5 below the run directory. MeshGraphNets, HI-MeshGraphNets, and
Transolver each receive a separate, pristine, hash-verified writable copy;
they never share a writable file or point at `plasticity.h5`. Runtime configs,
one-epoch checkpoints, logs, and all other outputs remain below:

```text
output/benchmarks/plasticity/resource_probe/runs/<run-id>/
```

Available profiles retain nominal effective batch size four. Baseline MGN,
DeepONet, and FNO provide `canonical`, `fallback_2x2`, and `fallback_1x4`;
Point-DeepONet is fixed to its BatchNorm-preserving canonical `2x2`; GINO and
Transolver remain `1x4`. HI-MGN defaults to `1x4` even though its checked-in
canonical profile is retained for comparison.

CPU mode writes a runtime-only `gpu_ids -1`, disables AMP, sets
`CUDA_VISIBLE_DEVICES=-1`, and caps `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS`. Only a complete matching DeepONet CPU record is marked
CPU-eligible. It contains a conservative 500-epoch wall-time projection based
on the end-to-end 32-case probe; missing, stale, or invalid CPU evidence means
GPU placement.

An explicit two-model GPU pair requires two completed individual GPU result
records supplied with repeated `--baseline-record`. It is certified only when
both children complete the same work, total observed GPU use is at most 6,656
MiB, monitoring is complete, and aggregate throughput improves by at least
10% over the two individual sequential wall times. Non-certified pair runs
are recorded but return a nonzero status.

Every completed result is written atomically as
`resource_probe_result.json`. Scheduler discovery uses only the atomic
`output/benchmarks/plasticity/resource_probe/index.json`, whose
`latest_completed_single`, `latest_cpu_eligible`, and `latest_certified_pair`
entries contain a result path, SHA-256, and exact dataset/model/profile
identity. Consumers must re-hash and revalidate the referenced result; they
must not select records by filesystem modification time.

Within each index identity, `models_key` is the comma-joined sorted model list,
and `profile_key` is the no-whitespace, comma-joined sorted mapping
`model=profile` (for example,
`deeponet=fallback_1x4,fno=canonical`). `identity_key` is the lowercase SHA-256
of the compact sorted-key JSON encoding of the full identity object; consumers
should match the explicit fields and use the hash for integrity and deduplication.
