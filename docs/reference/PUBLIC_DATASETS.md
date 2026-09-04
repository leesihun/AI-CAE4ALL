# Public test datasets

Candidate public CAE/ML datasets for exercising this repo, why each was picked, and how to
rebuild it into the shared mesh HDF5 contract ([DATASET_FORMAT.md](DATASET_FORMAT.md)).

Everything here is **< 100 GB** and redistributable for research.

> **2026-08-18 restructuring note:** `ex4`/`ex5`/`ex6` (cylinder_flow/deforming_plate/flag_simple)
> used to ship `smoke`/`arrt` tiers -- strided subsamples of the raw trajectory, built purely as
> dev-time shortcuts (faster preflight, tractable AR-RT backprop length). They were never
> genuinely useful as *test* sets: every tier read from the same `train.tfrecord` pool with no
> held-out samples and no distributional difference from the training data, and by 2026-08-18 the
> `_smoke.h5` files referenced by every ex4-ex6 config had silently gone missing (same
> gitignored-`.h5`-plus-uncommitted-script vanishing pattern documented below for the 2026-08-09
> recovery) while nothing consumed the unused full-size `_arrt`/plain files sitting next to them.
> Removed the tier concept entirely. `build_public_mgn.py` now reads a single pool of trajectories
> per dataset (default 120: 100 train + 20 infer) at full temporal resolution, scores each one on
> a dataset-specific physical axis, and assigns the **highest** values to the infer split so
> evaluation is a genuine extrapolation test rather than a same-distribution held-out sample:
> inflow speed for cylinder_flow, actuator indentation depth for deforming_plate, and (since
> flag_simple stores no per-trajectory control parameter -- the HANDLE nodes barely move at all)
> overall cloth-motion magnitude as an outcome-based proxy for flag_simple. Every `exN` now has
> exactly two files, matching the `ex1.h5`/`ex1_infer.h5` convention: `exN.h5` (train) and
> `exN_infer.h5` (extrapolation test). See the per-dataset sections below and
> `build_public_mgn.py`'s module docstring for the exact axis definitions.
>
> **2026-08-09 recovery note:** `build_public_mgn.py`, `build_public_airfrans.py`, and every
> `configs/{MeshGraphNets,Transolver,Neural_Operator}/{ex4,ex5}/*.txt` referenced below had been
> deleted from disk (never committed -- `*.h5` is gitignored so the built files survived, the
> scripts and configs that made and consumed them did not). Both converters were reconstructed
> from scratch by reverse-engineering the raw TFRecord/VTU schemas and validated **byte-exact**
> against the surviving `.h5` files (`nodal_data`/`mesh_edge` `np.allclose`/`array_equal` against
> real samples). AirfRANS surface normals are **zero at volume nodes**, not
> nearest-neighbor-propagated. flag_simple was also newly built (raw TFRecords were already
> staged, just never converted) and the AirfRANS extrapolation splits (the dataset's actual
> selling point) were built for the first time.

## What was already covered, and the gap

| Existing | Kind | Exercises |
| --- | --- | --- |
| `ex1.h5`, `ex2.h5` | proprietary warpage | temporal, node types, multiscale |
| `ex3_NASA_CRM_*.h5` | public, static surface aero, 149 samples | `cond_var`, large meshes (454k nodes) |
| `deepjeb.h5` | structural, fixed geometry | SimulGenVAE dense FOM |
| `benchmarks/*` | paper-replication (Darcy, elasticity, …) | Neural_Operator / Transolver baselines |

NASA-CRM is a good pick and is already built. Its limit is that it is **static (T=1)** with a
**fixed topology**, so nothing public in `dataset/` exercised the autoregressive rollout,
`use_world_edges`, or per-sample varying meshes. The datasets below fill exactly that gap.

## Staged and built

Raw sources are staged on `D:/CAE_datasets_raw` (not `C:`, which has ~115 GB free) by
`dataset/fetch_public_datasets.sh`. Converted HDF5 land in `dataset/` so the configs' relative
`../dataset/...` paths resolve.

### 1. DeepMind `cylinder_flow` — 15.2 GB raw

Incompressible flow past a cylinder. **600 timesteps**, ~1.9k nodes, triangular, and each
trajectory carries **its own mesh**, so it covers varied topology as well as time.

    rows: [x, y, z=0 | vel_x, vel_y, pressure | node_type]      input_var = output_var = 3

Node types are the canonical MGN set: 0 normal, 4 inflow, 5 outflow, 6 wall.

### 2. DeepMind `deforming_plate` — 10.7 GB raw

A rigid actuator pressed into a hyperelastic plate. **400 timesteps**, ~0.8-1.2k tetrahedral
nodes. This is the only dataset here with **contact**, so it is the only one that exercises
`use_world_edges`.

    rows: [x, y, z | disp_x, disp_y, disp_z, stress | node_type]  input_var = output_var = 4

Node types: 0 normal, 1 obstacle (actuator), 3 handle. The source stores absolute
`world_pos`; the converter writes **displacement** (`world_pos - mesh_pos`) because the loader
integrates state as `x[t+1] = x[t] + delta`.

Rebuild either with:

```bash
python dataset/build_public_mgn.py --dataset cylinder_flow
python dataset/build_public_mgn.py --dataset deforming_plate
```

`build_public_mgn.py` needs **TensorFlow** (CPU-only is fine) purely to decode the raw
`tf.train.Example` TFRecords -- no GPU, no `tf.data` training ops.

Each run reads `--n-train + --n-infer` trajectories (default 100 + 20) from `train.tfrecord` at
**full temporal resolution** (no striding -- the old `smoke`/`arrt` tiers are gone), scores every
trajectory on a dataset-specific physical axis, and writes the lowest-scoring `n-train` to
`exN.h5` and the highest-scoring `n-infer` to `exN_infer.h5`:

| Dataset | Axis | Kind |
| --- | --- | --- |
| `cylinder_flow` | mean inflow speed at t=0 (`node_type==4`) | true control parameter |
| `deforming_plate` | z-range of the OBSTACLE centroid (`node_type==1`) over the trajectory, i.e. indentation depth | true control parameter |
| `flag_simple` | mean cloth displacement magnitude (`node_type==0`) | outcome-based proxy -- see section 3 |

Each sample's `metadata/extrapolation_axis` attribute stores its raw axis value if you need to
re-derive the split boundary. `time_integration ar_rt` unrolls the **whole** stored trajectory and
backprops through every step
([time_integration.py](../../methods/MeshGraphNets/general_modules/time_integration.py)); at full resolution
that is 599/399/400 chained steps per item for cylinder_flow/deforming_plate/flag_simple, so
AR-RT training runs on these full-length files should use a small `Batch_size` and expect a long
backward pass -- there is no shorter-trajectory tier to fall back to anymore.

### 3. DeepMind `flag_simple` — 11.4 GB raw

A flag (cloth) pinned to a pole in the wind, with self-contact. **401 timesteps**, triangular.
Unlike cylinder_flow/deforming_plate, **every one of the ~1000 trajectories shares the same
1579-node/4606-edge template mesh** (verified across all sampled records) -- nothing else in
`ex1`-`ex4` exercises a temporal, contact (`use_world_edges`) dataset with **fixed topology**
across samples. That also makes it, unlike cylinder_flow/deforming_plate, structurally eligible
for `simulgenvae`'s fixed-node/fixed-timestep dense-FOM constraint (untested here; `deepjeb.h5`
already covers that method).

    rows: [x, y, z=0 | disp_x, disp_y, disp_z | node_type]      input_var = output_var = 3

**Correction to this file's own earlier candidate-table entry:** `node_type` is tagged `"type":
"dynamic"` in DeepMind's `meta.json` (shape `[401, 1579, 1]`), which earlier read as "dynamic
node types (they change per timestep), which nothing else here covers." Checked against the raw
data directly: across every sampled trajectory, `node_type` **never actually changes over time**
-- 0 nodes out of 1579 differ from their t=0 value. The dynamic-shaped storage looks like a
DeepMind schema convention, not a real per-timestep signal in this dataset. Time-varying node
types remain an uncovered *code path* (nothing in this repo's loaders reads node type at
anything but t=0) as well as an uncovered *data* path -- flag_simple does not fill it after all.
7 rows, so it has the same `num_features > 7` node-type-guard caveat as `cylinder_flow` (see
below).

Unlike cylinder_flow/deforming_plate, flag_simple has **no stored per-trajectory control
parameter to extrapolate on**: the 2 HANDLE nodes (`node_type==3`) move by ~0.01-0.015 units
(path length) across every trajectory sampled -- essentially constant, not the driver of the
5x variation seen in overall cloth motion. That variation (presumably from a hidden per-trajectory
initial condition or wind parameter that isn't serialized into the TFRecord) can only be split on
as an *outcome* -- `exN_infer.h5` holds out the highest-motion trajectories, testing
generalization to more energetic dynamics rather than true out-of-range-parameter extrapolation.

```bash
python dataset/build_public_mgn.py --dataset flag_simple
```

### 4. AirfRANS — 9.3 GB raw

1000 steady RANS solutions over NACA 4/5-digit airfoils, Re 2-6M, AoA -5°..15°, ~155k nodes
each (2D slice, all quad cells). Static like NASA-CRM but with **6.7x more samples**, a
different mesh per sample, and published **extrapolation** splits — the only dataset here
that tests generalization rather than interpolation.

    rows 0:3    x, y, z(=0)
    rows 3:7    u_x, u_y, p, nu_t                 state       input_var = output_var = 4
    rows 7:12   u_inf_x, u_inf_y, sdf, n_x, n_y   conditions  cond_var = 5
    row  12     node_type                         0 volume, 1 airfoil surface

The condition rows are exactly the AirfRANS paper's input features. `u_inf_x/y` are recovered
from the folder name (`airFoil2D_SST_<U_inf>_<AoA>_<naca…>`, first two numeric tokens) and are
per-sample constants (`u_inf_x = U_inf * cos(AoA)`, `u_inf_y = U_inf * sin(AoA)`, AoA in
degrees). `sdf` is the VTU's own `implicit_distance` field, unmodified (its sign convention is
whatever OpenFOAM/AirfRANS wrote -- `<= 0` empirically, not a true signed distance flipping at
the surface). **`n_x`/`n_y` are zero at every volume node and only nonzero at the 1007
airfoil-surface nodes** (`node_type == 1`, itself derived from `implicit_distance == 0`) --
verified against real data; they are *not* nearest-neighbor-propagated into the volume the way
`sdf` effectively is. Do not assume a smooth normal field away from the surface.

> **`augment_geometry` must stay `False`.** `u_inf` and the normals are vector *components*,
> so a z-rotation would rotate the geometry without rotating them — the caveat
> DATASET_FORMAT.md raises for conditioning rows.

```bash
python dataset/build_public_airfrans.py --task smoke          # 40 samples @ 16k nodes, 34 MB
python dataset/build_public_airfrans.py --task scarce_train   # 200 samples, full resolution
python dataset/build_public_airfrans.py --task full_test      # 200 samples, full resolution
```

The `--task` names come straight from the dataset's own `manifest.json`, so
`reynolds_train`/`reynolds_test` and `aoa_train`/`aoa_test` build the extrapolation studies --
**this is the actual reason AirfRANS was picked** (published Re/AoA extrapolation splits, the
only dataset here that tests generalization rather than interpolation). Because `split_strategy
hdf5` is not implemented in this runtime, each task is written as its **own file** — point
`dataset_dir` at the train task and `infer_dataset` at the test task rather than relying on the
seeded 80/10/10 split.

**2026-08-18: `ex7`'s configs switched to the AoA extrapolation split** (804/196 samples), fixing
a self-reference bug where `dataset_dir`/`infer_dataset` both pointed at the identical
`scarce_train.h5`, so "inference" evaluated on the same 200 samples the model trained on -- no
real generalization test at all.

**2026-08-19: consolidated to a single `ex7.h5`/`ex7_infer.h5` pair.** Checked whether the
`scarce_train`/`full_test`/`reynolds_{train,test}`/`aoa_{train,test}` splits were independent or
overlapping data by comparing sample IDs across all six built files: they are **six different
train/test partitions of the exact same 1000 unique AirfRANS samples** -- the aoa-split union
alone already equals the union of all six files, zero unique samples anywhere else. So nothing was
lost by keeping only one: `ex7_airfrans_aoa_train.h5` -> `ex7.h5`, `ex7_airfrans_aoa_test.h5` ->
`ex7_infer.h5` (rename, not rebuild), and the other four files (~7.7 GB, confirmed fully
redundant) were deleted. `ex7` now matches every other `exN` slot's plain two-file convention.
If a different extrapolation axis (Reynolds) or the plain interpolation/data-scarce regime is
wanted later, rebuild it with `build_public_airfrans.py --task reynolds_train` etc. --the
`--task` names still work, they just don't get kept as permanent files by default anymore.

The `smoke` tier (not rebuilt in the 2026-08-09 recovery) decimates to 16k nodes by random
subsampling but **always keeps every airfoil-surface node**, since those carry the geometry the
model conditions on; its edges are a k=6 proximity graph. The full-resolution tasks
(`scarce_train`, `full_test`, and the extrapolation splits) use the real quad connectivity --
`build_public_airfrans.py`'s edge extractor takes each quad's 4 **ring** edges, not
all `C(4,2)` vertex pairs (which would wrongly include the two diagonals).

## Configs -- one dataset per `exN` (renumbered 2026-08-10)

`ex4` originally bundled cylinder_flow + deforming_plate + flag_simple together (the "ex4 has two
[really three] datasets" confusion) and `ex6`/`ex7` briefly meant elasticity/plasticity while
airfrans sat at `ex5`. Split so **every `exN` is exactly one dataset**, matching every other
method's convention; the dataset filenames on disk were renamed to match (`ex4_deforming_plate*`
-> `ex5_deforming_plate*`, etc.) rather than just the config folders, so `dataset_dir` strings and
filenames agree:

| `exN` | Dataset | File(s) |
| --- | --- | --- |
| `ex4` | `cylinder_flow` | `dataset/ex4.h5`, `dataset/ex4_infer.h5` |
| `ex5` | `deforming_plate` | `dataset/ex5.h5`, `dataset/ex5_infer.h5` |
| `ex6` | `flag_simple` | `dataset/ex6.h5`, `dataset/ex6_infer.h5` |
| `ex7` | AirfRANS | `dataset/ex7.h5` (804 train), `dataset/ex7_infer.h5` (196 test, AoA extrapolation) |
| `ex8` | Geo-FNO elasticity | `dataset/ex8.h5` (1000 train), `dataset/ex8_infer.h5` (200 test) -- rebuilt 2026-08-19 via `dataset/build_geo_fno_elasticity.py`, canonical ntrain/ntest split (not extrapolation, see the script's docstring) |
| `ex9` | Geo-FNO plasticity | `dataset/ex9.h5` (900 train), `dataset/ex9_infer.h5` (87 test) -- rebuilt 2026-08-19 via `dataset/build_geo_fno_plasticity.py`, plain held-out split (not extrapolation). **Read the channel-semantics note below before writing an `ex9` config.** |

### `ex9` channel semantics -- `cond_var 2`, not `input_var 4` (corrected 2026-08-19)

`ex9.h5`'s 7 rows are `[x, y, z | ux, uy | uz, die_profile]`. Only **two** of those carry a
predictable signal:

| row | content | verified property |
| --- | --- | --- |
| 3, 4 | `ux`, `uy` | real 2D displacement, varies over time -- the actual target |
| 5 | `uz` | **identically zero everywhere** (plasticity is 2D; the builder emits a third displacement row only to fit the 3-component convention) |
| 6 | `die_profile` | the benchmark's **driving input parameter** (Geo-FNO's `input` array), constant over the trajectory |

Every `ex9` config shipped before 2026-08-19 used `input_var 4` / `output_var 4` / `cond_var 0`,
i.e. it asked the model to *predict* `uz` and `die_profile`. Both have a per-step delta of exactly
zero, so with `feature_loss_weights 1.0, 1.0, 1.0, 1.0` **half the loss budget was being spent on
targets that are trivially zero** -- and predicting the die profile is meaningless anyway, since it
is the input the benchmark conditions on. This is precisely what the
`CRITICAL: Near-zero delta variance - targets are constant!` line in the training logs was
reporting; it was a real defect, not cosmetic noise.

The correct form (now used by all seven mesh-method `ex9` configs) is
`input_var 2`, `output_var 2`, `cond_var 2`, `feature_loss_weights 1.0, 1.0`. `cond_var` consumes
the **trailing** rows, which is exactly where `uz`/`die_profile` sit, so the layout works without
touching the data. Confirmed at runtime: the loader now reports
`cond_dim: 2 (input-only conditioning rows 5:7)` and non-degenerate delta stats
(`std: [0.132, 0.085]` instead of two `1e-8` floors), and the constant-target CRITICAL is gone.
`configs/SimulGenVAE/ex9` uses `num_var 2` for the same reason (SimulGenVAE has no `cond_var`
mechanism, so those rows are excluded rather than re-tagged).

| Directory | Contents |
| --- | --- |
| `configs/MeshGraphNets/ex4..ex9/` | one train config per dataset above |
| `configs/Transolver/ex4..ex9/` | same, `use_world_edges False` everywhere (see below); `use_node_types False` on `ex4`/`ex6` (7-row files) |
| `configs/Neural_Operator/ex4..ex7/` | `gino` (one representative alias); `ex8`/`ex9` additionally have `deeponet`/`fno`/`point_deeponet` |
| `configs/SimulGenVAE/ex6,ex8,ex9/` | standalone `train_vae`, fixed-topology datasets only (see below) |

**`configs/MeshGraphNets_Variational/` intentionally has no `ex4`-`ex9` entries.** The variational tree
exists for **one-to-many** problems (stochastic process variation -> a distribution of plausible
outcomes for the *same* nominal input, hence the VAE + conditional-prior machinery and
`num_vae_samples`) -- that's what `SAOI_all_input`/`b8_all_warpage_input` actually are. `ex4`-`ex9`
are all **deterministic one-to-one** simulations (same input always gives the same output); MGN-V
configs were built and confirmed to *technically train* on several of them during investigation,
but were removed afterward as a wrong fit for the tool, not because they failed to run.

`ex4`-`ex6` train configs set both `dataset_dir` (train split) and `infer_dataset` (extrapolation
split) in the same file, matching the `ex1`/`ex2` convention -- no separate `config_infer_*.txt`
needed since MeshGraphNets/Transolver/GINO all run an inference pass against `infer_dataset` from
the same config. No `_arrt`/`_smoke` tiers or configs exist anymore (see the 2026-08-18 note
above).

### Verified -- real training launches, not just `--check` (2026-08-10)

`--check` passed for every config below and still missed a real crash in a different method
(SimulGenVAE, see `dataset/benchmarks/README.md`), so nothing here is asserted from preflight
alone -- every ✅ actually ran and printed a moving loss value. **Ran against the now-removed
`_smoke` tier files** (20 trajectories, strided to ~100/67/67 steps) -- the shapes and physics are
unchanged in the 2026-08-18 `exN.h5`/`exN_infer.h5` rebuild (same builder functions, same row
layout, just full resolution and no stride), so these results should still transfer, but they were
not re-run against the new files:

| Config | Result |
| --- | --- |
| MGN, all of `ex4`-`ex7` | ✅ (preflight only for `ex4`-`ex6`, matches the identical contract shape real-launched elsewhere) |
| Transolver, `ex4` (cylinder) | ✅ full epoch 0, loss 0.0216->0.0122 |
| Transolver, `ex5` (plate) | ✅ full epoch 0, TrainOpt 0.526->Valid 0.46 |
| Transolver, `ex6` (flag) | ✅ 98% epoch 0, no crash |
| Transolver, `ex7` (airfrans) | preflight only |
| GINO, `ex4` (cylinder) | ✅ full run incl. rollout viz -- caught and fixed a real config bug (`gino_grid_resolution` needs 2 entries for this planar dataset, not 3; z-coordinate is always 0 even though `disp_z` genuinely varies) |
| GINO, `ex5` (plate) | ✅ 75% epoch 0 |
| GINO, `ex6` (flag) | ✅ full 20 epochs, same 2D grid-resolution fix applied |
| GINO, `ex7` (airfrans) | preflight only |

### 2026-08-19: ex8/ex9 rebuilt, real-execution-verified across all four methods

`dataset/benchmarks/` (and the scripts that built it) had gone missing entirely -- see the
2026-08-10 recovery note above; this was never actually fixed until now. Raw Geo-FNO source was
still staged at `D:/CAE_datasets_raw/geo_fno/{elasticity,plasticity}` (untouched by the loss --
only the built `.h5`/scripts vanish, per the established pattern), so no re-download was needed.
Rebuilt `dataset/build_geo_fno_elasticity.py`/`build_geo_fno_plasticity.py` from git history
(commit `18e7e854`), renamed outputs to the `exN.h5`/`exN_infer.h5` convention (flat in `dataset/`,
not a `dataset/benchmarks/<name>/` subdirectory), and repointed all 15 `ex8`/`ex9` configs
(MeshGraphNets x2, Neural_Operator x4 x2, Transolver x2, SimulGenVAE x2). Also fixed three
pre-existing copy-paste bugs found along the way: `configs/MeshGraphNets/ex8` had its
`log_file_dir`/`modelpath` pointing at `ex6`, and `configs/SimulGenVAE/ex8`/`ex9` had theirs
pointing at `ex6`/`ex7` respectively.

Real (not just `--check`) training launches, one per method, watched until a full loss value
printed then stopped:

| Config | Result |
| --- | --- |
| MeshGraphNets, `ex8` (elasticity) | ✅ 2 full epochs, TrainOpt 1.20e+00 -> 7.27e-01 |
| MeshGraphNets, `ex9` (plasticity) | ✅ real per-step loss (500-epoch config, stopped mid-epoch-0 by design) |
| GINO, `ex8` (elasticity) | ✅ coverage preflight passed, real per-step loss |
| Transolver, `ex9` (plasticity) | ✅ real per-step loss |
| SimulGenVAE, `ex8` (elasticity) | ✅ Epoch 0 Recon: 4.77e-01, KL: 2.09e+01 |

**Correction (2026-08-19):** the "CRITICAL: Near-zero **delta** variance" warning in those `ex9`
runs was originally written off here as expected/cosmetic. That was wrong -- it was reporting a
real config defect (two of four predicted channels were constant), now fixed; see the `ex9`
channel-semantics section above. The remaining "Near-zero **node** variance" warning *is*
genuinely cosmetic: it refers to constant *input* features (`z_coord`, and `uz` in its new
conditioning role), which are read but never scored. Also: killing a
launched training run via a plain process-stop does **not** clean up its DataLoader worker
subprocesses on Windows (`num_workers`>0 spawns via `multiprocessing.spawn`, orphaned on a hard
kill) -- verify with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and kill the
whole tree, not just the top PID, or GPU memory stays pinned by zombie workers.

**Contact physics (`use_world_edges`) is MeshGraphNets-exclusive across this suite.** Transolver
refuses the flag at config-validation time (`TRANS-WORLD-001`); Neural_Operator accepts the flag
but "No model consumes MGN edge attributes" per its own CLAUDE.md, so it would silently ignore
world edges even if set. `deforming_plate` and `flag_simple`'s contact physics can only be
*meaningfully* modeled by MeshGraphNets today, even though Transolver/GINO can train on the same
files.

### SimulGenVAE and MLP: fixed-mesh-only, and what that means concretely

Both require every sample to share the same node count (SimulGenVAE: dense-FOM tensor; MLP:
doesn't touch mesh data at all, needs a pre-built tabular `X`/`Y` table). Checked against every
`exN` in this repo:

| `exN` | Fixed mesh? | SimulGenVAE | MLP |
| --- | --- | --- | --- |
| `ex1`/`ex2` (warpage) | ❌ (varying per-sample topology, confirmed by attempting `ex1.h5`) | ❌ | ❌ (no derived table exists; `configs/MLP/ex1` trains on an unrelated synthetic toy table, not `ex1.h5`) |
| `ex3` (NASA-CRM) | ✅ | ✅ | ✅ (`build_ex3_mlp_table.py` derives `X`/`Y` from the fixed mesh's global conditions + surface-integrated Cp/Cf) |
| `ex4`/`ex5`/`ex7` (cylinder/plate/airfrans) | ❌ (per-trajectory or per-sample mesh) | ❌ | ❌ |
| `ex6` (flag_simple) | ✅ (1579 nodes, every sample) | ✅ (after the 2026-08-10 fix, see below) | not built |
| `ex8`/`ex9` (elasticity/plasticity) | ✅ | ✅ (after the fix) | not built |

So **NASA-CRM was not the only fixed-mesh dataset** — `flag_simple`, elasticity, and plasticity
all qualify structurally too — but until 2026-08-10, SimulGenVAE crashed on all three anyway for
an unrelated reason: `SimulGenVAE/model/decoder.py`'s reconstruction layer hardcoded
`GroupNorm(num_groups=8, ...)`, which requires `num_var * num_nodes % 8 == 0`. NASA-CRM's channel
count happened to satisfy that by coincidence of its factor of 4 (`num_var=4`); elasticity (972),
plasticity (12524), and flag_simple (4737) all failed it. Fixed by replacing every
`min(8, max(1, X//4))` GroupNorm-groups computation (19 call sites across
`common.py`/`encoder.py`/`decoder.py`) with a `group_norm_groups()` helper that picks the largest
divisor of the channel count `<= 8` — behavior-identical for every already-divisible case (i.e.
`ex3` is unaffected), and no longer crashes for the rest. Verified: existing
`pytest tests/` (17 tests) still passes, and `ex6`/`ex8`/`ex9` all now train for real (multiple
epochs, decreasing reconstruction loss, no traceback). MLP was left alone per direction -- its
tabular contract is a data-*shape* difference, not a bug, and building `X`/`Y` tables for
elasticity/plasticity would be new scope (e.g. die-profile parameters -> integrated stress), not
a fix.

**2026-08-18:** that 2026-08-10 `ex6` verification never actually got a committed config --
`configs/SimulGenVAE/ex6/` didn't exist until this session (`config_train_vae.txt`, `num_var 3`
= `disp_x/disp_y/disp_z`, `field_start_row 3`, `--check` PASSED against the rebuilt `ex6.h5`).
`ex8`/`ex9`'s SimulGenVAE configs were blocked on missing `dataset/benchmarks/` data until the
2026-08-19 rebuild (see above); both now pass `--check` and were real-execution-verified (see the
2026-08-19 note below).

Inference configs, once written, are expected to fail preflight with `PATH-INPUT-001` until
their checkpoint exists — that is expected, not a defect.

## Two contract bugs this surfaced

Both come from the same stale assumption — that node types live at **feature index 7** — when
the loader actually reads the **last** row (`nodal_data[-1, 0, :]`). Those coincide only for
the 8-row ANSYS builder; `cylinder_flow` is 7 rows wide.

1. **`cae_suite/preflight.py`** rejected any `use_node_types True` dataset with `<= 7` rows.
   Now requires one row past the state/conditions block, which is the real constraint.
2. **`MeshGraphNets/inference_profiles/rollout.py`** gated on `num_features > 7` while reading
   `[-1]`. Training has no such guard, so a 7-row dataset trained fine and then **crashed at
   inference** with a node-encoder width error. Same fix.

### Still open (not changed)

The identical `num_features > 7` guard remains in 5 more places:

- `Transolver/inference_profiles/rollout.py:219`
- `Transolver/general_modules/mesh_dataset.py:121` (init check)
- `methods/MeshGraphNets_Variational/inference_profiles/rollout.py:630`
- `inference/cae_infer/families/{transolver,meshgraphnets,meshgraphnets_v}/driver.py`

Transolver is at least **self-consistent** — it raises a clear error at init instead of
crashing later — so this is a limitation, not a live bug. It was left alone rather than
silently editing four method trees. The practical effect: **`cylinder_flow` and `flag_simple`
(7 rows each) can only use node types under MeshGraphNets.** `deforming_plate` (8 rows) and
AirfRANS (13 rows) work everywhere.

## Adequacy summary: is any of this runnable by *every* method?

No, and it was never intended to be -- each method has a different data contract. As of
2026-08-09, cross-checked against real files (not just spec reading):

| Method | ex4 (cylinder/plate/flag) | ex5 (AirfRANS) | Why |
| --- | --- | --- | --- |
| `meshgraphnets`, `meshgraphnets-v` | Yes (config-verified) | Yes (config-verified) | native format |
| `transolver` | Yes, `use_world_edges` forced False | Yes | same HDF5, no edge consumption; TRANS-WORLD-001 rejects world edges |
| `point_deeponet`/`deeponet`/`fno`/`gino` | Yes (config-verified this session, `gino`) | Yes (config-verified this session, `gino`) | reads identical HDF5 per Neural_Operator/CLAUDE.md; simply never had a config before now |
| `simulgenvae` | **No** for cylinder/plate (per-trajectory mesh, node count varies 88-96 ways across 100 samples) | **No** (199 distinct node counts across 200 samples) | dense-FOM contract hard-requires one fixed node count for the whole file. `flag_simple`'s fixed 1579-node topology is the one exception, structurally, but untested |
| `mlp` | No | No | tabular X/Y contract, not mesh; nothing here is a scalar-in/scalar-out design table |
| `sdfflow` | No | No | different SDF-sidecar layout entirely, no relation to `data/{id}/nodal_data` |
| `geometry_ingest` | N/A | N/A | it's a producer of this contract from CAD, not a consumer of built HDF5 |

So: **3 of 8 registered `model` values were architecturally untested until this session** (the
four Neural_Operator aliases, deduplicated to one family); they now have passing preflight
configs. **3 of 8 are structurally incompatible** with ex4/ex5's per-sample-varying mesh
(`simulgenvae`, `mlp`, `sdfflow`) for reasons unrelated to how the files were built -- more
downloads of the same *kind* of dataset would not change that, only a fixed-topology multi-sample
dataset (for `simulgenvae`) or a scalar-parameter table (for `mlp`) would.

## Probabilistic / one-to-many datasets (added 2026-09-04)

Everything in `ex1`-`ex9` is **deterministic**: one input, one correct field. That makes them
useless for validating `meshgraphnets-v`, whose whole point is a distribution over outcomes --
per [the MGN-V scope note](../../methods/MeshGraphNets_Variational/), it is for one-to-many /
stochastic problems, not deterministic sims. These three fill that gap. The selection criterion
was strict: the *same* macroscopic condition must produce *different* valid outcomes, so a large
parametric sweep with one deterministic solve per parameter (PDEBench, PDEGym, CFDBench, the
classic Darcy-GRF sets) does **not** qualify no matter how big it is.

| Output (`X.h5` + `X_infer.h5`) | Source | License | What is stochastic | train + infer | Shape / size |
| --- | --- | --- | --- | --- | --- |
| `dataset/ex10.h5` | [The Well](https://polymathic-ai.org/the_well/) `turbulent_radiative_layer_2D` (Polymathic AI, NeurIPS 2024 D&B) | CC-BY-4.0 | 9 `t_cool` values x 10 random seeds; the paper itself cites seed sensitivity as motivating a probabilistic treatment | 72 + 18 | `[8, 101, 49152]`, 4.7 + 1.2 GB |
| `dataset/ex12.h5` | [ASME 2023 Hackathon SPPARKS dataset](https://zenodo.org/record/8241535) (Sandia, GrainPaint) | CC-BY-4.0 | 1000 Potts Monte Carlo runs, one random seed each, explicitly built to capture microstructure-induced aleatory uncertainty | 900 + 100 | `[4, 1, 125000]`, 56 + 7 MB |
| `dataset/ex11.h5` | [Mechanical MNIST Crack Path, extended](https://zenodo.org/records/5149019) (Lejeune Lab, Boston University) | CC0 | identical loading protocol for every case; only the random rigid-inclusion placement differs, so the crack path is the stochastic outcome | 1750 + 250 | `[6, 20, 65536]`, 9.6 + 1.4 GB |

Raw downloads live on `D:/CAE_datasets_raw/probabilistic/`; the one-off downloader and the three
converters are in `junk/` (gitignored).

### Channel layouts

```text
ex10   rows 0:3 x,y,z   3:7 density,pressure,vx,vy   7 tcool
                               input_var 4, output_var 4, cond_var 1, T=101
ex12        rows 0:3 x,y,z   3   grain_id
                               input_var 1, output_var 1, cond_var 0, T=1 (static)
ex11                rows 0:3 x,y,z   3:6 damage,x_disp,y_disp
                               input_var 3, output_var 3, cond_var 0, T=20
```

`tcool` is a per-sample constant broadcast across nodes -- the conditioning-row case the contract
already allows. The other two carry no conditions **on purpose**: crack path is determined by the
inclusion positions (shipped separately in `inclusions_test.7z`), and feeding them in would turn
a one-to-many problem into a near-deterministic microstructure -> crack-path regression, which is
the opposite of why the dataset was picked.

### Grids become graphs

All three are regular grids, so `mesh_edge` is grid adjacency, not element extraction:

- turbulent radiative layer: 128x384, 4-connected, **periodic in x** (the file's own
  `boundary_conditions/x_periodic` says so), open in y -> 98,176 edges
- grainpaint: 100^3, 6-connected, periodic in all three axes (SPPARKS Potts default) -> 3,000,000 edges
- crack path: 256x256, 4-connected, **non-periodic** -- the plate has a different BC on each edge,
  so wrapping would fuse unrelated boundaries -> 130,560 edges

Topology is identical across samples in all three, so the grainpaint and crack-path converters
write `mesh_edge` **once** and hard-link every sample group at it. This is not cosmetic: 1000
duplicated copies of grainpaint's 3M-edge array would be ~48 GB, and the whole file is 481 MB.
h5py hard links are transparent to readers -- `f["data"]["500"]["mesh_edge"][:]` works normally.

### Verified, not assumed

- Launcher probe clean on all three (`python cae_suite/dataset_probe.py mesh_hdf5 <file>`).
- **Crack-path grid orientation was read off the data**: at step 0 damage > 0.5 occupies columns
  0-64 in a single row band, matching the documented initial crack of length 0.25 on the left
  edge, which pins the flat index to C-order `row*256 + col`. `y_disp` maxes at exactly 0.02, the
  documented maximum applied displacement.
- **One-to-many actually holds**: crack overlap IoU between cases is 0.09-0.25 (same loading,
  genuinely different paths); grainpaint seeds give different grain counts (924 vs 921) and
  non-identical fields.


### Slots

These occupy `ex10` (turbulent radiative layer), `ex11` (crack path) and `ex12`
(grainpaint), two files each -- `exN.h5` + `exN_infer.h5` -- and **every mesh
method reads the same file**; only `mlp` (tabular X/Y) sits outside that contract.
Concurrent arms against one file are already handled: `write_preprocessing_to_hdf5`
is the sole writer and retries its `r+` open eight times with backoff, exactly
because a sweep launches several arms against one shared dataset seconds apart.

`ex12` is the faithful 100^3 conversion (1,000,000 nodes, 3,000,000 undirected
edges, `grain_id` as the field). Two things to know before training on it:
one graph does not fit a graph-net block on a 24 GB card (a single edge-latent
tensor at width 128 is 6e6*128*4 B = 3.1 GB and a block needs several live), and
`grain_id` is an arbitrary label, so an L2 loss on it punishes a correct
microstructure that merely permutes ids. `junk/crop_grainpaint.py` produces the
trainable variant -- a centre 50^3 crop at native resolution with a
permutation-invariant grain-boundary target -- at
`D:/CAE_datasets_raw/probabilistic/grainpaint_spparks/crop50*.h5`.

### Splits: same-condition holdout, not extrapolation

Each dataset ships as `<name>.h5` (train) + `<name>_infer.h5`, per the `exN` convention -- but the
split rule is **not** ex4-ex6's "highest value on a physical axis goes to infer". That rule builds
an extrapolation test, and extrapolation is the wrong question here: a variational model is judged
on whether its predicted *spread* at a condition matches the observed spread, so the held-out set
must contain **several ground-truth outcomes at the same condition**. Holding out the largest
`tcool` entirely would leave nothing to compare a predicted distribution against.

| Dataset | infer rule | Result |
| --- | --- | --- |
| turb radiative layer | the source's own valid+test split | 18 infer = 2 seeds at each of all 9 `tcool` values; every condition present in both halves |
| grainpaint | `seed % 10 == 0` | 100 / 900, no overlap |
| crack path | `case_id % 10 == 0` | 250 / 1750, no overlap (12.5%, not 10% -- the converter samples case ids ~5 apart, so multiples of 10 land more often than one in ten) |

All deterministic, no RNG. If a *generalization* test is wanted instead, a `tcool`-extrapolation
variant of the turbulent-radiative-layer split is the one that makes physical sense; the other two
have no physical axis to extrapolate along, since every sample is drawn from identical settings.

### Traps hit while building these

- Mechanical MNIST's per-case files are named `*.npy` but are **whitespace-separated ASCII text**
  ("text files with .npy format" in its readme). `np.load` fails with a *pickled object data*
  error, which reads like a security warning; the file simply is not an npy. Use
  `np.fromstring(f.read(), sep=" ")`.
- Zenodo throttles hard per connection (~0.3-1 MB/s) and drops long transfers; `curl --retry`
  does **not** retry exit 18 (partial file) without `--retry-all-errors`.
- Running two downloader instances against the same partial file corrupts it: both resume with
  `curl -C -` and their appends interleave, which grew a 4.29 GB Zenodo part to 5.76 GB of
  garbage. The downloader now takes an OS-level single-instance lock and re-fetches any file
  larger than its recorded size.
- Verify against the size the API reports, not "file exists and is non-empty" -- a truncated
  transfer that exits 0 otherwise gets recorded as complete.

### Not converted, and why

Only 2000 of Mechanical MNIST's 10,000 test cases are in the HDF5 (evenly spread across the ten
1000-case archives; `--max-samples` changes it). All 70,000 cases (60k train + 10k test) would be
**~2.2 TB** uncompressed at 31.5 MB/case, and 2000 graphs of 65,536 nodes is already a large mesh-GNN
training set. The full-mesh `dmg-init` / `last-step` FEniCS archives in the same record are
downloaded but unused -- they are single snapshots at native mesh resolution, redundant with the
uniform-grid time series.

## Other candidates, not downloaded

| Dataset | Size | Why it might be worth it |
| --- | --- | --- |
| ShapeNet-Car (Umetani & Bickel) | 26 GB | 889 cars, 32k pts; the standard Transolver/GINO surface benchmark — closest thing to an apples-to-apples published baseline |
| DeepMind `airfoil` | 56 GB | compressible, 5233 nodes, 600 steps — same converter, just bigger |
| AhmedML surface-only | ~500 MB/case | 3.08 TB total, but `boundary_i.vtp` alone is ~500 MB — 100 cases ≈ 50 GB of industrial-scale automotive aero |
| WindsorML / DrivAerML | 1 TB+ | over budget except as a small case subset |

`flag_simple` (was listed here previously) is now staged and built -- see section 3 above; its
"dynamic node type" framing turned out not to hold up against the real data.

Sources: [DeepMind meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets),
[AirfRANS](https://airfrans.readthedocs.io/en/latest/notes/dataset.html),
[CAE ML Datasets](https://caemldatasets.org/),
[ShapeNet-Car flow](https://zenodo.org/records/13737721).
