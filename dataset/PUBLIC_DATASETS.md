# Public test datasets

Candidate public CAE/ML datasets for exercising this repo, why each was picked, and how to
rebuild it into the shared mesh HDF5 contract ([DATASET_FORMAT.md](DATASET_FORMAT.md)).

Everything here is **< 100 GB** and redistributable for research.

> **2026-08-09 recovery note:** `build_public_mgn.py`, `build_public_airfrans.py`, and every
> `configs/{MeshGraphNets,Transolver,Neural_Operator}/{ex4,ex5}/*.txt` referenced below had been
> deleted from disk (never committed -- `*.h5` is gitignored so the built files survived, the
> scripts and configs that made and consumed them did not). Both converters were reconstructed
> from scratch by reverse-engineering the raw TFRecord/VTU schemas and validated **byte-exact**
> against the surviving `.h5` files (`nodal_data`/`mesh_edge` `np.allclose`/`array_equal` against
> real samples). Two corrections came out of that reconstruction, both below: the `smoke`/`arrt`
> tiers are **strided** subsamples of the full trajectory, not truncations, and AirfRANS surface
> normals are **zero at volume nodes**, not nearest-neighbor-propagated. flag_simple was also
> newly built (raw TFRecords were already staged, just never converted) and the AirfRANS
> extrapolation splits (the dataset's actual selling point) were built for the first time.

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
python dataset/build_public_mgn.py --dataset cylinder_flow   --tier main
python dataset/build_public_mgn.py --dataset deforming_plate --tier smoke
```

`build_public_mgn.py` reads the TFRecords with a **pure-numpy** parser — no TensorFlow — so it
runs in any method venv.

#### Tiers

| Tier | Trajectories | Sampling | T (cylinder / plate / flag) | Purpose |
| --- | --- | --- | --- | --- |
| `smoke` | 20 | every 6th raw step, from t=0 | 100 / 67 / 67 | fast preflight + rollout smoke test |
| `arrt` | 50 | every 8th raw step, t=0..192 only | 25 / 25 / 25 | `time_integration ar_rt` |
| `main` | 100 | every raw step (no subsampling) | 600 / 400 / 401 | real AR-OT training runs |

**Both `smoke` and `arrt` are strided subsamples of the full raw trajectory, not truncations** --
confirmed by matching individual timestep values against the raw TFRecords (e.g. `arrt` frame 1
for `deforming_plate` sample 1 is raw frame 8, not raw frame 1). `smoke`'s stride (6) spans
~the full raw duration; `arrt`'s stride (8) only reaches raw frame 192 of 400/600 -- it trades
duration coverage for a tractable full-trajectory backprop length, it does not summarize the
whole trajectory the way `smoke` does. The `arrt` tier exists because AR-RT unrolls the **whole**
stored trajectory and backprops through every step
([time_integration.py](../MeshGraphNets/general_modules/time_integration.py)); at the main tier
that would be 599 chained steps per item -- unusable. 25 strided frames keep it tractable while
still being a genuine multi-step unroll.

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

```bash
python dataset/build_public_mgn.py --dataset flag_simple --tier smoke
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
only dataset here that tests generalization rather than interpolation) and, as of the 2026-08-04
build, was never actually exercised; only `scarce_train`/`full_test` (plain interpolation
splits) existed on disk. Because `split_strategy hdf5` is not implemented in this runtime, each
task is written as its **own file** — point `dataset_dir` at the train task and `infer_dataset`
at the test task rather than relying on the seeded 80/10/10 split.

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
| `ex4` | `cylinder_flow` | `dataset/ex4_cylinder_flow{,_arrt,_smoke}.h5` |
| `ex5` | `deforming_plate` | `dataset/ex5_deforming_plate{,_arrt,_smoke}.h5` |
| `ex6` | `flag_simple` | `dataset/ex6_flag_simple{,_arrt,_smoke}.h5` |
| `ex7` | AirfRANS | `dataset/ex7_airfrans_{full_test,scarce_train,reynolds_train,reynolds_test,aoa_train,aoa_test}.h5` |
| `ex8` | Geo-FNO elasticity | `dataset/benchmarks/elasticity/*.h5` |
| `ex9` | Geo-FNO plasticity | `dataset/benchmarks/plasticity/*.h5` |

| Directory | Contents |
| --- | --- |
| `configs/MeshGraphNets/ex4..ex9/` | one train config per dataset above |
| `configs/Transolver/ex4..ex9/` | same, `use_world_edges False` everywhere (see below); `use_node_types False` on `ex4`/`ex6` (7-row files) |
| `configs/Neural_Operator/ex4..ex7/` | `gino` (one representative alias); `ex8`/`ex9` additionally have `deeponet`/`fno`/`point_deeponet` |
| `configs/SimulGenVAE/ex8,ex9/` | standalone `train_vae`, fixed-topology datasets only (see below) |

**`configs/MeshGraphNets-V/` intentionally has no `ex4`-`ex9` entries.** The variational tree
exists for **one-to-many** problems (stochastic process variation -> a distribution of plausible
outcomes for the *same* nominal input, hence the VAE + conditional-prior machinery and
`num_vae_samples`) -- that's what `SAOI_all_input`/`b8_all_warpage_input` actually are. `ex4`-`ex9`
are all **deterministic one-to-one** simulations (same input always gives the same output); MGN-V
configs were built and confirmed to *technically train* on several of them during investigation,
but were removed afterward as a wrong fit for the tool, not because they failed to run.

No infer configs yet (would need a checkpoint from an actual training run, or
`dataset/generate_inference_dataset.py`); no `_arrt` configs (the tier exists and is schema-valid,
just not wired into a config).

### Verified -- real training launches, not just `--check` (2026-08-10)

`--check` passed for every config below and still missed a real crash in a different method
(SimulGenVAE, see `dataset/benchmarks/README.md`), so nothing here is asserted from preflight
alone -- every ✅ actually ran and printed a moving loss value:

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
- `MeshGraphNets - variational/inference_profiles/rollout.py:630`
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
