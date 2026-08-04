# Public test datasets

Candidate public CAE/ML datasets for exercising this repo, why each was picked, and how to
rebuild it into the shared mesh HDF5 contract ([DATASET_FORMAT.md](DATASET_FORMAT.md)).

Everything here is **< 100 GB** and redistributable for research.

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

| Tier | Trajectories | T | Purpose |
| --- | --- | --- | --- |
| `smoke` | 20 | 100 / 67 | fast preflight + rollout smoke test (~15-40 MB) |
| `arrt` | 50 | 25 | `time_integration ar_rt` (~16-27 MB) |
| `main` | 100 | 600 / 400 | real AR-OT training runs (1.19 GB / 490 MB) |

The `arrt` tier exists because AR-RT unrolls the **whole** trajectory and backprops through
every step ([time_integration.py](../MeshGraphNets/general_modules/time_integration.py)). At the
main tier that is 599 chained steps per item — unusable. T=25 keeps it tractable while still
being a genuine multi-step unroll.

### 3. AirfRANS — 9.3 GB raw

1000 steady RANS solutions over NACA 4/5-digit airfoils, Re 2-6M, AoA -5°..15°, ~155k nodes
each (2D slice, all quad cells). Static like NASA-CRM but with **6.7x more samples**, a
different mesh per sample, and published **extrapolation** splits — the only dataset here
that tests generalization rather than interpolation.

    rows 0:3    x, y, z(=0)
    rows 3:7    u_x, u_y, p, nu_t                 state       input_var = output_var = 4
    rows 7:12   u_inf_x, u_inf_y, sdf, n_x, n_y   conditions  cond_var = 5
    row  12     node_type                         0 volume, 1 airfoil surface

The condition rows are exactly the AirfRANS paper's input features. `u_inf_x/y` are recovered
from the folder name (`airFoil2D_SST_<U_inf>_<AoA>_<naca…>`) and are per-sample constants;
sdf and the normals vary per node, which the mesh methods accept.

> **`augment_geometry` must stay `False`.** `u_inf` and the normals are vector *components*,
> so a z-rotation would rotate the geometry without rotating them — the caveat
> DATASET_FORMAT.md raises for conditioning rows.

```bash
python dataset/build_public_airfrans.py --task smoke          # 40 samples @ 16k nodes, 34 MB
python dataset/build_public_airfrans.py --task scarce_train   # 200 samples, full resolution
python dataset/build_public_airfrans.py --task full_test      # 200 samples, full resolution
```

The `--task` names come straight from the dataset's own `manifest.json`, so
`reynolds_train`/`reynolds_test` and `aoa_train`/`aoa_test` build the extrapolation studies.
Because `split_strategy hdf5` is not implemented in this runtime, each task is written as its
**own file** — point `dataset_dir` at the train task and `infer_dataset` at the test task
rather than relying on the seeded 80/10/10 split.

The smoke tier decimates to 16k nodes by random subsampling but **always keeps every
airfoil-surface node**, since those carry the geometry the model conditions on; its edges are
a k=6 proximity graph. The full-resolution tasks use the real quad connectivity.

## Configs

| Directory | Contents |
| --- | --- |
| `configs/MeshGraphNets/ex4/` | train + inference for `cylinder_smoke`, `cylinder_arrt`, `plate_smoke`, `plate_arrt` |
| `configs/Transolver/ex4/` | Transolver on the *same* `ex4_deforming_plate_smoke.h5` |
| `configs/MeshGraphNets/ex5/` | MeshGraphNets on AirfRANS smoke |
| `configs/Transolver/ex5/` | Transolver on AirfRANS smoke |

The ex4 Transolver config is the point of the exercise: **one HDF5, two methods, no
conversion step**.

### Verified

Each of these was actually run, not just preflighted:

| Run | Result |
| --- | --- |
| MGN train → rollout, `cylinder_smoke` | 50-step AR rollout over 20 scenes |
| MGN train → rollout, `plate_smoke` | world-edge radius resolved to 0.0291 vs DeepMind's documented `collision_radius` 0.03 |
| MGN AR-RT, `cylinder_arrt` | 24-step BPTT unroll, loss 3.32 → 1.99 |
| MGN AR-RT, `plate_arrt` | AR-RT + world edges + node types together |
| Transolver, `plate_smoke` | same file as MGN, `node_input_size` 11 |
| Transolver + MGN, `airfrans_smoke` | `cond_var 5` picked up as rows 7:12, input 15 |

Inference configs fail preflight with `PATH-INPUT-001` until their checkpoint exists — that
is expected, not a defect.

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
silently editing four method trees. The practical effect: **`cylinder_flow` (7 rows) can only
use node types under MeshGraphNets.** `deforming_plate` (8 rows) and AirfRANS work everywhere.

## Other candidates, not downloaded

| Dataset | Size | Why it might be worth it |
| --- | --- | --- |
| ShapeNet-Car (Umetani & Bickel) | 26 GB | 889 cars, 32k pts; the standard Transolver/GINO surface benchmark — closest thing to an apples-to-apples published baseline |
| DeepMind `airfoil` | 56 GB | compressible, 5233 nodes, 600 steps — same converter, just bigger |
| DeepMind `flag_simple` | 11.4 GB | cloth with **dynamic** node types (they change per timestep), which nothing else here covers |
| AhmedML surface-only | ~500 MB/case | 3.08 TB total, but `boundary_i.vtp` alone is ~500 MB — 100 cases ≈ 50 GB of industrial-scale automotive aero |
| WindsorML / DrivAerML | 1 TB+ | over budget except as a small case subset |

Sources: [DeepMind meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets),
[AirfRANS](https://airfrans.readthedocs.io/en/latest/notes/dataset.html),
[CAE ML Datasets](https://caemldatasets.org/),
[ShapeNet-Car flow](https://zenodo.org/records/13737721).
