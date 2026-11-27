# Dataset provenance

Where each dataset in `dataset/` comes from, how it was converted to the shared
mesh HDF5 ([DATASET_FORMAT.md](DATASET_FORMAT.md)), and what to know before
training on it.

`dataset/` holds only the cleaned files and is git-ignored. Each dataset is a
training `.h5` plus a held-out `_infer.h5`. Raw sources live outside the repo,
under `D:/CAE_datasets_raw/` on the workstation. All public sources here are
redistributable for research.

**Converters.** Most public-dataset converters were removed from the tree on
2026-09-03 (commit `b5ff338`). Recover one with
`git show b5ff338^:dataset/<name>.py`:

- `build_public_mgn.py`: cylinder_flow, deforming_plate, flag_simple
- `build_public_airfrans.py`: AirfRANS
- `build_geo_fno_elasticity.py`, `build_geo_fno_plasticity.py`: Geo-FNO
- `fetch_public_datasets.sh`: download script

Converters that still exist are named in the tables below.

---

## Deterministic (`dataset/deterministic/`)

One input, one correct field. Used by MGN, Transolver, Neural_Operator, and
SimulGenVAE where the mesh is fixed.

| Slot | Dataset | Source | Train / infer | Notes |
| --- | --- | --- | --- | --- |
| ex1 | static thermoelastic warpage | proprietary | — | Planar (z ≡ 0). The infer file is one hex-mesh sample whose state rows carry the ground truth |
| ex2 | dynamic contact warpage | proprietary | — | 3D, temporal. Infer = 5 unseen scenes × 50 steps, rolled out from t = 0 |
| ex3 | NASA CRM, full / mid | NASA Common Research Model, built upstream | 149 samples total | Static surface aero. Full = 454k nodes, mid = 123k (`configs/campaigns/dataset_matrix/rebuild_crm_mid.py`). Uses `cond_var`. Also `ex3_NASA_CRM_mlp_{train,test}.h5`, the tabular X/Y for `mlp` |
| ex4 | DeepMind `cylinder_flow` | [meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | 100 / 20 | 600 steps, own mesh per trajectory. Infer = highest inflow speed |
| ex5 | DeepMind `deforming_plate` | same | 100 / 20 | 400 steps, tets, contact (`use_world_edges`). Infer = deepest indentation |
| ex6 | DeepMind `flag_simple` | same | 100 / 20 | 401 steps, one shared 1579-node mesh. Infer = largest cloth motion |
| ex7 | AirfRANS | [AirfRANS](https://airfrans.readthedocs.io/en/latest/notes/dataset.html) | 804 / 196 | Steady RANS, ~155k nodes. Uses the published AoA extrapolation split |
| ex8 | Geo-FNO elasticity | Geo-FNO | 1000 / 200 | Canonical ntrain/ntest split, fixed mesh |
| ex9 | Geo-FNO plasticity | Geo-FNO | 900 / 87 | Plain held-out split, fixed mesh. **`cond_var 2`** (see below) |
| ex10 | DeepJEB vertical-load FEA | own OpenRadioss relabel of DeepJEB geometry | 2121 solved | Built by `D:/CAE_datasets_raw/deepjeb_ver_fea/src/build_ex10.py`. Replaces the earlier DeepJEB field set, whose fields could not be paired with its vertices |

### Row layouts and traps

**Rows by dataset:**

```text
ex4 cylinder_flow    x,y,z=0 | vel_x, vel_y, pressure        | node_type   input_var 3
ex5 deforming_plate  x,y,z   | disp_x, disp_y, disp_z, stress | node_type   input_var 4
ex6 flag_simple      x,y,z   | disp_x, disp_y, disp_z         | node_type   input_var 3
ex7 AirfRANS         x,y,z=0 | u_x, u_y, p, nu_t | u_inf_x, u_inf_y, sdf, n_x, n_y | node_type
                                                  input_var 4, cond_var 5
ex9 plasticity       x,y,z   | ux, uy | uz, die_profile        input_var 2, cond_var 2
```

**DeepMind sets (ex4–ex6):**

- Node types are the MGN set:
  - ex4: 0 normal, 4 inflow, 5 outflow, 6 wall
  - ex5: 0 normal, 1 obstacle, 3 handle
- The converter writes displacement (`world_pos − mesh_pos`), not absolute
  position, because the loaders integrate `x[t+1] = x[t] + delta`.
- The extrapolation axis of each sample is in `metadata/extrapolation_axis`.
- flag_simple stores no per-trajectory control parameter (the handle nodes
  barely move), so its split is by outcome, not by input.
- flag_simple's `node_type` is stored per timestep but never changes.
- Under `time_integration ar_rt`, the full trajectories mean 399–600 chained
  steps per item. Use a small batch.

**AirfRANS (ex7):**

- `n_x`/`n_y` are nonzero only at the 1007 surface nodes, and zero in the
  volume.
- `u_inf_x`/`u_inf_y` are per-sample constants parsed from the folder name.
- **`augment_geometry` must stay false.** The condition rows are vector
  components, and a rotation would not rotate them.
- Edges are each quad's 4 ring edges, not its diagonals.
- The six published splits are partitions of the same 1000 samples. To use the
  Reynolds or scarce-data split, rebuild with `--task reynolds_train`, etc.

**Geo-FNO plasticity (ex9):**

- `uz` is identically zero (the problem is 2D) and `die_profile` is the
  benchmark's input.
- Predicting them wastes half the loss on constant targets. So configs use
  `input_var 2`, `output_var 2`, `cond_var 2`, which makes them conditions.
- SimulGenVAE uses `num_var 2` instead.

**Contact:** `use_world_edges` is consumed only by MGN. Transolver rejects it
(`TRANS-WORLD-001`), and Neural_Operator ignores edge attributes.

### Which method can read which slot

SimulGenVAE needs one node count across every sample. MLP needs a tabular X/Y
file.

| Slot | Fixed mesh | SimulGenVAE | MLP |
| --- | --- | --- | --- |
| ex1, ex2, ex4, ex5, ex7 | no | no | no |
| ex3 | yes | yes | yes (`mlp_*` tables) |
| ex6, ex8, ex9 | yes | yes | no table built |

MGN-V configs exist only for one-to-many data (below). The deterministic slots
are the wrong fit for it, though it does train on them.

---

## Probabilistic (`dataset/probabilistic/`)

Each of these has the same macroscopic condition produce several valid
outcomes. A parametric sweep with one solve per parameter (PDEBench, CFDBench,
Darcy-GRF) does not qualify, however large. These datasets are what
`meshgraphnets-v` and the SAR/LDGN baselines are judged on.

| Slot | Dataset | Source | License | What is random | Train / infer |
| --- | --- | --- | --- | --- | --- |
| ex1 | turbulent radiative layer 2D | [The Well](https://polymathic-ai.org/the_well/) | CC-BY-4.0 | 9 `t_cool` values × 10 seeds | 72 / 18 |
| ex2 | Mechanical MNIST Crack Path (extended) | [Zenodo 5149019](https://zenodo.org/records/5149019) | CC0 | random rigid-inclusion placement under one loading protocol | 1750 / 250 |
| ex3 | shell buckling | own OpenRadioss campaign | — | imperfection draws per geometry | see below |

Rows:

```text
ex1  x,y,z | density, pressure, vx, vy | tcool     input_var 4, cond_var 1, T=101
ex2  x,y,z | damage, x_disp, y_disp                input_var 3, cond_var 0, T=20
```

- **Grids become graphs.**
  - ex1 is 128×384, 4-connected, periodic in x.
  - ex2 is 256×256, 4-connected, not periodic.
  - Topology is shared across samples, so each `mesh_edge` is written once and
    hard-linked into every sample.
- **The splits hold out outcomes at seen conditions, not extrapolation.** A
  distribution model is judged on predicted versus observed spread at one
  condition, so the held-out set needs several outcomes per condition.
  - ex1 uses the source's own valid+test split: 2 seeds at each of the 9
    `t_cool` values.
  - ex2 holds out `case_id % 10 == 0`.
- ex2 deliberately carries no conditions. Feeding the inclusion positions would
  turn it into a near-deterministic regression.
- Mechanical MNIST's per-case `*.npy` files are actually ASCII text. Read them
  with `np.fromstring(..., sep=" ")`.
- **ex3, shell buckling.**
  - Built by the campaign at `D:/CAE_datasets_raw/shell_buckling/` (pack step
    `src/pack_hdf5.py`); raw runs are on aarl at
    `/data/Lee/buckling_radioss/`.
  - `ex3_shell_buckling_infer.h5` holds unseen geometries, one factor at a time,
    so it is an extrapolation test.
  - The design is in
    [research/shell_buckling/OPENRADIOSS_BUILD_PLAN.md](../research/shell_buckling/OPENRADIOSS_BUILD_PLAN.md).
- An earlier ex3, GrainPaint (SPPARKS Potts), was dropped. Its raw data stays at
  `D:/CAE_datasets_raw/probabilistic/grainpaint_spparks/`.

---

## Geometry generation (`dataset/geometry_generation/`)

SDFFlow uses its own SDF layout (`shapes/{index}/{surface_points,
surface_normals, sdf_points, sdf_values, cond}`), built by
`methods/SDFFlow/scripts/build_dataset.py`. The condition sidecars are written
by the `scripts/add_*_conditions.py` tools. See
[methods/SDFFlow/CLAUDE.md](../../methods/SDFFlow/CLAUDE.md).

| Slot | Dataset | Source | Conditions |
| --- | --- | --- | --- |
| ex1 | DeepJEB brackets | DeepJEB (`D:/CAE_datasets_raw/deepjeb/`) | geometry descriptors, FEA results. The train/infer split is by parent design: siblings of one parent never straddle it |
| ex2 | DrivAerML cars | [DrivAerML](https://caemldatasets.org/drivaerml/), 484 cases | geometry parameters, CFD forces |
| ex3 | MCB part A | [Mechanical Components Benchmark](https://mcb.eecs.umich.edu/) | part category |

Each dataset needs its own condition writer. Do not reuse DeepJEB's FEA
condition schema for MCB (which has no simulation) or for DrivAerML (whose
labels are CFD).

Thingi10K was trained as `ex4` and removed on 2026-10-07. Its shapes are
unrelated objects, so in-between samples fall apart. The files are kept under
`_set_aside_20261007/` as `thingi10k(_infer).h5`.

The slots were compacted on 2026-11-27 (no gaps): deterministic `ex13` → `ex10`
and `ex14` → `ex11`; geometry `ex6` → `ex4`. Copies on aarl and scripts under
`paper/generative_design_optimization/` still use the old names.

SFEM (cmudrc, BrepGen-generated parts with linear-static FEA) was built as
deterministic `ex11` and dropped on 2026-11-27: the parts are AI-generated
shapes, not real designs, and the near-point-load stresses span ~13 decades.
The dataset, its configs and the raw download were deleted.

ShapeNet-Car (Umetani & Bickel 2018, geometry `ex4`) was dropped the same day
before training: every car is one shared 3,584-quad template fitted to the
ShapeNet model (no wheels, rounded detail), and the CFD ran on that smoothed
shape. DrivAerML is used for car aerodynamics instead.

---

## Validation (`dataset/validation/`)

`ellipse.h5`, `ellipse_infer.h5`, `ellipse_infer_t0.h5`: the Ellipse case from
Lino et al.'s dgn4cfd. It is used to check that the SAR and LDGN ports reproduce
upstream. Converter: `D:/CAE_datasets_raw/lino_dgn/convert_ellipse.py`.

---

## Candidates not downloaded

| Dataset | Size | Why |
| --- | --- | --- |
| DeepMind `airfoil` | 56 GB | Compressible, 600 steps; same converter as ex4 |
| AhmedML surface-only | ~500 MB/case | Industrial-scale automotive aero, as a case subset |
| WindsorML | 1 TB+ | Over budget except as a small subset |
