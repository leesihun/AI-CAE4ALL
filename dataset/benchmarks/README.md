# Paper-replication benchmarks

`configs/benchmarks/*` and this directory replicate specific published results (Geo-FNO/FNO,
GINO). Written 2026-08-10 after finding **every dataset under here was missing** despite
`configs/`, model code (`Transolver/model/paper_elasticity.py`,
`Neural_Operator/model/gino_carcfd.py`), and even old run artifacts (`output/benchmarks/**`)
all still being present — the same "`*.h5` is gitignored, the builder script wasn't, so both
silently disappear together while old outputs survive" pattern documented in
[../PUBLIC_DATASETS.md](../PUBLIC_DATASETS.md).

## Also live as `ex8`/`ex9` (2026-08-10, renumbered same day)

To match the `configs/<Method>/exN/` convention, `elasticity` is wired up as **`ex8`** and
`plasticity` as **`ex9`** under `configs/{MeshGraphNets,Transolver,Neural_Operator,
SimulGenVAE}/ex8,ex9/` (same underlying `dataset/benchmarks/{elasticity,plasticity}/*.h5` files —
paths inside configs resolve relative to the method repo root, not the config file's own folder,
so the same data works from either location with zero path changes). Originally landed as `ex6`/
`ex7`; renumbered to `ex8`/`ex9` the same day once `ex4` (which had bundled cylinder_flow +
deforming_plate + flag_simple together) was split into one dataset per `exN` — see
[../PUBLIC_DATASETS.md](../PUBLIC_DATASETS.md)'s Configs section for the full `ex4`-`ex9` table.
`configs/benchmarks/*` remains the canonical location for the paper-exact hyperparameters (500
epochs, paper's own LR schedule); `ex8`/`ex9` configs are quick smoke-test variants (~20 epochs).

**No `configs/MeshGraphNets-V/ex8,ex9/`** — built during investigation, confirmed to train, then
removed: the variational tree is for **one-to-many** problems (stochastic variation around a
fixed nominal input), which elasticity/plasticity are not (deterministic one-to-one). "Runs
without crashing" isn't the same bar as "is the right tool" — see PUBLIC_DATASETS.md.

**Every claim below is real-execution-verified, not preflight-only** — `--check` passed for
every one of these and then still missed a real crash (SimulGenVAE), so nothing here is asserted
without actually launching training and watching a loss value print.

| Model | ex8 (elasticity, 972 pts, T=1) | ex9 (plasticity, 3131-node grid, T=20) |
| --- | --- | --- |
| `meshgraphnets` | ✅ trains (Epoch 0 TrainOpt 1.71→ converging) | ✅ trains (loss 0.73→0.40 in first steps) |
| `transolver` | ✅ trains (Epoch 0 TrainOpt 0.526→Valid 0.46) | ✅ trains (loss 0.16→0.13, 13680 steps/epoch) |
| `gino` (Neural_Operator) | ✅ trains (75% through epoch 0, loss fluctuating normally) | preflight passes; not re-run standalone (shares GINO's pipeline) |
| `deeponet`/`fno`/`point_deeponet` | preflight passes; not individually real-launched (share the identical `OperatorWrapper` data pipeline as `gino`, which was real-launched) | same |
| `simulgenvae` | ✅ **fixed 2026-08-10** (was crashing at model construction) | ✅ fixed |
| `mlp` | ❌ structurally incompatible (no tabular `X`/`Y`) | ❌ same |
| `sdfflow` | ❌ structurally incompatible (`dataset_kind sdf_hdf5`, not `mesh_hdf5`) | ❌ same |
| `geometry_ingest` | N/A -- no `train` mode exists for this model at all | N/A |

### SimulGenVAE: fixed 2026-08-10 — was a real bug, not the fixed-topology limitation first assumed

Elasticity and plasticity both **do** satisfy SimulGenVAE's documented fixed-geometry
requirement (unlike cylinder_flow/deforming_plate) — every sample shares the same node count.
They still crashed, with the *identical* traceback in both cases:

```
File "model/decoder.py", line 119, in __init__
    nn.GroupNorm(min(8, max(1, num_node//4)), num_node),
ValueError: num_channels (972) must be divisible by num_groups (8)
```

`SimulGenVAE/model/decoder.py`'s final reconstruction layer hardcoded `num_groups=8` whenever
`num_var * num_nodes >= 32` (true for any realistic dataset). The real, previously undocumented
constraint was **`num_var * num_nodes` must be a multiple of 8** — unrelated to
fixed-vs-varying topology. Elasticity (`1 x 972 = 972`, `972 % 8 = 4`) and plasticity
(`4 x 3131 = 12524`, `% 8 = 4`) both failed it by coincidence of their real node counts, and so
would `flag_simple` (`3 x 1579 = 4737`, also not divisible).

**Fixed** by adding `group_norm_groups(num_channels, max_groups=8)` to
`SimulGenVAE/model/common.py` — the largest divisor of `num_channels` that is `<= 8` — and
replacing all 19 occurrences of the `min(8, max(1, X//4))` pattern across `common.py`,
`encoder.py`, and `decoder.py` with it. Behavior-identical whenever `X % 8 == 0` (every existing
config, including `ex3`, is unaffected), no longer crashes otherwise. Verified: `pytest tests/`
(17 tests) still passes; `ex8` and `ex9` both now train for real (multiple epochs, decreasing
reconstruction loss, checkpoints would save).

Separately: `configs/SimulGenVAE/ex1/config_train_vae.txt` (pointing at `dataset/ex1.h5`) still
does **not** work, for an *earlier*, different, and still-open reason: `ex1.h5` itself does not
have fixed geometry across samples (`ValueError: ... every sample must share the same (T,
N)=(1, 25203). Mismatched samples: [(1, (1, 48573)), (2, (1, 31749)), ...]`) — a data problem, not
a code bug, out of scope for this fix. `ex3`'s `dataset/ex3_train_reordered.h5` remains the one
dataset confirmed both fixed-geometry *and* (now, moot point since the divisor fix) clear of the
GroupNorm constraint. `--check`/`--dry-run` catch neither failure mode; both only ever surfaced by
actually running the native process.

## Built and config-verified this session (2026-08-10)

| Benchmark | Source | Built by | Configs passing `--check` |
| --- | --- | --- | --- |
| `elasticity/` | Zongyi Li's Geo-FNO Google Drive, `elasticity/Meshes/Random_UnitCell_{XY,sigma}_10.npy` | [`../build_geo_fno_elasticity.py`](../build_geo_fno_elasticity.py) | transolver_paper, deeponet, fno, gino, point_deeponet (5/5) |
| `plasticity/` | same Drive, `plasticity/plas_N987_T20.mat` | [`../build_geo_fno_plasticity.py`](../build_geo_fno_plasticity.py) | meshgraphnets, hi_meshgraphnets, transolver, deeponet, fno, gino, point_deeponet (7/7) |
| `fno_darcy/` | **partial** — only a 20-sample smoke fragment survived in `output/benchmarks/fno_darcy/smoke_20260719/`, promoted here as `darcy_smoke_{train,test}.h5`. Does **not** match the filenames the configs expect (`darcy_train.h5`/`darcy_paper_train.h5`) — configs will still fail path checks. | n/a | 0/4 (still broken; see below) |

Both `Random_UnitCell_*.npy` (elasticity) and `plas_N987_T20.mat` (plasticity) ship **no explicit
mesh connectivity**. Elasticity is a bare 972-point cloud -> `mesh_edge` is a deterministic k=6
proximity graph. Plasticity's `output` array is a genuine `101x31` **structured grid** (3131
nodes, matches the existing config comment exactly) -> `mesh_edge` is real 4-connected grid
adjacency, not KNN. Plasticity's `output` channels 0/1 are themselves grid coordinates that drift
slightly over time (an updated-Lagrangian mesh, like `deforming_plate`); channel-by-channel value
inspection (not the source paper) is what confirmed channels 2/3 are the true from-zero
displacement state and `input`'s 101 values are the per-sample die-profile broadcast across the
grid's other axis, before writing the converter — see the converter docstrings for the exact
verification each claim is based on.

## Found the source for, not yet built (2026-08-10)

| Benchmark | What's missing | Source found | Why not built |
| --- | --- | --- | --- |
| `fno_darcy/` full train/test | classic FNO Darcy dataset (`piececonst_r421_N1024_smooth{1,2}.mat`, 1024x421x421), would need downsampling to match the existing `darcy_85` naming | https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt- | different Google Drive folder from Geo-FNO's, not fetched this session — next step, not a dead end |
| `gino_carcfd/carcfd_paper_r64.h5` | GINO paper's car-CFD benchmark. `Neural_Operator/model/gino_carcfd.py` is a **bespoke, non-standard-contract** loader (`graph.latent_sdf [B,R,R,R,1]` + `graph.pos` surface queries normalized to `[-1,1]^3`, not the shared `nodal_data`/`mesh_edge` HDF5) -- ShapeNet-Car pressure benchmark per its own docstring | Google Drive folder `1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8` (same one as elasticity/plasticity), `car-cfd/ahmed.tar.xz` (489MB, **downloaded** to `D:/CAE_datasets_raw/geo_fno/car-cfd/`) and `car-cfd/car-cfd.tgz` (not yet fetched) | `ahmed.tar.xz` unpacks to pre-processed PyTorch `.pt` tensors (`test/info_NNN.pt`, 1658 files) whose keys were not cross-checked against `gino_carcfd.py`'s exact `latent_sdf`/`pos` contract -- needs reading that file in full (only its header was read this session) before a converter can be written correctly. `car-cfd.tgz` (not `ahmed.tar.xz`) may be the more direct ShapeNet-Car match, also unchecked. |
| `deeponet_fractional2d/fractional2d_released.h5` | 2D fractional-Laplacian-on-a-disk DeepONet benchmark, entirely **synthetic** (no external download needed at all) | n/a -- generator scripts `dataset/benchmarks/deeponet_fractional2d/{prepare_fractional2d,train_fractional2d}.py` are missing, same lost-work pattern | `Neural_Operator/tests/test_deeponet_fractional2d_data.py` proves the original generator reproduced an **exact MATLAB reference** (Joe-Kuo 2003 Sobol direction table specifically, not SciPy's newer default; MATLAB column-major meshgrid ordering; a specific flattened function/alpha/query index encoding). Reconstructing this from the test file's few assertions risks a generator that passes the visible tests but is still numerically wrong elsewhere in ways nothing here would catch -- too risky to reverse-engineer without the original reference implementation. Left alone rather than guessed. |

## A gap no amount of downloading fixes: adaptive remeshing

The MeshGraphNets paper's `flag_dynamic`/`sphere_dynamic_sizing` datasets exercise **adaptive
remeshing** (the model predicts a sizing field and a domain-independent remesher recomputes mesh
topology every step). This repo's shared contract stores one `mesh_edge` **per sample**, not per
timestep -- there is no way to represent a mesh whose connectivity itself changes over time, and
no loader here reads a sizing-field target. A stray `D:/flag_dynamic.h5` (18.85 GB, dated
2026-02-09, HDF5 "open for write" consistency flag never cleared -- left untouched, provenance and
integrity both unverified) is most likely a prior, abandoned attempt at exactly this. Fixing this
is a schema change plus a new remesher-training pipeline, not a dataset download -- out of scope
for a dataset-recovery pass, recorded here so it isn't rediscovered from scratch.
