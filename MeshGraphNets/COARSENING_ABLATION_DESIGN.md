# Coarsening Ablation — design

Four axes of the HI-MGN hierarchy, isolated one at a time:

1. **Seed selection** — FPS-Voronoi vs random sampling vs BSMS bi-stride (interpolator fixed)
2. **Prolongation** — learned interpolator vs nearest-seed copy vs inverse-distance weighting
3. **Coarse-edge construction** — mesh-crossing vs kNN proximity
4. **`k` sensitivity** — 100 → 500 → 2000

Companion to [ATTENTION_TRANSFER_DESIGN.md](ATTENTION_TRANSFER_DESIGN.md), which
ablates the *operators* (`pool_type`/`unpool_type` attention, `voronoi_branches`).
This document ablates the *hierarchy those operators run on*. The two studies
share the baseline, the runner, and the arm generator.

Status: **design only — nothing here is implemented yet.** All measurements
below are real, taken on `dataset/ex2.h5` sample 0 (N = 199,993, E = 763,224
undirected) with normalized stress at the last timestep, using a standalone
reimplementation of `model/coarsening.py`'s algorithms. The FPS/seedmean
number at k=5000 reproduces ATTENTION_TRANSFER_DESIGN.md §1's 0.2009 exactly,
which is the cross-check that the probe matches the shipped pipeline.

---

## 1. Verdict per axis

| Axis | Runnable today? | New config keys | Rebuilds `.mscache`? | Effort |
| --- | --- | --- | --- | --- |
| **4. `k` sensitivity** | **yes, no code at all** | — | yes, one per `k` | none |
| **2. Prolongation** | no | `unpool_type copy\|idw\|idw_mlp`, `idw_power` | **no — model-side only** | small (~1 file) |
| **3. Coarse edges** | no | `coarse_edge_mode`, `coarse_edge_knn` | yes | medium |
| **1. Seed selection** | partially | `coarsening_type random_voronoi`, `coarsening_seed`, `bfs_steps` | yes | medium |

Axis 4 needs zero code. Axis 2 is the cheapest to build **and** shares the
baseline's hierarchy cache, so its arms cost nothing but GPU time. Axes 1 and 3
change cached topology and each arm pays a fresh 2.35 GB cache build.

Run them in that order.

---

## 2. Measured baseline

Fine mesh, ex2 sample 0: **2 connected components** (159,744 + 40,249 nodes),
mean degree 7.63, diameter ≥ 102. The two bodies are separated by a
**40.15-unit gap = 2.5 mean edge lengths** — close enough that geometric
proximity will bridge them and mesh topology never will. Node/edge counts vary
per sample (187k–200k across the first 8), so every sample gets its own
hierarchy; nothing is shared.

### Level-0 partition of the fine mesh, FPS vs random

| `k` | method | deg | diam | cluster min/med/max | recon RMSE | peak kept | FPS time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | FPS | 9.70 | 5 | 656 / 1888 / 6229 | **0.4813** | 0.413 | 1.1 s |
| 100 | random | 11.14 | 5 | 554 / 1864 / 4442 | 0.5314 | 0.463 | 0.7 s |
| 500 | FPS | 11.44 | 10 | 86 / 382 / 1088 | **0.3612** | 0.591 | 2.7 s |
| 500 | random | 13.40 | 9 | 48 / 380 / 1234 | 0.3816 | 0.565 | 0.7 s |
| 2000 | FPS | 12.23 | 17 | 24 / 95 / 299 | **0.2558** | 0.645 | 8.2 s |
| 2000 | random | 13.94 | 13 | **1** / 94 / 321 | 0.2714 | 0.720 | 0.7 s |
| 5000 | FPS | 12.44 | 23 | 10 / 37 / 121 | **0.2009** | 0.726 | 19.5 s |
| 5000 | random | 13.78 | 19 | **1** / 37 / 179 | 0.2241 | 0.713 | 0.7 s |

### The real `k` sweep: `voronoi_clusters 5000, k` (level 0 held at 5000)

Level 1 is fixed: n_c = 5000, deg 12.44, diameter ≥ 23, recon 0.2009.

| coarsest `k` | reduction | deg | diam | comps | recon (fine→coarsest) | peak | `E_up` at level 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **100** (baseline) | 50.0× | 9.74 | **5** | 2 | 0.4906 | 0.430 | 61,408 |
| 500 | 10.0× | 11.67 | 9 | 2 | 0.3774 | 0.585 | 74,764 |
| 2000 | 2.5× | 11.80 | 12 | 2 | **0.2957** | 0.655 | 102,729 |

Reach vs information, cleanly: k=100 buys diameter 5 at recon 0.49; k=2000 buys
recon 0.30 at diameter 12. Compute barely moves — `E_up` at level 1 grows 61k →
103k against level 0's ~2.6 M, so **the whole `k` sweep is compute-neutral**
(< 2% of V-cycle work). Any accuracy difference is attributable to the
hierarchy, not to a bigger model or a slower one. That makes axis 4 the
cleanest of the four.

### BFS bi-stride cascade

| step | n_c | ratio | deg | diam | recon | peak |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 99,901 | 2.00× | 9.44 | 57 | 0.0722 | 0.948 |
| 2 | 49,871 | 2.00× | 10.33 | 36 | 0.1337 | 0.920 |
| 3 | 24,943 | 2.00× | 10.79 | 20 | 0.2248 | 0.920 |
| 4 | 12,765 | 1.95× | 11.26 | 11 | 0.3423 | 0.920 |
| 5 | 7,256 | 1.76× | 11.55 | 6 | 0.5383 | 0.762 |
| 6 | 3,428 | 2.12× | 11.49 | 2 | 0.7241 | 0.295 |
| 7 | **2** | 1714× | **0.00** | 0 | 0.7708 | 0.271 |

---

## 3. Five findings that change the design

### 3.1 Bi-stride is 2× per level, not 4× — and it cannot reach k=100 at all

`model/coarsening.py`'s docstring and [`docs/methods/03_BSMS-GNN.md`](../docs/methods/03_BSMS-GNN.md)
both state "~4× node reduction per level on triangular meshes." Measured:
**exactly 2.00×** for the first four steps. That is structural, not
mesh-specific — the rule keeps every even-depth node, which is ~half the graph
by construction, so 4× is unreachable by this implementation regardless of
mesh. Both docs should be corrected.

Consequences for the study:

- **The shipped BSMS config sketch is not the same experiment as the HI-MGN
  baseline.** `coarsening_type bfs` + `multiscale_levels 2` gives
  200k → 100k → 50k with a coarsest diameter of 36. The Voronoi baseline gives
  200k → 5k → 100 with a coarsest diameter of 5. Running those head to head
  measures "deep hierarchy vs shallow one", not "bi-stride vs FPS-Voronoi".
- **Matching the budget needs ~5 steps for level 1 and is impossible for
  level 2.** Step 5 lands at 7,256 (nearest to 5,000); step 6 at 3,428; step 7
  collapses to **n_c = 2 with zero coarse edges**. There is no bi-stride
  hierarchy whose coarsest level resembles k=100.
- **The collapse is silent.** `build_multiscale_hierarchy` appends the
  degenerate level and then breaks (`n_c <= 1 or c_ei.shape[1] == 0`), and
  `_forward_multiscale` runs `actual_levels = len(level_data)` — so the
  coarsest `GnBlock`s execute on a 2-node graph with no edges (a per-node MLP),
  and any `pre_blocks`/`post_blocks` past the truncation are allocated but never
  called. No warning is emitted. Anyone configuring a deep `bfs` hierarchy gets
  a different architecture than `mp_per_level` describes.

**Design decision.** Add `bfs_steps` (per level, default 1): compose *s*
bi-stride contractions into one V-cycle level (`ftc_total = ftc_2[ftc_1]`).
`bfs_steps 5, 1` gives 200k → 7,256 → 3,428, the closest bi-stride can get to
5,000 → 100. Report both arms and state the mismatch rather than pretending
the budget matched:

| arm | hierarchy | what it answers |
| --- | --- | --- |
| `s-bfs-doc` | `bfs`, L=2, no `bfs_steps` | what the shipped BSMS config actually does |
| `s-bfs-matched` | `bfs`, `bfs_steps 5, 1` | bi-stride at the closest reachable budget |

Also add a **warning** when `len(hierarchy) < multiscale_levels` or when a
level has zero coarse edges. That is a real latent bug, not just a study
concern.

### 3.2 Random seeding needs ≥ 3 seeds to be honest

Across 3 RNG seeds, recon varies **2.1–2.4%** of the mean; FPS across 3 start
points varies **1.0–1.9%**. The FPS-vs-random gap is 5–11%. So the effect is
real but only 2–5× the noise floor.

| `k` | random (3 seeds) | FPS (3 starts) |
| --- | --- | --- |
| 500 | 0.3816 / 0.3831 / 0.3634 → **0.3760 ± 0.0090** | 0.3612 / 0.3477 / 0.3633 → **0.3574 ± 0.0069** |
| 5000 | 0.2241 / 0.2146 / 0.2141 → **0.2176 ± 0.0046** | 0.2009 / 0.1978 / 0.2026 → **0.2004 ± 0.0020** |

Run the random arm at **3 RNG seeds** and report mean ± sd. A single random arm
that lands within ~2% of FPS proves nothing. Note that FPS is *also* stochastic
in effect (the start point matters), so the FPS baseline deserves the same
treatment if the gap comes out small.

Two secondary effects worth reporting, both visible above:

- **Random produces degenerate clusters** — minimum cluster size 1 at k≥2000
  (a seed that captured only itself). FPS's minimum is 10 at k=5000.
- **Random costs more at the coarse level** — deg 13.78 vs 12.44 at k=5000,
  i.e. ~11% more coarse edges and proportionally more unpool edges. The random
  arm is therefore *not* compute-matched; it is slightly more expensive. Say so.
- Random gives a *smaller* diameter (19 vs 23) because uneven cluster sizes
  bridge distance. So random trades information for reach — the same trade
  axis 4 sweeps. Do not report it as strictly worse.

### 3.3 A degree-matched kNN arm needs `knn_k ≈ 12`, not 6

Symmetrizing kNN dedups heavily, so `knn_k = mesh_degree / 2` badly
under-connects. At level 0 (k=5000, mesh-crossing = 62,184 edges, deg 12.44,
diameter 23):

| `knn_k` | edges | deg | diam | comps | IoU with mesh-crossing | frac of kNN edges also mesh-crossing |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 34,234 | 6.85 | 30 | 1 | 0.538 | 0.986 |
| 9 | 49,854 | 9.97 | 26 | 1 | 0.710 | 0.933 |
| **12** | **66,006** | **13.20** | **23** | 1 | **0.746** | 0.830 |
| 15 | 82,360 | 16.47 | 21 | 1 | 0.695 | 0.719 |

`knn_k 12` matches edge count (+6%) *and* diameter (23 = 23) while 17% of its
edges are new. That is the arm to run: same budget, same reach, different
*which*. Anything at `knn_k 6` confounds "kNN vs mesh" with "sparser graph".

### 3.4 kNN's real effect here is bridging the two bodies — which confounds it with `coarse_world_edges`

Mesh-crossing coarse graphs keep **2 components** at every level (the two
bodies never touch through mesh edges). Every kNN variant collapses to **1
component**. The far-away kNN edges are not scattered shortcuts — at k=5000
only 0.4% of kNN edges have a mesh-hop distance > 2, and *all* of those are
between the two components. With a 2.5-edge-length gap, kNN is effectively
inventing contact coupling.

That is exactly what `coarse_world_edges` does deliberately, and the ex2
configs currently run `use_world_edges True` with `coarse_world_edges` **unset
(False)** — so today the coarse levels carry *no* cross-body coupling at all. A
two-arm mesh-vs-kNN comparison would credit kNN for something the existing
`coarse_world_edges` flag already provides. Run three arms:

| arm | `coarse_edge_mode` | `coarse_world_edges` | isolates |
| --- | --- | --- | --- |
| `e-mesh` (baseline) | `mesh` | False | — |
| `e-mesh-cwe` | `mesh` | **True** | does the coarse level need cross-body coupling *at all* |
| `e-knn` | `knn`, `coarse_edge_knn 12` | False | does geometric proximity beat contact-derived coupling |

If `e-mesh-cwe` ≈ `e-knn`, kNN is buying nothing that a one-line flag doesn't
already give. That is the outcome to be prepared for.

Second decision: **build kNN from reference positions, once, in the cache** —
not per timestep from deformed positions. Rebuilding topology every rollout
step would change the graph under AR-RT and invalidate the per-level edge
normalization stats. Mesh edges are reference-derived; kNN edges must be too.

### 3.5 The IDW-vs-copy ordering flips with `k`, so axes 2 and 4 must be crossed

Pure geometry, no learning — reconstructing the fine stress field from cluster
means through each fixed prolongation:

| `k` | nearest-seed copy | IDW (p=2, unpool support) | Δ |
| --- | --- | --- | --- |
| 100 | 0.4813 | 0.4962 | **+3.1%** (IDW worse) |
| 500 | 0.3612 | 0.3585 | −0.8% |
| 2000 | 0.2558 | 0.2426 | −5.1% |
| 5000 | 0.2009 | **0.1829** | **−9.0%** |

At coarse `k` the clusters are so large that blending neighbours smears; at
fine `k` blending wins. **Running axis 2 at a single `k` would produce a result
that reverses at another `k`.** Run it at the baseline's coarsest k=100 **and**
at k=2000.

These numbers are also the floor the learned unpool must beat. If the learned
arm does not clear IDW by a comfortable margin, the 265k parameters it costs
(§4.2) are not earning their keep.

---

## 4. Arm definitions

Baseline for all four axes: `configs/MeshGraphNets/ex2/config_train_himgn.txt`
(AR-OT), **not** `_base.txt` (AR-RT). Rationale: 13 arms of AR-RT on ex2 is not
affordable, coarsening effects show up in one-step prediction, and AR-OT is
already the shipped reference. Promote the per-axis winner to AR-RT afterwards.
Everything else — `split_seed 42`, `Latent_dim 128`, `mp_per_level 2,3,5,3,2`,
100 epochs, `Batch_size 4`, EMA, `use_world_edges True` — is held fixed and
generated mechanically, exactly as the p1/p2/p12 arms are.

### 4.1 Axis 4 — `k` sensitivity (no code)

| arm | change | cache |
| --- | --- | --- |
| `k100` | = baseline (`voronoi_clusters 5000, 100`) | baseline |
| `k500` | `voronoi_clusters 5000, 500` | new |
| `k2000` | `voronoi_clusters 5000, 2000` | new |

Report against the §2 table: if accuracy tracks recon RMSE, information is the
binding constraint; if it tracks diameter, reach is. That single answer decides
whether `voronoi_branches` (which buys both at once — ATTENTION_TRANSFER_DESIGN.md
Part II) is worth pursuing, so **run this axis first**.

### 4.2 Axis 2 — prolongation (model-side only, shares the baseline cache)

Extend `unpool_type` from `sum | attention` to add:

| value | operator | learned params/level |
| --- | --- | --- |
| `sum` (baseline) | `node_mlp([skip, Σ edge_mlp([h_c, skip, rel_pos])])` | 132,736 |
| `copy` | `h_up = h_coarse[ftc]` — piecewise constant, own cluster only | 0 |
| `idw` | `h_up = Σ_c w_ic h_c / Σ_c w_ic`, `w = ‖pos_i − anchor_c‖^−p` over the existing unpool support | 0 |
| `idw_mlp` | `node_mlp([skip, idw_agg])` — fixed transfer, learned post-projection | 66,176 |

Plus `idw_power` (default 2). In every case `skip_projs[i]` still merges, so the
coarse→fine path is never fully unlearned and the arms stay comparable.

Two things to keep straight:

- **`copy` changes the support as well as the weights** (own cluster only,
  vs own cluster + all coarse neighbours). `idw` keeps the full support. So
  `copy` vs `idw` varies two things. If the gap is large, add a `uniform` arm
  (equal weights over the full support) to separate them — it is a two-line
  addition to the same dispatch.
- **`copy`/`idw` are 265,472 params lighter** than the baseline (8.4% of
  3,164,292). `idw_mlp` exists precisely so a loss can be attributed to *the
  fixed transfer* rather than *fewer parameters*. Run `idw_mlp` at whichever
  `k` `idw` does best.

Arms: `i-copy@k100`, `i-idw@k100`, `i-copy@k2000`, `i-idw@k2000`, `i-idwmlp@k*`.
All five share the two hierarchy caches axis 4 already built — **zero extra
cache cost**.

Note the honest caveat: IDW is a geometric interpolant applied to *learned
latent* features, which have no intrinsic distance-decay. It is the classical
multigrid prolongation and therefore the right baseline, but it is not
obviously well-posed in latent space, and a poor result partly reflects that.

### 4.3 Axis 1 — seed selection

New `coarsening_type random_voronoi`: identical to `fps_voronoi_coarsen` except
seeds come from `rng.choice(N, k, replace=False)` instead of FPS. Everything
downstream (multi-source BFS Voronoi, component guarantee, cluster compaction,
boundary coarse edges, seedmean anchoring) is unchanged, which is what makes
this a clean single-variable arm. Add `coarsening_seed` (int).

| arm | keys |
| --- | --- |
| `s-fps` | = baseline |
| `s-rand-0/1/2` | `coarsening_type random_voronoi`, `coarsening_seed 0 / 1 / 2` |
| `s-bfs-doc` | `coarsening_type bfs`, `voronoi_clusters 0` |
| `s-bfs-matched` | `coarsening_type bfs`, `bfs_steps 5, 1` |

Interpolator fixed at the baseline learned unpool throughout, per the ask.
`coarsening_seed` **must** enter the cache signature (§5.1) or the three random
arms silently share one cache and the seed sweep measures nothing.

### 4.4 Axis 3 — coarse-edge construction

New `coarse_edge_mode` (`mesh` default | `knn` | `mesh+knn`) and
`coarse_edge_knn` (per level). Arms as in §3.4, with `coarse_edge_knn 12`
degree-matched at level 0 — re-measure the matched value for the coarsest
level, don't reuse 12 blindly.

`mesh+knn` (union) is worth a fifth arm only if `e-knn` wins; it is the variant
that keeps mesh fidelity and adds proximity rather than replacing it.

---

## 5. Landmines

### 5.1 Cache signature — the silent-corruption one

`multiscale_cache._coarse_params` currently returns `{levels, types, clusters,
branches}`. **Every new key that changes cached topology must be added there**,
or two arms hash to the same `.mscache.*.h5` and one silently trains on the
other's hierarchy:

| key | in signature? |
| --- | --- |
| `coarsening_seed` | **required** |
| `bfs_steps` | **required** |
| `coarse_edge_mode`, `coarse_edge_knn` | **required** |
| `unpool_type`, `idw_power` | no — model-side, arms share a cache (this is the point) |
| `coarse_world_edges` | no — world edges are lifted at graph-build time, not cached |

Bump `FORMAT_VERSION` to 3 alongside. This failure mode produces no error and
no warning; it produces a plausible wrong number.

### 5.2 Every new key needs six edits

The existing keys show the full path. Missing any one of these fails late or
silently:

1. `model/coarsening.py` / `general_modules/multiscale_helpers.py` — the algorithm
2. `general_modules/mesh_dataset.py` — parse + validate (`_VALID_COARSENERS`)
3. `general_modules/multiscale_cache.py` — `_coarse_params` (topology keys only)
4. `model/MeshGraphNets.py` — `_build_multiscale_processor` validation (a model
   is rebuilt from a checkpoint's `model_config` without touching the dataset)
5. `training_profiles/setup.py::build_model_config` — or inference reconstructs
   the wrong architecture from the checkpoint
6. `cae_suite/specs/meshgraphnets.py` — `MGN_KEYS` + the `coarsening_modes` set,
   or `--audit-configs` rejects valid configs / accepts invalid ones

`parallelism/model_split.py` has its own multiscale forward and is already not
branch-aware; keep every arm on `parallel_mode ddp`.

### 5.3 Disk

`ex2.mscache.*.h5` is **2.35 GB per distinct coarsening config** against a 7.65
GB dataset, and the volume has **102 GB free (90% used)**. Distinct caches
needed: baseline + k500 + k2000 + 3 random + 2 bfs + knn = **8 × 2.35 ≈ 19 GB**
concurrent at peak. Axis 2's five arms add none. That fits, but:

- keep `hierarchy_cache_keep` at its default `false` so each run deletes its own
- do not start a second wave before the first wave's caches are released
- there is already a stale `dataset/ex2.mscache.d3f3d2305c.h5` on disk from an
  earlier run; `_prune_siblings` collects it on the next run, or delete it now

### 5.4 Scheduling

`run_ablation.sh`, `compare_ablation.py`, and
`configs/MeshGraphNets/gen_ablation_arms.py` were **deleted** (commits `3ae5930`
and `45c9482`) but are recoverable:

```bash
git show 45c9482^:configs/MeshGraphNets/gen_ablation_arms.py > configs/MeshGraphNets/gen_ablation_arms.py
git show 3ae5930^:run_ablation.sh        > run_ablation.sh
git show 3ae5930^:compare_ablation.py    > compare_ablation.py
```

Extend rather than rewrite them — the generator's whole premise (arms derive
mechanically from one baseline so they cannot drift) is exactly what a 13-arm
study needs. The runner assumes 8 GPUs, one arm per GPU, `gpu_ids` single-valued
so the native launcher takes the non-DDP path.

Note the p1/p2/p12 campaign from ATTENTION_TRANSFER_DESIGN.md **has not
actually run** — `output/meshgraphnets/ex2/ablation_runner.log` records four
0-minute dry runs on 2026-07-29 and nothing since. Decide whether that study
completes first; the two share GPUs and the same baseline.

### 5.5 What to report

Node-MSE alone will not separate these arms — every axis here trades
information against reach, and both show up in *rollout* error long before
they show up in one-step MSE. Report, per arm:

- one-step node MSE **and** 49-step rollout RMSE (error-vs-step curve)
- **peak stress error** — the §2 tables show peak retention swinging 0.29–0.95
  across configs while RMSE moves much less, and peak von Mises is the FEA
  quantity of interest
- wall-clock per epoch and cache build time (random is 27× faster to build than
  FPS at k=5000; `bfs` is faster still)
- the §2 geometric predictors (recon RMSE, coarse diameter) beside the trained
  result — the point of the cheap probe is to learn whether it *predicts*, which
  would let future hierarchy choices skip training entirely

---

## 6. Recommended order

| stage | arms | code needed | cache cost | answers |
| --- | --- | --- | --- | --- |
| **0** | 0 | none (probe already run) | 0 | the tables above, extended to all 50 samples |
| **1** | `k500`, `k2000` | **none** | 2 × 2.35 GB | information vs reach — gates everything else |
| **2** | 5 prolongation arms | `unpool_type` dispatch, 1 file | **0** | is the learned unpool earning its 8.4% params |
| **3** | 4 seed arms + 3 edge arms | 2 modes + 3 keys + 6-point wiring | 6 × 2.35 GB | the two topology axes |

Stage 1 is two configs and a `--check`. Stage 2 is the highest information per
unit of work in the whole plan: no cache, no dataset change, one dispatch site,
and a measured prediction (§3.5) that the answer flips with `k` — which is
falsifiable, so the run is worth something whichever way it lands.
