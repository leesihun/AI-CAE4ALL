# Multi-Partition Coarse Representation + Learned Transfer Operators

Design document for two coupled changes to the multiscale V-cycle:

- **Part I** — replace the fixed restriction (`mean`) and the fixed prolongation
  aggregation (`sum`) with learned attention, **without changing the model's
  behaviour at initialization**.
- **Part II** — represent each coarse level with **K parallel partitions**
  instead of one, decoupling how much information the coarse level carries from
  how far it can communicate.
- **Part III** — why neither works well without the other, and why multi-head
  attention is *not* a substitute for multiple partitions.

Status: **Parts I and II are implemented** (see the "As-built" sections below
for what shipped and where each deviates from the plan — Part II in
particular ships a deliberately narrower scope than §5-7 describe). Part III
remains an analysis/argument, not something with its own code to build.
All measurements are on `dataset/ex2.h5` sample 0 (N = 199,993), normalized
stress at the last timestep, with broadcast-reconstruction RMSE as the proxy
for how much field information the coarse level retains.

---

## As-built (Part I)

What actually shipped is simpler than §8 originally planned, in one respect:
**no dataset or cache changes were needed.** The plan called for precomputing
`log_scale` in `mesh_dataset.py` and caching it (like `x_pos`). Instead,
`EncoderProcessorDecoder._pool_log_scale` computes it on the fly, once per
level, from whatever connectivity already feeds that pool step:
`current_graph.edge_index` (the fine mesh at level 0, `coarse_edge_index_{i-1}`
at level `i>0`) and `ld['fine_pos']` — both already in scope in
`_forward_multiscale`, needed by nothing new. This sidesteps a cache
`FORMAT_VERSION` bump and the invalidation of every existing `.mscache.*.h5`
file entirely for Part I. The phase-1b query/anchor extension (§2) was not
built — `pool_type`/`unpool_type` ship as additive-attention only.

One refinement beyond the original plan: **`pool_type` and `unpool_type` are
independent config keys** (not one flag driving both), because §10.5's
pool-only / unpool-only / both ablation needs them decoupled.

Files touched:

| File | Change |
| --- | --- |
| [`model/blocks.py`](../../../methods/MeshGraphNets/model/blocks.py) | `AttentionPoolBlock` (new); `UnpoolBlock` gained a `use_attention` constructor flag and an `attn_mlp` |
| [`model/MeshGraphNets.py`](../../../methods/MeshGraphNets/model/MeshGraphNets.py) | `pool_type`/`unpool_type`/`pool_heads` config plumbing in `_build_multiscale_processor`; the pool branch in `_forward_multiscale`; `_pool_log_scale`; zero-init walk in `MeshGraphNets.__init__` |
| [`training_profiles/setup.py`](../../../methods/MeshGraphNets/training_profiles/setup.py) | `build_model_config` carries the three new keys so checkpoints reconstruct the right architecture at inference (`inference_profiles/rollout.py` already does a blanket `model_config` merge — no change needed there) |
| `cae_suite/specs/meshgraphnets.py` (parent repo) | `pool_type`, `pool_heads`, `unpool_type` added to `MGN_KEYS` |
| [`tests/test_attention_transfer.py`](../../../methods/MeshGraphNets/tests/test_attention_transfer.py) | equivalence (§10.1), pool-only/unpool-only ablation, a train-step divergence sanity check, and a batching/no-leakage check |

A numerics note beyond what §9 specified: both score computations
(`AttentionPoolBlock.forward`, `UnpoolBlock`'s attention branch) run inside a
nested `torch.autocast(..., enabled=False)` with `.float()`-cast inputs, and
the pool/unpool aggregation casts the resulting attention weights back to the
ambient (possibly bf16) activation dtype before the final `scatter` — done
this way (rather than leaving the whole aggregate in fp32) specifically to
avoid silently upgrading the rest of that level's compute to fp32 through
ordinary dtype promotion.

Validation performed: all 6 new tests plus the existing 10 (`test_ar_rollout.py`,
`test_multiscale_stats.py`) pass — 16/16. The equivalence tests cover 1- and
2-level hierarchies at 1e-4/1e-5 tolerance (not bit-exact: softmax's `1/n`
division and the unpool `deg_i * (1/deg_i)` round-trip aren't associative with
the plain mean/sum in floating point, so an exact-equality assertion would be
the wrong test). Not yet done: the actual A/B training run (§10.4) — the
equivalence property only proves the starting point is safe, not that
training moves anywhere useful.

To try it, add to a config: `pool_type attention`, `unpool_type attention`,
`pool_heads 4` (each independently optional; both default to today's behavior
if omitted).

---

## As-built (Part II)

Ships a narrower design than §5-7: **branching is only supported on the last
configured hierarchy level**, not arbitrary per-level combinations. §6's own
worked example already only branches the coarsest transition, but the text
otherwise reads as if any level could carry its own `K`. It can't, in this
implementation, and the reason is combinatorial: if a non-terminal level
branched, every level beneath it would need to fork into K independent
sub-hierarchies (K different downstream partitions, each needing its own
further coarsening), and K would compound multiplicatively with depth. Making
only the terminal level branch sidesteps that entirely — there is nothing
"beneath" it to fork, so one set of shared-weight processor/pool/unpool blocks
handles all K branches with no recursion. This is enforced in three places,
not just documented: `build_multiscale_hierarchy` raises before building
anything if a non-last level requests `voronoi_branches > 1`;
`EncoderProcessorDecoder._build_multiscale_processor` raises the same check
independently (a model can be reconstructed from a checkpoint's
`model_config` without going through the dataset); and `MeshGraphDataset.__init__`
raises it a third time, as early as possible, before any HDF5 or cache work
starts.

Two further scope cuts, each with a corresponding guard rather than a silent
gap:

- **No `coarse_world_edges`.** Which branch would receive a lifted contact
  edge is not a well-defined question (a contact could plausibly belong to
  several branches' clusters at once), so this is rejected at
  `MeshGraphDataset.__init__` rather than answered arbitrarily.
- **`parallel_mode model_split` is not branch-aware.** `parallelism/model_split.py`
  has its own independent multiscale forward pass (for splitting the model
  itself across GPUs) that was not touched. It reads `graph[f'fine_to_coarse_{level}']`
  unconditionally, which a branched level never writes (only the
  `_{level}_{branch}`-suffixed name), so it fails loudly (`KeyError`) rather
  than silently misbehaving — but it does fail. Only `parallel_mode ddp` (the
  default single-GPU/DDP path through `model/MeshGraphNets.py`) supports
  `voronoi_branches > 1` right now.

**AR-RT is supported** (added after the initial Part II landing, which had
shipped an AR-OT-only guard). `_refresh_multiscale` detects a branched level by
probing for `num_coarse_{level}_0` and refreshes every branch from the same
upstream positions; `_coarse_positions` and the new `_refresh_one_partition`
take a `{level}_{branch}` key instead of a level index. The subtle half was
`RolloutContext`: a branched level's entry in `coarse_edge_means` is a *list*
of per-branch arrays, and the deformed-half slice has to be applied to each
array — applied to the list it would slice the branch axis and yield a
silently wrong `[K, 8]` normalization constant rather than raising. That slice
is now `DEFORMED_SLICE` / `REFERENCE_SLICE` from
[`general_modules/edge_features.py`](../../../methods/MeshGraphNets/general_modules/edge_features.py) instead
of a bare `4` repeated at seven call sites, with `EDGE_FEATURE_DIM` derived
from the two half-widths and a test pinning the constants to what
`deformed_edge_attr_torch` actually emits.

Branch merging ships as **§7 option 1 only** (concat + `skip_projs`, widened
from `2*latent_dim` to `(1+K)*latent_dim` input — a plain `nn.Linear` learns
the combination). Option 2 (attention over branches) was not built; nothing
blocks adding it later as an alternative merge mode alongside this one.

The two features compose: `pool_type`/`unpool_type` `attention` (Part I) run
once per branch with shared weights, exactly as `mean`/`sum` would — no
special-casing needed, verified by a combined test.

Files touched beyond Part I's list:

| File | Change |
| --- | --- |
| [`model/coarsening.py`](../../../methods/MeshGraphNets/model/coarsening.py) | `_fps_euclidean`/`_fps_geodesic`/`fps_voronoi_coarsen`/`coarsen_graph` gain a `seed_start`/`start` param (default 0 = today's behavior exactly); `MultiscaleData`'s `_LEVEL_RE` and `__inc__` extended for an optional branch suffix |
| [`general_modules/multiscale_helpers.py`](../../../methods/MeshGraphNets/general_modules/multiscale_helpers.py) | `build_multiscale_hierarchy` gains `voronoi_branches`, produces `{'branches': [...]}` entries; `attach_coarse_levels_to_graph` refactored around a new `_attach_one_partition` helper, called once per branch when present |
| [`general_modules/multiscale_cache.py`](../../../methods/MeshGraphNets/general_modules/multiscale_cache.py) | `FORMAT_VERSION` bumped to 2 (defensive — the new `voronoi_branches` signature key already forces a fresh cache path); `_write_entry`/`_read_entry` refactored around `_write_one_branch`/`_read_one_branch`, keyed `L{l}B{b}_...` only for an actually-branched level |
| [`general_modules/mesh_dataset.py`](../../../methods/MeshGraphNets/general_modules/mesh_dataset.py) | `voronoi_branches` parsing + two validation guards (last-level-only, no `coarse_world_edges`); `_compute_coarse_edge_stats` rewritten around per-branch accumulator lists (a new `_accumulate_coarse_edge_stats` helper); `_create_subset`/`inherit_preprocessing_from` carry the per-level mean/std lists through splits (`_copy_edge_stat` handles the list-of-arrays case) |
| [`model/MeshGraphNets.py`](../../../methods/MeshGraphNets/model/MeshGraphNets.py) | `voronoi_branches` parsing + guard in `_build_multiscale_processor`; `skip_projs` width scales with branch count; `_extract_level_data` detects a branched level from the presence of a `_{i}_0`-suffixed attribute; new `_pool_one` (factored out, byte-identical to the old inline pooling logic) and `_unpool_merge_branched_level`; `_forward_multiscale`'s loops branch on `'branches' in ld` |
| [`general_modules/edge_features.py`](../../../methods/MeshGraphNets/general_modules/edge_features.py) | `DEFORMED_FEATURE_DIM` / `REFERENCE_FEATURE_DIM` / `DEFORMED_SLICE` / `REFERENCE_SLICE`; `EDGE_FEATURE_DIM` now derived from the two halves instead of a literal 8 |
| [`training_profiles/ar_rollout.py`](../../../methods/MeshGraphNets/training_profiles/ar_rollout.py) | branch-aware `_refresh_multiscale` + new `_refresh_one_partition`; `_coarse_positions` keyed by `{level}_{branch}`; `RolloutContext.coarse_edge_stat()` unwraps per-branch stat lists; all seven bare `4` slices replaced with the named slices |
| [`training_profiles/setup.py`](../../../methods/MeshGraphNets/training_profiles/setup.py) | `build_model_config` carries `voronoi_branches` |
| `cae_suite/specs/meshgraphnets.py` (parent repo) | `voronoi_branches` added to `MGN_KEYS` |
| [`tests/test_multi_partition.py`](../../../methods/MeshGraphNets/tests/test_multi_partition.py) (new) | K=1-explicit-vs-unset equivalence, single- and two-level forward pass, **batching/no-cross-sample-leakage** (the design's own flagged highest-risk item), a gradient step confirming every branch's slice of `skip_projs`' input gets a nonzero gradient every step (not just eventually), the two hierarchy-build validation guards, the model-construction guard, per-branch stat unwrapping in `RolloutContext`, and composition with Part I attention |
| [`tests/test_ar_rollout.py`](../../../methods/MeshGraphNets/tests/test_ar_rollout.py) | new `multiscale_branched` geometry case in `GEOMETRY_CASES` (rebuilt-vs-dataloader feature fidelity, per branch) with assertions that it cannot pass vacuously, plus a test pinning the edge-half constants to `deformed_edge_attr_torch`'s real output |

The `k_branches <= 1` code path in every touched function is, line for line,
the code that existed before Part II — not "equivalent to," literally the
same branch of the same `if`. `voronoi_branches` omitted or all-1s never
constructs a `{'branches': ...}` entry anywhere in the pipeline, so the
default configuration cannot regress by construction, not just by test
coverage. `test_explicit_single_branch_matches_unset_config` checks this at
the model level with `atol=0, rtol=0` (exact equality, not the 1e-4/1e-5
tolerance Part I's attention equivalence needed — there's no softmax
rounding here, K=1 truly is the old code path).

Validation performed: 28/28 tests pass. The AR-RT branch support was
additionally checked by sabotaging `_refresh_multiscale`'s branch path and
confirming the `multiscale_branched` case fails — an equivalence-style test
that cannot detect a no-op is worth very little.

To try it, add to a `himgn`-style config (2-level example):
```
multiscale_levels    2
voronoi_clusters     5000, 100
voronoi_branches     1, 4          # 4 partitions of the coarsest (100-cluster) level
```
`voronoi_branches` defaults to `1` at every level if omitted. Combine freely
with `pool_type`/`unpool_type attention` from Part I, and with either
`time_integration`. Not compatible with `coarse_world_edges True` or
`parallel_mode model_split` (both raise a clear error rather than running
incorrectly) — use `coarse_world_edges False` and `parallel_mode ddp`.

One operational note: since `voronoi_branches` is a new key in the cache
signature, the **first run after upgrading will rebuild `.mscache.*.h5`
even for existing configs that never set it** (the signature hash changes for
everyone, so a fresh cache file is built at a new path; old cache files are
left on disk, orphaned, not overwritten — clean them up manually if disk
space matters). This costs the same one-time FPS/Voronoi build time as any
other coarsening-config change already does.

---

# Part I — Learned transfer operators

## 1. Motivation

Inter-level transfer is the one part of the hierarchy that is *not* learned on
the way up:

| Direction | Operator | Learned? | Where |
| --- | --- | --- | --- |
| fine → coarse (restriction) | `scatter(reduce='mean')` or `x[seeds]` gather | **no** | [`pool_features`](../../../methods/MeshGraphNets/model/coarsening.py) L411, [`_forward_multiscale`](../../../methods/MeshGraphNets/model/MeshGraphNets.py) L257-260 |
| coarse → fine (prolongation) | learned MLPs, but `scatter(reduce='sum')` aggregation | partly | [`UnpoolBlock`](../../../methods/MeshGraphNets/model/blocks.py) L135-173 |

Measured, at k = 5,000 (~40 nodes/cluster):

```
seedmean (mean-pool)          recon RMSE 0.2009   global peak kept  72.6%
inherit  (gather @ FPS seed)  recon RMSE 0.2225   global peak kept  86.8%
inherit  (gather @ |max|)     recon RMSE 0.4475   global peak kept 100.0%
```

1. **Neither fixed operator wins on both axes.** The mean is the L2-optimal
   single representative of a cluster, so it minimises RMSE — and for exactly
   that reason it is a low-pass filter that destroys extrema. Gather preserves
   extrema but aliases. In FEA the quantity of interest is usually peak von
   Mises stress, so the RMSE-optimal operator loses the number that matters.
2. **The right choice is per-cluster.** Only ~13% of clusters have
   `|max| > 2·|mean|`. One fixed rule must be wrong on one of the two
   populations.

Both existing modes are endpoints of one operator — uniform weights
(`seedmean`) and one-hot weights (`inherit`) inside a cluster. Learning the
weights subsumes both and removes the pool/gather axis from `coarsening_type`.

## 2. Restriction: attention pool

For cluster `c` with members `{i : ftc[i] = c}`:

```
s_i  = score_mlp([ h_i , pos_i - anchor_c , log_scale_i ])      # [N, H]
a_i  = softmax(s_i, index=ftc)                                  # within-cluster
h_c  = scatter(a_i * value_proj(h_i), ftc, reduce='sum')        # [M, H*d_head]
```

**Multi-head (H = 4) is the point, not a detail.** With one head the operator
must choose between a mean-like and a peak-like summary — the same dilemma the
fixed operators have. With several heads it carries both, which is the only way
to improve RMSE and peak retention simultaneously. See Part III for what
multi-head does **not** buy.

Score inputs:
- `h_i` — the fine latent state.
- `pos_i - anchor_c` — geometric offset from the cluster anchor
  (`coarse_centroid_{l}`, already on the graph).
- `log_scale_i` — log of a local mesh-size proxy (sum of incident reference
  edge lengths). FEM restriction operators are mass-matrix weighted, not
  uniform, and cluster sizes here vary 3-10x. We do not hard-code the
  weighting; we supply the feature that makes it learnable.

Optional (phase 1b): use the anchor's features as a **query**, making this true
cross-attention. Requires `coarse_anchor_idx_{l}` at every level, not only
under AR-RT — flip the `expose_anchors` flag at
[`attach_coarse_levels_to_graph`](../../../methods/MeshGraphNets/general_modules/multiscale_helpers.py) L208.
Seeds are already stored in the cache, so **no cache rebuild is needed** for this.

## 3. Prolongation: attention unpool

`unpool_edge_index` connects each fine node to its own cluster plus every coarse
neighbour of that cluster (~13 sources per fine node at level 0, where the
coarse graph's mean degree is 12.4). Today all are summed with equal weight.

```
msg_ci   = edge_mlp([ h_c , h_i_skip , rel_pos ])       # unchanged
score_ci = attn_mlp([ h_c , h_i_skip , rel_pos ])
a_ci     = softmax(score_ci, index=dst_fine)            # over node i's sources
h_up_i   = deg_i * scatter(a_ci * msg_ci, dst_fine, reduce='sum')
```

A fine node between two coarse regions can now choose which to listen to.
Normalising over sources also makes prolongation an **interpolation** operator,
which is what a multigrid prolongation is supposed to be — the current sum is
not. The `deg_i` factor preserves the current scale; see §4.

## 4. Exact baseline recovery at initialization — the critical property

The current best configuration is `voronoi_seedmean`. If attention starts
random, training starts *below* that baseline and may never recover. Both
operators must reduce **exactly** to today's behaviour at step 0.

Zero-initialize the last `Linear` of `score_mlp` and `attn_mlp` (weight *and*
bias). Then:

- **Pool**: all scores equal → `a_i = 1/n_c` → `h_c = mean(h_i)`. Identical to
  `pool_features`, provided `value_proj` is identity-initialized (or omitted in v1).
- **Unpool**: all scores equal → `a_ci = 1/deg_i` → `deg_i * Σ a_ci·msg_ci =
  Σ msg_ci`. Identical to the current `reduce='sum'`.

The run therefore starts at the current model and departs only if the gradient
says to. This turns a risky rewrite into a strictly-safe generalization and
makes the A/B honest: any delta is attributable to the learned weights, not to
a different initialization basin. Same idea as the existing decoder trick at
[`MeshGraphNets.__init__`](../../../methods/MeshGraphNets/model/MeshGraphNets.py) L26.

---

# Part II — Multi-partition coarse representation

## 5. The problem: `k` controls two things at once

`voronoi_clusters` sets one number per level, and it couples:

- **information retained** at the coarse level (large `k` is better), and
- **coarse-graph diameter / global reach** (small `k` is better).

For an elliptic problem you need a small coarse graph for global reach — and
small is exactly what destroys the information. Measured, at matched coarse-node
budget:

| Configuration | coarse nodes | **diameter** | info ceiling (recon RMSE) |
| --- | --- | --- | --- |
| k=100 × 1 | 100 | 5 | 0.4831 |
| k=400 × 1 | 400 | 9 | 0.3716 |
| **k=100 × 4** | 400 | **6** | **0.3277** |
| k=1600 × 1 | 1600 | 16 | 0.2597 |
| **k=100 × 16** | 1600 | **6** | **0.1747** |
| k=5000 × 1 | 5000 | 23 | 0.1946 |
| k=20000 × 1 | 20000 | 37 | 0.1278 |
| **k=5000 × 4** | 20000 | **23** | **0.0903** |

At a 1,600-node budget, 16 partitions of 100 beat one partition of 1,600 on
**both** axes at once: diameter 6 vs 16, information 0.1747 vs 0.2597.

Mechanism: a single Voronoi partition draws one set of cluster boundaries, and
every boundary is a place where information is cut. With K partitions whose
boundaries fall elsewhere, one branch's blind spots are covered by another.

**The earlier negative result is what makes this work.** Seed *placement* barely
matters (Lloyd relocation: +1.5%), so K seed sets are all equally good — which
is precisely why K of them compose into complementary views at no quality cost.

## 6. Where to branch: coarser levels, where compute is nearly free

```
level 0 (200k nodes)  ->  K=1   1 partition,  5000 clusters
level 1 (5000 nodes)  ->  K=4   4 partitions,  500 clusters each
level 2 (100 nodes)   ->  K=16 16 partitions,  100 clusters each
```

Cost is dominated by the unpool edge set, `E_up ≈ (1 + coarse_degree) · N_level`
where `N_level` is the *finer* side of that level:

- level 0: `13 × 200,000 ≈ 2.6 M` edges
- level 1: `13 × 5,000 ≈ 65 K` edges — **2.5% of level 0**
- level 2: `13 × 100 ≈ 1.3 K` edges — negligible

So K=4 at level 1 costs ~+7% of level-0 work, and K=16 at level 2 costs
essentially nothing. **The multiplication lands where compute is cheapest.**
Keeping K=1 at level 0 also keeps the on-disk cache from blowing up (it is
already 679 MB for `ex1` against a 275 MB dataset).

Standard hierarchies collapse 200k → 5k → 100, so representational width decays
geometrically. Branching keeps the *total* width roughly constant per level
while each branch's graph stays small enough for global reach — closer to a
filter bank than to a single pyramid.

## 7. Combination across branches

Naive averaging of the K cluster means recovers little of the available
information:

```
k=100  × 16 :  naive average 0.4081   vs   ceiling 0.1747
k=5000 ×  4 :  naive average 0.1363   vs   ceiling 0.0903
```

The information is present but must be *extracted*. Two options at the merge
point in `_unpool_merge_level`:

1. **Concat + linear**: `skip_proj(cat([skip_x, h_up^1, ..., h_up^K]))`. Simple,
   and the projection is a learned linear combiner.
2. **Attention over branches**: score each branch's `h_up^b` against the fine
   skip state and take a weighted sum. Lets a node decide *which partition's
   view to trust*, which is the natural extension of Part I.

Start with (1) — it changes one `Linear`'s input width and gives a clean
measurement of whether branching helps at all. Move to (2) once (1) is positive.

**Gate before implementing any of this**: fit a small MLP mapping the K cluster
means to the fine field and see how close it gets to the ceiling. Minutes of
compute, no model training. If the achievable number sits near 0.4081 rather
than 0.1747, Part II is not worth building.

---

# Part III — Why multi-head is not a substitute

The obvious reviewer question is "isn't this just multi-head attention?" It is
not, and the distinction is provable.

**Claim.** For a single partition, every fine node in cluster `c` receives the
*identical* coarse feature vector, regardless of how many heads produced it. Any
decoder must therefore assign them all the same value, and the L2-optimal
constant per cluster is the mean. **The cluster mean is a hard floor that no
number of heads can cross.**

Measured (k = 100, one partition, H statistics per cluster with an optimal
linear readout):

```
--- A. ONE partition, H heads ---          --- B. K partitions, 1 statistic ---
    H= 1  -> 0.4910                            K= 1  -> 0.4831   (    100 groups)
    H= 2  -> 0.4868                            K= 2  -> 0.4170   (    642 groups)
    H= 4  -> 0.4838                            K= 4  -> 0.3277   (  4,171 groups)
    H= 8  -> 0.4833                            K=16  -> 0.1747   ( 76,073 groups)
    H=16  -> 0.4831   <- converges to the
             cluster mean and stops
```

|  | Multi-head | Multi-partition |
| --- | --- | --- |
| What varies | the **weighting** over a fixed set | the **set itself** |
| Support | identical for all heads | different per branch |
| Coarse graph | one topology | K topologies |
| Reconstruction floor | cluster mean | keeps falling with K |

In sampling terms: **heads increase the precision of each measurement;
partitions increase the number of independent measurements.** The partition
fixes the σ-algebra of what the coarse level can resolve; heads live inside it.

**They are complementary, not redundant:**

- Multi-head fixes *what a coarse node knows about its cluster* — the
  mean-vs-peak dilemma in §1. This does not show up in the reconstruction
  metric but determines the quality of coarse-level message passing.
- Multi-partition fixes *how finely the coarse level can localize*.

One honest caveat: `rel_pos` in [`_unpool_merge_level`](../../../methods/MeshGraphNets/model/MeshGraphNets.py)
L300 does distinguish same-cluster nodes geometrically. That is position
information, not field information — the field content remains cluster-constant
for any H.

---

# Implementation

## 8. File-by-file changes

### Part I — attention operators

**`model/blocks.py`**
- New `AttentionPoolBlock(latent_dim, num_heads, build_mlp_fn)`;
  `forward(h_fine, ftc, num_coarse, rel_pos, log_scale, query=None) -> [M, D]`.
  Zero-init the score head.
- `UnpoolBlock`: add `attn_mlp` beside `edge_mlp`, apply
  `torch_geometric.utils.softmax` over `dst_fine`, multiply by `deg`. Gate on a
  constructor flag. Keep the `_split_first_linear` treatment — `E_up ≈ 13N` is
  the largest activation buffer in the V-cycle and the concat must stay
  unmaterialized.

**`model/MeshGraphNets.py`**
- `_build_multiscale_processor` (L125): build `pool_blocks` when
  `pool_type == 'attention'`.
- `_forward_multiscale` (L256-260): replace the `if 'seeds' in ld` / `else
  pool_features` branch. Legacy paths stay under the other `pool_type` values.
- `_extract_level_data` (L306): surface `coarse_anchor_idx_{i}` and the
  per-node scale feature.

**`general_modules/mesh_dataset.py`**
- Compute per-node `log_scale` (sum of incident reference edge lengths) once in
  `_get_static_sample_data` (L683) and cache it beside `edge_index` / `x_pos`.
  Reference-geometry-only, so rotation-invariant and augmentation-safe.

### Part II — multi-partition

**`general_modules/multiscale_helpers.py`**
- `build_multiscale_hierarchy` returns, per level, a **list of K branch dicts**
  instead of one. Each branch uses a different deterministic FPS start
  (`seeds[0] = b * (N // K)`) — determinism matters, see the comment at
  [`_fps_euclidean`](../../../methods/MeshGraphNets/model/coarsening.py) L184.
- `attach_coarse_levels_to_graph` writes `*_{level}_{branch}` attributes.

**`general_modules/multiscale_cache.py`**
- Bump `FORMAT_VERSION` to 2; datasets `L{l}B{b}_ftc`, `L{l}B{b}_c_ei`, ...
- `_coarse_params` must include the per-level `K` so the signature hash changes
  and stale caches are invalidated rather than silently reused.

**`model/coarsening.py`**
- Extend `_LEVEL_RE` to `..._(\d+)_(\d+)$` and make `MultiscaleData.__inc__` /
  `__cat_dim__` offset by `num_coarse_{level}_{branch}`.
  **This is the highest-risk edit in the whole change.** The existing docstring
  (L527-537) already warns that PyG's default `'index'` heuristic silently mixes
  samples together at `batch_size > 1`. A missed branch key fails the same way —
  no exception, just wrong results. The configs currently use `Batch_size 1`,
  which would hide it. Add an explicit `batch_size=2` correctness test.

**`training_profiles/ar_rollout.py`**
- `_refresh_multiscale` (L133) loops over levels; it needs the branch loop too.

**Config / launcher**
- `pool_type`: `mean` (default) | `attention`; `pool_heads` (default 4).
- `voronoi_clusters` gains a companion `voronoi_branches` (per level, default
  all 1 → current behaviour exactly).
- Register all three in `cae_suite/specs/meshgraphnets.py` `known_keys` or
  `--audit-configs` flags them, and add them to `build_model_config` in
  [`training_profiles/setup.py`](../../../methods/MeshGraphNets/training_profiles/setup.py) L158 so inference
  reconstructs the same architecture from the checkpoint.

**Not affected by Part I**: attention changes only *features*, never coarse
*positions*, so `coarse_edge_attr`, the per-level normalization stats, and the
AR-RT position refresh are untouched. Part II *does* affect them — each branch
needs its own coarse edge statistics.

## 9. Numerics

- Compute attention scores in fp32 even under bf16 autocast; softmax over ~40
  (pool) or ~13 (unpool) elements is cheap and bf16 scores lose too much
  resolution. Wrap the score head in `torch.autocast(enabled=False)`.
- Use `torch_geometric.utils.softmax(src, index, num_nodes)` — it does the
  per-segment max subtraction. Do not hand-roll `exp` / `scatter_sum`.
- `ftc` is already batch-offset by `MultiscaleData.__inc__`, so segment-softmax
  over the batched `ftc` is correct with no extra work — *provided* the branch
  keys are wired correctly (§8).
- Assert `n_c == centroid.shape[0]` per branch to keep the compaction invariant
  explicit.

## 10. Validation plan

1. **Equivalence test** — with zero-init score heads and `voronoi_branches` all
   1, `attention` and `mean` produce identical outputs on a fixed batch to
   within 1e-6. Must pass before anything else.
2. **Batching test** — `batch_size=2` gives the same per-sample outputs as two
   `batch_size=1` passes. Guards §8's highest-risk edit.
3. **Gradient test** — loss decreases on a 20-step single-batch overfit.
4. **A/B at matched parameters** — report node-MSE **and** peak-stress error.
5. **Ablations**
   - heads ∈ {1, 4, 8} at K=1
   - K ∈ {1, 4, 16} at H=1
   - the 2-D grid of both — this is the table that answers "isn't it just
     multi-head?", and it will be asked.
   - pool-only vs unpool-only vs both; branch merge (1) concat vs (2) attention.

## 11. Analysis output

- **Attention entropy per cluster vs within-cluster field variance.** Entropy
  falling as variance rises = the operator learned to switch from averaging to
  selecting. Entropy pinned at maximum = uniform mean was already optimal, a
  clean negative result.
- **Branch agreement**: how often do the K branches' `h_up` disagree, and does
  the merge weight correlate with local field variance?
- **Peak-stress retention** before/after, against the 72.6% / 86.8% / 100%
  reference numbers in §1.

## 12. Risks

- **Ceiling vs achievable (Part II's main risk).** 0.1747 assumes a perfect
  readout; naive averaging gives 0.4081. Where a learned combiner lands is
  unknown. `16 × 128 = 2048` latent dims is ample capacity to identify the
  intersection cell, so the ceiling is not obviously unreachable — but run the
  §7 gate first.
- **No gain from Part I.** Plausible if the fine skip connection already carries
  the extrema the coarse branch drops. The coarse branch is known to matter here
  (parameter-matched vanilla is much worse), but this is the main risk. The
  entropy figure diagnoses it either way.
- **Batching correctness.** See §8 — silent, not loud. Test at `batch_size=2`.
- **Cache growth.** Keep K=1 at level 0. Verify the `.mscache` size before and
  after; `ex1` is already 2.5× its dataset.
- **Cost.** The unpool score runs on `E_up` rows and is the expensive half of
  Part I. Measure rather than assume; reuse `_split_first_linear`.
- **`use_checkpointing` interaction.** `_unpool_merge_level` is already wrapped
  in `run_checkpointed`; the new attention and the branch loop must stay inside
  that boundary so activations are recomputed rather than stored.
