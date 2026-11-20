# 02 — HI-MGN (hierarchical MeshGraphNets)

| | |
| --- | --- |
| Route | `model meshgraphnets` with `use_multiscale true` and a `voronoi_*` `coarsening_type` |
| Directory | [methods/MeshGraphNets/](../../methods/MeshGraphNets/) |
| Entrypoint | [MeshGraphNets_main.py](../../methods/MeshGraphNets/MeshGraphNets_main.py) |
| Spec | [ai_cae4all/specs/meshgraphnets.py](../../ai_cae4all/specs/meshgraphnets.py) |

HI-MGN swaps the processor only. The encoder, decoder, `GnBlock`, features,
training and rollout are the same as in [flat MGN](01_MeshGraphNets_MGN.md).

The flat processor is replaced by a U-Net-style **V-cycle** over a hierarchy
of FPS-Voronoi coarse graphs. Long-range coupling therefore crosses the mesh
in a few blocks instead of one block per hop. Use it on large meshes with
global coupling, such as whole-body deformation or warpage. With
`coarsening_type bfs`, the same machinery becomes [BSMS-GNN](03_BSMS-GNN.md).

Prerequisites: [00](00_shared_foundations.md) and [01](01_MeshGraphNets_MGN.md).

## Architecture

The code is in
[MeshGraphNets.py](../../methods/MeshGraphNets/model/MeshGraphNets.py)
(`_build_multiscale_processor`, `_forward_multiscale`). With
`L = multiscale_levels`, `mp_per_level` has `2L + 1` entries:

```text
[ pre_0 … pre_(L-1),  coarsest,  post_(L-1) … post_0 ]
```

For example, `4, 6, 8, 6, 4` with `L = 2` gives 28 blocks.
`message_passing_num` is not used.

1. **Down arm.** At each level `i`, `pre_blocks[i]` runs and the node, edge
   and world states are saved as a skip connection. The graph is then pooled
   to level `i+1`, and the coarse edges are re-encoded by
   `coarse_eb_encoders[i]`. Pooling (`pool_type`) works like this:
   - `mean` (default): scatter-mean over the cluster. `voronoi_inherit`
     gathers the seed node instead.
   - `attention`: `AttentionPoolBlock`
     ([blocks.py](../../methods/MeshGraphNets/model/blocks.py)) with
     `pool_heads` heads (default 4).
2. **Coarsest blocks.**
3. **Up arm.** At each level `i`, the coarse state is unpooled and merged with
   the skip by `skip_projs[i]`, then `post_blocks[i]` runs.
   - With `learned_interpolation true` (default), unpooling is the
     `UnpoolBlock`, a bipartite coarse-to-fine message pass over
     `unpool_edge_index_i`:

     ```text
     msg  = EdgeMLP([h_coarse, h_fine_skip, fine_pos − coarse_centroid])   # 2D+3 → D
     h_up = NodeMLP([h_fine_skip, Σ msg])                                  # 2D   → D
     ```

     `unpool_type` sets how messages are aggregated: `sum` (default) or
     `attention`.
   - With `learned_interpolation false`, the coarse state is broadcast over
     its cluster.
4. **Decode** the finest level.

**Multi-partition.** `voronoi_branches` can be greater than 1 on the last
level only. That level then builds K independent Voronoi partitions with
shared weights. The skip projection becomes `Linear((1+K)·D → D)`.

## Coarsening

The coarsening code is in
[coarsening.py](../../methods/MeshGraphNets/model/coarsening.py)
(`fps_voronoi_coarsen`):

1. Farthest-point sampling picks `voronoi_clusters[i]` seeds. It uses
   Euclidean distance when positions exist and falls back to graph distance.
   It starts at node 0, so it is deterministic.
2. Every connected component gets at least one seed.
3. A multi-source BFS from the seeds assigns each node to its nearest seed,
   which gives a Voronoi partition in hops.
4. Two clusters are joined by a coarse edge if any of their members share a
   fine edge.

The three `voronoi_*` modes share this partition. They differ downstream:

| Mode | Coarse position | Pooling |
| --- | --- | --- |
| `voronoi_centroid` | cluster centroid | mean |
| `voronoi_seedmean` | seed node | mean |
| `voronoi_inherit` | seed node | seed gather |

Changing between seed-anchored and centroid modes changes the statistics of
the coarse edges, so the checkpoints cannot be swapped. `coarsening_type`
may also be a comma list with one mode per level.

Each level adds these attributes to the batch:

- `fine_to_coarse_i`, `coarse_edge_index_i`, `coarse_edge_attr_i`,
  `coarse_centroid_i` and `unpool_edge_index_i`;
- `coarse_seed_idx_i` (`voronoi_inherit` only);
- `coarse_world_edge_*_i` (with `coarse_world_edges`).

`MultiscaleData` offsets these per-level indices when samples are batched.

**Saturation.** The hierarchy stops early, without a warning, when a level
reaches one node or has no coarse edges. The model then runs only
`len(level_data)` levels. Blocks configured for levels past that point stay
unused.

**Cache.** Hierarchies are built once per run into a shared
`*.mscache.<digest>.h5` file
([multiscale_cache.py](../../methods/MeshGraphNets/general_modules/multiscale_cache.py)).
The build is locked, so concurrent jobs wait for it, up to
`hierarchy_cache_wait_timeout`. Workers then stream from the file.

The digest includes the source HDF5's size and mtime. Training rewrites that
file with its normalizers, so a cache cannot be reused across runs. It is
therefore deleted after training unless `hierarchy_cache_keep true` is set,
and the next start prunes any leftovers.

## Modes and keys

Every key from [01](01_MeshGraphNets_MGN.md#modes-and-keys) applies, plus
these:

| Key | Meaning |
| --- | --- |
| `use_multiscale` | `true` |
| `multiscale_levels` | `L` (> 0) |
| `mp_per_level` | `2L + 1` block counts |
| `coarsening_type` | `voronoi_seedmean`, `voronoi_centroid`, `voronoi_inherit`, `bfs`, or one mode per level |
| `voronoi_clusters` | one count, or one per level |
| `pool_type`, `pool_heads` | `mean` / `attention`; heads for attention pooling |
| `unpool_type`, `learned_interpolation` | `sum` / `attention`; learned unpool or broadcast |
| `voronoi_branches` | partitions per level; > 1 on the last level only |
| `coarse_world_edges` | world edges on coarse levels (needs `use_world_edges`) |
| `hierarchy_cache_dir`, `hierarchy_cache_build_workers`, `hierarchy_cache_wait_timeout`, `hierarchy_cache_keep` | cache control |
| `coarse_cache_per_worker` | in-memory hierarchy cache per loader worker |

Spec checks:

- `MGN-MULTI-REQ`: `coarsening_type`, `multiscale_levels`, `voronoi_clusters`
  and `mp_per_level` are all required.
- `MGN-MULTI-LEVEL`: `multiscale_levels` must be greater than 0.
- `MGN-MULTI-001`: the length of `mp_per_level`.
- `MGN-MULTI-002`: the length of `voronoi_clusters`.
- `MGN-COARSEN-001`: a bare `voronoi` is rejected.
- `MGN-COARSEN-002`: unknown mode.
- `MGN-UNPOOL-001`: `unpool_type attention` needs `learned_interpolation true`.
- `MGN-UNPOOL-002`: `learned_interpolation false` is not supported with
  `model_split`.

The shipped configs are
`configs/MeshGraphNets/deterministic/<slot>/baseline/config_{train,infer}_himgn.txt`.
The ex1 reference is
[config_train_himgn.txt](../../configs/MeshGraphNets/deterministic/ex1/baseline/config_train_himgn.txt):

```text
use_multiscale         True
coarsening_type        voronoi_seedmean
multiscale_levels      2
voronoi_clusters       5000, 100
mp_per_level           4, 6, 8, 6, 4
pool_type              mean
unpool_type            sum
learned_interpolation  True
voronoi_branches       1, 1
```

## Outputs, tests, validation

Outputs are the same as [01](01_MeshGraphNets_MGN.md#outputs). The hierarchy
cache file sits in `hierarchy_cache_dir` while training runs.

The tests that cover the multiscale path are in
[methods/MeshGraphNets/tests/](../../methods/MeshGraphNets/tests/):

- `test_multiscale_stats`
- `test_hierarchy_cache_lifecycle`
- `test_attention_transfer`
- `test_learned_interpolation`
- `test_multi_partition`

## Known limitations

- **Expensive coarsening.** FPS plus BFS is costly per sample, so the cache
  build time and its disk space are real costs.
- **Node-order dependence.** FPS starts at node 0, so relabelling the nodes
  changes the hierarchy and, with it, the prediction.
- **More hyperparameters.** The level count, block counts, cluster counts
  and mode all need tuning.
- **Silent saturation.** A hierarchy that saturates is truncated without a
  warning.
- **Blurring.** Coarse clusters with mean pooling can blur sharp features.
