# 03 — BSMS-GNN (bi-stride multiscale GNN)

| | |
| --- | --- |
| Route | `model meshgraphnets` with `use_multiscale true` and `coarsening_type bfs` |
| Directory | [methods/MeshGraphNets/](../../methods/MeshGraphNets/) |
| Entrypoint | [MeshGraphNets_main.py](../../methods/MeshGraphNets/MeshGraphNets_main.py) |
| Paper | Cao et al., *Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network*, ICML 2023 |

BSMS-GNN is not a separate model. It is [HI-MGN](02_HI-MGN.md) with the BFS
bi-stride coarsener in place of FPS-Voronoi. The V-cycle, pooling, unpooling,
decoder, training and rollout are all identical. Only the choice of coarse
nodes differs.

## Coarsening

The coarsener is `bfs_bistride_coarsen` in
[coarsening.py](../../methods/MeshGraphNets/model/coarsening.py):

1. A BFS gives every node a hop depth. It runs on `scipy.sparse.csgraph` and
   restarts at every unvisited node, so disconnected multi-part meshes work.
2. Nodes at even depth become coarse nodes.
3. Each node at odd depth joins the cluster of its BFS parent, which is at
   even depth.
4. Two clusters are joined by a coarse edge if any of their members share a
   fine edge.

The coarsener returns the same contract as the Voronoi coarsener
(`fine_to_coarse`, `coarse_edge_index`, `num_coarse`, `seeds`), so the two can
be mixed per level. For example, `coarsening_type bfs, voronoi_seedmean`.

**The reduction is about 2× per level, not 4×.** On mesh graphs, roughly half
the nodes sit at even depth. A small coarsest graph therefore needs many
levels. When the graph collapses to one node, or loses its coarse edges, the
hierarchy is truncated without a warning, as described in
[02](02_HI-MGN.md#coarsening).

## Keys

Every [HI-MGN key](02_HI-MGN.md#modes-and-keys) applies:

| Key | BSMS value |
| --- | --- |
| `use_multiscale` | `true` |
| `coarsening_type` | `bfs` |
| `multiscale_levels` | `L`; each level halves the node count |
| `mp_per_level` | `2L + 1` block counts |
| `voronoi_clusters` | required by the spec (`MGN-MULTI-REQ`) but not read by `bfs`; set it to `0` |

No checked-in config uses `bfs`. To try it, copy a
`config_*_himgn.txt` file, change the keys above, and run
`python AI_CAE4ALL_main.py --config <file> --check`.

## When to use it

- **Advantages:**
  - There is no cluster budget to tune.
  - Coarse edges follow the real mesh adjacency, so thin walls and folded
    surfaces are never linked across a gap.
  - Building the hierarchy is fast.
- **Disadvantages:**
  - The coarse size cannot be controlled directly, because the reduction is
    fixed at about 2× per level.
  - Irregular meshes give uneven clusters.
  - The clusters follow BFS shells rather than physical regions.
  - Like FPS, the BFS starts at node 0, so the hierarchy depends on node
    order.

Outputs, tests and the remaining limitations are the same as for
[HI-MGN](02_HI-MGN.md).
