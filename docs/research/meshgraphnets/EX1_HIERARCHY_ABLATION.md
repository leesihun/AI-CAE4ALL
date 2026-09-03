# ex1 hierarchy-shape ablation — design

How deep should the V-cycle be, where should the message-passing budget sit,
and which coarsener builds it. Companion to
[COARSENING_ABLATION_DESIGN.md](COARSENING_ABLATION_DESIGN.md) (which ablates
the coarsening *operators* on ex2) and
[ATTENTION_TRANSFER_DESIGN.md](ATTENTION_TRANSFER_DESIGN.md) (transfer
operators).

Axes:

1. **`multiscale_levels`** — 1, 2, 3, 4
2. **`mp_per_level`** — total fixed at 28; flat / deepest-large / deepest-shallow
3. **`coarsening_type`** — `bfs`, `voronoi_seedmean`, `voronoi_inherit`
4. **`voronoi_clusters`** — the ladder that makes axis 1 a controlled experiment
5. **total message-passing budget** — 14 / 28 / 56, plus the flat-MGN control

Status: **design only.** Measurements are real, on `dataset/ex1.h5`
(100 samples) with a standalone reimplementation of `model/coarsening.py`.
Representative sample for per-level numbers: sid 24, N = 38,682 (the median).

---

## 1. ex1 is a different problem from ex2, and it favours this study

| | ex1 | ex2 |
| --- | --- | --- |
| samples | 100 | 50 |
| timesteps | **1 (static)** | 50 |
| N | **14,605 – 88,582** (median 38,821, **6.07× spread**) | ~187k – 200k (~1.07×) |
| components | **1** | 2 |
| mean degree | 5.96 | 7.63 |
| **mesh diameter** | **297 median, 377 max** | 102 |
| `use_world_edges` | False | True |
| training | `ar_ot`, `Batch_size 1`, 2000 epochs | `ar_rt`, `Batch_size 4`, 100 epochs |

Two consequences:

**ex1 is static, so the hierarchy carries more of the load.** With T=1 the state
rows are zeroed at input — the model predicts the field from geometry, node
types, positional features and conditions alone. There is no temporal signal to
lean on, so whatever long-range coupling exists must come from message passing.

**Flat MGN structurally cannot span an ex1 mesh.** Receptive field is one hop
per block, against a diameter of 297:

| `message_passing_num` | params | GFLOP-ish/fwd | receptive field |
| --- | --- | --- | --- |
| 15 (shipped) | 2,318,468 | 643 | 15 hops = **5%** of diameter |
| 28 | 4,252,036 | 1,200 | 9% |
| 56 | 8,416,644 | 2,399 | 19% |
| 112 | 16,745,860 | 4,798 | 38% |

You cannot buy global reach with depth here — 112 blocks still sees a third of
the mesh and costs 4,798 GFLOP against HI-MGN's 476 at L=2. That makes ex1 the
right dataset for a levels sweep, and it makes the flat-MGN arms a genuinely
falsifiable control rather than a formality.

---

## 2. `voronoi_clusters` — what to do

### The problem with what's shipped

`configs/MeshGraphNets/ex1/config_train_himgn.txt` carries
`voronoi_clusters 5000, 100`, copied from ex2 where N ≈ 200k. On ex1 that means:

| | N | level-0 reduction at k=5000 | nodes per cluster |
| --- | --- | --- | --- |
| smallest sample | 14,605 | **2.9×** | 2.9 |
| median | 38,821 | 7.8× | 7.8 |
| largest | 88,582 | 17.7× | 17.7 |

At the small end, level 0 is barely a coarsening at all — ~3 nodes per cluster,
so the "coarse" graph is nearly a copy of the fine one and that level of the
V-cycle does almost nothing. At the large end it is a 17.7× jump. **The same
config produces a 6× different hierarchy depending on which sample the loader
hands you**, which is not a property you want in the baseline of a study whose
whole subject is hierarchy shape.

### The fix: fix the endpoints, vary only the descent

Anchor the coarsest level at 100 and interpolate geometrically from the median
N. Then L=1 and L=4 differ **only** in how gradually they get there — same
fine mesh, same coarsest size — which is what makes the levels sweep a
controlled experiment instead of a confounded one.

| L | `voronoi_clusters` | per-level ratio |
| --- | --- | --- |
| 1 | `100` | 388× |
| 2 | `1970, 100` | 19.7× |
| 3 | `5320, 730, 100` | 7.29× |
| 4 | `8750, 1970, 440, 100` | 4.44× |

Measured on the median sample, chained exactly as `build_multiscale_hierarchy`
does (seedmean → next level's `ref_pos` = seed positions):

| L | per-level `n_c` (deg, diam, recon, peak) | total `E_up` |
| --- | --- | --- |
| 1 | 100 (5.20, **15**, 0.3849, 0.439) | 251,434 |
| 2 | 1970 (5.81, 61, 0.1585, 0.677) → 100 (5.14, **15**, 0.3988, 0.458) | 286,336 |
| 3 | 5320 (5.88, 99, 0.1210, 0.698) → 730 (5.68, 35, 0.2219, 0.582) → 100 (5.16, **13**, 0.4302, 0.420) | 323,461 |
| 4 | 8750 (5.91, 121, 0.1058, 0.820) → 1970 (5.82, 53, 0.1745, 0.694) → 440 (5.62, 25, 0.2680, 0.495) → 100 (5.22, **12**, 0.4119, 0.464) | 385,292 |

The ladder does its job: coarsest diameter stays 12–15 and coarsest recon stays
0.38–0.43 across all four. Total unpool edges grow only 53% from L=1 to L=4.
So the sweep really is asking one question — *does a gradual descent beat an
abrupt one?* — and nothing else.

### Still open: absolute counts vs. a ratio

Even with this ladder, `voronoi_clusters` is an absolute count against a 6×
N spread, so the per-sample reduction ratio still varies 6×. The principled fix
is a `voronoi_ratio` key (clusters as a fraction of N, resolved per sample).
That is a small new key — worth **one arm** at L=2 (`ratio` vs `absolute`,
matched at the median) to find out whether the variance is costing anything
before building it out. See §6.

### And a separate coarsest-`k` sweep

Orthogonal to L: hold `multiscale_levels 2` and sweep the coarsest at
`50 / 100 / 400 / 1600`. ex1's meshes are 5× smaller than ex2's, so ex2's
k-sweep values do not transfer. Run this **after** the L sweep — L is the
bigger effect and its outcome tells you which region of k to sample.

---

## 3. `multiscale_levels` × `mp_per_level` — the grid, and its confound

Symmetric V-cycles (`pre[i] == post[i]`), `2L+1` entries, every row summing to
**28**:

| L | shape | `mp_per_level` | params | GFLOP-ish | vs L1-flat |
| --- | --- | --- | --- | --- | --- |
| 1 | flat | `9, 10, 9` | 4,452,100 | 791 | 1.00× |
| 1 | deep-large | `4, 20, 4` | 4,452,100 | 363 | 0.46× |
| 1 | deep-shallow | `13, 2, 13` | 4,452,100 | **1,133** | **1.43×** |
| 2 | flat | `5, 6, 6, 6, 5` | 4,652,164 | 476 | 0.60× |
| 2 | deep-large | `3, 4, 14, 4, 3` | 4,652,164 | 296 | 0.37× |
| 2 | deep-shallow | `8, 5, 2, 5, 8` | 4,652,164 | 728 | 0.92× |
| 3 | flat | `4, 4, 4, 4, 4, 4, 4` | 4,852,228 | 419 | 0.53× |
| 3 | deep-large | `2, 3, 3, 12, 3, 3, 2` | 4,852,228 | **235** | **0.30×** |
| 3 | deep-shallow | `6, 4, 3, 2, 3, 4, 6` | 4,852,228 | 589 | 0.74× |
| 4 | flat | `3, 3, 3, 3, 4, 3, 3, 3, 3` | 5,052,292 | 357 | 0.45× |
| 4 | deep-large | `2, 2, 2, 3, 10, 3, 2, 2, 2` | 5,052,292 | 248 | 0.31× |
| 4 | deep-shallow | `6, 4, 2, 1, 2, 1, 2, 4, 6` | 5,052,292 | 627 | 0.79× |

**Fixing the block count fixes neither compute nor parameters.**

- **Compute spans 3.8×** across the grid (235 → 1,133). A block at level 0
  costs ~450× a block at the 100-node coarsest, so every block you move
  downward is nearly free. "deep-large" configurations are cheap by
  construction; "deep-shallow" are expensive.
- **Parameters grow 13.5%** with L (4.45M → 5.05M), because each level adds
  `UnpoolBlock` (132,736) + `coarse_eb_encoder` (34,432) + `skip_proj`
  (32,896) = **200,064 params/level**. The 28 GnBlocks (148,736 each) are the
  constant part.

This does not invalidate the grid — "same depth budget, different placement" is
a legitimate question, and it is the one the user asked. But **report all three
columns next to every result.** "L=3 deep-large wins" is a much stronger claim
once you can add "at 0.30× the compute of L=1 flat"; conversely "L=1
deep-shallow wins" is a much weaker one at 1.43×. If a single winner emerges,
add one compute-matched rerun (scale `Latent_dim` or the block count until
GFLOP matches) before publishing it.

Note `L=1 deep-shallow` (`13, 2, 13`) puts 2 blocks on a 100-node coarsest
graph — nearly a no-op coarse level. That is the intended extreme, not a
mistake; it is close to a flat 26-block model with a vestigial coarse branch,
which makes it a useful bridge to the flat-MGN control.

---

## 4. `coarsening_type` — two different questions, not three variants of one

The three requested values do not sit on one axis:

| pair | what differs |
| --- | --- |
| `bfs` vs `voronoi_*` | **the hierarchy topology** — different clusters, different coarse graphs, different sizes |
| `voronoi_seedmean` vs `voronoi_inherit` | **nothing about the hierarchy** — identical `fps_voronoi_coarsen` call, identical `ftc`/`c_ei`/`seeds`, identical seed-anchored coarse positions. The *only* difference is that `inherit` writes `coarse_seed_idx_{i}`, which switches the model from mean-pool to gather-at-seed |

So `seedmean` vs `inherit` is a **restriction-operator** ablation, and it has a
natural third arm that costs nothing extra: `pool_type attention`, which
reduces to mean-pool exactly at init. Run it as a 3-way:

| arm | operator | measured on ex1 sid 24 |
| --- | --- | --- |
| `voronoi_seedmean` | scatter-mean over the cluster | recon 0.3849 / 0.1936 / 0.1226 at k = 100 / 1000 / 5000 |
| `voronoi_inherit` | gather at the FPS seed | recon 0.4804 / 0.2247 / 0.1465 — **+16 to +25% RMSE** |
| `pool_type attention` | learned per-member weights | — |

`inherit` is uniformly worse on RMSE and uniformly **better on peak retention
(+8 to +14%)** — the mean-vs-extrema trade from
ATTENTION_TRANSFER_DESIGN.md §1, and *larger* on ex1 than on ex2 (+10.8%).
Since `plot_feature_idx -1` makes stress the headline output, report both
metrics or this arm reads as a flat loss when it isn't.

> Cache note: `seedmean` and `inherit` produce byte-identical topology but hash
> to different `.mscache.*.h5` files, because `_coarse_params` puts the raw
> type string in the signature. Two identical multi-GB caches get built. Worth
> normalizing the signature (map both to `voronoi`) if this study runs at
> scale — the pool mode is a model-side property, not a topology one.

### The `bfs` arm needs `bfs_steps`

Measured bi-stride cascade on ex1 sid 24 — **2.00× per step**, not the
documented ~4× (see [COARSENING_ABLATION_DESIGN.md §3.1](COARSENING_ABLATION_DESIGN.md)):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `n_c` | 19,357 | 9,646 | 4,807 | 2,395 | 1,191 | 600 | 323 | **148** | **1** |
| diam | 158 | 88 | 52 | 25 | 13 | 7 | 4 | 2 | 0 |

Unlike ex2, coarse degree stays ~5.9 the whole way (ex2's grew to 11.5), and
**step 8 reaches 148 — close enough to the k=100 anchor that a budget-matched
BSMS arm is actually feasible on ex1.** Two usable arms:

| arm | config | coarsest |
| --- | --- | --- |
| `ct-bfs-doc` | `coarsening_type bfs`, L=2, no `bfs_steps` | 9,646 — *not* comparable, this is what the shipped BSMS sketch does |
| `ct-bfs-matched` | `coarsening_type bfs`, L=4, `bfs_steps 2, 2, 2, 2` | ~151 — comparable to the k=100 anchor |

Step 9 collapses to `n_c = 1`, and a saturated hierarchy runs silently with
dead `pre_blocks`/`post_blocks` (COARSENING_ABLATION_DESIGN.md §3.1). Do not
configure past step 8 without the warning being added first.

---

## 5. Total message-passing budget — the control that makes everything else mean something

At L=2 flat, sweeping the total:

| total | `mp_per_level` | params | GFLOP-ish |
| --- | --- | --- | --- |
| 14 | `3, 3, 2, 3, 3` | 2,569,860 | 291 |
| 28 | `5, 6, 6, 6, 5` | 4,652,164 | 476 |
| 56 | `11, 11, 12, 11, 11` | 8,816,772 | 1,012 |

Plus the flat-MGN arms from §1 (`use_multiscale False`, `message_passing_num`
15 / 28 / 56). Together these answer the question every reviewer asks first:
**is the hierarchy buying reach, or just capacity?**

The prediction worth writing down before running: HI-MGN at L=2/28 blocks
(476 GFLOP, 4.65M params) should beat flat MGN at 56 blocks (2,399 GFLOP,
8.42M params), because 56 hops is still only 19% of the mesh diameter. If it
does not, the levels sweep is measuring capacity and the whole study needs
reframing. This is cheap to run and it gates everything else.

---

## 6. Other axes worth adding

Ranked by information per GPU-hour.

**1. Size-generalization split (not an ablation — an evaluation protocol).**
ex1's N spans 6.07×. Train on the small half, test on the large half. This is
free (a different split, no code) and it is the sharpest possible test of
whether a hierarchy transfers across mesh resolution — the property that
actually matters for deployment, and the one an absolute `voronoi_clusters`
count is most likely to break (§2). Arguably higher value than half the arms
above.

**2. `positional_features` 0 / 4 / 8.** ex1 is static with zeroed state rows,
so the model's input is essentially geometry + node types + RWPE. Positional
features may be doing work the hierarchy is being credited with. Three cheap
arms, and a null result is worth knowing.

**3. `voronoi_ratio` vs absolute counts.** §2 — one arm at L=2 to size the
problem before building the key.

**4. Coarsest-`k` sweep at L=2** — `50 / 100 / 400 / 1600`. Run after the L
sweep.

**5. `Latent_dim` 64 / 128 / 256** at fixed blocks. Orthogonal capacity axis;
lower priority than the total-MP sweep, which covers similar ground more
directly.

**6. `std_noise 0.01` → `0.0`.** Input-noise injection exists for
autoregressive stability. ex1 has T=1 and no rollout, so it is plausibly inert
or mildly harmful here. One arm, and it is the kind of thing that quietly costs
accuracy in every other ex1 run if it turns out to matter.

**7. The operator axes already designed** — `pool_type`/`unpool_type attention`
(ATTENTION_TRANSFER_DESIGN.md Part I), `voronoi_branches` (Part II), and the
prolongation arms (`copy`/`idw`, COARSENING_ABLATION_DESIGN.md §4.2). None have
run on ex1.

**8. `augment_geometry` True/False.** 100 samples is small and augmentation may
be carrying the run — but this is a data question, not a hierarchy one. Park it.

---

## 7. Suggested wave schedule (8 GPUs, one arm each)

| wave | arms | answers |
| --- | --- | --- |
| **1** | L=1/2/3/4 at flat + flat-MGN at 15/28/56 + baseline | is the hierarchy real, and how deep |
| **2** | shape sweep (deep-large / deep-shallow) at the two best L + total-MP 14/56 + size-generalization split | where the budget goes, and capacity vs reach |
| **3** | `inherit` + `pool_type attention` + `bfs-doc` + `bfs-matched` + coarsest-`k` sweep | coarsener and restriction operator |

Wave 1 is 8 arms exactly and needs **no new code** — only new configs. Waves 2
and 3 need `bfs_steps` (§4) and, optionally, `voronoi_ratio` (§2).

Reuse the deleted-but-recoverable tooling rather than rewriting it:

```bash
git show 45c9482^:configs/MeshGraphNets/gen_ablation_arms.py > configs/MeshGraphNets/gen_ablation_arms.py
git show 3ae5930^:run_ablation.sh     > run_ablation.sh
git show 3ae5930^:compare_ablation.py > compare_ablation.py
```

Report per arm: node MSE, **peak stress error** (§4 — the operator arms trade
these against each other), params, GFLOP, wall-clock per epoch, and the §2
geometric predictors (coarsest recon + diameter) so it becomes measurable
whether the cheap probe predicts the trained result.
