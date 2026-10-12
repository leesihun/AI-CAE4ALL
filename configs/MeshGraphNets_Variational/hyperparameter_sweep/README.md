# MeshGraphNets-V parametric study (B300 round 2)

Round 2 runs on two 8-GPU B300 nodes, one study per node. Each node runs
this method's runner and the cHI-MGNflow runner side by side, so every card
holds one MGN-V job and one cHI job:

| node | study | data | MGN-V jobs | cHI-MGNflow jobs |
| --- | --- | --- | --- | --- |
| 1 | [`SAOI/`](SAOI/) | `dataset/SAOI/saoi_train_{bot,top}.h5` | 4 arms x 2 board halves | 4 arms x 2 board halves |
| 2 | [`probabilistic/`](probabilistic/) | `dataset/probabilistic/ex3_shell_buckling{,_infer}.h5` | 8 arms | 8 arms |

The cHI-MGNflow half of each study lives in
[`configs/HI_MGNFlow/hyperparameter_sweep/`](../../HI_MGNFlow/hyperparameter_sweep/),
with a byte-identical copy of every script here. Its README also has the
full run instructions, the overrides and how to read the type-back block;
they apply to this runner unchanged, with `mgnv` in place of `chi`.

The results come back as **typed text only**. Each runner ends with a block
of short lines between `==== TYPE THIS BACK ====` and `==== END ====` that
holds its whole result.

| File | Purpose |
| --- | --- |
| `run_sweep.sh SAOI\|probabilistic` | one method on one node, end to end: checks, data, preflight, the 8 jobs, the summary |
| `brief.py SAOI\|probabilistic` | prints the type-back block at any time, during or after the run |
| `SAOI/saoi_heldout.py` | SAOI only: geometry labels, the held-out parts and their inference configs |
| `SAOI/*.txt`, `probabilistic/*.txt` | the configs; every header names the arm's change and its question |

## SAOI: does the model carry a part's scatter to a new geometry?

cHI-MGNflow's round 1 found that its best arm fits parts it trained on and
misses unseen ones, even those inside the training range. Every arm then
split **samples** 80/10/10, so the checkpoint was selected on more
realizations of the training parts. Round 2 asks why the unseen parts miss,
and asks it of both methods under identical splits:

| arm (MGN-V / cHI) | change | question |
| --- | --- | --- |
| `v_smp` / `c_smp` | none: [`../SAOI_run/`](../SAOI_run/) / cHI round-1 `tok`, sample split | the reference |
| `v_grp` / `c_grp` | `split_group_attr geometry_id` on the labelled `_bygeom.h5` copy | does selecting the checkpoint on **unseen geometries** help on unseen parts? |
| `v_grp_rep` / `c_grp_rep` | `v_grp` + `training_seed 20261012` / `c_grp` repeated | the noise floor |
| `v_nopart` / `c_nopart` | `*_grp` + `cond_var 2` (the Part No. row is no longer read) | is the arbitrary part ID, fed in as a scalar, what lets the model memorize parts? |

A geometry is the node coordinates plus the mesh. Variants that differ only in
thickness or part number stay on one side of the split.

Each job is scored on the three official eval parts (**O**, unseen by every
arm) and on up to 6 held-out test parts per half (**H**, cut by
`saoi_heldout.py`). H is SEEN by `v_smp`, which trained on about 80% of
their realizations, and UNSEEN by `v_grp` and `v_nopart`. The parts are the
same for every arm and both methods, so `H(smp) - H(grp)` is the
generalization gap at fixed parts.

| result | meaning |
| --- | --- |
| `v_grp` beats `v_smp` on O by more than the `v_grp` vs `v_grp_rep` gap | checkpoint selection on seen parts was part of the miss |
| H of `v_grp` close to H of `v_smp` | a new in-range geometry costs nothing; the official parts are hard for another reason |
| H of `v_grp` far worse than H of `v_smp` | the model does not transfer scatter to a new geometry |
| `v_nopart` beats `v_grp` | the part-ID row is a memorization channel; drop it from the recipe |
| MGN-V vs cHI on the same arm | which method transfers better |

Every arm differs from `SAOI_run` only by its change and the shared layout:
`gpu_ids 0`, outputs under `output/b300_r2/saoi/<arm>/<half>/`,
`write_preprocessing False`, and a private hierarchy cache. It runs 1000
epochs with the encoder frozen at 700. Inference draws 2000 samples per
official part and 500 per held-out part.

## probabilistic (ex3 buckling): what limits the latent?

Each arm is the ex3 baseline
([`../probabilistic/ex3/baseline/config_train_himgn_v.txt`](../probabilistic/ex3/baseline/))
plus one change. The deeper posterior encoder comes first: on a 20-geometry
subset it was what let `z` rise above the z-free floor.

| arm | change | question |
| --- | --- | --- |
| `v_base` | none, at batch 8 x 1 instead of 2 x 4 (MMD sees 8 graphs) | the reference |
| `v_rep` | `training_seed 43` | the seed noise floor |
| `v_mp8` | `vae_mp_layers 8` | does the subset result hold on all 150 geometries? |
| `v_mp12` | `vae_mp_layers 12` | does depth keep paying past 8? |
| `v_mp8_pmin01` | `v_mp8` + `posterior_min_std 0.01` | does a lower posterior floor let `z` carry more of the draw? |
| `v_mp8_aux100` | `v_mp8` + `beta_aux 100` | does a stronger pull on the decoded peak-to-valley undo the shrunken amplitude (0.6-0.8x of FEA)? |
| `v_mp8_mmd10` | `v_mp8` + `lambda_mmd 10` | does a stronger aggregate-posterior MMD reduce posterior drift? |
| `v_mp8_w256` | `v_mp8` + `latent_dim 256` | is the decoder too narrow to turn `z` into amplitude? |

Inference draws 64 samples per scene of `ex3_shell_buckling_infer.h5`. The
MGN-V jobs go lightest first and the cHI jobs heaviest first, so the two
biggest jobs never share a card.

## Run

From the repo root, both runners of a node together:

```bash
# node 1 (SAOI)
nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh SAOI > output/b300_r2_saoi_chi.out 2>&1 &
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh SAOI > output/b300_r2_saoi_mgnv.out 2>&1 &

# node 2 (shell buckling)
nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh probabilistic > output/b300_r2_ex3_chi.out 2>&1 &
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh probabilistic > output/b300_r2_ex3_mgnv.out 2>&1 &
```

The MGN-V runner needs a MeshGraphNets-V interpreter with a Blackwell torch
(sm_100, CUDA 12.8 or newer), set in `ai_cae4all.local.toml` or through
`METHOD_PYTHON=`, and a checkout with `split_group_attr`. Prerequisites,
abort stages and overrides are in the
[cHI-MGNflow README](../../HI_MGNFlow/hyperparameter_sweep/README.md#run).
To run one job again:
`JOBS=v_grp.bot bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh SAOI`.

The block at the end of this runner reads `R2 <saoi|ex3> mgnv ...`, and
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/brief.py SAOI`
(or `probabilistic`) prints it at any time. The columns are explained in
[Reading the block](../../HI_MGNFlow/hyperparameter_sweep/README.md#reading-the-block).

Outputs go to `output/b300_r2/<saoi|ex3>/<arm>/[<half>/]`, with this
runner's logs and status under `output/b300_r2/<saoi|ex3>/run_logs/mgnv/`
and its held-out inference configs under
`output/b300_r2/saoi/heldout/configs/mgnv/`.

## Earlier sweeps

The latent-capacity sweep that used to live in this directory is retired.
That was the 2026-09-28 redesign: arms `base`, `seed`, `zdim4`, `pmin30`,
`arecon`, `aux100`, `fmmom` and `g2e`, plus the benched `zdim8`, `pmin15` and
`mmd10`. It is in git history:

```bash
git show d4c63ba:configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md
```

The production recipe these arms start from is in [`../SAOI_run/`](../SAOI_run/).
