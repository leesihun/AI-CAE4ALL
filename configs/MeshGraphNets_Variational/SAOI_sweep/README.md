# MeshGraphNets-V SAOI FM-v2 sweep

Run the complete campaign from the repository root:

```bash
bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh
```

The command generates every config, checks the training and evaluation data,
trains all models from scratch, performs stochastic and deterministic
inference, creates the same-condition GT/posterior/prior diagnostics, ranks the
arms and writes `docs/research/SAOI_FM_V2_SWEEP_MGNV.md`.

Outputs are isolated from the previous campaign under:

```text
output/meshgraphnets-v/saoi_fm_v2_sweep/
```

## Controlled comparison

Each board half uses four arms. The graph conditioner remains 256-wide with
five message-passing layers in every arm.

| Variant | Velocity | Conditional base | Question answered |
|---|---|---|---|
| P0 | legacy MLP, width 256 | global standardization | current FM baseline |
| P1 | legacy MLP, width 512 | global standardization | does velocity capacity help? |
| P2 | four residual FiLM blocks, width 512 | global standardization | does time/condition modulation help? |
| P3 | same as P2 | `mu(c)`, diagonal `L(c)` plus residual FM | does explicit conditional location/scale help? |

Arms 1–4 are `bot` P0–P3 and arms 5–8 are `top` P0–P3. All arms use the
same `training_seed=20260916`, split, batch size, loss weights, graph
conditioner, 1000-epoch schedule and prior-only tail from epoch 700. The FM
gradient to the posterior encoder is disabled in every arm.

During P3 training, the moment head minimizes the analytic Gaussian
cross-entropy against posterior `mu/logvar`. The residual FM receives

```text
r* = (z* - mu(c)) / L(c)
```

with `mu(c), L(c)` detached from the FM loss. This keeps conditional moments
and nonlinear residual transport from redefining each other's scale.
After the simulator freezes at epoch 700, P3 uses epochs 700–749 to calibrate
the graph conditioner and moment head against the fixed posterior. It then
freezes both and uses epochs 750–999 for residual-velocity fitting on another
fresh cosine schedule. Checkpoint selection restarts at each boundary; the
joint and moment-stage bests remain available as `.joint.pth` and
`.moment.pth` controls.

## Boundary conditions and comparison contract

For one evaluation case, the fixed condition is

```text
c = graph geometry + mesh topology + thickness/recorded condition rows
    + configured node/B.C. types
```

The `_compare_` file supplies multiple physical realizations under that same
condition. Before any comparison, `posterior_vs_prior.py` verifies coordinates,
topology, condition rows and configured B.C. types against the `_infer_` graph.
A mismatch stops the diagnostic.

The report then compares:

1. GT realizations;
2. posterior-mean decode;
3. posterior-sample decode;
4. graph-conditioned FM-prior decode.

PNG is embedded in the report. Matching PDF and SVG files are written for the
paper, with the latent slot/PCA numbers in JSON:

```text
output/meshgraphnets-v/saoi_fm_v2_sweep/diag/<arm>/<eval-set>/
```

## Operational controls

The normal command runs everything. Selected stages can be repeated without
editing configs:

```bash
TRAIN=0 INFER=0 PVP=1 SCORE=1 PVP_FORCE=1 \
bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh
```

Useful variables are `ARMS`, `INFER_TAGS`, `PVP_N`, `PVP_CHUNK`, `STAGGER`,
and `PYTHON`. Config regeneration and both preflights default to on. A
preflight failure aborts before GPU work so a partial grid cannot be mistaken
for a controlled sweep.

Training arms run in parallel, one per GPU. After training, the script accepts
only checkpoints newer than each arm's launch marker, preventing an old
checkpoint from being inferred after a failed fresh run. Inference and the
posterior/prior diagnostic also run in parallel by arm.
