# cHI-MGNflow SAOI hyperparameter sweep

`SAOI_run/` next door trains the fixed recipe: one model per board half, both
stages in one process. This directory takes the same data apart into **three
phases on eight cards** and asks which settings actually move the number the
campaign cares about — the width of the warpage distribution.

The previous campaign's verdict was that cHI-MGNflow's draws come out about
**2× too narrow**. That is what this sweep exists to fix, and every choice
below is aimed at it.

## The three phases

| Phase | Mode | What it trains | Arms | Cards |
|---|---|---|---|---|
| **A** | `train_ae` | **the VAE** — the KL-regularized multiscale compressor (`HierarchyEncoder` + `MultiscaleDecoder`) that puts the field on the coarsest mesh level | 4 per half | 8 |
| **B** | `train_prior` | **the flow model** — the flow-matching velocity prior over that coarse latent, against a **frozen** stage 1 | 4 per half | 8 |
| **C** | `inference` | nothing; draws 2000 samples per part and writes the histogram | 8 × 3 parts | 8 |

Phase A is the VAE part and Phase B is the flow part. They are separated
because stage 2 never sees the fine mesh except through stage 1's latent, and
stage 1 does not move while stage 2 trains — so training them together would
spend eight GPUs re-learning the same compressor four times per half.

## Running it

```bash
bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh
```

That one command is the whole campaign, unattended: Phase A on both halves,
then for each half an automatic pick of which compressor to promote, then
Phase B and Phase C for whichever halves came out of that pick usable, then
the two reports (`ae_report.py`, `rank_arms.py`) — the ranked table also lands
in `output/chi-mgnflow/saoi_sweep/RESULTS.txt`. No step in between needs a
human. `bash run_sweep.sh all` is the same thing spelled out.

**Picking the compressor to promote is not "take the lowest reconstruction
loss".** Phase B fans four prior arms out from *one* frozen compressor per
half, so something has to choose it — and the four compressors differ in
`latent_ch`, so the best reconstruction loss belongs to the largest one almost
by definition; taking it blindly buys capacity the prior then has to model for
no reason. `select_ae.py` runs this decision automatically between Phase A and
Phase B:

1. drop any arm whose `ae_ceiling_check` ratio is FATAL — it cannot represent
   the spread being measured, and no prior setting downstream fixes that;
2. inside the surviving `c4 → c8 → c16` ladder, take the smallest `latent_ch`
   within 10% of the ladder's best validation recon (the knee, not the min);
3. prefer `c8kl` over `c8` at that same `latent_ch` if it beats it by more
   than 10% — same cost, better-regularized latent;
4. if every arm is FATAL, still promote the least-bad one so the pipeline
   finishes and reports a number, but flag that half `NEEDS_ATTENTION` and
   skip Phase B/C for it rather than spend eight more GPU-hours training a
   prior against a proven-broken compressor.

`select_ae.py`'s own docstring has the full reasoning. To override the
decision for one half — e.g. you inspected a `NEEDS_ATTENTION` half by hand
and want a specific arm anyway:

```bash
PROMOTE_BOT=c8 PROMOTE_TOP=c16 bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh
```

### Running one phase by hand instead

Useful for debugging a single arm, or re-running just Phase C after changing
an inference-only setting.

```bash
bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh A

python configs/HI_MGNFlow/hyperparameter_sweep/ae_report.py
python configs/HI_MGNFlow/hyperparameter_sweep/ae_ceiling_check.py --half bot --arm c8
python configs/HI_MGNFlow/hyperparameter_sweep/ae_ceiling_check.py --half top --arm c8

bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh promote   # auto-selects too
bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh B
bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh C

python configs/HI_MGNFlow/hyperparameter_sweep/rank_arms.py
```

`ae_report.py` and `ae_ceiling_check.py` are still worth reading by hand even
in the automatic flow — they explain *why* `select_ae.py` picked what it
picked, and are the tool for inspecting a `NEEDS_ATTENTION` half.

## Phase A — the compressor (`mode train_ae`)

| Arm | `latent_ch` | `ae_kl_weight` | GPU bot / top | Question it answers |
|---|---|---|---|---|
| `c4` | 4 | 1e-6 | 0 / 4 | is the current width enough? |
| `c8` | 8 | 1e-6 | 1 / 5 | the SAOI_run setting, as the reference |
| `c16` | 16 | 1e-6 | 2 / 6 | does the ceiling keep rising? |
| `c8kl` | 8 | **1e-4** | 3 / 7 | is the latent under- or over-regularized? |

`c4 → c8 → c16` is a capacity ladder, not a grid: it finds the knee. `c8kl`
sits beside `c8` and changes one thing, so their difference is attributable.

Everything else is held at the SAOI_run recipe: `latent_dim 128`,
`voronoi_clusters 1000, 100`, `multiscale_levels 2`, `mp_per_level 4, 6, 8, 6, 4`,
`batch_size 16`, `learningr 0.0001`, `hierarchy_variants 4`.

### `ae_ceiling_check.py` — the gate that can end the campaign early

Stage 1 sets a hard ceiling on everything downstream. If the compressor's own
reconstruction distorts peak-to-valley by as much as the physics varies it,
then no prior setting can produce the right spread and Phase B is 8 GPUs of
wasted time. The script reads the periodic stage-1 dumps (`predicted_denorm`
vs `target_denorm`, already in physical units) and the `_compare_` files, and
reduces both sides to a dimensionless fraction of peak-to-valley so the
training parts and the evaluation part can be compared at all. It needs no
GPU, no torch and no model.

A **FATAL** verdict is the one outcome that invalidates the whole campaign
rather than one cell of it — raise `latent_ch` or widen `voronoi_clusters` and
re-run Phase A instead of continuing.

## Phase B — the flow prior (`mode train_prior`)

| Arm | `prior_blocks` | `flow_loss_weighting` | GPU bot / top |
|---|---|---|---|
| `p4u` | 4 | uniform | 0 / 4 |
| `p8u` | 8 | uniform | 1 / 5 |
| `p4x` | 4 | x0 | 2 / 6 |
| `p8x` | 8 | x0 | 3 / 7 |

This one **is** a full 2×2, because the two axes plausibly interact: `x0`
weighting emphasises the data-prediction end of the trajectory, and whether
that helps may depend on having the depth to exploit it. Four cells is the
whole grid, so it costs nothing to be complete here.

All four load the same `ae_checkpoint` (`<half>.ae.pth`) and differ only in
keys that live inside `model.prior`.

### Which keys are free and which are locked

`CHiMGNFlow._load_frozen_ae` loads the promoted checkpoint with every
`model.prior.*` key stripped and then **raises on any remaining missing or
unexpected key**. That is what draws the line:

| Free — may vary across Phase-B arms | Locked — must match the promoted compressor |
|---|---|
| `prior_blocks`, `flow_time_freqs` | `latent_ch`, `latent_dim` |
| `flow_t_sampling`, `flow_loss_weighting` | the channel block (`input_var`, `output_var`, `cond_var`, `edge_var`, `positional_features`, `use_node_types`, `use_world_edges`) |
| `batch_size`, `learningr` | the hierarchy block (`use_multiscale`, `multiscale_levels`, `mp_per_level`, `coarsening_type`, `voronoi_clusters`) |
| | `ae_kl_weight` (provenance, not a shape — see below) |

`flow_time_freqs` is free *between* arms but architecture-defining *within*
one: it sets the AdaLN input width, so a Phase-B checkpoint only ever reloads
under the value it trained with. It is held at 16 everywhere here.

### `promote_ae.py`, and why `--check` cannot replace it

`_probe_checkpoints` in `cae_suite/preflight.py` only probes fields whose name
ends in `modelpath`. **`ae_checkpoint` is existence-checked and nothing more**,
so a `latent_ch` that disagrees with the promoted file passes `--check` cleanly
and dies as a `RuntimeError` minutes into the run, on eight cards at once.

`promote_ae.py` is the only place that comparison happens. It reads the
checkpoint's own `model_config`, checks every locked key against every
Phase-B and Phase-C config of that half, refuses a checkpoint whose
`training_stage` is not `train_ae`, and copies nothing if anything disagrees —
printing the exact `sed` that fixes all of them at once.

`ae_kl_weight` is checked even though the AE takes no gradient in stage 2 and a
mismatch would break nothing at load time: `build_model_config` copies it into
the Phase-B checkpoint's own `model_config`, so a wrong value there ships a
checkpoint that misdescribes the compressor inside it.

## Phase C — inference and the histogram

24 runs: 8 arms × 3 evaluation parts (`s26fe_main`, `s26fe_sec`,
`sm_l345u_main`). One card per `(half, arm)`; that arm's three parts run in
series on it.

Inference is also what draws the figure. With `make_histogram True`,
`rollout.py` writes the GT-vs-generated peak-to-valley histogram,
`spread_values.npz` and the machine-readable `[SPREAD]` log lines itself.
There is no separate figure stage.

Each `_infer_` file is **one part's** geometry and its `_compare_` file holds
that same part's 125 physical realizations, so the condition is fixed and the
target width ratio is exactly **1**. This is a calibration measurement, not an
accuracy measurement. `check_eval_inputs.py` verifies the pairing before any
GPU work, because a mismatch compares two different parts and reads as a
spread defect.

`rank_arms.py` ranks by the mean of `|log(sd ratio)|` over the three parts —
the log is what makes 2× too narrow and 2× too wide count equally — and
reports the mean bias in GT sigmas alongside without ranking on it. It also
prints the `SAOI_run` baseline row when that campaign's results are present,
which is the only way to see whether the sweep bought anything at all.

Note on margins: `std(gt)` comes from the same 125 realizations for every arm,
so its ~6% sampling error shifts the whole column together and cancels out of
the ranking. The generated side (2000 draws, ~1.6%) does not. Two arms within
about 0.02 of each other are tied.

## Why 2000 epochs and not 600

`SAOI_run` uses `ae_epochs 600` for stage 1. This sweep uses
`training_epochs 2000` for both phases. The reason is the LR schedule, not a
hunch about difficulty.

`build_optimizer_scheduler` sets `cosine_T0 = training_epochs - warmup_epochs`
and `CosineAnnealingWarmRestarts(T_0=cosine_T0, T_mult=1, eta_min=1e-8)`, after
a `LinearLR` warmup, inside a `SequentialLR`. So there is exactly **one** cosine
cycle, the learning rate reaches `1e-8` precisely at the final epoch, and **no
warm restart ever lands inside a run**.

Two consequences:

1. **Every run looks converged at its own end, whatever its length.** You
   cannot decide "was 600 enough" by staring at the tail of a 600-epoch run —
   it is flat by construction. `training_epochs` is not a length dial; it
   reshapes the entire LR trajectory.
2. **Screening short would bias the ladder.** `c16` has the most capacity and
   therefore converges last, so a short Phase A would systematically flatter
   `c4` — which is exactly the comparison Phase A exists to make.

So Phase A runs long once and `ae_report.py` reads the answer off the curve:
it reports each arm's best reconstruction, where it happened, and how much the
run improved after `--probe-epoch` (default 600).

**That gain column is an upper bound.** At epoch 600 of a 2000-epoch run the
learning rate is still high, so the value there is *worse* than what a
dedicated 600-epoch run would have reached by annealing into its own floor.
The improvement reported is therefore the most that lengthening could possibly
have bought — which makes it decisive in one direction: if the upper bound is
already small, 600 was enough and future campaigns can go back to it.

There is **no resume** (`cosine_T0` depends on the total), so a killed job
restarts from zero. One launch is one complete model, and the runner's launch
markers refuse to promote or infer from a checkpoint older than its own launch.

## What is deliberately not swept

| Key | Held at | Why |
|---|---|---|
| `flow_t_sampling` | `uniform` | already measured: beat logit-normal by 0.333 on this data |
| `batch_size` | `16` | already measured: beat 32 by 0.288 |
| `learningr` | `0.0001` | the contrast was a 0.083 tie — not worth a card |
| `flow_time_freqs` | `16` | architecture-defining; a change forces a retrain to compare |
| `latent_dim`, `voronoi_clusters`, `mp_per_level` | SAOI_run values | changing them changes the compressor, so they belong in Phase A's locked block, not a Phase-B axis — sweep 2 (below) moves the last two |
| `flow_steps`, `flow_solver` | `30`, `heun` | sampling-time only — `rollout.py` lets the config win on exactly these two keys, so they can be re-dialled at inference with no retraining |

A full factorial over Phase A × Phase B would be 4 × 4 = 16 models per half and
roughly 4× the wall clock, to answer a question the staged design already
answers: the compressor and the prior interact only through the latent, and
`ae_ceiling_check.py` measures that interface directly.

## Output layout

```text
output/chi-mgnflow/saoi_sweep/
    <half>.ae_<arm>.pth           Phase A checkpoints (8)
    <half>.ae.pth                 the promoted one, per half (2)
    <half>.prior_<arm>.pth        Phase B checkpoints (8)
    ae/<half>.ae_<arm>.log        Phase A training logs
    ae/test/<gpu>/<epoch>/        Phase A reconstruction dumps
    prior/<half>.prior_<arm>.log  Phase B training logs
    infer/<half>/<arm>/<part>/    spread_values.npz, histogram_compare.png
    run_logs/                     per-job launcher stdout, preflight logs, markers
```

Phase A and Phase B reuse the same cards, which is why their `log_file_dir`
values sit in **different subdirectories**: the periodic dumps land under
`<log_dir>/test/<gpu_ids>/<epoch>/`, so a shared `log_dir` would have Phase B
overwriting Phase A's reconstructions at every epoch the two intervals share.

Job stdout goes to `run_logs/<label>.out`, deliberately *not* to the file the
trainer owns via `log_file_dir`.

## Files

| File | Purpose |
|---|---|
| `config_train_ae_<half>_<arm>.txt` | 8 = 2 halves × 4 compressor arms |
| `config_train_prior_<half>_<arm>.txt` | 8 = 2 halves × 4 prior arms |
| `config_infer_<half>_<arm>_<tag>.txt` | 24 = 8 arms × 3 evaluation parts |
| `run_sweep.sh` | the runner: bare / `all` = the whole campaign unattended; `A`, `promote`, `B`, `C` run one phase by hand |
| `select_ae.py` | between A and B: auto-picks which compressor to promote per half (knee + ceiling gate), then calls `promote_ae.py` |
| `check_eval_inputs.py` | verifies every `_infer_`/`_compare_` pair before GPU work |
| `ae_report.py` | Phase A: rank the compressors, answer the epoch-length question |
| `ae_ceiling_check.py` | Phase A gate: can the compressor represent the spread at all? |
| `promote_ae.py` | copy one Phase-A checkpoint to `<half>.ae.pth`, with a strict config check |
| `rank_arms.py` | Phase C: rank arms by spread calibration, against the SAOI_run baseline |

## Operational controls

```bash
# one half only
HALVES=bot bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh A

# re-run one Phase-C cell
HALVES=bot PRIOR_ARMS=p8x INFER_TAGS=s26fe_main \
bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh C

# launch without validating (not recommended; --check is cheap)
PREFLIGHT=0 bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh B
```

Variables: `PYTHON`, `HALVES`, `AE_ARMS`, `PRIOR_ARMS`, `INFER_TAGS`,
`STAGGER`, `LOG_ROOT`, and the switches `PREFLIGHT`, `STRICT_PREFLIGHT`,
`EVAL_PREFLIGHT`. `STRICT_PREFLIGHT=1` (the default) refuses to launch a
partial phase — with four arms per half feeding one comparison, a missing cell
is worse than no phase. `PROMOTE_BOT`/`PROMOTE_TOP` force `select_ae.py` to
skip its own policy for that half and promote the named arm instead — they
apply the same way whether you run `promote` alone or the default `all`.

`STAGGER` only reduces contention: the four arms of a half share one multiscale
cache, the first to arrive builds it under an exclusive lock and the rest
block. Correctness does not depend on the delay.

`hierarchy_cache_keep True` leaves each half's cache beside its training HDF5.
Nothing deletes it for you, and it is worth keeping until Phase B is done:

```bash
rm -f dataset/SAOI/saoi_train_*.mscache.*.h5
```

## Dataset paths are case-sensitive — and `ae_checkpoint` is the exception

`dataset_dir`, `infer_dataset`, `eval_dataset`, `modelpath`, `log_file_dir` and
`inference_output_dir` are in `PATH_KEYS`, so the case written in the config is
the case opened. Keep `SAOI/`, `S26FE-MAIN`, `S26FE-SEC` and `SM-L345U-MAIN`
uppercase as written.

**`ae_checkpoint` is not in `PATH_KEYS`** (neither in `cae_suite/config_parser.py`
nor in `general_modules/load_config.py`), so its value is lowercased like any
other string. Every path in this directory is already lowercase, so nothing
breaks — but an uppercase directory introduced into that one key would be
silently folded and then fail to open on Linux. Keep it lowercase, or add the
key to both `PATH_KEYS` sets first.

## Sweep 2 — the compressor axes sweep 1 holds fixed

Sweep 1 varies only `latent_ch` at a fixed 100-node coarsest level, so it cannot
tell whether the latent budget `B = coarsest nodes × latent_ch` is better spent
on **space** or on **channels**, or whether the coarsest stack is deep enough.
Sweep 2 asks exactly that, with everything else held at sweep 1's c8 recipe:

| Arm | `voronoi_clusters` | `latent_ch` | `mp_per_level` | B | Read against |
|---|---|---|---|---|---|
| `n200c4` | `1000, 200` | 4 | `4, 6, 8, 6, 4` | 800 | sweep-1 `c8` (same B) |
| `n400c4` | `1000, 400` | 4 | `4, 6, 8, 6, 4` | 1600 | sweep-1 `c16` (same B) |
| `n400c8` | `1000, 400` | 8 | `4, 6, 8, 6, 4` | 3200 | headroom: does recon still improve past c16? |
| `c8m16` | `1000, 100` | 8 | `4, 6, 16, 6, 4` | 800 | sweep-1 `c8`, depth only |

`mp_per_level[L]` sizes both the encoder's coarsest stack and the decoder's
post-z stack, and the coarsest level is ~0.05% of the compute, so `c8m16` is
nearly free. There is **no selection step**: every compressor gets its own
prior at one fixed recipe (`p8u` — prior_blocks 8, since the n400 graphs are 4×
larger than sweep 1's and 4 blocks reach a smaller fraction of them) and its own
Phase C, so every arm is measured end to end: 8 AE + 8 prior + 24 inference jobs.

```bash
GPUS="4 5 6 7" ./run_sweep2.sh   # recommended: name the free cards
./run_sweep2.sh                  # else: every card under IDLE_MB (2000 MiB) now
./run_sweep2.sh A|B|C|report|clean|help
```

**Running beside sweep 1.** Every config says `gpu_ids 0`; the runner launches
each job with `CUDA_VISIBLE_DEVICES=<card>` (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`)
from a pool, one job per pool entry (list a card twice to pack two jobs). The
isolation is in the configs: all artifacts under `output/chi-mgnflow/saoi_sweep2/`,
one log directory per arm (the dumps are keyed by `gpu_ids`, which is 0
everywhere), `write_preprocessing False` (sweep 2 never writes the shared HDF5),
and `hierarchy_cache_dir` pointing at `saoi_sweep2/mscache/<hierarchy>/`, out of
reach of the same-stem prune next to the dataset. `all` deletes that cache
directory at the end (`KEEP_CACHE=1` to keep it).

| File | Purpose |
|---|---|
| `config_train_ae2_<half>_<arm>.txt` | Phase A, 8 = 2 halves × 4 arms |
| `config_train_prior2_<half>_<arm>.txt` | Phase B, 8; `ae_checkpoint` = that arm's own Phase-A file |
| `config_infer2_<half>_<arm>_<tag>.txt` | Phase C, 24 |
| `run_sweep2.sh` | the runner and GPU pool |
| `report_sweep2.py` | one table per half — B, best recon, ceiling, sd/bias per part, score — with sweep 1's compressors beside it; written to `saoi_sweep2/RESULTS.txt` |

The `2` in the names keeps sweep 1's globs (`select_ae.py`,
`check_eval_inputs.py`, `rank_arms.py`) from ever picking these files up.
