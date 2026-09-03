# SAOI wave 3 — MeshGraphNets-V sweep

A 2^(4-1) resolution-IV half fraction: **four factors in eight arms**, one arm
per GPU across GPUs 0–7. Trains on `saoi_train_bot.h5`, then infers every arm
against the three held-out `*_bot` eval sets and scores the whole grid into one
report.

Everything here except the three scripts and this file is **generated**. Do not
hand-edit a config — change `gen_sweep_configs.py` and regenerate.

```bash
python configs/MeshGraphNets_Variational/SAOI_sweep3/gen_sweep_configs.py
```

## Layout

| Path | Authored? | What |
| --- | --- | --- |
| `gen_sweep_configs.py` | yes | Emits all 32 configs from the production ones next door |
| `run_sweep.sh` | yes | preflight → cache warm → train → infer → score, one command |
| `score_sweep.py` | yes | Builds `sweep_results.md` + the warpage overlay figures |
| `config_train_<arm>.txt` | generated (8) | One training arm |
| `config_infer_<arm>_<tag>.txt` | generated (24) | One arm × one eval set |

The base is **`../SAOI_all_input/config_train_bot.txt`**, and the inference
configs are derived from that folder's three `*_bot` infer configs. That is
deliberate: an arm is the production config with a handful of keys overridden,
so **a non-swept key that needs to change should change in the production
config**, not here. `../SAOI_all_input/` holds production only — the 2 training
and 6 inference configs — and nothing in it is generated.

## The design

Defining relation **`I = ABCD`**: the fourth factor is `A xor B xor C`, not
free. All 4 main effects are estimable clean; the six 2-factor interactions
collapse into **three confounded pairs** — `AB=CD`, `AC=BD`, `AD=BC` — so a
large effect on one pair cannot be attributed to either half without a
follow-up run. With one run per cell there is no replication either way, so
3-factor terms were never trustworthy anyway — this is the shape that keeps
main effects clean at the lowest run count.

| | factor | level 0 | level 1 |
| --- | --- | --- | --- |
| A | `z_conditioning` | `cc` concat fuser (legacy) | `ad` AdaLN-Zero |
| B | `prior_grad_to_encoder` | `g0` 0.0 — detached, no CVAE rate term | `g1` 1.0 — end-to-end |
| C | capacity | `c0` 128 / `4,6,8,6,4` (28 blocks) | `c1` 128 / `6,8,12,8,6` (40 blocks) |
| D | regularizer scale (generated) | `r001` `lambda_mmd` 1 + `prior_nll_weight` 1 | `r100` both 100 |

Arm names encode the cell: `<cc|ad>_<g0|g1>_<c0|c1>_<r001|r100>`.

**This replaced an earlier 5-factor / 16-arm / 2-per-GPU design** (`vae_latent_dim`
as a fifth axis). At a measured **~576 s/epoch**, 24h of wall time only reached
epoch 150 on the old 2000-epoch budget — multiple GPU-weeks to finish, and
two-per-GPU sharing was a real OOM risk on the `c1` (40-block) arms. Eight arms
at one-per-GPU and **1000 epochs** is a `~345 s/epoch × 1000 ≈ 4-day`,
**BUDGET-LIMITED comparison, NOT a converged one** — read the report that way.
The twin cHI-MGNflow sweep (`SAOI_sweepB`) made the identical trade for the
identical reason.

`vae_latent_dim` was dropped rather than any of the other four: it is the
lowest-priority axis (wave-1's small-z win was measured with MMD statistically
dead, so it matters, but less than the three architectural axes kept here) and
costs almost nothing in VRAM or wall time, so it was never going to be what made
the old budget unaffordable. It is now **fixed at 16** in every arm.

**The pair to read first is `z_conditioning × capacity`.** Under `cc` every
extra processor block compounds the concat fuser's ~1.33x gain, so depth
should HURT; under `ad` the residual highway is intact, so depth should HELP.
A large confounded-pair value also means the corresponding MAIN effects are
averages over two opposite behaviours and must not be read alone.

**`r001` is the "regularizers effectively off" control.** `alpha_recon` is 1000
while both regularizers sit at ~1 there, i.e. ~0.1% of the objective each — and
that matters most for `prior_grad_to_encoder`, whose rate term competes with
reconstruction inside the encoder. The `g0`/`g1` contrast can only show force in
the `r100` half.

`beta_aux` is fixed at 1.0 in every arm and **must stay > 0**: it is the I(z;y)
floor that guards the four `g1` arms against collapsing to a deterministic
`z = h(g)`, which MMD does not prevent.

## GPU packing

One arm per GPU (`gpu_ids` = arm index) — no card sharing, so there is no VRAM
co-residency exposure and no complement-pairing logic is needed.

```
gpu 0  arm 1  (cc g0 c0 r001)      gpu 4  arm 5  (ad g0 c0 r100)
gpu 1  arm 2  (cc g0 c1 r100)      gpu 5  arm 6  (ad g0 c1 r001)
gpu 2  arm 3  (cc g1 c0 r100)      gpu 6  arm 7  (ad g1 c0 r001)
gpu 3  arm 4  (cc g1 c1 r001)      gpu 7  arm 8  (ad g1 c1 r100)
```

None of the swept keys enter the coarsening cache signature (that is
`multiscale_levels` / `coarsening_type` / `voronoi_clusters` /
`hierarchy_variants` / `positional_features` + the source file), so all 8 arms
share ONE `*.mscache.*.h5`.

## Running it

```bash
# Delete any leftover cache FIRST: cache_ready() only globs the filename, so a
# stale one makes the script skip the warm-up and launch all 8 into a MISS.
rm -f dataset/saoi/saoi_train_bot.mscache.*.h5

nohup bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep.sh > sweep.out 2>&1 &
tail -f sweep.out
```

`run_sweep.sh` preflights all 32 configs, launches ONE arm to build the shared
cache (aborting the batch if it dies, so a config error costs minutes not days),
then the other 7, then inference, then scoring.

| env | default | effect |
| --- | --- | --- |
| `ARMS` | all 8 | subset to run |
| `PREFLIGHT` | 1 | `--check` every arm before launching any |
| `TRAIN` | 1 | `0` skips training entirely and goes straight to infer + score on the checkpoints already on disk |
| `INFER` | 1 | run the per-arm inference stage |
| `INFER_TAGS` | all 3 | which eval sets to infer |
| `SCORE` | 1 | build the report when training ends |
| `SCORE_K` | 50 | draws per geometry for the rank histogram |
| `WARM_TIMEOUT` | 21600 | seconds to wait for the shared cache |

## What comes out

```
output/meshgraphnets-v/saoi_sweep3/sweep_results.md      the report — read/paste this
output/meshgraphnets-v/saoi_sweep3/sweep_results.json    everything, incl. full rank histograms
output/meshgraphnets-v/saoi_sweep3/warpage_<tag>.png     GT + all 8 arms on ONE axis, ranked by W1
output/meshgraphnets-v/saoi_sweep3/run_logs/<arm>.log    per-arm transcripts
output/meshgraphnets-v/saoi_sweep3/<arm>.pth
output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/histogram_compare.png
output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/spread_values.npz
```

The report carries four things: the per-arm table (CRPS, wild rate, rank
calibration), **main effects** (4 vs 4 per factor), **confounded two-factor
effects** (3 pairs, each the sum of its alias), and the **warpage spread
table** — `max(z_disp) − min(z_disp)` per realization, generated against
ground truth, normalized by the GT spread's own std so the three eval sets
are comparable. `W1/sd` is the ranking column; `sd ratio < 1` is the classic
under-dispersion failure.

The inference configs set `save_rollouts False`, so **no trajectory HDF5s are
written** — scene × draws would be thousands of files across the grid.
`num_vae_samples` is **2000** draws per scene, so each histogram carries
`scenes × 2000` generated realizations against the eval set's one-per-scene
ground truth. Total inference forwards are `8 arms × 3 eval sets × scenes ×
2000`; lower `INFER_SAMPLES` in the generator first if the stage overruns.
`vae_batch_vram_fraction` is the native default `0.70` — one arm per GPU now,
so there is no card-sharing partner to halve it for.

## Watch these in the first hour

1. **`VRAM peak=` on a `c1` arm.** One arm per card now, so there is real
   headroom, but `c1` (40 processor blocks vs 28) is still the heavier level.
2. **`mmd` and `fm_p` against `total`** on the progress bar. If BOTH `r001` and
   `r100` look negligible the axis has to move UP (1000/10000), not sideways.
3. **`aux` on the `g1` arms.** A steady rise means `y` is being bleached out of
   `z` — lower `prior_grad_to_encoder` or raise `beta_aux`.
4. **A "Unable to synchronously open file" error on any one arm.** Every arm's
   setup phase rewrites normalization stats into the SHARED `saoi_train_bot.h5`
   (unconditionally, no "already present" guard), and 8 arms launch seconds
   apart against that one file. `general_modules/mesh_dataset.py` now retries
   that specific open with backoff, so this should be rare; if it still
   happens, just relaunch the one arm:
   ```bash
   ARMS="<that arm>" PREFLIGHT=0 TRAIN=1 INFER=0 SCORE=0 bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep.sh
   ```

## Caveats

- 1000 epochs at a measured ~345 s/epoch is a **budget-limited**, not converged,
  comparison — treat every ranking here as "best at this budget."
- MMD sees 16 samples per arm, not production's 64: `mmd_gather_ranks` stays
  `True` but is inert at `world_size 1`. A 4x smaller sample makes the
  V-statistic more biased AND its gradient noisier, which can push the optimum
  either way — **re-check the winning regularizer scale at the production
  batch** before adopting it.
- `LearningR` is held at the production `1e-4` for every arm, which keeps the
  comparison internally valid but means the winner needs its LR revisited for
  the 4-GPU production run (global batch 16 → 64).
- Only the three `*_bot` eval sets are inferred: the sweep trains on
  `saoi_train_bot.h5`, so no `*_top` checkpoint exists for these arms.
- The generated inference configs spell the dataset directory `saoi`, matching
  the training configs. The six production infer configs in `../SAOI_all_input/`
  still spell it `SAOI`; both keys are `PATH_KEYS` so case is preserved, and on
  a case-sensitive filesystem only one spelling can resolve.
