# SAOI wave 3 — MeshGraphNets-V sweep

A 2^(5-1) resolution-V half fraction: **five factors in sixteen arms**, two arms
per GPU across GPUs 0–7. Trains on `saoi_train_bot.h5`, then infers every arm
against the three held-out `*_bot` eval sets and scores the whole grid into one
report.

Everything here except the three scripts and this file is **generated**. Do not
hand-edit a config — change `gen_sweep_configs.py` and regenerate.

```bash
python configs/MeshGraphNets-V/SAOI_sweep3/gen_sweep_configs.py
```

## Layout

| Path | Authored? | What |
| --- | --- | --- |
| `gen_sweep_configs.py` | yes | Emits all 64 configs from the production ones next door |
| `run_sweep.sh` | yes | preflight → cache warm → train → infer → score, one command |
| `score_sweep.py` | yes | Builds `sweep_results.md` + the warpage overlay figures |
| `config_train_<arm>.txt` | generated (16) | One training arm |
| `config_infer_<arm>_<tag>.txt` | generated (48) | One arm × one eval set |

The base is **`../SAOI_all_input/config_train_bot.txt`**, and the inference
configs are derived from that folder's three `*_bot` infer configs. That is
deliberate: an arm is the production config with a handful of keys overridden,
so **a non-swept key that needs to change should change in the production
config**, not here. `../SAOI_all_input/` holds production only — the 2 training
and 6 inference configs — and nothing in it is generated.

## The design

Defining relation **`I = ABCDE`**: the fifth factor is `A xor B xor C xor D`,
not free. All 5 main effects and all 10 two-factor interactions are estimable
clean; only 3-factor and higher alias with them. With one run per cell there is
no replication either way, so three-factor terms were never trustworthy — the
half fraction gives up nothing real and buys a whole extra factor.

| | factor | level 0 | level 1 |
| --- | --- | --- | --- |
| A | `z_conditioning` | `cc` concat fuser (legacy) | `ad` AdaLN-Zero |
| B | `prior_grad_to_encoder` | `g0` 0.0 — detached, no CVAE rate term | `g1` 1.0 — end-to-end |
| C | `vae_latent_dim` | `z16` | `z64` |
| D | capacity | `c0` 128 / `4,6,8,6,4` | `c1` 192 / `6,8,12,8,6` (+VAE/prior depth) |
| E | regularizer scale | `r001` `lambda_mmd` 1 + `prior_nll_weight` 1 | `r100` both 100 |

Arm names encode the cell: `<cc|ad>_<g0|g1>_<z16|z64>_<c0|c1>_<r001|r100>`.

**The interaction to read first is `z_conditioning × capacity`.** Under `cc`
every extra processor block compounds the concat fuser's ~1.33x gain, so depth
should HURT; under `ad` the residual highway is intact, so depth should HELP. A
large interaction also means the corresponding MAIN effect is an average over
two opposite behaviours and must not be read alone.

**`r001` is the "regularizers effectively off" control.** `alpha_recon` is 1000
while both regularizers sit at ~1 there, i.e. ~0.1% of the objective each — and
that matters most for `prior_grad_to_encoder`, whose rate term competes with
reconstruction inside the encoder. The `g0`/`g1` contrast can only show force in
the `r100` half.

`beta_aux` is fixed at 1.0 in every arm and **must stay > 0**: it is the I(z;y)
floor that guards the eight `g1` arms against collapsing to a deterministic
`z = h(g)`, which MMD does not prevent.

## GPU packing

Arms are paired by complementing the four free factors, so every GPU carries one
`c0` and one `c1` — capacity is the only factor that moves memory much, so this
is what keeps VRAM balanced. The regularizer scale is constant within a pair,
which costs nothing: it changes no memory or runtime, and eight identical cards
in one node carry no batch/day/operator effect for it to confound with.

```
GPU 0 : cc_g0_z16_c0_r001  +  ad_g1_z64_c1_r001
GPU 1 : cc_g0_z16_c1_r100  +  ad_g1_z64_c0_r100
GPU 2 : cc_g0_z64_c0_r100  +  ad_g1_z16_c1_r100
GPU 3 : cc_g0_z64_c1_r001  +  ad_g1_z16_c0_r001
GPU 4 : cc_g1_z16_c0_r100  +  ad_g0_z64_c1_r100
GPU 5 : cc_g1_z16_c1_r001  +  ad_g0_z64_c0_r001
GPU 6 : cc_g1_z64_c0_r001  +  ad_g0_z16_c1_r001
GPU 7 : cc_g1_z64_c1_r100  +  ad_g0_z16_c0_r100
```

None of the swept keys enter the coarsening cache signature (that is
`multiscale_levels` / `coarsening_type` / `voronoi_clusters` /
`hierarchy_variants` / `positional_features` + the source file), so all 16 arms
share ONE `*.mscache.*.h5`.

## Running it

```bash
# Delete any leftover cache FIRST: cache_ready() only globs the filename, so a
# stale one makes the script skip the warm-up and launch all 16 into a MISS.
rm -f dataset/saoi/saoi_train_bot.mscache.*.h5

nohup bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh > sweep.out 2>&1 &
tail -f sweep.out
```

`run_sweep.sh` preflights all 64 configs, launches ONE arm to build the shared
cache (aborting the batch if it dies, so a config error costs minutes not days),
then the other 15, then inference, then scoring.

| env | default | effect |
| --- | --- | --- |
| `ARMS` | all 16 | subset to run |
| `PREFLIGHT` | 1 | `--check` every arm before launching any |
| `TRAIN` | 1 | `0` skips training entirely and goes straight to infer + score on the checkpoints already on disk — how you finish a two-wave run |
| `INFER` | 1 | run the per-arm inference stage |
| `INFER_TAGS` | all 3 | which eval sets to infer |
| `SCORE` | 1 | build the report when training ends |
| `SCORE_K` | 50 | draws per geometry for the rank histogram |
| `WARM_TIMEOUT` | 21600 | seconds to wait for the shared cache |

## What comes out

```
outputs/saoi_sweep3/sweep_results.md      the report — read/paste this
outputs/saoi_sweep3/sweep_results.json    everything, incl. full rank histograms
outputs/saoi_sweep3/warpage_<tag>.png     GT + all 16 arms on ONE axis, ranked by W1
outputs/saoi_sweep3/run_logs/<arm>.log    per-arm transcripts
output/meshgraphnets-v/saoi_sweep3/<arm>.pth
output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/histogram_compare.png
output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/spread_values.npz
```

The report carries four things: the per-arm table (CRPS, wild rate, rank
calibration), **main effects** (8 vs 8 per factor), **two-factor interactions**,
and the **warpage spread table** — `max(z_disp) − min(z_disp)` per realization,
generated against ground truth, normalized by the GT spread's own std so the
three eval sets are comparable. `W1/sd` is the ranking column; `sd ratio < 1`
is the classic under-dispersion failure.

The inference configs set `save_rollouts False`, so **no trajectory HDF5s are
written** — scene × draws would be tens of thousands of files across the grid.
`num_vae_samples` is **2000** draws per scene, so each histogram carries
`scenes x 2000` generated realizations against the eval set's one-per-scene
ground truth. **This is the sweep's dominant cost**: total forwards are
`16 arms x 3 eval sets x scenes x 2000`. Lower `INFER_SAMPLES` in the generator
first if the inference stage overruns. `vae_batch_vram_fraction` is 0.35 rather
than the default 0.70 because two arms share each card during inference too, and
both auto-size against the same *free* VRAM reading.

## If batch 16 does not fit two-per-card

It did not, the first time. The three options, and why the first one is the
default answer:

| | batch | MMD samples | wall clock | design |
| --- | --- | --- | --- | --- |
| **two waves of 8** | **16 kept** | **16 kept** | 2x | intact |
| all arms at batch 8 | 8 | 8 | unchanged | intact, axes all survive |
| shrink the `c1` level | 16 | 16 | unchanged | capacity axis weakened |

Batch size matters here for a SPECIFIC reason, not the usual one: **MMD is a
two-sample statistic and its effective sample count is the per-rank batch**,
because `mmd_loss` runs inside the model forward. Halving the batch halves the
sample count of the very term the `r` axis exists to measure. For the
reconstruction gradient alone, 8 vs 16 would barely matter.

**`grad_accum_steps 2` does NOT fix this.** It restores the optimizer batch but
MMD still only ever sees one micro-batch — it fixes gradient noise and leaves
the thing we care about untouched.

Splitting by name prefix does not work: the first eight arms are all `cc`, so
the wave would be fully confounded with `z_conditioning`. These two halves are
**4/4 balanced on all five factors** and each fills GPUs 0–7 one arm per card:

```bash
ARMS="cc_g0_z16_c0_r001 cc_g0_z16_c1_r100 cc_g1_z64_c0_r001 cc_g1_z64_c1_r100 ad_g0_z64_c0_r001 ad_g0_z64_c1_r100 ad_g1_z16_c0_r001 ad_g1_z16_c1_r100" \
  INFER=0 SCORE=0 bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh

ARMS="cc_g0_z64_c0_r100 cc_g0_z64_c1_r001 cc_g1_z16_c0_r100 cc_g1_z16_c1_r001 ad_g0_z16_c0_r100 ad_g0_z16_c1_r001 ad_g1_z64_c0_r100 ad_g1_z64_c1_r001" \
  INFER=0 SCORE=0 bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh

# both waves trained -> one inference + scoring pass over all 16
TRAIN=0 INFER=1 SCORE=1 bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh
```

`INFER=0 SCORE=0` on the waves because the inference stage also packs two arms
per card, so it should run once over the full grid at the end, not twice over
halves.

## Watch these in the first hour

1. **`VRAM peak=` on a `c1` arm.** Two arms share each card at the full
   production `Batch_size 16`, and `c1` is ~1.5x the width with 40 processor
   blocks instead of 28. If the pair does not fit: drop `Batch_size` to 8 in the
   generator and regenerate, or shrink the `c1` level. Splitting into two waves
   of 8 does *not* work by name prefix — the first eight arms are all `cc`.
2. **`mmd` and `fm_p` against `total`** on the progress bar. If BOTH `r001` and
   `r100` look negligible the axis has to move UP (1000/10000), not sideways.
3. **`aux` on the `g1` arms.** A steady rise means `y` is being bleached out of
   `z` — lower `prior_grad_to_encoder` or raise `beta_aux`.

## Caveats

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
