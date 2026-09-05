# SDFFlow VAE ablation arms

Ten `mode train_vae` configs around the recommended v3 recipe
([`../config_train_v3.txt`](../config_train_v3.txt)), on identical data settings
(`dataset/deepjeb.h5`, `split_seed 42`, `split_by_parent True`, 6144 encoder /
8192 query points, `seed 0`) and an identical 500-epoch, batch-4 budget. The arms
train **only the VAE**; the flow-matching stage is trained once, for the winner,
afterwards.

Most arms move **one axis** off A0, but two do not, and reading the table as if
they all did will mislead you:

- **A1 is an omnibus control, not an axis.** It restores the whole ex1
  architecture and loss recipe at once (token count, decoder type, query type,
  posterior schedule, KL weight, hybrid terms, AMP). A1-vs-A0 answers "is v3
  worth it at all"; it attributes nothing to any single change.
- **A9 is not a treatment.** It is A0 at a second `seed`, so `|A9 - A0|` on each
  metric is the sweep's run-to-run noise band. Read every other arm's gap
  against it first: a difference smaller than the A0/A9 spread is not evidence.

Every arm carries the same `seed` key (A9 excepted), so the arms differ by their
axis rather than by which shuffle and initialization they happened to draw.

| Arm | Label in `roster.tsv` | Differs from A0 by | Question it answers |
| --- | --- | --- | --- |
| [`A0.txt`](A0.txt) | `sdfflow_vae_A0` | nothing | the v3 control |
| [`A1.txt`](A1.txt) | `sdfflow_vae_A1` | **many axes at once** -- the whole ex1 recipe: 1 x 256 token, MLP decoder, learned query, legacy noise (0.1, no floor), `kl_weight 0.00001`, no hybrid terms, AMP on | is the v3 architecture worth it at all, on the same data? (not attributable to any one change) |
| [`A2.txt`](A2.txt) | `sdfflow_vae_A2` | `posterior_noise_max_scale 0.1`, `posterior_min_std_rel 0.0` | did full-scale reparameterization + the std floor help or hurt? |
| [`A3.txt`](A3.txt) | `sdfflow_vae_A3` | `kl_weight 0.000025` (10x) | KL pressure sensitivity |
| [`A4.txt`](A4.txt) | `sdfflow_vae_A4` | `encoder_query_type learned` | do FPS-anchored queries beat free parameters? |
| [`A5.txt`](A5.txt) | `sdfflow_vae_A5` | `surface_weight 0`, `normal_weight 0`, `eikonal_weight 0` (`use_amp` stays False) | do the hybrid geometry terms earn their cost? AMP is left off even though nothing here needs fp32: flipping precision alongside the loss composition would make the gap unattributable |
| [`A6.txt`](A6.txt) | `sdfflow_vae_A6` | `normal_weight 1.0` | TripoSG-like normal emphasis |
| [`A7.txt`](A7.txt) | `sdfflow_vae_A7` | `latent_dim 64` (32 x 64 = 2048 flat), `kl_weight 0.00000125` | latent capacity (the v2 latent size, v3 everything else). The KL is *summed* over tokens x dims, so the weight is halved alongside the doubled latent to hold `weight x D = 2.56e-3`; without that this arm would vary capacity and KL pressure together |
| [`A8.txt`](A8.txt) | `sdfflow_vae_A8` | `num_encoder_points 4096` (A0: 6144 of the 8192 stored) | encoder input density: is more surface supervision per step worth more than the extra stochasticity of a smaller draw? Both ends are stochastic per-epoch subsamples. The deterministic 8192-of-8192 end is deliberately absent -- it is a permutation of one fixed point set, the defect v3 was corrected for, not a candidate recipe |
| [`A9.txt`](A9.txt) | `sdfflow_vae_A9` | `seed 1` -- nothing else | the sweep's noise floor; `|A9 - A0|` is the smallest gap any other arm must beat |

Every arm writes to `output/geometry_generation/arms/<A#>/`:
`train_vae.log`, `sdfflow_vae.pth` (final epoch), `sdfflow_vae_best.pth` (best
validation SDF loss), and the periodic reconstruction-test meshes.

The arm files are named `A<n>.txt`, so the suite's `--audit-configs` (which
globs `config*.txt`) does not see them; validate them per file:

```bash
for f in configs/SDFFlow/arms/A*.txt; do python AI_CAE4ALL_main.py --config "$f" --check; done
```

## Launching on 8 GPUs

`roster.tsv` is in the `(label, train_config, ex_slot, light)` format the
benchmark campaign runner reads. `ex_slot` is `deepjeb` -- for `--mode train`
the runner uses it only as a label (it never resolves a dataset from it), and
the roster is **train-only**: the runner's `--mode infer` rewrites `mode train`
to `mode inference`, which SDFFlow does not have. Arms are scored with
`evaluate` instead (below).

From the repo root, one arm per GPU:

```bash
ROSTER=configs/SDFFlow/arms/roster.tsv \
LOG_ROOT=output/geometry_generation/arms/campaign \
MAX_CONCURRENCY_PER_GPU=1 \
GPUS="0 1 2 3 4 5 6 7" \
bash configs/campaigns/benchmarks_all/train_all.sh
```

or, without bash:

```bash
python configs/campaigns/benchmarks_all/campaign_runner.py --mode train \
    --roster configs/SDFFlow/arms/roster.tsv \
    --log-root output/geometry_generation/arms/campaign \
    --gpus "0 1 2 3 4 5 6 7" --max-concurrency-per-gpu 1
```

The runner rewrites each arm's `gpu_ids` line to its assigned GPU, preflights
the runtime copy with `--check --strict`, and launches it through the suite
launcher. `MAX_CONCURRENCY_PER_GPU=1` keeps the timing comparison fair (the
default admits up to three jobs per GPU by free VRAM, and the fp32 hybrid
double-backward at 6144 encoder / 8192 query points would otherwise share a card). Add
`CHECK_ONLY=1` to preflight the whole roster without training, and
`LABELS="sdfflow_vae_A0 sdfflow_vae_A4"` to run a subset.

Manual equivalent for one arm on GPU 3 (the config is edited in place, so copy
it first if the original must stay at `gpu_ids 0`):

```bash
sed -i 's/^gpu_ids\t0/gpu_ids\t3/' configs/SDFFlow/arms/A4.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/arms/A4.txt
```

Every arm prints an EMA warning at startup if `(1 - ema_decay) * total_updates
< 10`; at ~428 updates/epoch x 500 epochs it does not fire.

## Scoring an arm

Score each arm with a copy of [`../config_evaluate.txt`](../config_evaluate.txt)
whose `vae_modelpath` points at that arm's checkpoint (use
`sdfflow_vae_best.pth` for model selection; note which one you scored) and
whose `output_dir` is the arm's directory. The split keys already match every
arm (`split_seed 42`, `split_by_parent True`); leave `eval_split val` for
selection and touch `test` once, for the final winner.

```bash
for a in A0 A1 A2 A3 A4 A5 A6 A7 A8 A9; do
  sed -e "s#ex4/sdfflow_vae.pth#arms/$a/sdfflow_vae_best.pth#" \
      -e "s#ex4/eval#arms/$a/eval#" \
      configs/SDFFlow/config_evaluate.txt > "output/geometry_generation/arms/eval_$a.txt"
  python AI_CAE4ALL_main.py --config "output/geometry_generation/arms/eval_$a.txt"
done
```

Each run writes `eval_val.json` / `eval_val.csv` with per-shape rows and the
aggregate mean/median of `surface_mean` / `surface_p95` / `surface_max` (GT
surface point -> reconstructed mesh), `pred_to_gt_mean` / `pred_to_gt_p95`
(points sampled on the reconstruction -> GT surface) and their average
`chamfer_mean`, `sdf_l1`, `sign_accuracy` with `sign_balanced_accuracy` and
`positive_fraction`, `body_count_raw`, `watertight`, `valid` -- once for the
encoder mean (`enc_*`) and once after 300 steps of decoder-frozen latent
refinement (`ref_*`).

Reading them:

- **Compare arms on `enc_chamfer_mean` and `enc_sign_balanced_accuracy`.**
  `surface_mean` alone is one-sided: a noisy space-filling reconstruction scores
  well on it because every GT point still finds *some* nearby predicted surface,
  which is what `pred_to_gt_*` closes. Raw `sign_accuracy` has a majority-class
  floor near the `positive_fraction` the rows record (~0.64 outside points); the
  balanced form averages the inside and outside rates, so its trivial baseline
  is 0.5.
- `enc_*` is what the FM will see, so it decides the winner. The `enc_* - ref_*`
  gap is the encoder's own error: large means the encoder, not the decoder, is
  the bottleneck. With refinement on, the stored query points are halved and
  **both** prefixes are scored on the half the refinement never saw, so that gap
  is a held-out comparison; `ref_*_insample` are the fit-half numbers and are fit,
  not accuracy.
- Read `ActiveUnits: k/D` from each `train_vae.log` **together with
  `ActiveSNR: k/D`** on the same line. `ActiveUnits` counts latent scalars whose
  encoder-mean variance clears a fixed absolute threshold, so it moves with the
  latent's overall scale and is not comparable across arms that change
  `latent_dim` or KL pressure (A3, A7). `ActiveSNR` is the scale-free form --
  `Var_x(mu_d) / mean_x(sigma_d^2) > 1`, signal over posterior noise -- and is
  the one to compare. An arm with many dead units is buying its reconstruction
  with a smaller effective latent than it declares. The same line's
  `ValidSignBal` is the balanced sign accuracy; prefer it to raw `ValidSign`,
  whose floor is the majority-outside class.
- Check every gap against `|A9 - A0|` before believing it.

## Training the FM for the winner

Copy `../config_train_v3.txt`, set its VAE-side keys to the winning arm's
values (architecture, losses, noise schedule, and the `vae_*` training keys must
match what the arm trained with, because `skip_completed_stages` compares them
against the checkpoint's saved config), point `vae_modelpath` at the arm's
**final** `sdfflow_vae.pth` (the pipeline's completeness check requires the
final-epoch checkpoint, not the best one), and run `mode train`. The VAE stage
is reused and only the FM trains.
