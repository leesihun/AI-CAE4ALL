# Miscellaneous Utilities

This directory contains optional debugging and visualization helpers. They are
not part of the main training or rollout path.

## Files

| File | Purpose |
| --- | --- |
| `eval_distribution.py` | Generative-distribution diagnostic for a trained VAE checkpoint: draws K samples per held-out geometry and builds the verification-rank (PIT) histogram plus the "wild rate". Flat histogram = calibrated; U-shaped = under-dispersed; dome = over-dispersed. Needs no synthetic labels — real data has one ground truth per geometry, which is exactly what the rank test is built for. |
| `field_intrinsic_dim.py` | Answers whether a GLOBAL latent `z` is a bottleneck or a well-matched inductive bias, before any model is built. (A) what fraction of each sample's field a low-order polynomial in the reference coordinates explains; (B) how many PCA components of the sample-to-sample variation carry 95% of the variance — directly comparable to `vae_latent_dim`; (C) whether those leading variation modes are global (smooth, needs multiscale reach) or local. Verified against synthetic fields of known rank. |
| `posterior_vs_prior.py` | SAOI sweep (global latent): decodes posterior and prior draws of each part and asks whether the generated distribution's missing width is lost in the prior or in the decoder. |
| `posterior_vs_prior_groups.py` | The same question on a grouped held-out file (ex3, `split_group_attr`), any `latent_type`. Per geometry it decodes every FE draw as a posterior sample, posterior mean and prior sample, and scores each against the held-out half exactly like `hyperparameter_sweep/plots.py` (W2 / sd, 200 splits, floor included). floor → posterior sample is the encoder/decoder's cost; posterior sample → prior is the prior's. For `latent_type hybrid` it also decodes each draw with z_g, then z_l, swapped for another draw's of the same geometry (`g_shuffle`, `l_shuffle` in `paired_within_geometry`): a correlation that stays at the posterior mean's means the decoder ignores that part. Writes a JSON, per-geometry histograms and a `==== TYPE THIS BACK ====` block. Runners for the B300 sweeps: `hyperparameter_sweep/r8/diagnose.sh`, `r9/diagnose_135.sh`. |
| `prior_diagnosis.py` | What is wrong with the conditional prior, and how much (`latent_type` hybrid or local; grouped file). The reference is exact: a geometry's FE draws give n posteriors, their mixture is the distribution the prior should draw, and that mixture's flow-matching velocity has a closed form. Per geometry it scores, like `plots.py`, the FE floor, posterior samples (`P`), the exact flow through the model's own sampler (`O`, must equal `P`: the method's check), the prior (`Q`), the prior at 4x the steps, the prior with its mean / width / per-coordinate distribution corrected in turn, the exact flow before t = 0.5 and the model after, and the prior's draws read off at the nearest posterior component with no decoding. It also measures the velocity error per time against the exact field (beside a field that ignores the geometry and the best per-geometry Gaussian), how far the prior's draws sit from the posterior's support, the distance of each geometry's input embedding from the training geometries, the latent's scale after standardization, PIT tails and mode shares. Beyond the prior's own error it runs the checkpoint's trained (non-EMA) weights beside the EMA ones, only `z_g` or only `z_l` from the prior (against two unrelated posterior samples), and reads the epoch log of the training config beside the inference config (was the kept epoch too early or past the best). It also decodes the midpoint of two posteriors of the same geometry (`M`, `mid_out`, `mid_snap`: does the decoder give a valid result between two posteriors, or is a draw off the posterior wrong by itself; causes 39 `holes` and 40 `blend`, typed line `de`). `--split heldout|train|test`; `--tag jt --modelpath <stem>.joint.pth` diagnoses the checkpoint kept from before the prior freeze, which tells what the prior-only epochs bought. Writes `prior_diagnosis_<tag>.json`, a verdict in English sentences, a `CAUSES` table (every candidate `YES` / `no` / `?`) and a `==== TYPE THIS BACK ====` block. The shares are fixed-order splits of one gap, not independent effects. Runners: `hyperparameter_sweep/r11/prior_135.sh`; from round 13 on the round's watcher (`r13/diagnose_135.sh`) runs it on every finished job. |
| `prior_verdict.py` | Standard library only, English output: turns `prior_diagnosis.py`'s JSONs (one per part, seeds averaged) into the sentences, the `CAUSES` table (every candidate cause `YES` / `no` / `?` with the number behind it, numbered in a fixed order) and the typed lines with a check digit. Imported by `prior_diagnosis.py`, by `hyperparameter_sweep/r11/prior_brief.py` and by `r13/brief.py`, which turns each setting's diagnosis into one decision about the cause that setting tests (CONFIRMED / RULED OUT / OPEN / WORSE / MIXED). |
| `plot_loss.py` | Legacy static training-loss plotter. |
| `plot_loss_realtime.py` | Legacy FastAPI dashboard for live loss plotting. |
| `debug_model_output.py` | Loads a checkpoint and dataset sample to inspect model output magnitudes. |
| `analyze_mesh_topology.py` | Mesh topology inspection helper. |
| `requirements_plotting.txt` | Extra packages for the plotting dashboard. |
| `templates/dashboard.html` | HTML template used by `plot_loss_realtime.py`. |

## Main Runtime Logs

The trainer treats `log_file_dir` as a plain path relative to the method working
directory. Checked-in configs therefore point back to the suite-wide artifact
root:

```text
log_file_dir ../../output/meshgraphnets-v/<run>/train.log
```

For example, when the launcher runs the native process from
`methods/MeshGraphNets_Variational/`, this config line:

```text
log_file_dir ../../output/meshgraphnets-v/b8_all/train1.log
```

writes to the root-facing path:

```text
output/meshgraphnets-v/b8_all/train1.log
```

Current VAE logs look like:

```text
Elapsed: 123.45s Epoch 10 LR: 1.0000e-04 | Train recon=... mmd=... total=... | Valid recon=... mmd=... total=... | CRPS ...
```

Current non-VAE logs look like:

```text
Elapsed: 123.45s Epoch 10 Train 1.2345e-02 Valid 1.5678e-02 LR: 1.0000e-04
```

## Loss Plotters Are Legacy

`plot_loss.py` and `plot_loss_realtime.py` now use `log_file_dir` verbatim too,
so their path resolution agrees with training. They still parse only the older
log pattern:

```text
Epoch N Train Loss: ... Valid Loss: ...
```

That does not match the current VAE or deterministic log formats shown above.
Use these scripts only for older logs with that format, or update their parsers
before using them for current training runs. `gpu_ids` no longer changes the
log path.

## Legacy Plotter Usage

Install optional plotting dependencies:

```bash
pip install -r misc/requirements_plotting.txt
```

Static plotter:

```bash
python misc/plot_loss.py config.txt
python misc/plot_loss.py config.txt --output ../../output/meshgraphnets-v/loss_plot.png
```

Realtime dashboard:

```bash
python misc/plot_loss_realtime.py config.txt
python misc/plot_loss_realtime.py config.txt --port 8080
```

The dashboard serves:

```text
http://localhost:5000
http://localhost:5000/docs
```

or the port supplied with `--port`.

## Debug Output Helper

`debug_model_output.py` is a targeted inspection script. Use it when predictions
look collapsed or badly scaled and you need to compare normalized model output
against target tensors. It prints aggregate diagnostics rather than running a
full training loop.
