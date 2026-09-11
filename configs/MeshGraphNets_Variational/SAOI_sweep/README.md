# MeshGraphNets-V — SAOI warpage sweep

Eight arms, one per card 0–7. **A `2^3` full factorial over the prior:
section × prior fit × prior trunk.**

```bash
nohup bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run.out 2>&1 &
tail -f output/meshgraphnets-v/saoi_sweep/run.out
```

One command: preflight → train (8 parallel) → inference + deterministic
control → ranking → document.

| arm | card | section | prior fit | prior trunk |
| --- | --- | --- | --- | --- |
| 1 | 0 | bot | joint | small — 256 / 5 MP |
| 2 | 1 | bot | joint | large — 512 / 8 MP |
| 3 | 2 | bot | **tail** (freeze at 700) | small |
| 4 | 3 | bot | **tail** | large |
| 5 | 4 | top | joint | small |
| 6 | 5 | top | joint | large |
| 7 | 6 | top | **tail** | small |
| 8 | 7 | top | **tail** | large |

1000 epochs, batch 16, `best_by crps`, `beta_aux 10` held constant,
`prior_grad_to_encoder 0` on every arm.

## Why the prior

Posterior reconstruction is excellent while the generated distribution is ~2×
too narrow (`sd_ratio` 0.45–0.53 on every arm of the previous grid). Training
decodes `z ~ q(z | y, g)`; deployment decodes `z ~ p(z | g)`. **The decoder is
common to both paths.** If the posterior path reproduces a part's spread and the
prior path does not, the prior's conditional for that part is what is wrong —
and `misc/posterior_vs_prior.py` measures exactly that split on the eval sets.

Three mechanisms can make the prior's conditional too narrow; the two factors
here address all three.

**`prior fit`: joint vs tail.** In a joint run the encoder keeps receiving
reconstruction gradients, so `q` drifts for the whole run and the prior chases a
*moving target*. The `tail` level sets `prior_freeze_epoch 700`: at that epoch
every simulator parameter is frozen, **latent standardization** is fitted from
the now-fixed posterior (`z_shift` / `z_scale` — MMD pins the *aggregate*
`q(z)` to N(0,I), and with ~100 realizations of each of a few parts that
aggregate is a mixture whose per-part clouds are far smaller and off-origin),
and a fresh cosine over the prior's own parameters runs the last 300 epochs.
Those epochs are *cheaper* than joint ones: posterior encoder under `no_grad`,
no decoder at all.

**`prior trunk`: small vs large.** `prior_hidden_dim 256 / prior_mp_layers 5`
is much smaller than the main network. A trunk that cannot separate parts learns
a conditional smeared across neighbours.

`prior_grad_to_encoder` is closed everywhere: open, the FM objective can lower
itself by *shrinking the target distribution* (integrated CFM loss is `πa/2` for
target scale `a`), and a sweep about fitting the prior *to* the posterior must
not be moving the posterior *toward* the prior. The previous grid measured the
flag at Δ 0.001.

## What is held fixed, and why

`beta_aux 10` — the peak-to-valley term on the decoded field, at the balance
point estimated from realistic recon/aux magnitudes. It is the `I(z;y)` floor
that keeps `z` informative now that the old z-readout head is gone, so it stays
**on**; it stays **constant** so it is not a factor. Its own dose-response is a
different sweep.

The previous `2^(4-1)` grid moved `W1/sd` by 0.001–0.054 across
`z_conditioning`, `prior_grad_to_encoder`, capacity and `lambda_mmd`, against a
0.10 arm-to-arm range. None of those is spent again.

## Reading the result

`docs/research/SAOI_SWEEP_MGNV.md`, written automatically when the run ends.
The metric is **`sd_ratio`, target 1** — each eval set is one geometry with 125
realizations, so the truth *is* the conditional distribution.

Because the design is a full factorial, read the two main effects and their
interaction directly: does `tail` widen the ensemble on both sections? does
`large` help on its own, or only together with `tail`? The **Deterministic
control** section shows each arm's pure bias (prior at temperature ~0), which is
what separates a shifted conditional mean from a narrow ensemble.

For any arm, `misc/posterior_vs_prior.py --config <its infer config>` gives the
posterior-vs-prior split on one geometry, in latent space and in decoded
peak-to-valley.

## Files

| | |
| --- | --- |
| `gen_configs.py` | generates every config from `../SAOI_all_input/`. **Do not hand-edit the configs** — regenerate |
| `run_sweep.sh` | the one command above |
| `check_eval_inputs.py` | comprehensive validator for the `_infer_`/`_compare_` pairing |

Shared tooling in `configs/campaigns/`: `sweep_common.py` (the renderer both
sweeps use), `rank_arms.py`, `write_report.py`.

## If an arm dies

A failing arm is **dropped, not fatal**; everything dropped is reprinted at the
end so a shortened sweep cannot be mistaken for a complete one. Re-run just
those: `ARMS="3 7" bash .../run_sweep.sh`. There is no resume, and
`cosine_T0 = epochs − warmup`, so a killed arm restarts from zero.

When the sweep is done, delete the shared hierarchy cache:

```bash
rm dataset/SAOI/saoi_train_{bot,top}.mscache.*.h5
```
