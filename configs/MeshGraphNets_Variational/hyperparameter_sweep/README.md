# MeshGraphNets-V SAOI parametric study

This directory now holds only the B300 round-2 SAOI arms, in
[`b300_r2/`](b300_r2/): `v_smp`, `v_grp`, `v_grp_rep` and `v_nopart`, each
on both board halves, with 3 official eval sets per half. They run beside
the matching cHI-MGNflow arms on one 8-GPU node, and none of them is run from
here.

The study design, the per-arm questions, the run command and how to read the
results are in
[`configs/campaigns/b300_r2/README.md`](../../campaigns/b300_r2/README.md).

The earlier latent-capacity sweep in this directory is retired. That was the
2026-09-28 redesign: arms `base`, `seed`, `zdim4`, `pmin30`, `arecon`,
`aux100`, `fmmom` and `g2e`, plus the benched `zdim8`, `pmin15` and `mmd10`.
It is in git history:

```bash
git show d4c63ba:configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md
```

The production recipe these arms start from is in [`../SAOI_run/`](../SAOI_run/).
