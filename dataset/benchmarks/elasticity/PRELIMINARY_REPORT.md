# Supplemental Elasticity run (archived status)

The earlier five-method shared-Elasticity exercise is no longer the primary
accuracy validation. Only Transolver uses this dataset in its own paper; using
it for Point-DeepONet, DeepONet, FNO, or GINO cannot establish direct agreement
with those implementations' papers.

The common run was stopped after preserving these supplemental checkpoints:

| Method | Epoch | Train normalized MSE | Validation normalized MSE |
|---|---:|---:|---:|
| Point-DeepONet | 499 | 0.0036094970 | 0.0082491396 |
| DeepONet | 499 | 0.0265681289 | 0.0375064024 |
| FNO | 499 | 0.0012305783 | 0.0062580291 |
| GINO | 50 | 0.0094984498 | 0.0139381740 |

These are normalized MSE values and must not be compared numerically with
paper relative-L2 results. The exact 200-case Transolver own-paper run is
queued separately.

See the current authoritative status and per-paper applicability decisions in
[`../PER_PAPER_VALIDATION_REPORT.md`](../PER_PAPER_VALIDATION_REPORT.md).
