#!/usr/bin/env bash
# All-in-one TRAIN campaign for the ex4-ex9 public-dataset benchmark roster
# (configs/campaigns/benchmarks_all/roster.tsv): 42 arms across cylinder_flow,
# deforming_plate, flag_simple, AirfRANS (AoA extrapolation), Geo-FNO
# elasticity, and Geo-FNO plasticity.
#
# As of 2026-08-24 the roster is a COMPLETE 7-model x 6-dataset grid -- every
# mesh-contract method runs on every slot:
#
#     meshgraphnets  himgn  transolver  gino  fno  deeponet  point_deeponet
#
# It used to be skewed 3/3/3/3/6/7: ex4-ex7 carried only gino out of the four
# Neural_Operator aliases, and himgn existed on ex9 alone. Neither gap was
# structural (training_profiles/ar_rollout.py drives all four aliases
# identically, and the grid/sensor adapters handle per-sample varying node
# counts), so the 12 missing operator arms and 5 missing himgn arms were
# written; each new arm inherits its slot sibling's optimization block
# verbatim so the comparison isolates the architecture.
#
# NOT included, on purpose (same precedent as configs/campaigns/ex1/train_all.sh):
#   - SimulGenVAE: two independent blockers. (a) It has no `inference` mode at
#     all (`train`/`train_vae`/`train_lc`/`reconstruct`), so the `mode train`
#     -> `mode inference` patch campaign/runtime_config.py applies for the
#     infer campaign would emit an invalid config; and (b) its modes don't
#     write the rollout_sampleN_steps.h5 files score_rollouts.py reads.
#     Structurally it could only ever cover ex6/ex8/ex9 (fixed node count)
#     anyway -- ex4/ex5/ex7 have per-sample varying meshes. Configs for those
#     three slots exist under configs/SimulGenVAE/ and are run directly.
#   - MLP: no exN table exists for ex4-ex9 (tabular-only method).
#   - MeshGraphNets-variational: no ex4-ex9 configs by an earlier deliberate
#     call -- it targets one-to-many/stochastic problems, not the
#     deterministic simulations these six datasets contain.
#
# Fairness caveat when reading the scores: ex5/ex6 are contact problems and
# `use_world_edges` is MeshGraphNets-exclusive across this suite (Transolver
# refuses the flag at config-validation time; Neural_Operator accepts but
# ignores it). The non-MGN arms on those two slots train and score fine but
# are not modeling the contact physics.
#
# Thin wrapper: all the actual scheduling lives in campaign_runner.py, a
# dynamic shared-queue scheduler (every selected GPU pulls from one pending
# queue, admission gated on live nvidia-smi free VRAM, stuck jobs detected by
# log-file staleness and hard-killed) that replaced the old static
# round-robin "one job per GPU lane" design -- see campaign_runner.py's
# module docstring for the full rationale and every env var it reads.
#
# Environment overrides (unchanged from before, plus new scheduler knobs):
#   PYTHON          root launcher interpreter (default: python)
#   ROSTER          path to the label/config TSV (default: configs/campaigns/benchmarks_all/roster.tsv)
#   LABELS          space-separated subset of roster labels (default: all)
#   GPUS            space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PREFLIGHT       validate every runtime config before any training (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate the campaign without training (default: 0)
#   LOG_ROOT        campaign root (default: output/benchmarks_all/train_runs)
#   RUN_ID          run directory name (default: timestamp + PID)
#   STALL_TIMEOUT_MIN       no log growth for this long -> kill the job (default: 60)
#   VRAM_TARGET_UTIL_PERCENT admit a new job onto a GPU only while its current
#                           usage is below this % of total VRAM (default: 50)
#   MAX_CONCURRENCY_PER_GPU hard concurrency cap per GPU (default: 3)
#   ADMIT_WARMUP_SEC        cooldown after an admission before that GPU is
#                           reconsidered, so a job's real peak VRAM has time
#                           to materialize before the next check (default: 90)
#   POLL_INTERVAL_SEC       scheduler tick cadence (default: 20)
#
# Examples:
#   bash configs/campaigns/benchmarks_all/train_all.sh
#   CHECK_ONLY=1 bash configs/campaigns/benchmarks_all/train_all.sh
#   GPUS="0 1" bash configs/campaigns/benchmarks_all/train_all.sh
#   LABELS="meshgraphnets_ex8 transolver_ex8" bash configs/campaigns/benchmarks_all/train_all.sh
#   STALL_TIMEOUT_MIN=90 MAX_CONCURRENCY_PER_GPU=2 bash configs/campaigns/benchmarks_all/train_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

exec "$PYTHON" "$SCRIPT_DIR/campaign_runner.py" --mode train "$@"
