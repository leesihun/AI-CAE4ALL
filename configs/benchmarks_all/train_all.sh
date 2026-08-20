#!/usr/bin/env bash
# All-in-one TRAIN campaign for the ex4-ex9 public-dataset benchmark roster
# (configs/benchmarks_all/roster.tsv): 25 arms across cylinder_flow,
# deforming_plate, flag_simple, AirfRANS (AoA extrapolation), Geo-FNO
# elasticity, and Geo-FNO plasticity -- one config per method per dataset
# (ex8/ex9 additionally cover all four Neural_Operator aliases).
#
# NOT included, on purpose (same precedent as configs/ex1/train_all.sh):
#   - SimulGenVAE: its `train_vae`/`reconstruct` modes don't write the
#     rollout_sampleN_steps.h5 files configs/benchmarks_all/score_rollouts.py
#     expects, and ex4/ex5/ex7 are structurally incompatible anyway
#     (per-sample varying mesh) -- see dataset/PUBLIC_DATASETS.md.
#   - MLP: no exN table exists for ex4-ex9 (tabular-only method).
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
#   ROSTER          path to the label/config TSV (default: configs/benchmarks_all/roster.tsv)
#   LABELS          space-separated subset of roster labels (default: all)
#   GPUS            space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PREFLIGHT       validate every runtime config before any training (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate the campaign without training (default: 0)
#   LOG_ROOT        campaign root (default: output/benchmarks_all/train_runs)
#   RUN_ID          run directory name (default: timestamp + PID)
#   STALL_TIMEOUT_MIN       no log growth for this long -> kill the job (default: 30)
#   VRAM_TARGET_UTIL_PERCENT admit a new job onto a GPU only while its current
#                           usage is below this % of total VRAM (default: 50)
#   MAX_CONCURRENCY_PER_GPU hard concurrency cap per GPU (default: 3)
#   ADMIT_WARMUP_SEC        cooldown after an admission before that GPU is
#                           reconsidered, so a job's real peak VRAM has time
#                           to materialize before the next check (default: 90)
#   POLL_INTERVAL_SEC       scheduler tick cadence (default: 20)
#
# Examples:
#   bash configs/benchmarks_all/train_all.sh
#   CHECK_ONLY=1 bash configs/benchmarks_all/train_all.sh
#   GPUS="0 1" bash configs/benchmarks_all/train_all.sh
#   LABELS="meshgraphnets_ex8 transolver_ex8" bash configs/benchmarks_all/train_all.sh
#   STALL_TIMEOUT_MIN=45 MAX_CONCURRENCY_PER_GPU=2 bash configs/benchmarks_all/train_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

exec "$PYTHON" "$SCRIPT_DIR/campaign_runner.py" --mode train "$@"
