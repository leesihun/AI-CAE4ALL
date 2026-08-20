#!/usr/bin/env bash
# All-in-one INFERENCE campaign for the ex4-ex9 benchmark roster
# (configs/benchmarks_all/roster.tsv). Requires checkpoints already produced
# by configs/benchmarks_all/train_all.sh (or equivalent) at each config's
# `modelpath`.
#
# Each arm's canonical TRAIN config is never edited: campaign_runner.py
# derives a run-scoped inference copy by rewriting `mode train` ->
# `mode inference` and patching `gpu_ids`, same as before. `dataset_dir`
# stays the train file (inert in inference mode) and `infer_dataset`/
# `inference_output_dir` are already the exN_infer.h5 ground truth and a
# unique per-arm rollout directory in every roster config, so nothing else
# needs to change. Each method's rollout.py writes
# `<inference_output_dir>/rollout_sample<id>_steps<N>.h5`, which
# configs/benchmarks_all/score_rollouts.py then reads.
#
# Thin wrapper: all the actual scheduling lives in campaign_runner.py, a
# dynamic shared-queue scheduler shared with train_all.sh -- see its module
# docstring for the full design (live-VRAM-aware admission across every GPU,
# log-staleness stall detection + forceful process-tree kill) and every env
# var it reads.
#
# Deliberate behavior change from before: this script now supports
# PREFLIGHT/CHECK_ONLY (it never did previously), and per-job logs now live
# under LOG_ROOT/RUN_ID like train_all.sh instead of flat under LOG_ROOT --
# both are direct consequences of sharing one scheduler core with train_all.sh.
#
# Environment overrides:
#   PYTHON          python interpreter (default: python)
#   ROSTER          path to the label/config TSV (default: configs/benchmarks_all/roster.tsv)
#   LABELS          space-separated subset of roster labels (default: all)
#   GPUS            space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PREFLIGHT       validate every runtime config before inference (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate without running inference (default: 0)
#   LOG_ROOT        campaign root (default: output/benchmarks_all/infer_runs)
#   RUN_ID          run directory name (default: timestamp + PID)
#   STALL_TIMEOUT_MIN       no log growth for this long -> kill the job (default: 60)
#   VRAM_TARGET_UTIL_PERCENT admit a new job onto a GPU only while its current
#                           usage is below this % of total VRAM (default: 50)
#   MAX_CONCURRENCY_PER_GPU hard concurrency cap per GPU (default: 3)
#   ADMIT_WARMUP_SEC        cooldown after an admission before that GPU is
#                           reconsidered (default: 90)
#   POLL_INTERVAL_SEC       scheduler tick cadence (default: 20)
#
# Usage:
#   bash configs/benchmarks_all/infer_all.sh
#   LABELS="meshgraphnets_ex8 transolver_ex8" bash configs/benchmarks_all/infer_all.sh
#
# Then: python configs/benchmarks_all/score_rollouts.py

set -uo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

exec "$PYTHON" "$SCRIPT_DIR/campaign_runner.py" --mode infer "$@"
