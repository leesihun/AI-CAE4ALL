#!/usr/bin/env bash
# All-in-one INFERENCE campaign for the ex4-ex9 benchmark roster
# (configs/benchmarks_all/roster.tsv). Requires checkpoints already produced
# by configs/benchmarks_all/train_all.sh (or equivalent) at each config's
# `modelpath`.
#
# Each arm's canonical TRAIN config is never edited: this script derives a
# run-scoped inference copy by rewriting `mode train` -> `mode inference` and
# patching `gpu_ids`, same as configs/ex1/infer_all.sh's `runtime_config()`.
# `dataset_dir` stays the train file (inert in inference mode) and
# `infer_dataset`/`inference_output_dir` are already the exN_infer.h5 ground
# truth and a unique per-arm rollout directory in every roster config, so
# nothing else needs to change. Each method's rollout.py writes
# `<inference_output_dir>/rollout_sample<id>_steps<N>.h5`, which
# configs/benchmarks_all/score_rollouts.py then reads.
#
# Lane semantics (parallel across GPUS, queued within a lane) are identical
# to train_all.sh -- see its header for the full explanation.
#
# Environment overrides:
#   PYTHON   = python interpreter (default: python)
#   ROSTER   = path to the label/config TSV (default: configs/benchmarks_all/roster.tsv)
#   LABELS   = space-separated subset of roster labels (default: all)
#   GPUS     = space-separated CUDA IDs, one lane each (default: 0 1 2 3 4 5 6 7)
#   LOG_ROOT = directory for transcript logs (default: output/benchmarks_all/infer_runs)
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

ROSTER="${ROSTER:-configs/benchmarks_all/roster.tsv}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

LOG_ROOT="${LOG_ROOT:-output/benchmarks_all/infer_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

if [ ! -f "$ROSTER" ]; then
    echo "ERROR: roster not found: $ROSTER" >&2
    exit 2
fi

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
    if ! grep -Eq '^[[:space:]]*(mode)[[:space:]]+train[[:space:]]*$' "$source_cfg"; then
        echo "ERROR: no 'mode ... train' line found in $source_cfg (already non-canonical?)" >&2
        return 1
    fi
    sed -E \
        -e "s/^([[:space:]]*mode[[:space:]]+)train([[:space:]]*)\$/\1inference\2/" \
        -e "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" \
        "$source_cfg" > "$out_cfg"
}

declare -a ALL_LABELS ALL_CONFIGS
{
    read -r _header
    while IFS=$'\t' read -r label cfg _slot _light; do
        [ -z "$label" ] && continue
        ALL_LABELS+=("$label")
        ALL_CONFIGS+=("$cfg")
    done
} < "$ROSTER"

declare -a LABEL_LIST CONFIG_LIST
if [ -n "${LABELS:-}" ]; then
    read -r -a WANT <<< "$LABELS"
    for want in "${WANT[@]}"; do
        found=0
        for i in "${!ALL_LABELS[@]}"; do
            if [ "${ALL_LABELS[$i]}" = "$want" ]; then
                LABEL_LIST+=("${ALL_LABELS[$i]}")
                CONFIG_LIST+=("${ALL_CONFIGS[$i]}")
                found=1
                break
            fi
        done
        [ "$found" = 0 ] && { echo "ERROR: unknown label '$want' (see $ROSTER)" >&2; exit 2; }
    done
else
    LABEL_LIST=("${ALL_LABELS[@]}")
    CONFIG_LIST=("${ALL_CONFIGS[@]}")
fi

n="${#LABEL_LIST[@]}"
[ "$n" -eq 0 ] && { echo "ERROR: no roster entries selected" >&2; exit 2; }

declare -a ASSIGNED_GPU RUNTIME_CFG LOG_FILE
declare -a LANE_MEMBERS
for g in "${!GPU_LIST[@]}"; do LANE_MEMBERS[$g]=""; done

for i in "${!LABEL_LIST[@]}"; do
    label="${LABEL_LIST[$i]}"
    cfg="${CONFIG_LIST[$i]}"
    if [ ! -f "$cfg" ]; then
        echo "[$label] SKIP: config not found ($cfg)" >&2
        continue
    fi
    lane=$(( i % ${#GPU_LIST[@]} ))
    gpu="${GPU_LIST[$lane]}"
    rt_cfg="$RUNTIME_CONFIG_ROOT/infer_${label}.txt"
    if ! runtime_config "$cfg" "$gpu" "$rt_cfg"; then
        continue
    fi
    ASSIGNED_GPU[$i]="$gpu"
    RUNTIME_CFG[$i]="$rt_cfg"
    LOG_FILE[$i]="$LOG_ROOT/infer_${label}.log"
    LANE_MEMBERS[$lane]="${LANE_MEMBERS[$lane]} $i"
done

echo "benchmarks_all inference campaign ($n arms)"
echo "  PYTHON   = $PYTHON"
echo "  GPUS     = $GPUS  (${#GPU_LIST[@]} lanes, queued round-robin)"
echo "  LOG_ROOT = $LOG_ROOT"
echo "  RUN_ID   = $RUN_ID"
echo "  lane assignment:"
for g in "${!GPU_LIST[@]}"; do
    members="${LANE_MEMBERS[$g]}"
    [ -z "$members" ] && continue
    names=""
    for idx in $members; do names="$names ${LABEL_LIST[$idx]}"; done
    printf '    gpu=%-3s :%s\n' "${GPU_LIST[$g]}" "$names"
done

lane_worker() {
    local lane=$1
    local rc=0
    for idx in ${LANE_MEMBERS[$lane]}; do
        local label="${LABEL_LIST[$idx]}"
        local rt_cfg="${RUNTIME_CFG[$idx]}"
        local log="${LOG_FILE[$idx]}"
        echo "[$label] INFER START gpu=${ASSIGNED_GPU[$idx]} cfg=$rt_cfg -> $log"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" > "$log" 2>&1; then
            echo "[$label] INFER DONE"
        else
            echo "[$label] INFER FAILED -- see $log" >&2
            rc=1
        fi
    done
    return $rc
}

started=$(date +%s)
rc=0
pids=()
for g in "${!GPU_LIST[@]}"; do
    [ -z "${LANE_MEMBERS[$g]}" ] && continue
    lane_worker "$g" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

ended=$(date +%s)
echo ""
echo "benchmarks_all inference finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:     $LOG_ROOT/infer_<label>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
if [ "$rc" = "0" ]; then
    echo "Next: python configs/benchmarks_all/score_rollouts.py"
fi
exit "$rc"
