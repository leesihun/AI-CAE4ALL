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
# Every arm is assigned to one of GPUS "lanes". Jobs sharing a lane run
# SEQUENTIALLY (queued, one at a time, to avoid VRAM contention); different
# lanes run in PARALLEL -- this is the "simultaneously across 8 GPUs"
# semantics. Assignment is round-robin over the roster order, independent of
# the roster's `light` column; light (low-VRAM Neural_Operator) arms are only
# used to break ties when hand-tuning -- see LIGHT_GPUS below if you want them
# packed onto dedicated lanes instead.
#
# Environment overrides:
#   PYTHON          root launcher interpreter (default: python)
#   ROSTER          path to the label/config TSV (default: configs/benchmarks_all/roster.tsv)
#   LABELS          space-separated subset of roster labels (default: all)
#   GPUS            space-separated CUDA IDs, one lane each (default: 0 1 2 3 4 5 6 7)
#   PREFLIGHT       validate every runtime config before any training (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate the campaign without training (default: 0)
#   LOG_ROOT        campaign root (default: output/benchmarks_all/train_runs)
#   RUN_ID          run directory name (default: timestamp + shell PID)
#
# Examples:
#   bash configs/benchmarks_all/train_all.sh
#   CHECK_ONLY=1 bash configs/benchmarks_all/train_all.sh
#   GPUS="0 1" bash configs/benchmarks_all/train_all.sh          # 25 arms queued across 2 lanes
#   LABELS="meshgraphnets_ex8 transolver_ex8" bash configs/benchmarks_all/train_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ROSTER="${ROSTER:-configs/benchmarks_all/roster.tsv}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_FLAGS="${PREFLIGHT_FLAGS:---strict}"
CHECK_ONLY="${CHECK_ONLY:-0}"

LOG_ROOT="${LOG_ROOT:-output/benchmarks_all/train_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_ROOT="$LOG_ROOT/$RUN_ID"
RUNTIME_CONFIG_ROOT="$RUN_ROOT/runtime_configs"
mkdir -p "$RUNTIME_CONFIG_ROOT"

if [ ! -f "$ROSTER" ]; then
    echo "ERROR: roster not found: $ROSTER" >&2
    exit 2
fi

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi

case "$PREFLIGHT:$CHECK_ONLY" in
    [01]:[01]) ;;
    *) echo "ERROR: PREFLIGHT and CHECK_ONLY must each be 0 or 1" >&2; exit 2 ;;
esac

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
    if ! grep -Eq '^[[:space:]]*gpu_ids[[:space:]]+' "$source_cfg"; then
        echo "ERROR: no gpu_ids entry in $source_cfg" >&2
        return 1
    fi
    sed -E "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" \
        "$source_cfg" > "$out_cfg"
}

# ── load roster, filter to LABELS if given, assign round-robin GPU lanes ──
declare -a ALL_LABELS ALL_CONFIGS ALL_SLOTS
{
    read -r _header
    while IFS=$'\t' read -r label cfg slot light; do
        [ -z "$label" ] && continue
        ALL_LABELS+=("$label")
        ALL_CONFIGS+=("$cfg")
        ALL_SLOTS+=("$slot")
    done
} < "$ROSTER"

declare -a LABEL_LIST CONFIG_LIST SLOT_LIST
if [ -n "${LABELS:-}" ]; then
    read -r -a WANT <<< "$LABELS"
    for want in "${WANT[@]}"; do
        found=0
        for i in "${!ALL_LABELS[@]}"; do
            if [ "${ALL_LABELS[$i]}" = "$want" ]; then
                LABEL_LIST+=("${ALL_LABELS[$i]}")
                CONFIG_LIST+=("${ALL_CONFIGS[$i]}")
                SLOT_LIST+=("${ALL_SLOTS[$i]}")
                found=1
                break
            fi
        done
        [ "$found" = 0 ] && { echo "ERROR: unknown label '$want' (see $ROSTER)" >&2; exit 2; }
    done
else
    LABEL_LIST=("${ALL_LABELS[@]}")
    CONFIG_LIST=("${ALL_CONFIGS[@]}")
    SLOT_LIST=("${ALL_SLOTS[@]}")
fi

n="${#LABEL_LIST[@]}"
if [ "$n" -eq 0 ]; then
    echo "ERROR: no roster entries selected" >&2
    exit 2
fi

declare -a ASSIGNED_GPU RUNTIME_CFG LOG_FILE
declare -a LANE_MEMBERS  # LANE_MEMBERS[g] = space-separated indices queued on GPU_LIST[g]
for g in "${!GPU_LIST[@]}"; do LANE_MEMBERS[$g]=""; done

MANIFEST="$RUN_ROOT/campaign.tsv"
printf 'label\tex_slot\tgpu\tcanonical_config\truntime_config\tlog\n' > "$MANIFEST"

for i in "${!LABEL_LIST[@]}"; do
    label="${LABEL_LIST[$i]}"
    cfg="${CONFIG_LIST[$i]}"
    if [ ! -f "$cfg" ]; then
        echo "ERROR: config not found for '$label': $cfg" >&2
        exit 2
    fi
    lane=$(( i % ${#GPU_LIST[@]} ))
    gpu="${GPU_LIST[$lane]}"
    rt_cfg="$RUNTIME_CONFIG_ROOT/train_${label}.txt"
    log="$RUN_ROOT/train_${label}.log"
    runtime_config "$cfg" "$gpu" "$rt_cfg" || exit 2

    ASSIGNED_GPU[$i]="$gpu"
    RUNTIME_CFG[$i]="$rt_cfg"
    LOG_FILE[$i]="$log"
    LANE_MEMBERS[$lane]="${LANE_MEMBERS[$lane]} $i"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "${SLOT_LIST[$i]}" "$gpu" "$cfg" "$rt_cfg" "$log" >> "$MANIFEST"
done

echo "benchmarks_all training campaign ($n arms)"
echo "  PYTHON       = $PYTHON"
echo "  ROSTER       = $ROSTER"
echo "  GPUS         = $GPUS  (${#GPU_LIST[@]} lanes, queued round-robin)"
echo "  PREFLIGHT    = $PREFLIGHT ($PREFLIGHT_FLAGS)"
echo "  CHECK_ONLY   = $CHECK_ONLY"
echo "  RUN_ROOT     = $RUN_ROOT"
echo "  lane assignment:"
for g in "${!GPU_LIST[@]}"; do
    members="${LANE_MEMBERS[$g]}"
    [ -z "$members" ] && continue
    names=""
    for idx in $members; do names="$names ${LABEL_LIST[$idx]}"; done
    printf '    gpu=%-3s :%s\n' "${GPU_LIST[$g]}" "$names"
done

if [ "$PREFLIGHT" = "1" ]; then
    echo ""
    echo "Preflighting all $n runtime configs before launch..."
    read -r -a PREFLIGHT_ARG_LIST <<< "$PREFLIGHT_FLAGS"
    for i in "${!LABEL_LIST[@]}"; do
        label="${LABEL_LIST[$i]}"
        if ! "$PYTHON" AI_CAE4ALL_main.py \
            --config "${RUNTIME_CFG[$i]}" --check \
            "${PREFLIGHT_ARG_LIST[@]}" > "${LOG_FILE[$i]}.preflight" 2>&1; then
            echo "[$label] PREFLIGHT FAILED -- see ${LOG_FILE[$i]}.preflight" >&2
            exit 1
        fi
        echo "[$label] PREFLIGHT PASS"
    done
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY complete; no training was launched."
    echo "Manifest: $MANIFEST"
    exit 0
fi

lane_worker() {
    local lane=$1
    local rc=0
    for idx in ${LANE_MEMBERS[$lane]}; do
        local label="${LABEL_LIST[$idx]}"
        local rt_cfg="${RUNTIME_CFG[$idx]}"
        local log="${LOG_FILE[$idx]}"
        echo "[$label] TRAIN START gpu=${ASSIGNED_GPU[$idx]} cfg=$rt_cfg -> $log"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" > "$log" 2>&1; then
            echo "[$label] TRAIN DONE"
        else
            echo "[$label] TRAIN FAILED -- see $log" >&2
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
echo "benchmarks_all training finished in $((ended - started))s (rc=$rc)."
echo "Manifest:        $MANIFEST"
echo "Transcripts:     $RUN_ROOT/train_<label>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
if [ "$rc" = "0" ]; then
    echo "Next: bash configs/benchmarks_all/infer_all.sh"
fi
exit "$rc"
