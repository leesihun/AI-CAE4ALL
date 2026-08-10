#!/usr/bin/env bash
# All-in-one TRAIN campaign for the transient ex2 dataset (dataset/ex2.h5).
#
# The default campaign contains eleven directly auditable arms:
#   - vanilla MeshGraphNets (AR-OT)
#   - HI-MGN as-is (AR-OT)
#   - HI-MGN base (AR-RT) -- the ancestor the P1/P2/P12 arms are generated from
#   - HI-MGN P1, P2, and P12 (AR-OT; transfer-operator / multi-partition study,
#     see MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md)
#   - Point-DeepONet, DeepONet, FNO, GINO, and Transolver (AR-OT)
#
# NOT included: SimulGenVAE and MLP have no ex2 config directory (ex2 is a
# transient point-cloud/graph dataset, not the fixed-mesh dense-FOM or tabular
# shape those two methods need). The config_train_abl_*.txt coarsening-ablation
# study lives under configs/MeshGraphNets/ex1/ only -- not applicable here.
#
# Every arm is launched through AI_CAE4ALL_main.py. Canonical configs are never
# edited: this script writes run-scoped copies with only gpu_ids substituted.
# With PARALLEL=1, one worker lane is started per requested GPU and jobs assigned
# to the same GPU run sequentially. The eleven defaults therefore run in two
# waves on an eight-GPU machine instead of oversubscribing any GPU.
#
# Environment overrides:
#   PYTHON          root launcher interpreter (default: python)
#   METHODS         space-separated campaign arms (default: all eleven)
#   GPUS            space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PARALLEL        1 = concurrent GPU lanes; 0 = fully sequential (default: 1)
#   PREFLIGHT       validate every runtime config before any training (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate the campaign without training (default: 0)
#   LOG_ROOT        campaign root (default: output/ex2_all/train_runs)
#   RUN_ID          run directory name (default: timestamp + shell PID)
#
# Examples:
#   bash configs/ex2/train_all.sh
#   CHECK_ONLY=1 bash configs/ex2/train_all.sh
#   GPUS="0 1 2 3" bash configs/ex2/train_all.sh
#   METHODS="meshgraphnets himgn-as-is himgn-base" PARALLEL=0 \
#       bash configs/ex2/train_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets himgn-as-is himgn-base himgn-p1 himgn-p2 himgn-p12 point_deeponet deeponet fno gino transolver}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PARALLEL="${PARALLEL:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_FLAGS="${PREFLIGHT_FLAGS:---strict}"
CHECK_ONLY="${CHECK_ONLY:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex2_all/train_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_ROOT="$LOG_ROOT/$RUN_ID"
RUNTIME_CONFIG_ROOT="$RUN_ROOT/runtime_configs"
mkdir -p "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)                 echo "configs/MeshGraphNets/ex2/config_train_meshgraphnets.txt" ;;
        himgn-as-is|meshgraphnets-hi)  echo "configs/MeshGraphNets/ex2/config_train_himgn.txt" ;;
        himgn-base|himgn-ar-rt)        echo "configs/MeshGraphNets/ex2/config_train_himgn_base.txt" ;;
        himgn-p1)                      echo "configs/MeshGraphNets/ex2/config_train_himgn_p1.txt" ;;
        himgn-p2)                      echo "configs/MeshGraphNets/ex2/config_train_himgn_p2.txt" ;;
        himgn-p12)                     echo "configs/MeshGraphNets/ex2/config_train_himgn_p12.txt" ;;
        point_deeponet)                echo "configs/Neural_Operator/ex2/config_train_point_deeponet.txt" ;;
        deeponet)                      echo "configs/Neural_Operator/ex2/config_train_deeponet.txt" ;;
        fno)                           echo "configs/Neural_Operator/ex2/config_train_fno.txt" ;;
        gino)                          echo "configs/Neural_Operator/ex2/config_train_gino.txt" ;;
        transolver)                    echo "configs/Transolver/ex2/config_train_transolver.txt" ;;
        *)                             return 1 ;;
    esac
}

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
    if ! grep -Eq '^[[:space:]]*gpu_ids[[:space:]]+' "$source_cfg"; then
        echo "ERROR: no gpu_ids entry in $source_cfg" >&2
        return 1
    fi
    # Rewrite only the value, preserving whitespace and any trailing comment.
    sed -E "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" \
        "$source_cfg" > "$out_cfg"
}

case "$PARALLEL:$PREFLIGHT:$CHECK_ONLY" in
    [01]:[01]:[01]) ;;
    *) echo "ERROR: PARALLEL, PREFLIGHT, and CHECK_ONLY must each be 0 or 1" >&2; exit 2 ;;
esac

read -r -a METHOD_LIST <<< "$METHODS"
read -r -a GPU_LIST <<< "$GPUS"
read -r -a PREFLIGHT_ARG_LIST <<< "$PREFLIGHT_FLAGS"

if [ "${#METHOD_LIST[@]}" -eq 0 ]; then
    echo "ERROR: METHODS is empty" >&2
    exit 2
fi
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi

declare -a CONFIG_LIST ASSIGNED_GPU RUNTIME_CONFIG_LIST LOG_LIST
MANIFEST="$RUN_ROOT/campaign.tsv"
printf 'method\tgpu\tcanonical_config\truntime_config\tlog\n' > "$MANIFEST"

for index in "${!METHOD_LIST[@]}"; do
    method="${METHOD_LIST[$index]}"
    if ! cfg="$(config_for "$method")"; then
        echo "ERROR: unknown method '$method'" >&2
        exit 2
    fi
    if [ ! -f "$cfg" ]; then
        echo "ERROR: config not found for '$method': $cfg" >&2
        exit 2
    fi

    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    rt_cfg="$RUNTIME_CONFIG_ROOT/train_${method}.txt"
    log="$RUN_ROOT/train_${method}.log"
    runtime_config "$cfg" "$gpu" "$rt_cfg" || exit 2

    CONFIG_LIST[$index]="$cfg"
    ASSIGNED_GPU[$index]="$gpu"
    RUNTIME_CONFIG_LIST[$index]="$rt_cfg"
    LOG_LIST[$index]="$log"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$method" "$gpu" "$cfg" "$rt_cfg" "$log" >> "$MANIFEST"
done

echo "ex2 all-in-one training campaign"
echo "  PYTHON    = $PYTHON"
echo "  METHODS   = $METHODS"
echo "  GPUS      = $GPUS"
echo "  PARALLEL  = $PARALLEL (one sequential lane per GPU)"
echo "  PREFLIGHT = $PREFLIGHT ($PREFLIGHT_FLAGS)"
echo "  CHECK_ONLY= $CHECK_ONLY"
echo "  RUN_ROOT  = $RUN_ROOT"
echo "  assignment:"
for index in "${!METHOD_LIST[@]}"; do
    printf '    gpu=%-3s %-18s %s\n' \
        "${ASSIGNED_GPU[$index]}" "${METHOD_LIST[$index]}" "${CONFIG_LIST[$index]}"
done

if [ "$PREFLIGHT" = "1" ]; then
    echo ""
    echo "Preflighting all ${#METHOD_LIST[@]} runtime configs before launch..."
    for index in "${!METHOD_LIST[@]}"; do
        method="${METHOD_LIST[$index]}"
        if ! "$PYTHON" AI_CAE4ALL_main.py \
            --config "${RUNTIME_CONFIG_LIST[$index]}" --check \
            "${PREFLIGHT_ARG_LIST[@]}" > "${LOG_LIST[$index]}.preflight" 2>&1; then
            echo "[$method] PREFLIGHT FAILED -- see ${LOG_LIST[$index]}.preflight" >&2
            exit 1
        fi
        echo "[$method] PREFLIGHT PASS"
    done
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY complete; no training was launched."
    echo "Manifest: $MANIFEST"
    exit 0
fi

train_one() {
    local index=$1
    local method="${METHOD_LIST[$index]}"
    local gpu="${ASSIGNED_GPU[$index]}"
    local rt_cfg="${RUNTIME_CONFIG_LIST[$index]}"
    local log="${LOG_LIST[$index]}"

    echo "[$method] TRAIN START gpu=$gpu cfg=$rt_cfg -> $log"
    if [ "$PARALLEL" = "1" ]; then
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" > "$log" 2>&1
    else
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" 2>&1 | tee "$log"
    fi
}

run_lane() {
    local lane=$1
    local lane_rc=0
    local index
    for index in "${!METHOD_LIST[@]}"; do
        if [ $((index % ${#GPU_LIST[@]})) -ne "$lane" ]; then
            continue
        fi
        if train_one "$index"; then
            echo "[${METHOD_LIST[$index]}] TRAIN DONE"
        else
            echo "[${METHOD_LIST[$index]}] TRAIN FAILED -- see ${LOG_LIST[$index]}" >&2
            lane_rc=1
        fi
    done
    return "$lane_rc"
}

started=$(date +%s)
rc=0
if [ "$PARALLEL" = "1" ]; then
    lane_count=${#GPU_LIST[@]}
    if [ "$lane_count" -gt "${#METHOD_LIST[@]}" ]; then
        lane_count=${#METHOD_LIST[@]}
    fi
    pids=()
    lanes=()
    for ((lane=0; lane<lane_count; lane++)); do
        run_lane "$lane" &
        pids+=("$!")
        lanes+=("$lane")
        echo "  launched lane=$lane gpu=${GPU_LIST[$lane]} pid=$!"
    done
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "GPU lane ${lanes[$index]} failed" >&2
            rc=1
        fi
    done
else
    for index in "${!METHOD_LIST[@]}"; do
        if ! train_one "$index"; then
            echo "[${METHOD_LIST[$index]}] TRAIN FAILED -- see ${LOG_LIST[$index]}" >&2
            rc=1
        else
            echo "[${METHOD_LIST[$index]}] TRAIN DONE"
        fi
    done
fi

ended=$(date +%s)
echo ""
echo "ex2 all-in-one training finished in $((ended - started))s (rc=$rc)."
echo "Manifest:        $MANIFEST"
echo "Transcripts:     $RUN_ROOT/train_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
if [ "$rc" = "0" ]; then
    echo "Next: bash configs/ex2/infer_all.sh"
fi
exit "$rc"
