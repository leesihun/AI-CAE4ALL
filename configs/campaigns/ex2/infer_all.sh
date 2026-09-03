#!/usr/bin/env bash
# Head-to-head INFERENCE runner for the transient ex2 dataset (dataset/ex2.h5).
#
# Runs the same eleven arms as configs/campaigns/ex2/train_all.sh (must match exactly,
# so every checkpoint that gets trained also gets scored), ALL AT ONCE, against
# the held-out dataset/ex2_infer.h5: a 49-step autoregressive rollout seeded
# from t=0 alone, with the stored t=1..49 trajectory serving as untouched
# ground truth. Outputs land in output/<method>/rollout/ex2/... in a directly
# comparable layout. Requires checkpoints already produced by
# configs/campaigns/ex2/train_all.sh (or equivalent).
#
# Score the rollouts against ground truth with:
#   python configs/campaigns/ex2/score_rollouts.py
#
# GPU assignment mirrors configs/campaigns/ex2/train_all.sh: Point-DeepONet, DeepONet,
# and FNO are the low-VRAM arms and are packed onto GPUs from LIGHT_GPUS
# independently of the "heavy" arms, so no GPU hosts more than one heavy
# process by default.
#
# Environment overrides:
#   PYTHON        = python interpreter (default: python)
#   METHODS       = space-separated method list (default: all eleven, see config_for())
#   GPUS          = space-separated CUDA IDs for heavy arms (default: 0 1 2 3 4 5 6 7)
#   LIGHT_METHODS = space-separated method keys treated as low-VRAM
#                   (default: point_deeponet deeponet fno)
#   LIGHT_GPUS    = space-separated CUDA IDs for light arms (default: same as GPUS)
#   PARALLEL      = 1 launch all at once then wait (default); 0 run sequentially
#   LOG_ROOT      = directory for transcript logs (default: output/ex2_all/infer_runs)
#
# Usage:
#   bash configs/campaigns/ex2/infer_all.sh
#   METHODS="meshgraphnets himgn-as-is himgn-base" bash configs/campaigns/ex2/infer_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets himgn-as-is himgn-base himgn-p1 himgn-p2 himgn-p12 point_deeponet deeponet fno gino transolver}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LIGHT_METHODS="${LIGHT_METHODS:-point_deeponet deeponet fno}"
LIGHT_GPUS="${LIGHT_GPUS:-$GPUS}"
PARALLEL="${PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex2_all/infer_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)                 echo "configs/MeshGraphNets/ex2/config_infer_meshgraphnets.txt" ;;
        himgn-as-is|meshgraphnets-hi)  echo "configs/MeshGraphNets/ex2/config_infer_himgn.txt" ;;
        himgn-base|himgn-ar-rt)        echo "configs/MeshGraphNets/ex2/config_infer_himgn_base.txt" ;;
        himgn-p1)                      echo "configs/MeshGraphNets/ex2/config_infer_himgn_p1.txt" ;;
        himgn-p2)                      echo "configs/MeshGraphNets/ex2/config_infer_himgn_p2.txt" ;;
        himgn-p12)                     echo "configs/MeshGraphNets/ex2/config_infer_himgn_p12.txt" ;;
        point_deeponet)                echo "configs/Neural_Operator/ex2/config_infer_point_deeponet.txt" ;;
        deeponet)                      echo "configs/Neural_Operator/ex2/config_infer_deeponet.txt" ;;
        fno)                           echo "configs/Neural_Operator/ex2/config_infer_fno.txt" ;;
        gino)                          echo "configs/Neural_Operator/ex2/config_infer_gino.txt" ;;
        transolver)                    echo "configs/Transolver/ex2/config_infer_transolver.txt" ;;
        *) echo "" ;;
    esac
}

is_light() {
    local method=$1 m
    for m in $LIGHT_METHODS; do
        [ "$m" = "$method" ] && return 0
    done
    return 1
}

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
    sed -E "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" "$source_cfg" > "$out_cfg"
}

read -r -a GPU_LIST <<< "$GPUS"
read -r -a LIGHT_GPU_LIST <<< "$LIGHT_GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi
if [ "${#LIGHT_GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: LIGHT_GPUS is empty" >&2
    exit 2
fi

read -r -a METHOD_LIST <<< "$METHODS"
declare -a ASSIGNED_GPU WEIGHT_LIST
heavy_idx=0
light_idx=0
for index in "${!METHOD_LIST[@]}"; do
    method="${METHOD_LIST[$index]}"
    if is_light "$method"; then
        WEIGHT_LIST[$index]="light"
        ASSIGNED_GPU[$index]="${LIGHT_GPU_LIST[$((light_idx % ${#LIGHT_GPU_LIST[@]}))]}"
        light_idx=$((light_idx + 1))
    else
        WEIGHT_LIST[$index]="heavy"
        ASSIGNED_GPU[$index]="${GPU_LIST[$((heavy_idx % ${#GPU_LIST[@]}))]}"
        heavy_idx=$((heavy_idx + 1))
    fi
done

infer_one() {
    local index=$1
    local method="${METHOD_LIST[$index]}"
    local gpu="${ASSIGNED_GPU[$index]}"
    local cfg rt_cfg log
    cfg="$(config_for "$method")"
    if [ -z "$cfg" ]; then
        echo "[$method] SKIP: unknown method" >&2
        return 0
    fi
    if [ ! -f "$cfg" ]; then
        echo "[$method] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    rt_cfg="$RUNTIME_CONFIG_ROOT/infer_${method}.txt"
    runtime_config "$cfg" "$gpu" "$rt_cfg"
    log="$LOG_ROOT/infer_${method}.log"
    echo "[$method] INFER START  gpu=$gpu  cfg=$rt_cfg (from $cfg)  -> $log"
    if [ "$PARALLEL" = "1" ]; then
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" > "$log" 2>&1
    else
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" 2>&1 | tee "$log"
    fi
}

started=$(date +%s)
echo "ex2 infer-all"
echo "  PYTHON       = $PYTHON"
echo "  METHODS      = $METHODS"
echo "  GPUS         = $GPUS"
echo "  LIGHT_METHODS= $LIGHT_METHODS"
echo "  LIGHT_GPUS   = $LIGHT_GPUS"
echo "  PARALLEL     = $PARALLEL"
echo "  LOG_ROOT     = $LOG_ROOT"
echo "  RUN_ID       = $RUN_ID"

rc=0
if [ "$PARALLEL" = "1" ]; then
    pids=()
    for index in "${!METHOD_LIST[@]}"; do
        infer_one "$index" &
        pids+=("$!")
        echo "  launched ${METHOD_LIST[$index]} (pid $!, gpu ${ASSIGNED_GPU[$index]}, ${WEIGHT_LIST[$index]})"
    done
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "[${METHOD_LIST[$index]}] INFER FAILED -- see $LOG_ROOT/infer_${METHOD_LIST[$index]}.log" >&2
            rc=1
        else
            echo "[${METHOD_LIST[$index]}] INFER DONE"
        fi
    done
else
    for index in "${!METHOD_LIST[@]}"; do
        infer_one "$index" || rc=1
    done
fi

ended=$(date +%s)
echo ""
echo "ex2 infer-all finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:     $LOG_ROOT/infer_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
exit $rc
