#!/usr/bin/env bash
# Head-to-head INFERENCE runner for the static full-resolution ex3 dataset
# (dataset/ex3_test_reordered.h5, the held-out NASA-CRM-style split).
#
# Runs inference/reconstruction for all twelve ex3 arms (see
# configs/ex3/train_all.sh for the matching train campaign), each against its
# own held-out test config. Outputs land in output/<method>/rollout/ex3/full/...
# (or output/simulgenvae/ex3/full/reconstruct for SimulGenVAE, output/mlp/ex3/full/predictions
# for MLP). Requires checkpoints already produced by configs/ex3/train_all.sh
# (or equivalent).
#
# GPUs are assigned round-robin from GPUS, one lane per GPU, same as
# configs/ex3/train_all.sh. MLP is CPU-only (gpu_ids -1) and is copied
# unchanged -- it is never rewritten onto a GPU lane.
#
# Environment overrides:
#   PYTHON   = python interpreter (default: python)
#   METHODS  = space-separated method list (default: all twelve, see config_for())
#   GPUS     = space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PARALLEL = 1 launch all at once then wait (default); 0 run sequentially
#   LOG_ROOT = directory for transcript logs (default: output/ex3_all/infer_runs)
#
# Usage:
#   bash configs/ex3/infer_all.sh
#   METHODS="himgn-as-is himgn-p12 mlp" bash configs/ex3/infer_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets himgn-as-is himgn-p1 himgn-p2 himgn-p12 point_deeponet deeponet fno gino transolver mlp simulgenvae}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PARALLEL="${PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex3_all/infer_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)  echo "configs/MeshGraphNets/ex3/config_infer_meshgraphnets_full.txt" ;;
        himgn-as-is)    echo "configs/MeshGraphNets/ex3/config_infer_himgn_full.txt" ;;
        himgn-p1)       echo "configs/MeshGraphNets/ex3/config_infer_himgn_p1_full.txt" ;;
        himgn-p2)       echo "configs/MeshGraphNets/ex3/config_infer_himgn_p2_full.txt" ;;
        himgn-p12)      echo "configs/MeshGraphNets/ex3/config_infer_himgn_p12_full.txt" ;;
        point_deeponet) echo "configs/Neural_Operator/ex3/config_infer_point_deeponet_full.txt" ;;
        deeponet)       echo "configs/Neural_Operator/ex3/config_infer_deeponet_full.txt" ;;
        fno)            echo "configs/Neural_Operator/ex3/config_infer_fno_full.txt" ;;
        gino)           echo "configs/Neural_Operator/ex3/config_infer_gino_full.txt" ;;
        transolver)     echo "configs/Transolver/ex3/config_infer_transolver_full.txt" ;;
        mlp)            echo "configs/MLP/ex3/config_infer_mlp.txt" ;;
        simulgenvae)    echo "configs/SimulGenVAE/ex3/config_reconstruct.txt" ;;
        *) echo "" ;;
    esac
}

# MLP is CPU-only (gpu_ids -1) -- never rewrite it onto a GPU lane.
is_cpu_only() {
    [ "$1" = "mlp" ]
}

runtime_config() {
    local method=$1 source_cfg=$2 gpu=$3 out_cfg=$4
    if is_cpu_only "$method"; then
        cp "$source_cfg" "$out_cfg"
        return 0
    fi
    sed -E "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" "$source_cfg" > "$out_cfg"
}

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi

infer_one() {
    local method=$1 index=$2
    local cfg gpu rt_cfg log
    cfg="$(config_for "$method")"
    if [ -z "$cfg" ]; then
        echo "[$method] SKIP: unknown method" >&2
        return 0
    fi
    if [ ! -f "$cfg" ]; then
        echo "[$method] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    if is_cpu_only "$method"; then
        gpu="cpu"
    else
        gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    fi
    rt_cfg="$RUNTIME_CONFIG_ROOT/infer_${method}.txt"
    runtime_config "$method" "$cfg" "$gpu" "$rt_cfg"
    log="$LOG_ROOT/infer_${method}.log"
    echo "[$method] INFER START  gpu=$gpu  cfg=$rt_cfg (from $cfg)  -> $log"
    if [ "$PARALLEL" = "1" ]; then
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" > "$log" 2>&1
    else
        "$PYTHON" AI_CAE4ALL_main.py --config "$rt_cfg" 2>&1 | tee "$log"
    fi
}

started=$(date +%s)
echo "ex3 infer-all"
echo "  PYTHON   = $PYTHON"
echo "  METHODS  = $METHODS"
echo "  GPUS     = $GPUS"
echo "  PARALLEL = $PARALLEL"
echo "  LOG_ROOT = $LOG_ROOT"
echo "  RUN_ID   = $RUN_ID"

read -r -a METHOD_LIST <<< "$METHODS"

rc=0
if [ "$PARALLEL" = "1" ]; then
    pids=()
    names=()
    for index in "${!METHOD_LIST[@]}"; do
        m="${METHOD_LIST[$index]}"
        infer_one "$m" "$index" &
        pids+=("$!")
        names+=("$m")
        echo "  launched $m (pid $!)"
    done
    for k in "${!pids[@]}"; do
        if ! wait "${pids[$k]}"; then
            echo "[${names[$k]}] INFER FAILED -- see $LOG_ROOT/infer_${names[$k]}.log" >&2
            rc=1
        else
            echo "[${names[$k]}] INFER DONE"
        fi
    done
else
    for index in "${!METHOD_LIST[@]}"; do
        infer_one "${METHOD_LIST[$index]}" "$index" || rc=1
    done
fi

ended=$(date +%s)
echo ""
echo "ex3 infer-all finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:     $LOG_ROOT/infer_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
exit $rc
