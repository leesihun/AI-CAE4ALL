#!/usr/bin/env bash
# Head-to-head INFERENCE runner for the static hex-mesh ex1 dataset
# (dataset/ex1_infer.h5, a single held-out sample; state rows are real ground
# truth, NOT dataset/hex_dataset.h5 whose state rows are all zero).
#
# Runs the same eleven arms as configs/ex1/train_all.sh (must match exactly,
# so every checkpoint that gets trained also gets scored). Outputs land in
# output/<method>/rollout/ex1/... in a directly comparable layout. Requires
# checkpoints already produced by configs/ex1/train_all.sh (or equivalent).
#
# GPUs are assigned round-robin from GPUS, one lane per GPU, same as
# configs/ex1/train_all.sh. Not included: SimulGenVAE (ex1.h5 has no fixed
# geometry), MLP (its ex1 config trains on an unrelated toy table), and the
# config_infer_abl_*.txt coarsening-ablation study (its own runner:
# configs/MeshGraphNets/ex1/run_ablation.sh).
#
# Environment overrides:
#   PYTHON   = python interpreter (default: python)
#   METHODS  = space-separated method list (default: all eleven, see config_for())
#   GPUS     = space-separated CUDA IDs (default: 0 1 2 3 4 5 6 7)
#   PARALLEL = 1 launch all at once then wait (default); 0 run sequentially
#   LOG_ROOT = directory for transcript logs (default: output/ex1_all/infer_runs)
#
# Usage:
#   bash configs/ex1/infer_all.sh
#   METHODS="meshgraphnets himgn-as-is himgn-base" bash configs/ex1/infer_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets himgn-as-is himgn-base himgn-p1 himgn-p2 himgn-p12 point_deeponet deeponet fno gino transolver}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PARALLEL="${PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex1_all/infer_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)                echo "configs/MeshGraphNets/ex1/config_infer_meshgraphnets.txt" ;;
        himgn-as-is|meshgraphnets-hi) echo "configs/MeshGraphNets/ex1/config_infer_himgn.txt" ;;
        himgn-base)                   echo "configs/MeshGraphNets/ex1/config_infer_himgn_base.txt" ;;
        himgn-p1)                     echo "configs/MeshGraphNets/ex1/config_infer_himgn_p1.txt" ;;
        himgn-p2)                     echo "configs/MeshGraphNets/ex1/config_infer_himgn_p2.txt" ;;
        himgn-p12)                    echo "configs/MeshGraphNets/ex1/config_infer_himgn_p12.txt" ;;
        point_deeponet)               echo "configs/Neural_Operator/ex1/config_infer_point_deeponet.txt" ;;
        deeponet)                     echo "configs/Neural_Operator/ex1/config_infer_deeponet.txt" ;;
        fno)                          echo "configs/Neural_Operator/ex1/config_infer_fno.txt" ;;
        gino)                         echo "configs/Neural_Operator/ex1/config_infer_gino.txt" ;;
        transolver)                   echo "configs/Transolver/ex1/config_infer1.txt" ;;
        *) echo "" ;;
    esac
}

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
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
    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
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
echo "ex1 infer-all"
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
echo "ex1 infer-all finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:     $LOG_ROOT/infer_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
exit $rc
