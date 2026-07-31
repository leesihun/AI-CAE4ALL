#!/usr/bin/env bash
# Head-to-head INFERENCE runner for the ex3 dataset (NASA-CRM, dataset/ex3_NASA_CRM_{coarse,orig}.h5).
#
# Runs inference for all seven baselines IN PARALLEL, each pinned to its own GPU
# (see gpu_for() below), against the ex3 dataset's own test split (ex3 has no
# separate held-out geometry file, unlike ex1's hex_dataset.h5 -- infer_dataset
# reuses dataset_dir, same convention as ex2), so rollout outputs land in
# output/<method>/rollout/ex3/<variant>/... in a directly comparable layout.
# Requires checkpoints already produced by configs/ex3/train_all.sh (or
# equivalent) for the SAME DATASET_VARIANT.
#
# GPU assignment (fixed, matches configs/ex3/train_all.sh):
#   0 meshgraphnets   1 meshgraphnets-hi   2 deeponet   3 fno
#   4 gino            5 point_deeponet     6 transolver
#
# Environment overrides:
#   DATASET_VARIANT = coarse | orig (default: coarse)
#   PYTHON   = python interpreter (default: python)
#   METHODS  = space-separated method list (default: all seven, see config_for())
#   PARALLEL = 1 launch all at once then wait (default); 0 run sequentially
#   LOG_ROOT = directory for transcript logs (default: output/ex3_head_to_head/<variant>/infer_logs)
#
# Usage:
#   bash configs/ex3/infer_all.sh
#   DATASET_VARIANT=orig bash configs/ex3/infer_all.sh
#   METHODS="meshgraphnets meshgraphnets-hi" bash configs/ex3/infer_all.sh

set -uo pipefail

DATASET_VARIANT="${DATASET_VARIANT:-coarse}"
if [ "$DATASET_VARIANT" != "coarse" ] && [ "$DATASET_VARIANT" != "orig" ]; then
    echo "DATASET_VARIANT must be 'coarse' or 'orig' (got '$DATASET_VARIANT')" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets meshgraphnets-hi deeponet fno gino point_deeponet transolver}"
PARALLEL="${PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex3_head_to_head/$DATASET_VARIANT/infer_logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)     echo "configs/MeshGraphNets/ex3/config_infer_meshgraphnets_${DATASET_VARIANT}.txt" ;;
        meshgraphnets-hi)  echo "configs/MeshGraphNets/ex3/config_infer_himgn_${DATASET_VARIANT}.txt" ;;
        deeponet)          echo "configs/Neural_Operator/ex3/config_infer_deeponet_${DATASET_VARIANT}.txt" ;;
        fno)               echo "configs/Neural_Operator/ex3/config_infer_fno_${DATASET_VARIANT}.txt" ;;
        gino)              echo "configs/Neural_Operator/ex3/config_infer_gino_${DATASET_VARIANT}.txt" ;;
        point_deeponet)    echo "configs/Neural_Operator/ex3/config_infer_point_deeponet_${DATASET_VARIANT}.txt" ;;
        transolver)        echo "configs/Transolver/ex3/config_infer1_${DATASET_VARIANT}.txt" ;;
        *) echo "" ;;
    esac
}

gpu_for() {
    case "$1" in
        meshgraphnets)     echo 0 ;;
        meshgraphnets-hi)  echo 1 ;;
        deeponet)          echo 2 ;;
        fno)               echo 3 ;;
        gino)              echo 4 ;;
        point_deeponet)    echo 5 ;;
        transolver)        echo 6 ;;
        *) echo "" ;;
    esac
}

runtime_config() {
    local source_cfg=$1 gpu=$2 out_cfg=$3
    sed -E "s/^([[:space:]]*gpu_ids[[:space:]]+)[^[:space:]]+/\1${gpu}/" "$source_cfg" > "$out_cfg"
}

infer_one() {
    local method=$1
    local cfg gpu rt_cfg log
    cfg="$(config_for "$method")"
    gpu="$(gpu_for "$method")"
    if [ -z "$cfg" ] || [ -z "$gpu" ]; then
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
echo "ex3 infer-all"
echo "  DATASET_VARIANT = $DATASET_VARIANT"
echo "  PYTHON   = $PYTHON"
echo "  METHODS  = $METHODS"
echo "  PARALLEL = $PARALLEL"
echo "  LOG_ROOT = $LOG_ROOT"
echo "  RUN_ID   = $RUN_ID"

rc=0
if [ "$PARALLEL" = "1" ]; then
    pids=()
    names=()
    for m in $METHODS; do
        infer_one "$m" &
        pids+=("$!")
        names+=("$m")
        echo "  launched $m (pid $!, gpu $(gpu_for "$m"))"
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
    for m in $METHODS; do
        infer_one "$m" || rc=1
    done
fi

ended=$(date +%s)
echo ""
echo "ex3 infer-all ($DATASET_VARIANT) finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:     $LOG_ROOT/infer_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
exit $rc
