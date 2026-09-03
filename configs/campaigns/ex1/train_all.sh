#!/usr/bin/env bash
# All-in-one TRAIN campaign for the static hex-mesh ex1 dataset
# (dataset/ex1.h5, T=1 -- no rollout, every arm trains AR-OT).
#
# The default campaign is the same eleven directly auditable arms used by
# configs/campaigns/ex2/train_all.sh and configs/campaigns/ex3/train_all.sh:
#   - vanilla MeshGraphNets (flat/vanilla processor, no multiscale)
#   - HI-MGN as-is (hierarchical voronoi-multiscale backbone, use_world_edges False)
#   - HI-MGN base (same backbone, use_world_edges True -- the ancestor the P1/P2/P12
#     arms are generated from; would be the AR-RT arm on a transient dataset, but
#     ex1 is num_timesteps=1 so AR-RT is impossible and it trains AR-OT instead)
#   - HI-MGN P1, P2, and P12 (transfer-operator / multi-partition study arms,
#     see MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md)
#   - Point-DeepONet, DeepONet, FNO, and GINO
#   - Transolver (reuses its existing flagship ex1 config, config_train1.txt,
#     rather than a duplicated config_train_transolver.txt, to avoid config drift)
#
# NOT included, on purpose:
#   - SimulGenVAE: ex1.h5 has no fixed geometry across samples (node counts vary
#     25203..88582), fails at dataset loading -- a data problem, not this script's
#     concern (see configs/SimulGenVAE/ex1/).
#   - MLP: configs/MLP/ex1 trains on an unrelated synthetic toy table, not derived
#     from ex1.h5.
#   - The config_train_abl_*.txt coarsening-ablation study (22 configs, a
#     different axis of comparison: voronoi stages / mp distribution / coarsening
#     type / interpolation). That study has its own runner:
#     configs/MeshGraphNets/ex1/run_ablation.sh. This script never touches it.
#
# Every arm is launched through AI_CAE4ALL_main.py, ALL AT ONCE (no queueing --
# with PARALLEL=1 every arm starts immediately and the script waits on all of
# them together). Canonical configs are never edited: this script writes
# run-scoped copies with only gpu_ids substituted.
#
# GPU assignment is VRAM-aware, not a flat round robin: Point-DeepONet, DeepONet,
# and FNO are the low-VRAM neural-operator arms, so they are packed onto GPUs
# from LIGHT_GPUS (default: same list as GPUS) independently of the "heavy" arms
# (every MeshGraphNets/HI-MGN variant, GINO, Transolver) -- by default this
# means each heavy arm gets its own GPU and the three light arms co-locate with
# the first three heavy arms' GPUs, so no GPU ever hosts more than one heavy
# process. Override LIGHT_METHODS / LIGHT_GPUS if a different split is wanted
# (e.g. LIGHT_GPUS="7" to pile all light arms onto one otherwise-idle GPU).
#
# Environment overrides:
#   PYTHON          root launcher interpreter (default: python)
#   METHODS         space-separated campaign arms (default: all eleven)
#   GPUS            space-separated CUDA IDs for heavy arms (default: 0 1 2 3 4 5 6 7)
#   LIGHT_METHODS   space-separated method keys treated as low-VRAM
#                   (default: point_deeponet deeponet fno)
#   LIGHT_GPUS      space-separated CUDA IDs for light arms (default: same as GPUS)
#   PARALLEL        1 = launch every arm at once, concurrently (default); 0 = fully
#                   sequential, one arm at a time (debugging only)
#   PREFLIGHT       validate every runtime config before any training (default: 1)
#   PREFLIGHT_FLAGS flags appended to --check (default: --strict)
#   CHECK_ONLY      prepare + validate the campaign without training (default: 0)
#   LOG_ROOT        campaign root (default: output/ex1_all/train_runs)
#   RUN_ID          run directory name (default: timestamp + shell PID)
#
# Examples:
#   bash configs/campaigns/ex1/train_all.sh
#   CHECK_ONLY=1 bash configs/campaigns/ex1/train_all.sh
#   GPUS="0 1 2 3" bash configs/campaigns/ex1/train_all.sh
#   METHODS="meshgraphnets himgn-as-is himgn-base" PARALLEL=0 \
#       bash configs/campaigns/ex1/train_all.sh

set -uo pipefail

PYTHON="${PYTHON:-python}"
METHODS="${METHODS:-meshgraphnets himgn-as-is himgn-base himgn-p1 himgn-p2 himgn-p12 point_deeponet deeponet fno gino transolver}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LIGHT_METHODS="${LIGHT_METHODS:-point_deeponet deeponet fno}"
LIGHT_GPUS="${LIGHT_GPUS:-$GPUS}"
PARALLEL="${PARALLEL:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_FLAGS="${PREFLIGHT_FLAGS:---strict}"
CHECK_ONLY="${CHECK_ONLY:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-output/ex1_all/train_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_ROOT="$LOG_ROOT/$RUN_ID"
RUNTIME_CONFIG_ROOT="$RUN_ROOT/runtime_configs"
mkdir -p "$RUNTIME_CONFIG_ROOT"

config_for() {
    case "$1" in
        meshgraphnets)                echo "configs/MeshGraphNets/ex1/config_train_meshgraphnets.txt" ;;
        himgn-as-is|meshgraphnets-hi) echo "configs/MeshGraphNets/ex1/config_train_himgn.txt" ;;
        himgn-base)                   echo "configs/MeshGraphNets/ex1/config_train_himgn_base.txt" ;;
        himgn-p1)                     echo "configs/MeshGraphNets/ex1/config_train_himgn_p1.txt" ;;
        himgn-p2)                     echo "configs/MeshGraphNets/ex1/config_train_himgn_p2.txt" ;;
        himgn-p12)                    echo "configs/MeshGraphNets/ex1/config_train_himgn_p12.txt" ;;
        point_deeponet)               echo "configs/Neural_Operator/ex1/config_train_point_deeponet.txt" ;;
        deeponet)                     echo "configs/Neural_Operator/ex1/config_train_deeponet.txt" ;;
        fno)                          echo "configs/Neural_Operator/ex1/config_train_fno.txt" ;;
        gino)                         echo "configs/Neural_Operator/ex1/config_train_gino.txt" ;;
        transolver)                   echo "configs/Transolver/ex1/config_train1.txt" ;;
        *)                            return 1 ;;
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
read -r -a LIGHT_GPU_LIST <<< "$LIGHT_GPUS"
read -r -a PREFLIGHT_ARG_LIST <<< "$PREFLIGHT_FLAGS"

if [ "${#METHOD_LIST[@]}" -eq 0 ]; then
    echo "ERROR: METHODS is empty" >&2
    exit 2
fi
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 2
fi
if [ "${#LIGHT_GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: LIGHT_GPUS is empty" >&2
    exit 2
fi

declare -a CONFIG_LIST ASSIGNED_GPU RUNTIME_CONFIG_LIST LOG_LIST WEIGHT_LIST
MANIFEST="$RUN_ROOT/campaign.tsv"
printf 'method\tweight\tgpu\tcanonical_config\truntime_config\tlog\n' > "$MANIFEST"

heavy_idx=0
light_idx=0
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

    if is_light "$method"; then
        weight="light"
        gpu="${LIGHT_GPU_LIST[$((light_idx % ${#LIGHT_GPU_LIST[@]}))]}"
        light_idx=$((light_idx + 1))
    else
        weight="heavy"
        gpu="${GPU_LIST[$((heavy_idx % ${#GPU_LIST[@]}))]}"
        heavy_idx=$((heavy_idx + 1))
    fi
    rt_cfg="$RUNTIME_CONFIG_ROOT/train_${method}.txt"
    log="$RUN_ROOT/train_${method}.log"
    runtime_config "$cfg" "$gpu" "$rt_cfg" || exit 2

    CONFIG_LIST[$index]="$cfg"
    ASSIGNED_GPU[$index]="$gpu"
    RUNTIME_CONFIG_LIST[$index]="$rt_cfg"
    LOG_LIST[$index]="$log"
    WEIGHT_LIST[$index]="$weight"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$method" "$weight" "$gpu" "$cfg" "$rt_cfg" "$log" >> "$MANIFEST"
done

echo "ex1 all-in-one training campaign"
echo "  PYTHON       = $PYTHON"
echo "  METHODS      = $METHODS"
echo "  GPUS         = $GPUS"
echo "  LIGHT_METHODS= $LIGHT_METHODS"
echo "  LIGHT_GPUS   = $LIGHT_GPUS"
echo "  PARALLEL     = $PARALLEL (every arm launches at once, no per-GPU queueing)"
echo "  PREFLIGHT    = $PREFLIGHT ($PREFLIGHT_FLAGS)"
echo "  CHECK_ONLY   = $CHECK_ONLY"
echo "  RUN_ROOT     = $RUN_ROOT"
echo "  assignment:"
for index in "${!METHOD_LIST[@]}"; do
    printf '    gpu=%-3s [%-5s] %-18s %s\n' \
        "${ASSIGNED_GPU[$index]}" "${WEIGHT_LIST[$index]}" "${METHOD_LIST[$index]}" "${CONFIG_LIST[$index]}"
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

started=$(date +%s)
rc=0
if [ "$PARALLEL" = "1" ]; then
    pids=()
    for index in "${!METHOD_LIST[@]}"; do
        train_one "$index" &
        pids+=("$!")
        echo "  launched ${METHOD_LIST[$index]} (pid $!, gpu ${ASSIGNED_GPU[$index]}, ${WEIGHT_LIST[$index]})"
    done
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "[${METHOD_LIST[$index]}] TRAIN FAILED -- see ${LOG_LIST[$index]}" >&2
            rc=1
        else
            echo "[${METHOD_LIST[$index]}] TRAIN DONE"
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
echo "ex1 all-in-one training finished in $((ended - started))s (rc=$rc)."
echo "Manifest:        $MANIFEST"
echo "Transcripts:     $RUN_ROOT/train_<method>.log"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
if [ "$rc" = "0" ]; then
    echo "Next: bash configs/campaigns/ex1/infer_all.sh"
fi
exit "$rc"
