#!/usr/bin/env bash
# Parallel ablation runner for the Hi-MGN transfer-operator / multi-partition
# study (MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md).
#
# Ten training runs across eight GPUs, ALL launched at once. Each config uses
# exactly one GPU (gpu_ids is a single value, so the native launcher takes the
# single-process path -- no DDP). The `orig` and `base` reference runs share a
# GPU and run CONCURRENTLY on it, so all ten processes are live together:
#
#   GPU 0 : ex1/orig + ex1/base           GPU 4 : ex2/orig + ex2/base
#   GPU 1 : ex1/p1                        GPU 5 : ex2/p1
#   GPU 2 : ex1/p2                        GPU 6 : ex2/p2
#   GPU 3 : ex1/p12                       GPU 7 : ex2/p12
#
# The two shared GPUs therefore hold two processes each; on a B300 (~288 GB)
# that is comfortable, and for ex2 the pair is one AR-OT job (orig, light) plus
# one AR-RT job (base, heavy) rather than two heavy ones.
#
#   orig  config_train_himgn.txt       pre-existing reference
#   base  config_train_himgn_base.txt  the study baseline
#   p1    ...._p1.txt                  pool_type/unpool_type attention, pool_heads 4
#   p2    ...._p2.txt                  voronoi_branches 1, 4
#   p12   ...._p12.txt                 both
#
# GPU assignment lives in configs/MeshGraphNets/gen_ablation_arms.py and is
# written into the configs; this script only reads it back, so the two cannot
# disagree.
#
# ex1 is AR-OT and ex2 is AR-RT -- forced by the data (ex1 has one timestep, so
# there is no trajectory to unroll). Compare arms WITHIN an ex, never across.
#
# Usage:
#   ./run_ablation.sh              # all 8 lanes (both datasets)
#   ./run_ablation.sh ex1          # only ex1's 4 lanes
#   DRY=1 ./run_ablation.sh        # print the plan without launching
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY_EX="${1:-}"

# lane := "<ex>:<arm>[,<arm>...]"  -- arms in a lane run one after another.
LANES=(
    "ex1:orig,base"
    "ex1:p1"
    "ex1:p2"
    "ex1:p12"
    "ex2:orig,base"
    "ex2:p1"
    "ex2:p2"
    "ex2:p12"
)

config_for() {
    case "$2" in
        orig) echo "configs/MeshGraphNets/$1/config_train_himgn.txt" ;;
        base) echo "configs/MeshGraphNets/$1/config_train_himgn_base.txt" ;;
        *)    echo "configs/MeshGraphNets/$1/config_train_himgn_$2.txt" ;;
    esac
}

gpu_of() {  # read the assignment back out of the config itself
    grep -m1 '^gpu_ids' "$ROOT/$1" | awk '{print $2}' | tr -d '\r'
}

# ---- How many GPUs can we actually see? ---------------------------------
# Reported up front so "requested GPU 7, box has 1" reads as an environment
# mismatch rather than eight identical config errors.
VISIBLE=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "Visible CUDA devices: $VISIBLE"
NEED=8
[[ -n "$ONLY_EX" ]] && NEED=4
if [[ "$VISIBLE" -lt "$NEED" ]]; then
    echo "  NOTE: this sweep is laid out for 8 GPUs (4 per dataset) and this host"
    echo "        reports $VISIBLE. Preflight will reject the high-numbered lanes here;"
    echo "        run it on the 8-GPU node, or re-map GPU{} in"
    echo "        configs/MeshGraphNets/gen_ablation_arms.py and regenerate."
fi
echo

# ---- Plan + preflight everything BEFORE launching anything --------------
# A typo in lane 8 should not surface hours into lane 1, and two lanes
# accidentally sharing a GPU should be caught here, not by an OOM later.
# DRY skips the environment layer so the plan is inspectable off-node; a real
# launch keeps it, because that layer is what catches a bad GPU index.
CHECK_FLAGS=""
[[ "${DRY:-0}" == "1" ]] && CHECK_FLAGS="--skip-environment-check"
declare -A LANE_GPU
PLAN=()
FAIL=0
echo "Ablation plan"
echo "-------------"
for lane in "${LANES[@]}"; do
    ex="${lane%%:*}"; arms="${lane##*:}"
    [[ -n "$ONLY_EX" && "$ex" != "$ONLY_EX" ]] && continue

    lane_gpu=""
    IFS=',' read -ra arm_list <<< "$arms"
    for arm in "${arm_list[@]}"; do
        cfg="$(config_for "$ex" "$arm")"
        if [[ ! -f "$ROOT/$cfg" ]]; then
            echo "  MISSING: $cfg" >&2; FAIL=1; continue
        fi
        g="$(gpu_of "$cfg")"
        if [[ -z "$lane_gpu" ]]; then lane_gpu="$g"; fi
        if [[ "$g" != "$lane_gpu" ]]; then
            echo "  ERROR: $ex/$arm is on GPU $g but its lane is GPU $lane_gpu" >&2
            FAIL=1
        fi
        if ! python "$ROOT/AI_CAE4ALL_main.py" --config "$ROOT/$cfg" --check $CHECK_FLAGS >/dev/null 2>&1; then
            echo "  PREFLIGHT FAILED: $ex/$arm  ->  python AI_CAE4ALL_main.py --config $cfg --check" >&2
            FAIL=1
        fi
    done

    if [[ -n "${LANE_GPU[$lane_gpu]+x}" ]]; then
        echo "  ERROR: GPU $lane_gpu claimed by both '${LANE_GPU[$lane_gpu]}' and '$lane'" >&2
        FAIL=1
    fi
    LANE_GPU[$lane_gpu]="$lane"
    PLAN+=("$lane_gpu|$ex|$arms")
    n_on_gpu=$(( $(tr -cd ',' <<< "$arms" | wc -c) + 1 ))
    if [[ $n_on_gpu -gt 1 ]]; then
        printf '  GPU %-3s %s : %s   (%d processes concurrently)\n' \
            "$lane_gpu" "$ex" "${arms//,/ + }" "$n_on_gpu"
    else
        printf '  GPU %-3s %s : %s\n' "$lane_gpu" "$ex" "$arms"
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo; echo "Refusing to launch: fix the errors above." >&2
    exit 1
fi
if [[ ${#PLAN[@]} -eq 0 ]]; then
    echo "Nothing to run." >&2; exit 2
fi
echo
TOTAL=0
for entry in "${PLAN[@]}"; do
    a="${entry##*|}"
    TOTAL=$(( TOTAL + $(tr -cd ',' <<< "$a" | wc -c) + 1 ))
done
echo "${#PLAN[@]} GPUs, $TOTAL training runs, all launched at once."

if [[ "${DRY:-0}" == "1" ]]; then
    echo "(DRY=1: not launching)"
    exit 0
fi

# ---- Launch one background lane per GPU --------------------------------
PIDS=()
LABELS=()
for entry in "${PLAN[@]}"; do
    IFS='|' read -r gpu ex arms <<< "$entry"
    logdir="$ROOT/output/meshgraphnets/$ex"
    mkdir -p "$logdir"
    runlog="$logdir/ablation_gpu${gpu}.log"

    # Every config in the lane starts now, concurrently, on the same device --
    # they are separate processes sharing one GPU, not a queue.
    IFS=',' read -ra arm_list <<< "$arms"
    for arm in "${arm_list[@]}"; do
        cfg="$(config_for "$ex" "$arm")"
        armlog="$logdir/ablation_gpu${gpu}_${arm}.log"
        (
            start=$(date +%s)
            echo "[$(date '+%F %T')] START $ex/$arm (GPU $gpu)" | tee -a "$armlog"
            python "$ROOT/AI_CAE4ALL_main.py" --config "$ROOT/$cfg" >>"$armlog" 2>&1
            rc=$?
            el=$(( $(date +%s) - start ))
            printf '[%s] END   %s/%s rc=%d elapsed=%dh%02dm\n' \
                "$(date '+%F %T')" "$ex" "$arm" "$rc" $((el/3600)) $((el%3600/60)) | tee -a "$armlog"
            [[ $rc -ne 0 ]] && echo "  ($ex/$arm FAILED -- see $armlog)" >&2
            exit $rc
        ) &
        PIDS+=($!)
        LABELS+=("GPU$gpu:$ex:$arm")
        echo "  launched $ex/$arm on GPU $gpu  pid $!  log $armlog"
    done
done

echo
echo "Waiting for ${#PIDS[@]} lanes. Per-lane logs: output/meshgraphnets/*/ablation_gpu*.log"
echo "Per-epoch logs: output/meshgraphnets/*/train_himgn_*.log"

STATUS=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "lane ${LABELS[$i]} exited non-zero" >&2
        STATUS=1
    fi
done

echo
echo "All lanes finished. Compare with:"
echo "  python compare_ablation.py ex1"
echo "  python compare_ablation.py ex2"
exit $STATUS
