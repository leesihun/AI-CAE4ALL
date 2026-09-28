#!/usr/bin/env bash
# cHI-MGNflow SAOI sweep 3 -- the KL ladder, one arm per GPU, end to end.
#
#   ./run_sweep3.sh                     # 8 arms on GPUs 0-7, both halves
#   HALVES=bot ./run_sweep3.sh          # one board half only
#   PARALLEL_HALVES=1 ./run_sweep3.sh   # bot and top chains together on each card
#   GPUS="4 5 6 7" ./run_sweep3.sh      # fewer cards: arms are dealt round-robin
#   ARMS="c8k1e6 c8k1e3" ./run_sweep3.sh
#   ./run_sweep3.sh report              # just re-print the table
#   ./run_sweep3.sh clean               # remove the shared hierarchy cache
#   ./run_sweep3.sh help
#
# Each arm is a LANE on its own card. A lane runs, per half, the whole chain
#   train_ae -> train_prior (on that arm's own frozen compressor) -> 3 inferences
# so every row of the final table is one compressor carried all the way to a
# spread histogram; nothing is promoted between arms, and lanes never wait on
# each other. A stage that fails (or leaves no checkpoint newer than its own
# launch) skips the rest of THAT chain only.
#
# Cards are used whether or not something else is already on them;
# nvidia-smi is printed first so you can see what they are sharing with.
# gpu_ids in every config is the LOGICAL 0 -- this script picks the physical
# card with CUDA_VISIBLE_DEVICES (PCI bus order, the numbering nvidia-smi shows).
#
# Overrides: PYTHON, ARMS, HALVES, GPUS, PARALLEL_HALVES, INFER_TAGS, STAGGER,
#            PREFLIGHT=0, EVAL_PREFLIGHT=0, KEEP_CACHE=1
set -uo pipefail

PHASE="${1:-all}"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/chi-mgnflow/saoi_sweep3"
LOG_ROOT="$OUT_ROOT/run_logs"

ARMS="${ARMS:-c8k1e6 c8k1e4 c8k3e4 c8k1e3 c8k3e3 c8k1e2 c8k3e2 c16k3e3}"
HALVES="${HALVES:-bot top}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PARALLEL_HALVES="${PARALLEL_HALVES:-0}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
PREFLIGHT="${PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
# All training runs of a half share one hierarchy cache; the first to arrive
# builds it under an exclusive lock and the rest block. The stagger only keeps
# eight lanes from reaching the lock in the same second.
STAGGER="${STAGGER:-20}"
KEEP_CACHE="${KEEP_CACHE:-0}"

rc=0

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF" >&2
}

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*"
}

kill_tree() {
    # Children first: a lane is a subshell whose launcher starts the native
    # trainer in its own process group, so signalling the lane alone would
    # leave the trainer running on the card.
    for child in $(pgrep -P "$1" 2>/dev/null); do
        kill_tree "$child"
    done
    kill -TERM "$1" 2>/dev/null
}

cleanup() {
    trap - INT TERM
    echo "interrupted -- stopping every lane" >&2
    for child in $(pgrep -P $$ 2>/dev/null); do
        kill_tree "$child"
    done
    exit 130
}

# --- helpers -------------------------------------------------------------

show_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
    else
        echo "  (no nvidia-smi on PATH; cards not listed)"
    fi
}

check_one() {
    # check_one <gpu> <config> <label> [extra --check flags...] -> 0 if clean
    gpu="$1"; cfg="$2"; label="$3"; shift 3
    if [ ! -f "$cfg" ]; then
        echo "  $label  MISSING CONFIG ($cfg)"
        return 1
    fi
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check "$@" \
            > "$LOG_ROOT/${label}.check.log" 2>&1; then
        return 0
    fi
    echo "  $label  FAILED -- $LOG_ROOT/${label}.check.log"
    sed -n '/ERRORS/,$p' "$LOG_ROOT/${label}.check.log" | head -8 | sed 's/^/      /'
    return 1
}

run_job() {
    # run_job <gpu> <label> <config> <product> -> 0 if the run left a product
    # newer than its own launch. The launcher runs the full --check (now with
    # the upstream checkpoint present) before it starts the native process.
    gpu="$1"; label="$2"; cfg="$3"; product="$4"
    marker="$LOG_ROOT/${label}.marker"
    : > "$marker"
    log "  gpu $gpu  $label  start"
    # stdout goes to run_logs, not log_file_dir: the trainer owns that file.
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" \
        > "$LOG_ROOT/${label}.out" 2>&1
    code=$?
    if [ "$code" = "0" ] && [ -f "$product" ] && [ "$product" -nt "$marker" ]; then
        log "  gpu $gpu  $label  done"
        return 0
    fi
    log "  gpu $gpu  $label  FAILED (rc=$code) -- $LOG_ROOT/${label}.out"
    return 1
}

run_chain() {
    # run_chain <gpu> <half> <arm>: AE -> prior -> three inferences.
    gpu="$1"; half="$2"; arm="$3"
    key="${half}_${arm}"
    run_job "$gpu" "A.$key" "$CFG_DIR/config_train_ae3_${key}.txt" \
        "$OUT_ROOT/${half}.ae_${arm}.pth" || { log "  $key: AE failed, prior + inference skipped"; return 1; }
    run_job "$gpu" "B.$key" "$CFG_DIR/config_train_prior3_${key}.txt" \
        "$OUT_ROOT/${half}.prior_${arm}.pth" || { log "  $key: prior failed, inference skipped"; return 1; }
    chain_rc=0
    for tag in $INFER_TAGS; do
        run_job "$gpu" "C.$key.$tag" "$CFG_DIR/config_infer3_${key}_${tag}.txt" \
            "$OUT_ROOT/infer/$half/$arm/$tag/spread_values.npz" || chain_rc=1
    done
    return "$chain_rc"
}

run_lane() {
    # run_lane <gpu> <arm>...: this card's arms in turn, each arm's halves
    # in series (or side by side with PARALLEL_HALVES=1).
    gpu="$1"; shift
    lane_rc=0
    for arm in "$@"; do
        if [ "$PARALLEL_HALVES" = "1" ]; then
            hpids=""
            for half in $HALVES; do
                run_chain "$gpu" "$half" "$arm" &
                hpids="$hpids $!"
                sleep "$STAGGER"
            done
            for p in $hpids; do wait "$p" || lane_rc=1; done
        else
            for half in $HALVES; do
                run_chain "$gpu" "$half" "$arm" || lane_rc=1
            done
        fi
    done
    return "$lane_rc"
}

# --- phases --------------------------------------------------------------

preflight_all() {
    # Everything is checked before any card is committed. The AE configs get
    # the full check; the prior and inference configs read checkpoints that
    # do not exist yet, so their filesystem layer (paths, dataset, checkpoint)
    # is skipped here and runs for real when each job launches.
    first_gpu="${GPU_LIST[0]}"
    bad=0
    echo "---- preflight ---------------------------------------------------"
    for arm in $ARMS; do
        for half in $HALVES; do
            key="${half}_${arm}"
            ok=1
            check_one "$first_gpu" "$CFG_DIR/config_train_ae3_${key}.txt" "A.$key" || ok=0
            check_one "$first_gpu" "$CFG_DIR/config_train_prior3_${key}.txt" "B.$key" \
                --skip-filesystem-check || ok=0
            for tag in $INFER_TAGS; do
                check_one "$first_gpu" "$CFG_DIR/config_infer3_${key}_${tag}.txt" "C.$key.$tag" \
                    --skip-filesystem-check || ok=0
            done
            if [ "$ok" = "1" ]; then echo "  $key  OK"; else bad=1; fi
        done
    done
    if [ "$EVAL_PREFLIGHT" = "1" ]; then
        # An _infer_/_compare_ mismatch compares two different parts and reads
        # as a spread defect. Sweep 3 uses sweep 1's pairs, so sweep 1's
        # configs are what this checks.
        if "$PYTHON" "$CFG_DIR/check_eval_inputs.py" --config-dir "$CFG_DIR" \
                > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
            echo "  infer/compare pairs OK"
        else
            echo "  infer/compare pairs FAILED -- $LOG_ROOT/eval_inputs.check.log"
            bad=1
        fi
    fi
    return "$bad"
}

run_all() {
    echo "---- cards -------------------------------------------------------"
    show_gpus
    if [ "$PREFLIGHT" = "1" ]; then
        preflight_all || { echo "Preflight failed; nothing launched." >&2; return 1; }
    fi

    arm_list=($ARMS)
    n_gpu=${#GPU_LIST[@]}
    echo "---- lanes -------------------------------------------------------"
    pids=""
    for ((i = 0; i < n_gpu && i < ${#arm_list[@]}; i++)); do
        lane_arms=""
        for ((j = i; j < ${#arm_list[@]}; j += n_gpu)); do
            lane_arms="$lane_arms ${arm_list[j]}"
        done
        gpu="${GPU_LIST[i]}"
        echo "  gpu $gpu :$lane_arms  (halves: $HALVES)"
        run_lane "$gpu" $lane_arms > "$LOG_ROOT/lane_gpu${gpu}.log" 2>&1 &
        pids="$pids $!"
        sleep "$STAGGER"
    done
    echo "  lane logs: $LOG_ROOT/lane_gpu<N>.log"
    tail_pid=""
    if command -v tail >/dev/null 2>&1; then
        tail -q -n 0 -F "$LOG_ROOT"/lane_gpu*.log 2>/dev/null &
        tail_pid=$!
    fi
    for pid in $pids; do
        wait "$pid" || rc=1
    done
    [ -n "$tail_pid" ] && kill "$tail_pid" 2>/dev/null
    return 0
}

run_report() {
    echo
    echo "==== sweep 3 report ================================================"
    "$PYTHON" "$CFG_DIR/report_sweep3.py" --out-root "$OUT_ROOT" 2>&1 \
        | tee "$OUT_ROOT/RESULTS.txt"
    echo "saved to $OUT_ROOT/RESULTS.txt"
}

clean_cache() {
    if [ "$KEEP_CACHE" = "1" ]; then
        echo "  KEEP_CACHE=1: hierarchy cache kept at $OUT_ROOT/mscache"
        return 0
    fi
    rm -rf "$OUT_ROOT/mscache" && echo "  removed $OUT_ROOT/mscache"
}

# --- dispatch ------------------------------------------------------------

case "$PHASE" in
    all|ALL|report|clean) ;;
    -h|--help|help) usage; exit 2 ;;
    *) echo "unknown phase: $PHASE" >&2; usage; exit 2 ;;
esac

GPU_LIST=($GPUS)
if [ "${#GPU_LIST[@]}" = "0" ]; then
    echo "GPUS is empty" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
trap cleanup INT TERM

echo "=================================================================="
echo " cHI-MGNflow SAOI sweep 3 (KL ladder) -- phase $PHASE"
echo "=================================================================="
echo "  arms   : $ARMS"
echo "  halves : $HALVES"
echo "  gpus   : $GPUS"
echo "  output : $OUT_ROOT"

case "$PHASE" in
    all|ALL)
        if run_all; then
            run_report
            clean_cache
        else
            rc=1
        fi
        ;;
    report)
        run_report
        ;;
    clean)
        clean_cache
        ;;
esac

echo "Finished phase $PHASE with rc=$rc"
exit "$rc"
