#!/usr/bin/env bash
# SAOI cHI-MGNflow SWEEP 2 -- the compressor axes sweep 1 holds fixed.
#
#   ./run_sweep2.sh                    # A -> B -> C -> report, unattended
#   GPUS="4 5 6 7" ./run_sweep2.sh     # pin the physical cards to use
#   GPUS="4 4 5 5" ./run_sweep2.sh     # a card listed twice runs two jobs
#   ./run_sweep2.sh A | B | C | report | clean | help
#
# Safe to run BESIDE run_sweep.sh (sweep 1): every artifact goes under
# output/chi-mgnflow/saoi_sweep2, the multiscale caches live in their own
# directories there (sweep 1's same-stem prune can never reach them), and
# write_preprocessing False means sweep 2 never writes the shared dataset.
#
# ARMS, per board half (bot, top) -- B = coarsest nodes x latent_ch:
#   n200c4  voronoi 1000,200  latent_ch 4   B  800  same budget as sweep-1 c8
#   n400c4  voronoi 1000,400  latent_ch 4   B 1600  same budget as sweep-1 c16
#   n400c8  voronoi 1000,400  latent_ch 8   B 3200  headroom past c16
#   c8m16   voronoi 1000,100  latent_ch 8   B  800  sweep-1 c8 + 16 coarsest blocks
#
# PHASES. There is no selection step: every compressor gets its own prior.
#   A  8 jobs  config_train_ae2_<half>_<arm>.txt      (2000 epochs, recon)
#   B  8 jobs  config_train_prior2_<half>_<arm>.txt   (prior p8u, against A)
#   C 24 jobs  config_infer2_<half>_<arm>_<tag>.txt   (2000 draws per part)
#   report     report_sweep2.py -> $OUT_ROOT/RESULTS.txt, sweep 1 alongside
# B runs only for arms whose A checkpoint is from this campaign; C only for
# arms whose B checkpoint is. A failed arm costs its own chain, nothing else.
#
# GPU POOL. Every config says gpu_ids 0; each job is launched with
# CUDA_VISIBLE_DEVICES=<physical card>, so the pool is just a list of cards.
# Without GPUS the script takes every card using less than IDLE_MB right now --
# check the printed pool if sweep 1 is between phases (its cards look idle then).
#
# Environment:
#   PYTHON=python   GPUS=<auto>   IDLE_MB=2000   STAGGER=20   POLL=30
#   HALVES="bot top"   ARMS="n200c4 n400c4 n400c8 c8m16"
#   PREFLIGHT=1, STRICT_PREFLIGHT=1, EVAL_PREFLIGHT=1, KEEP_CACHE=0

set -uo pipefail

PHASE="${1:-all}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA indices == nvidia-smi indices

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="$REPO_ROOT/output/chi-mgnflow/saoi_sweep2"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"

HALVES="${HALVES:-bot top}"
ARMS="${ARMS:-n200c4 n400c4 n400c8 c8m16}"
TAGS="s26fe_main s26fe_sec sm_l345u_main"
HIER_TAGS="v1000_100 v1000_200 v1000_400"

IDLE_MB="${IDLE_MB:-2000}"
STAGGER="${STAGGER:-20}"
POLL="${POLL:-30}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
KEEP_CACHE="${KEEP_CACHE:-0}"

MARK_A="$LOG_ROOT/.phaseA.start"
MARK_B="$LOG_ROOT/.phaseB.start"
MARK_C="$LOG_ROOT/.phaseC.start"

rc=0
SKIPPED=""
POOL=()
PIDS=()

usage() {
    sed -n '2,36p' "$SELF" | sed 's/^# \{0,1\}//'
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

cleanup() {
    log "Interrupted - stopping workers."
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

build_pool() {
    if [ -n "${GPUS:-}" ]; then
        read -r -a POOL <<< "$GPUS"
    else
        if ! command -v nvidia-smi > /dev/null 2>&1; then
            echo "nvidia-smi not found; set GPUS=\"<card ids>\" explicitly." >&2
            return 1
        fi
        read -r -a POOL <<< "$(nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits \
            | awk -F', *' -v lim="$IDLE_MB" '$2 + 0 < lim { printf "%s ", $1 }')"
    fi
    if [ "${#POOL[@]}" -eq 0 ]; then
        echo "No idle GPU (memory.used < ${IDLE_MB} MiB). Set GPUS=\"<card ids>\"." >&2
        return 1
    fi
    log "GPU pool (physical ids): ${POOL[*]}"
}

ae_ckpt()    { echo "$OUT_ROOT/$1.ae_$2.pth"; }
prior_ckpt() { echo "$OUT_ROOT/$1.prior_$2.pth"; }
spreads()    { echo "$OUT_ROOT/infer/$1/$2/$3/spread_values.npz"; }

fresh() {
    # fresh <file> <marker> -> the file exists and was written after <marker>
    # (or no campaign marker exists yet, i.e. a phase run on its own).
    [ -f "$1" ] || return 1
    [ ! -f "$2" ] || [ "$1" -nt "$2" ]
}

preflight_one() {
    # preflight_one <config> <label> -> 0 if the launcher validates it
    cfg="$1"
    label="$2"
    if CUDA_VISIBLE_DEVICES="${POOL[0]}" "$PYTHON" AI_CAE4ALL_main.py \
            --config "$cfg" --check > "$LOG_ROOT/${label}.check.log" 2>&1; then
        return 0
    fi
    echo "  PREFLIGHT FAILED: $label -- $LOG_ROOT/${label}.check.log" >&2
    sed -n '/ERRORS/,$p' "$LOG_ROOT/${label}.check.log" | head -8 >&2
    SKIPPED="$SKIPPED $label"
    return 1
}

run_pool() {
    # run_pool <marker> <cfg|label|done-file>... -> runs the jobs over POOL,
    # at most one per pool entry at a time, then reports which finished.
    marker="$1"
    shift
    jobs_ok=()
    if [ "$PREFLIGHT" = "1" ]; then
        log "preflight: $# job(s)"
        for job in "$@"; do
            IFS='|' read -r cfg label done_file <<< "$job"
            preflight_one "$cfg" "$label" && jobs_ok+=("$job")
        done
        if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
            echo "STRICT_PREFLIGHT=1; no partial phase will be launched." >&2
            return 1
        fi
    else
        jobs_ok=("$@")
    fi
    [ "${#jobs_ok[@]}" -gt 0 ] || { echo "Nothing to launch." >&2; return 1; }

    : > "$marker"
    n="${#POOL[@]}"
    slot_pid=()
    for ((i = 0; i < n; i++)); do slot_pid[i]=""; done

    for job in "${jobs_ok[@]}"; do
        IFS='|' read -r cfg label done_file <<< "$job"
        launched=0
        while [ "$launched" = "0" ]; do
            for ((i = 0; i < n; i++)); do
                pid="${slot_pid[i]}"
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    continue
                fi
                [ -z "$pid" ] || wait "$pid" 2>/dev/null
                gpu="${POOL[i]}"
                log "launch $label on GPU $gpu -> $LOG_ROOT/${label}.out"
                CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py \
                    --config "$cfg" > "$LOG_ROOT/${label}.out" 2>&1 &
                slot_pid[i]=$!
                PIDS+=("$!")
                launched=1
                sleep "$STAGGER"
                break
            done
            [ "$launched" = "1" ] || sleep "$POLL"
        done
    done
    for pid in "${slot_pid[@]}"; do
        [ -z "$pid" ] || wait "$pid" 2>/dev/null
    done

    failed=0
    for job in "${jobs_ok[@]}"; do
        IFS='|' read -r cfg label done_file <<< "$job"
        if [ -f "$done_file" ] && [ "$done_file" -nt "$marker" ]; then
            log "  done   $label"
        else
            log "  FAILED $label -- no fresh $(basename "$done_file"); see $LOG_ROOT/${label}.out"
            SKIPPED="$SKIPPED $label"
            failed=1
        fi
    done
    return "$failed"
}

run_phase_a() {
    echo "==== Phase A: compressors ==========================================="
    joblist=()
    for half in $HALVES; do
        for arm in $ARMS; do
            joblist+=("$CFG_DIR/config_train_ae2_${half}_${arm}.txt|A.${half}_${arm}|$(ae_ckpt "$half" "$arm")")
        done
    done
    run_pool "$MARK_A" "${joblist[@]}"
}

run_phase_b() {
    echo "==== Phase B: one p8u prior per compressor =========================="
    joblist=()
    for half in $HALVES; do
        for arm in $ARMS; do
            if fresh "$(ae_ckpt "$half" "$arm")" "$MARK_A"; then
                joblist+=("$CFG_DIR/config_train_prior2_${half}_${arm}.txt|B.${half}_${arm}|$(prior_ckpt "$half" "$arm")")
            else
                log "  skip B.${half}_${arm}: no compressor from Phase A"
                SKIPPED="$SKIPPED B.${half}_${arm}"
            fi
        done
    done
    [ "${#joblist[@]}" -gt 0 ] || { echo "No compressor to train a prior on." >&2; return 1; }
    run_pool "$MARK_B" "${joblist[@]}"
}

run_phase_c() {
    echo "==== Phase C: 2000-draw calibration, 3 parts per arm ================"
    if [ "$EVAL_PREFLIGHT" = "1" ]; then
        # Sweep 1's infer configs name the same infer/compare pairs.
        if ! "$PYTHON" "$CFG_DIR/check_eval_inputs.py" --config-dir "$CFG_DIR" \
                > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
            echo "Inference data preflight FAILED -- $LOG_ROOT/eval_inputs.check.log" >&2
            return 1
        fi
    fi
    joblist=()
    for half in $HALVES; do
        for arm in $ARMS; do
            if ! fresh "$(prior_ckpt "$half" "$arm")" "$MARK_B"; then
                log "  skip C.${half}_${arm}: no prior from Phase B"
                SKIPPED="$SKIPPED C.${half}_${arm}"
                continue
            fi
            for tag in $TAGS; do
                joblist+=("$CFG_DIR/config_infer2_${half}_${arm}_${tag}.txt|C.${half}_${arm}_${tag}|$(spreads "$half" "$arm" "$tag")")
            done
        done
    done
    [ "${#joblist[@]}" -gt 0 ] || { echo "No prior to sample from." >&2; return 1; }
    run_pool "$MARK_C" "${joblist[@]}"
}

run_report() {
    echo "==== Report ========================================================="
    "$PYTHON" "$CFG_DIR/report_sweep2.py" --out-root "$OUT_ROOT" \
        | tee "$OUT_ROOT/RESULTS.txt"
}

clean_cache() {
    if [ "$KEEP_CACHE" = "1" ]; then
        log "KEEP_CACHE=1: leaving $OUT_ROOT/mscache"
        return 0
    fi
    rm -rf "$OUT_ROOT/mscache"
    log "removed $OUT_ROOT/mscache"
}

run_all() {
    run_phase_a || rc=1
    run_phase_b || rc=1
    run_phase_c || rc=1
    run_report || rc=1
    clean_cache
}

case "$PHASE" in
    all|A|B|C|report|clean) ;;
    -h|--help|help) usage; exit 2 ;;
    *) echo "Unknown phase: $PHASE" >&2; usage; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
for tag in $HIER_TAGS; do mkdir -p "$OUT_ROOT/mscache/$tag"; done
for half in $HALVES; do
    for arm in $ARMS; do
        mkdir -p "$OUT_ROOT/ae/${half}_${arm}" "$OUT_ROOT/prior/${half}_${arm}"
        for tag in $TAGS; do mkdir -p "$OUT_ROOT/infer/$half/$arm/$tag"; done
    done
done

echo "SAOI cHI-MGNflow sweep 2 -- phase $PHASE"
echo "  repo   : $REPO_ROOT"
echo "  output : $OUT_ROOT"
echo "  halves : $HALVES   arms: $ARMS"

case "$PHASE" in
    report) run_report || rc=1 ;;
    clean) clean_cache ;;
    *)
        build_pool || exit 4
        case "$PHASE" in
            all) run_all ;;
            A) run_phase_a || rc=1 ;;
            B) run_phase_b || rc=1 ;;
            C) run_phase_c || rc=1 ;;
        esac
        ;;
esac

if [ -n "$SKIPPED" ]; then
    echo "Skipped or failed:$SKIPPED"
fi
echo "Finished phase $PHASE with rc=$rc"
exit "$rc"
