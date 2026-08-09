#!/usr/bin/env bash
# HI-MGN ex1 ablation runner.
#
# Runs the ALREADY-GENERATED configs in this directory. It never regenerates
# them -- `ablation.py gen` is a separate, deliberate step, so a re-run here can
# never silently change what is being compared mid-study.
#
# Stages, in order:
#   cost    params + FLOPs for every arm (CPU only)   -> cost.json
#   train   all arms at once, one process per arm     -> checkpoints
#   infer   each checkpoint rolled out on ex1_infer.h5 -> rollout HDF5s
#   eval    R2 / RMSE / peak error                     -> scores.json
#   report  final table                                -> report.md, report.csv
#
# The 11 arms are packed onto 8 GPUs, so three GPUs host two arms each and run
# them CONCURRENTLY. Which GPU an arm uses is read back out of its config's
# `gpu_ids` line -- the config is the single source of truth, so the runner and
# the config cannot disagree.
#
# Usage:
#   ./run_ablation.sh                # every stage in order
#   ./run_ablation.sh train          # one stage
#   ./run_ablation.sh infer eval report
#   DRY=1 ./run_ablation.sh          # print the plan, launch nothing
#   PY=/path/to/venv/bin/python ./run_ablation.sh
#
# See ABLATION.md in this directory for the study design.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULT_DIR="$REPO_ROOT/output/meshgraphnets/ex1/ablation"
ABLATION_PY="$HERE/ablation.py"
PY="${PY:-python}"
DRY="${DRY:-0}"

mkdir -p "$RESULT_DIR"
RUNNER_LOG="$RESULT_DIR/runner.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUNNER_LOG"; }
die() { log "ERROR: $*"; exit 1; }

[ -f "$ABLATION_PY" ] || die "missing $ABLATION_PY"

# ---------------------------------------------------------------------------
# Arm discovery -- from the config files themselves, not a hardcoded list, so
# adding an arm to ablation.py's ARMS table and re-running `gen` is enough.
# ---------------------------------------------------------------------------
ARMS=()
for cfg in "$HERE"/config_train_abl_*.txt; do
    [ -e "$cfg" ] || die "no generated configs found; run '$PY $ABLATION_PY gen' first"
    name="$(basename "$cfg")"; name="${name#config_train_abl_}"; name="${name%.txt}"
    ARMS+=("$name")
done
[ ${#ARMS[@]} -gt 0 ] || die "no arms discovered"

# gpu_ids from a config; the value may carry a trailing '# comment'.
gpu_of() {
    sed -n 's/^gpu_ids[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$1" | head -1
}

# ---------------------------------------------------------------------------
# Launch every arm at once and wait for all of them.
# ---------------------------------------------------------------------------
run_stage() {
    local stage="$1" prefix="$2"
    local pids=() names=() gpus=() starts=()
    local missing=0

    log "=== stage: $stage (${#ARMS[@]} arms) ==="
    for arm in "${ARMS[@]}"; do
        local cfg="$HERE/config_${prefix}_abl_${arm}.txt"
        if [ ! -f "$cfg" ]; then
            log "  MISSING config for $arm: $cfg"; missing=1; continue
        fi
        local gpu; gpu="$(gpu_of "$cfg")"
        [ -n "$gpu" ] || die "no gpu_ids in $cfg"
        local out="$RESULT_DIR/${stage}_${arm}.stdout"

        if [ "$DRY" = "1" ]; then
            log "  DRY  $arm  gpu=$gpu  -> $PY AI_CAE4ALL_main.py --config $cfg"
            continue
        fi

        ( cd "$REPO_ROOT" && "$PY" AI_CAE4ALL_main.py --config "$cfg" ) >"$out" 2>&1 &
        pids+=("$!"); names+=("$arm"); gpus+=("$gpu"); starts+=("$(date +%s)")
        log "  START $arm  gpu=$gpu  pid=$! -> $(basename "$out")"
    done
    [ "$missing" = "0" ] || die "$stage aborted: some configs are missing"
    [ "$DRY" = "1" ] && { log "  (dry run: nothing launched)"; return 0; }

    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"; local rc=$?
        local elapsed=$(( $(date +%s) - starts[i] ))
        if [ "$rc" -eq 0 ]; then
            log "  DONE  ${names[$i]}  gpu=${gpus[$i]}  $((elapsed/3600))h$(( (elapsed%3600)/60 ))m"
        else
            log "  FAIL  ${names[$i]}  gpu=${gpus[$i]}  rc=$rc  see ${stage}_${names[$i]}.stdout"
            failed=$((failed+1))
        fi
    done

    if [ "$failed" -gt 0 ]; then
        # Not fatal on purpose: one dead arm should not throw away the other
        # ten. Later stages simply skip an arm whose artifacts are absent, and
        # `report` prints n/a for it.
        log "  $failed/${#pids[@]} arm(s) failed in $stage -- continuing with the rest"
    fi
    return 0
}

run_py_stage() {
    local stage="$1"
    log "=== stage: $stage ==="
    if [ "$DRY" = "1" ]; then
        log "  DRY  $PY $ABLATION_PY $stage"; return 0
    fi
    ( cd "$REPO_ROOT" && "$PY" "$ABLATION_PY" "$stage" ) 2>&1 | tee -a "$RUNNER_LOG"
    return "${PIPESTATUS[0]}"
}

stage_cost()   { run_py_stage cost; }
stage_train()  { run_stage train train; }
stage_infer()  { run_stage infer infer; }
stage_eval()   { run_py_stage eval; }
stage_report() { run_py_stage report; }

# ---------------------------------------------------------------------------
main() {
    local stages=("$@")
    if [ ${#stages[@]} -eq 0 ]; then
        stages=(cost train infer eval report)
    fi

    log "ex1 ablation runner  repo=$REPO_ROOT  python=$PY  arms=${#ARMS[@]}"
    if [ "$DRY" = "1" ]; then log "DRY=1 -- printing the plan only"; fi

    # Show the GPU packing up front: with 11 arms on 8 GPUs three GPUs carry
    # two concurrent processes, and it is worth seeing which before committing
    # hours of B300 time to it.
    log "GPU packing (read from the configs):"
    for arm in "${ARMS[@]}"; do
        printf '    gpu %s  %s\n' "$(gpu_of "$HERE/config_train_abl_${arm}.txt")" "$arm"
    done | sort | tee -a "$RUNNER_LOG"

    for s in "${stages[@]}"; do
        case "$s" in
            cost)   stage_cost   || die "cost failed" ;;
            train)  stage_train ;;
            infer)  stage_infer ;;
            eval)   stage_eval   || die "eval failed" ;;
            report) stage_report || die "report failed" ;;
            *) die "unknown stage '$s' (cost|train|infer|eval|report)" ;;
        esac
    done

    log "runner finished. Artifacts in $RESULT_DIR"
    [ -f "$RESULT_DIR/report.md" ] && log "  report: $RESULT_DIR/report.md"
    return 0
}

main "$@"
