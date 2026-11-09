#!/usr/bin/env bash
# B300 round 8, node 135: where the prior loses, and why -- in one command:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/diagnose_deep.sh
#
# It goes to the background on its own, so the terminal can close. Forward
# passes only, no training. For each finished job of deep_brief.py's list
# (base, e360, e720, lr2e-4, pms0.1, fmbig, mmd100) it runs
# methods/MeshGraphNets_Variational/misc/posterior_vs_prior_groups.py four times:
#
#   ho  the held-out file (8 geometries x 256 FE draws), the final checkpoint
#   jt  the same, the joint-phase checkpoint (<model>.joint.pth, before the
#       prior-only tail), when the job kept one
#   ts  the training file's 15 test geometries (never trained on, not used to
#       pick the checkpoint) x 64 draws
#   tr  16 of its 120 training geometries x 64 draws
#
# tr against ts/ho separates "the prior is under-fitted" from "the prior does
# not carry over to new geometries"; jt against ho shows what the prior-only
# tail did. The jobs share the round's cards (run_135.sh's GPUS, the ones with
# room), one job per card at a time; about an hour in all.
#
#   progress:  tail -f output/b300_r8/_diag/deep.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/deep_brief.py
#              (also the end of deep.out)
#   stop:      kill_all.sh (or kill the pid in output/b300_r8/deep_135.pid)
#
# Run it again after a stop: finished parts (their JSON exists) are skipped.
# DEEP_REDO=1 runs every part again.
#
# The script calls itself: DEEP_PHASE=run is the background driver.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
ROUND="$(basename "$SCRIPT_DIR")"   # r8
NODE=135
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES = nvidia-smi's index

OUT_ROOT="output/b300_$ROUND"
DIAG_DIR="$OUT_ROOT/_diag"
LOG="$DIAG_DIR/deep.out"
FAIL="$DIAG_DIR/deep_fail.txt"
PID_FILE="$OUT_ROOT/deep_$NODE.pid"
MIN_FREE_MIB=40000   # a card with less free memory is busy (the diagnosis needs ~10-20 GB)
METHOD=methods/MeshGraphNets_Variational

pid_alive() { local p; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
cfg_path() { sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1; }

# ================================================================ the terminal part
if [ "${DEEP_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "diagnose_deep is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
        echo "If it is not, delete $PID_FILE and start again."
        exit 0
    fi
    "$PYTHON" -c "import cae_suite" 2>/dev/null \
        || { echo "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."; exit 2; }
    MGNV_PYTHON="${MGNV_PYTHON:-}"
    if [ -z "$MGNV_PYTHON" ]; then
        MGNV_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets_variational'))
")" || { echo "could not resolve the MGN-V interpreter (ai_cae4all.local.toml); set MGNV_PYTHON."; exit 2; }
    fi
    [ -x "$MGNV_PYTHON" ] || command -v "$MGNV_PYTHON" >/dev/null 2>&1 \
        || { echo "MGN-V interpreter '$MGNV_PYTHON' does not exist."; exit 2; }
    "$MGNV_PYTHON" -B "$METHOD/misc/posterior_vs_prior_groups.py" --help 2>/dev/null | grep -q -- "--n-groups" \
        || { echo "posterior_vs_prior_groups.py has no --split/--n-groups: git pull first"; exit 2; }
    n=0
    for job in $("$PYTHON" -B "$SWEEP_DIR/deep_brief.py" --list | tr -d '\r'); do
        cfg="$SWEEP_DIR/config_infer_$job.txt"
        if [ ! -f "$cfg" ]; then echo "  skip $job: no $cfg"; continue; fi
        ck="$METHOD/$(cfg_path "$cfg" modelpath)"
        if [ ! -f "$ck" ]; then echo "  skip $job: no checkpoint $ck"; continue; fi
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] || { echo "no finished job to diagnose"; exit 2; }
    mkdir -p "$DIAG_DIR"
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    DEEP_PHASE=run MGNV_PYTHON="$MGNV_PYTHON" nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $n jobs; the terminal can close."
    echo "  progress : tail -f $LOG"
    echo "  result   : python $SWEEP_DIR/deep_brief.py   (also the end of $LOG)"
    exit 0
fi

# ================================================================ the background driver
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
echo "=================================================================="
log "diagnose_deep started (host $(hostname), MGN-V python $MGNV_PYTHON)"
: > "$FAIL"

# One part: <job> <tag> [extra args]; skipped when its JSON exists.
part() {
    local job="$1" tag="$2" gpu="$3"; shift 3
    local cfg="$SWEEP_DIR/config_infer_$job.txt"
    local out="$OUT_ROOT/$job/diag"
    local json="$out/posterior_vs_prior_groups_deep_$tag.json"
    if [ -f "$json" ] && [ -z "${DEEP_REDO:-}" ]; then log "$job $tag: kept"; return 0; fi
    mkdir -p "$out"
    local plog="$out/deep_$tag.log"
    log "$job $tag on GPU $gpu"
    ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
        misc/posterior_vs_prior_groups.py --config "../../$cfg" --gpu 0 \
        --out "../../$out" --tag "deep_$tag" "$@" ) > "$plog" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
        log "$job $tag FAILED (exit $rc): $plog"
        echo "FAIL $job $tag rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-100)" >> "$FAIL"
    fi
}

one_job() {
    local job="$1" gpu="$2"
    local cfg="$SWEEP_DIR/config_infer_$job.txt"
    local ck; ck="$(cfg_path "$cfg" modelpath)"
    part "$job" ho "$gpu"
    local joint="${ck%.pth}.joint.pth"
    if [ -f "$METHOD/$joint" ]; then
        part "$job" jt "$gpu" --modelpath "$joint"
    else
        log "$job jt: no $METHOD/$joint (the job kept no joint checkpoint)"
    fi
    part "$job" ts "$gpu" --split test
    part "$job" tr "$gpu" --split train --n-groups 16
}

JOBS=()
for job in $("$PYTHON" -B "$SWEEP_DIR/deep_brief.py" --list | tr -d '\r'); do
    cfg="$SWEEP_DIR/config_infer_$job.txt"
    [ -f "$cfg" ] && [ -f "$METHOD/$(cfg_path "$cfg" modelpath)" ] && JOBS+=("$job") \
        || log "skip $job: no config or checkpoint"   # deep_brief.py prints it as "<n> none"
done

# Cards: the round's, with room; none free -> wait.
GPUS="$(sed -n 's/^GPUS="\${GPUS:-\([^}]*\)}".*/\1/p' "$SWEEP_DIR/run_$NODE.sh")"
GPUS="${GPUS:-0}"
while :; do
    FREE=()
    while IFS=', ' read -r i f; do
        case " $GPUS " in *" $i "*) [ "${f:-0}" -ge "$MIN_FREE_MIB" ] 2>/dev/null && FREE+=("$i") ;; esac
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '\r')
    command -v nvidia-smi >/dev/null 2>&1 || FREE=(${GPUS%% *})
    [ "${#FREE[@]}" -gt 0 ] && break
    log "no card of '$GPUS' has ${MIN_FREE_MIB} MiB free; waiting"
    sleep 300
done
log "jobs: ${JOBS[*]}"
log "cards: ${FREE[*]}"

# Round-robin: card k runs jobs k, k+C, ... one after the other.
C=${#FREE[@]}
for ((k = 0; k < C; k++)); do
    (
        for ((j = k; j < ${#JOBS[@]}; j += C)); do one_job "${JOBS[$j]}" "${FREE[$k]}"; done
    ) &
done
wait
log "diagnose_deep done"

echo
"$PYTHON" -B "$SWEEP_DIR/deep_brief.py"
