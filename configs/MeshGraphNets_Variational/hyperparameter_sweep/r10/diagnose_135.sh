#!/usr/bin/env bash
# B300 round 10, node 135: the cause diagnosis of every finished job, without being asked.
#
# run_135.sh starts it after the jobs have started; by hand (again) it is
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r10/diagnose_135.sh
#
# It goes to the background on its own and waits. Each time a job has its
# spread_metrics.json (trained and scored), it runs
# methods/MeshGraphNets_Variational/misc/posterior_vs_prior_groups.py twice on
# that job's final checkpoint (forward passes only, ~10-20 GB, minutes):
#
#   ho  the held-out file (8 geometries x 256 FE draws)
#   tr  16 of the 120 training geometries x 64 draws
#
# Only this round's jobs: the reference runs brief.py shows beside them (r8's
# hybrid_e360, r9's base and gfix_tf_k8) were diagnosed by r9's watcher, under
# output/b300_r9/. brief.py turns the JSONs into one line per arm. A part
# goes on a card of run_135.sh's GPUS with at least MIN_FREE_MIB free, one
# part per card at a time, beside the trainings.
#
# It stops by itself once the round's runner is gone and every finished job is
# diagnosed. A part that fails is not retried (DIAG_REDO=1 runs every part again).
#
#   progress:  tail -f output/b300_r10/_diag/diag.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r10/brief.py
#   stop:      kill_all.sh (or kill the pid in output/b300_r10/deep_135.pid)
#
# The script calls itself: DIAG_PHASE=run is the background driver.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
ROUND="$(basename "$SCRIPT_DIR")"   # r10
NODE=135
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES = nvidia-smi's index

OUT_ROOT="output/b300_$ROUND"
DIAG_DIR="$OUT_ROOT/_diag"
LOG="$DIAG_DIR/diag.out"
FAIL="$DIAG_DIR/diag_fail.txt"
PID_FILE="$OUT_ROOT/deep_$NODE.pid"   # deep_*.pid: kill_all.sh stops it with the round
RUN_PID="$OUT_ROOT/run_$NODE.pid"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
POLL="${DIAG_POLL:-600}"
METHOD=methods/MeshGraphNets_Variational

pid_alive() { local p; [ -f "$1" ] || return 1; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
cfg_path() { sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1; }

# ================================================================ the terminal part
if [ "${DIAG_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "diagnose_135 is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
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
    grep -q "l_shuffle" "$METHOD/misc/posterior_vs_prior_groups.py" \
        || { echo "posterior_vs_prior_groups.py has no g_shuffle/l_shuffle: git pull first"; exit 2; }
    [ -f "$SWEEP_DIR/jobs.txt" ] || { echo "no $SWEEP_DIR/jobs.txt: run make_sweep.py first"; exit 2; }
    mkdir -p "$DIAG_DIR"
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    DIAG_PHASE=run MGNV_PYTHON="$MGNV_PYTHON" nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "diagnosis watcher started in the background (pid $!); it diagnoses each job as it finishes."
    echo "  progress : tail -f $LOG"
    exit 0
fi

# ================================================================ the background driver
mkdir -p "$DIAG_DIR"
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
echo "=================================================================="
log "diagnose_135 started (host $(hostname), MGN-V python $MGNV_PYTHON)"
[ -n "${DIAG_REDO:-}" ] && : > "$FAIL"
touch "$FAIL"

# The jobs: name, infer config, diag dir.
NAMES=(); CFGS=(); OUTS=()
while read -r node jid _rest; do
    [ "$node" = "$NODE" ] || continue
    NAMES+=("$jid"); CFGS+=("$SWEEP_DIR/config_infer_$jid.txt"); OUTS+=("$OUT_ROOT/$jid/diag")
done < <(grep -v '^[[:space:]]*#' "$SWEEP_DIR/jobs.txt" | tr -d '\r')
log "${#NAMES[@]} jobs"

json_of() { echo "${OUTS[$1]}/posterior_vs_prior_groups_$2.json"; }
finished() {   # trained and scored
    local cfg="${CFGS[$1]}" ck out
    [ -f "$cfg" ] || return 1
    ck="$METHOD/$(cfg_path "$cfg" modelpath)"
    out="$METHOD/$(cfg_path "$cfg" inference_output_dir)/spread_metrics.json"
    [ -f "$ck" ] && [ -f "$out" ] && [ "$out" -nt "$ck" ]
}
todo() {   # 0 when a part of job $1 still has to run
    local tag
    for tag in ho tr; do
        [ -f "$(json_of "$1" "$tag")" ] && [ -z "${DIAG_REDO:-}" ] && continue
        grep -q "^FAIL ${NAMES[$1]} $tag " "$FAIL" && continue
        return 0
    done
    return 1
}

one_job() {   # <index> <gpu>: both parts in turn
    local i="$1" gpu="$2" tag json plog rc
    local job="${NAMES[$i]}" cfg="${CFGS[$i]}" out="${OUTS[$i]}"
    mkdir -p "$out"
    for tag in ho tr; do
        json="$(json_of "$i" "$tag")"
        if [ -f "$json" ] && [ -z "${DIAG_REDO:-}" ]; then continue; fi
        plog="$out/diag_$tag.log"
        log "$job $tag on GPU $gpu"
        local extra=()
        [ "$tag" = "tr" ] && extra=(--split train --n-groups 16)
        ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
            misc/posterior_vs_prior_groups.py --config "../../$cfg" --gpu 0 \
            --out "../../$out" --tag "$tag" "${extra[@]}" ) > "$plog" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
            log "$job $tag FAILED (exit $rc): $plog"
            echo "FAIL $job $tag rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-80)" >> "$FAIL"
        else
            log "$job $tag done"
        fi
    done
}

GPUS="$(sed -n 's/^GPUS="\${GPUS:-\([^}]*\)}".*/\1/p' "$SWEEP_DIR/run_$NODE.sh")"
GPUS="$(printf '%s\n' ${GPUS:-0} | awk '!seen[$0]++' | tr '\n' ' ')"   # each card once
declare -A BUSY=()    # card -> pid of its part
declare -A ACTIVE=()  # job index -> card
dead=0
while :; do
    # reap finished parts
    for i in "${!ACTIVE[@]}"; do
        g="${ACTIVE[$i]}"
        kill -0 "${BUSY[$g]}" 2>/dev/null && continue
        wait "${BUSY[$g]}" 2>/dev/null
        unset "BUSY[$g]" "ACTIVE[$i]"
    done
    # free cards of the round: not running a part here, enough memory
    FREE=()
    while IFS=', ' read -r g f; do
        case " $GPUS " in *" $g "*) ;; *) continue ;; esac
        [ -n "${BUSY[$g]:-}" ] && continue
        [ "${f:-0}" -ge "$MIN_FREE_MIB" ] 2>/dev/null && FREE+=("$g")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '\r')
    pending=0
    for ((i = 0; i < ${#NAMES[@]}; i++)); do
        [ -n "${ACTIVE[$i]:-}" ] && continue
        finished "$i" && todo "$i" || continue
        pending=$((pending + 1))
        [ "${#FREE[@]}" -gt 0 ] || continue
        g="${FREE[0]}"; FREE=("${FREE[@]:1}")
        one_job "$i" "$g" &
        BUSY[$g]=$!; ACTIVE[$i]=$g
        pending=$((pending - 1))
    done
    if pid_alive "$RUN_PID"; then dead=0; else dead=$((dead + 1)); fi
    # the runner gone twice in a row (not a restart) and nothing left: done
    if [ "$dead" -ge 2 ] && [ "${#ACTIVE[@]}" -eq 0 ] && [ "$pending" -eq 0 ]; then break; fi
    [ "$pending" -gt 0 ] && [ "${#ACTIVE[@]}" -eq 0 ] && [ "${#FREE[@]}" -eq 0 ] \
        && log "$pending jobs wait for a card of '$GPUS' with ${MIN_FREE_MIB} MiB free"
    sleep "$POLL"
done
log "diagnose_135 done ($(grep -c '^FAIL' "$FAIL") failed parts)"
echo
"$PYTHON" -B "$SWEEP_DIR/brief.py"
