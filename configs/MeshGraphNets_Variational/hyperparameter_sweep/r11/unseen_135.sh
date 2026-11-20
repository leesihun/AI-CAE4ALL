#!/usr/bin/env bash
# B300 round 11, node 135: the prior on unseen geometries INSIDE the training range.
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_135.sh
#
# No training. On the final checkpoint of each of this round's 16 jobs and of
# the five reference runs (R1-R5 of brief.py: the base's three seeds and r9's
# gfix_tf_k8), it runs methods/MeshGraphNets_Variational/misc/
# posterior_vs_prior_groups.py once more (forward passes only, ~10-20 GB,
# minutes), on the one set of geometries the round has not scored:
#
#   ts  the test split of the training file: 15 geometries x 64 FE draws,
#       never trained on, drawn from the same ranges as the 120 training ones
#
# beside the two parts diagnose_135.sh left (ho: 8 held-out geometries outside
# those ranges; tr: 16 training geometries). unseen_brief.py then prints one
# line per arm with all three, the seeds pooled and seed against seed.
#
# It goes to the background on its own: one part per card of run_135.sh's GPUS
# with at least MIN_FREE_MIB free, and it ends when every run is done. A part
# whose JSON exists is skipped; a part that fails is not retried
# (UNSEEN_REDO=1 runs every part again).
#
#   progress:  tail -f output/b300_r11/_diag/unseen.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_brief.py
#   stop:      kill_all.sh (or kill the pid in output/b300_r11/deep_135_unseen.pid)
#
# The script calls itself: UNSEEN_PHASE=run is the background driver.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
ROUND="$(basename "$SCRIPT_DIR")"   # r11
NODE=135
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES = nvidia-smi's index

OUT_ROOT="output/b300_$ROUND"
DIAG_DIR="$OUT_ROOT/_diag"
LOG="$DIAG_DIR/unseen.out"
FAIL="$DIAG_DIR/unseen_fail.txt"
PID_FILE="$OUT_ROOT/deep_${NODE}_unseen.pid"   # deep_*.pid: kill_all.sh stops it
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
POLL="${UNSEEN_POLL:-30}"
METHOD=methods/MeshGraphNets_Variational
TAG=ts

pid_alive() { local p; [ -f "$1" ] || return 1; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

# ================================================================ the terminal part
if [ "${UNSEEN_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "unseen_135 is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
        exit 0
    fi
    "$PYTHON" -c "import ai_cae4all" 2>/dev/null \
        || { echo "launcher python '$PYTHON' cannot import ai_cae4all. Set PYTHON=/path/to/python."; exit 2; }
    MGNV_PYTHON="${MGNV_PYTHON:-}"
    if [ -z "$MGNV_PYTHON" ]; then
        MGNV_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from ai_cae4all.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets_variational'))
")" || { echo "could not resolve the MGN-V interpreter (ai_cae4all.local.toml); set MGNV_PYTHON."; exit 2; }
    fi
    [ -x "$MGNV_PYTHON" ] || command -v "$MGNV_PYTHON" >/dev/null 2>&1 \
        || { echo "MGN-V interpreter '$MGNV_PYTHON' does not exist."; exit 2; }
    [ -f "$SWEEP_DIR/unseen_brief.py" ] || { echo "no $SWEEP_DIR/unseen_brief.py: git pull first"; exit 2; }
    RUNS="$("$PYTHON" -B "$SWEEP_DIR/unseen_brief.py" --runs | tr -d '\r')" && [ -n "$RUNS" ] \
        || { echo "unseen_brief.py --runs listed no run (jobs.txt missing?)"; exit 2; }
    n=0; have=0; gone=0; failed=0
    while read -r label cfg out model; do
        n=$((n + 1))
        if [ -f "$out/posterior_vs_prior_groups_$TAG.json" ] && [ -z "${UNSEEN_REDO:-}" ]; then have=$((have + 1))
        elif [ ! -f "$model" ]; then gone=$((gone + 1)); echo "  no checkpoint for $label ($model): it is left out"
        elif [ -z "${UNSEEN_REDO:-}" ] && grep -q "^FAIL $label $TAG " "$FAIL" 2>/dev/null; then
            failed=$((failed + 1)); echo "  $label failed before: UNSEEN_REDO=1 runs it again"
        fi
    done <<< "$RUNS"
    echo "$n runs: $have already have the $TAG part, $gone have no checkpoint, $failed failed before, $((n - have - gone - failed)) to run."
    mkdir -p "$DIAG_DIR"
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    UNSEEN_PHASE=run MGNV_PYTHON="$MGNV_PYTHON" nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (pid $!)."
    echo "  progress : tail -f $LOG"
    echo "  result   : $PYTHON $SWEEP_DIR/unseen_brief.py"
    exit 0
fi

# ================================================================ the background driver
mkdir -p "$DIAG_DIR"
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
echo "=================================================================="
log "unseen_135 started (host $(hostname), MGN-V python $MGNV_PYTHON)"
[ -n "${UNSEEN_REDO:-}" ] && : > "$FAIL"
touch "$FAIL"

# The runs: label, infer config, diag dir, checkpoint (unseen_brief.py owns the list).
NAMES=(); CFGS=(); OUTS=(); MODELS=()
while read -r label cfg out model; do
    [ -n "$label" ] || continue
    NAMES+=("$label"); CFGS+=("$cfg"); OUTS+=("$out"); MODELS+=("$model")
done < <("$PYTHON" -B "$SWEEP_DIR/unseen_brief.py" --runs | tr -d '\r')
log "${#NAMES[@]} runs"

json_of() { echo "${OUTS[$1]}/posterior_vs_prior_groups_$TAG.json"; }
declare -A STARTED=()   # run index -> started in this pass (so UNSEEN_REDO runs each part once)
todo() {   # 0 when run $1 still has to run
    [ -n "${STARTED[$1]:-}" ] && return 1
    [ -f "$(json_of "$1")" ] && [ -z "${UNSEEN_REDO:-}" ] && return 1
    grep -q "^FAIL ${NAMES[$1]} $TAG " "$FAIL" && return 1
    return 0
}

one_run() {   # <index> <gpu>
    local i="$1" gpu="$2" json plog rc
    local name="${NAMES[$i]}" cfg="${CFGS[$i]}" out="${OUTS[$i]}"
    mkdir -p "$out"
    json="$(json_of "$i")"
    plog="$out/diag_$TAG.log"
    log "$name $TAG on GPU $gpu"
    [ -n "${UNSEEN_REDO:-}" ] && rm -f "$json"
    ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
        misc/posterior_vs_prior_groups.py --config "../../$cfg" --gpu 0 \
        --out "../../$out" --tag "$TAG" --split test ) > "$plog" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
        log "$name $TAG FAILED (exit $rc): $plog"
        echo "FAIL $name $TAG rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-80)" >> "$FAIL"
    else
        log "$name $TAG done"
    fi
}

GPUS="$(sed -n 's/^GPUS="\${GPUS:-\([^}]*\)}".*/\1/p' "$SWEEP_DIR/run_$NODE.sh")"
GPUS="$(printf '%s\n' ${GPUS:-0} | awk '!seen[$0]++' | tr '\n' ' ')"   # each card once
declare -A BUSY=()    # card -> pid of its part
declare -A ACTIVE=()  # run index -> card
waiting=0
while :; do
    # reap finished parts
    for i in "${!ACTIVE[@]}"; do
        g="${ACTIVE[$i]}"
        kill -0 "${BUSY[$g]}" 2>/dev/null && continue
        wait "${BUSY[$g]}" 2>/dev/null
        unset "BUSY[$g]" "ACTIVE[$i]"
    done
    # free cards: not running a part here, enough memory
    FREE=()
    while IFS=', ' read -r g f; do
        case " $GPUS " in *" $g "*) ;; *) continue ;; esac
        [ -n "${BUSY[$g]:-}" ] && continue
        [ "${f:-0}" -ge "$MIN_FREE_MIB" ] 2>/dev/null && FREE+=("$g")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '\r')
    pending=0
    for ((i = 0; i < ${#NAMES[@]}; i++)); do
        [ -n "${ACTIVE[$i]:-}" ] && continue
        todo "$i" || continue
        if [ ! -f "${MODELS[$i]}" ]; then
            log "${NAMES[$i]}: no checkpoint (${MODELS[$i]})"
            echo "FAIL ${NAMES[$i]} $TAG rc=- no checkpoint" >> "$FAIL"
            continue
        fi
        pending=$((pending + 1))
        [ "${#FREE[@]}" -gt 0 ] || continue
        g="${FREE[0]}"; FREE=("${FREE[@]:1}")
        STARTED[$i]=1
        one_run "$i" "$g" &
        BUSY[$g]=$!; ACTIVE[$i]=$g
        pending=$((pending - 1))
    done
    if [ "${#ACTIVE[@]}" -eq 0 ] && [ "$pending" -eq 0 ]; then break; fi
    if [ "$pending" -gt 0 ] && [ "${#ACTIVE[@]}" -eq 0 ]; then
        [ "$waiting" = "0" ] && log "$pending runs wait for a card of '$GPUS' with ${MIN_FREE_MIB} MiB free"
        waiting=1
    else
        waiting=0
    fi
    sleep "$POLL"
done
log "unseen_135 done ($(grep -c '^FAIL' "$FAIL") failed parts)"
echo
# The MGN-V interpreter has matplotlib: the same block plus one histogram per geometry.
"$MGNV_PYTHON" -B "$SWEEP_DIR/unseen_brief.py" --figures
