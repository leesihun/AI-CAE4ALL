#!/usr/bin/env bash
# B300 round 11, node 135: what is wrong with the prior, and how much.
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/prior_135.sh
#
# No training. On the final checkpoint of the five reference runs (R1-R5 of
# brief.py: the base's three seeds and r9's gfix_tf_k8) and of this round's
# local / hybrid jobs it runs methods/MeshGraphNets_Variational/misc/
# prior_diagnosis.py up to four times (forward passes only):
#
#   ho  the held-out file: 8 geometries x 256 FE draws, outside the training ranges
#   tr  16 of the 120 training geometries x 64 draws
#   ts  the test split: 15 geometries x 64 draws, unseen, inside the ranges
#   jt  ho again with <checkpoint stem>.joint.pth, the best checkpoint of the epochs
#       before the prior freeze: what the prior-only epochs bought. Left out
#       without a word when a run has no such file.
#
# prior_brief.py then prints the verdict in English sentences, the table of every
# candidate cause (YES / no / ?) and the few lines to type back.
#
# It goes to the background on its own: one run (its parts in turn) per
# card of run_135.sh's GPUS with at least MIN_FREE_MIB free; the base and tfk8
# runs go first. A part whose JSON exists is skipped; a part that fails is not
# retried (PRIOR_REDO=1 runs every part again). PRIOR_ONLY="base tfk8" limits
# it to those settings.
#
#   progress:  tail -f output/b300_r11/_diag/prior.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/prior_brief.py
#   stop:      kill_all.sh (or kill the pid in output/b300_r11/deep_135_prior.pid)
#
# The script calls itself: PRIOR_PHASE=run is the background driver.
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
LOG="$DIAG_DIR/prior.out"
FAIL="$DIAG_DIR/prior_fail.txt"
PID_FILE="$OUT_ROOT/deep_${NODE}_prior.pid"   # deep_*.pid: kill_all.sh stops it
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
POLL="${PRIOR_POLL:-30}"
BATCH="${PRIOR_BATCH:-8}"
METHOD=methods/MeshGraphNets_Variational
BASE_TAGS="ho tr ts"
TAGS="$BASE_TAGS jt"

joint_of() { echo "${1%.*}.joint.pth"; }   # training_loop.bank_joint_checkpoint
has_part() {   # <checkpoint> <tag>: jt needs the joint-phase checkpoint
    [ "$2" != "jt" ] || [ -f "$(joint_of "$1")" ]
}

pid_alive() { local p; [ -f "$1" ] || return 1; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

run_list() {   # label, infer config, diag dir, checkpoint; PRIOR_ONLY keeps the named settings
    "$PYTHON" -B "$SWEEP_DIR/prior_brief.py" --runs | tr -d '\r' | while read -r label cfg out model arm; do
        [ -n "$label" ] || continue
        if [ -n "${PRIOR_ONLY:-}" ]; then   # the setting's name, exactly (brief.py's ARMS)
            case " $PRIOR_ONLY " in *" $arm "*) ;; *) continue ;; esac
        fi
        echo "$label $cfg $out $model"
    done
}

# ================================================================ the terminal part
if [ "${PRIOR_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "prior_135 is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
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
    for f in "$SWEEP_DIR/prior_brief.py" "$METHOD/misc/prior_diagnosis.py" "$METHOD/misc/prior_verdict.py"; do
        [ -f "$f" ] || { echo "no $f: git pull first"; exit 2; }
    done
    RUNS="$(run_list)" && [ -n "$RUNS" ] \
        || { echo "prior_brief.py --runs listed no run (jobs.txt missing, or PRIOR_ONLY matches nothing)"; exit 2; }
    n=0; parts=0; have=0; gone=0
    while read -r label cfg out model; do
        n=$((n + 1))
        if [ ! -f "$model" ]; then gone=$((gone + 1)); echo "  no checkpoint for $label ($model): it is left out"; continue; fi
        for tag in $TAGS; do
            has_part "$model" "$tag" || continue
            parts=$((parts + 1))
            [ -f "$out/prior_diagnosis_$tag.json" ] && [ -z "${PRIOR_REDO:-}" ] && have=$((have + 1))
        done
    done <<< "$RUNS"
    echo "$n runs ($gone without a checkpoint): $parts parts, $have already there, $((parts - have)) to run."
    mkdir -p "$DIAG_DIR"
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    PRIOR_PHASE=run MGNV_PYTHON="$MGNV_PYTHON" nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (pid $!)."
    echo "  progress : tail -f $LOG"
    echo "  result   : $PYTHON $SWEEP_DIR/prior_brief.py"
    exit 0
fi

# ================================================================ the background driver
mkdir -p "$DIAG_DIR"
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
echo "=================================================================="
log "prior_135 started (host $(hostname), MGN-V python $MGNV_PYTHON)"
[ -n "${PRIOR_REDO:-}" ] && : > "$FAIL"
touch "$FAIL"

NAMES=(); CFGS=(); OUTS=(); MODELS=()
while read -r label cfg out model; do
    [ -n "$label" ] || continue
    NAMES+=("$label"); CFGS+=("$cfg"); OUTS+=("$out"); MODELS+=("$model")
done < <(run_list)
log "${#NAMES[@]} runs"

json_of() { echo "${OUTS[$1]}/prior_diagnosis_$2.json"; }
declare -A STARTED=()   # run index -> started in this pass (so PRIOR_REDO runs each run once)
todo() {   # 0 when run $1 still has a part to run
    local tag
    [ -n "${STARTED[$1]:-}" ] && return 1
    for tag in $TAGS; do
        has_part "${MODELS[$1]}" "$tag" || continue
        grep -q "^FAIL ${NAMES[$1]} $tag " "$FAIL" && continue
        [ -f "$(json_of "$1" "$tag")" ] && [ -z "${PRIOR_REDO:-}" ] && continue
        return 0
    done
    return 1
}

one_run() {   # <index> <gpu>: the parts in turn
    local i="$1" gpu="$2" tag json plog rc
    local name="${NAMES[$i]}" cfg="${CFGS[$i]}" out="${OUTS[$i]}" model="${MODELS[$i]}"
    mkdir -p "$out"
    for tag in $TAGS; do
        has_part "$model" "$tag" || continue
        json="$(json_of "$i" "$tag")"
        if [ -f "$json" ] && [ -z "${PRIOR_REDO:-}" ]; then continue; fi
        grep -q "^FAIL $name $tag " "$FAIL" && continue
        plog="$out/prior_$tag.log"
        log "$name $tag on GPU $gpu"
        [ -n "${PRIOR_REDO:-}" ] && rm -f "$json"
        local extra=(--split heldout)
        [ "$tag" = "tr" ] && extra=(--split train --n-groups 16)
        [ "$tag" = "ts" ] && extra=(--split test)
        [ "$tag" = "jt" ] && extra=(--split heldout --modelpath "../../$(joint_of "$model")" --train-ref 0 --no-raw)
        ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
            misc/prior_diagnosis.py --config "../../$cfg" --gpu 0 --batch "$BATCH" \
            --out "../../$out" --tag "$tag" "${extra[@]}" ) > "$plog" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
            log "$name $tag FAILED (exit $rc): $plog"
            echo "FAIL $name $tag rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-80)" >> "$FAIL"
        else
            log "$name $tag done"
        fi
    done
}

GPUS="$(sed -n 's/^GPUS="\${GPUS:-\([^}]*\)}".*/\1/p' "$SWEEP_DIR/run_$NODE.sh")"
GPUS="$(printf '%s\n' ${GPUS:-0} | awk '!seen[$0]++' | tr '\n' ' ')"   # each card once
declare -A BUSY=()    # card -> pid of its run
declare -A ACTIVE=()  # run index -> card
waiting=0
while :; do
    # reap finished runs
    for i in "${!ACTIVE[@]}"; do
        g="${ACTIVE[$i]}"
        kill -0 "${BUSY[$g]}" 2>/dev/null && continue
        wait "${BUSY[$g]}" 2>/dev/null
        unset "BUSY[$g]" "ACTIVE[$i]"
    done
    # free cards: not running a run here, enough memory
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
            for tag in $BASE_TAGS; do echo "FAIL ${NAMES[$i]} $tag rc=- no checkpoint" >> "$FAIL"; done
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
log "prior_135 done ($(grep -c '^FAIL' "$FAIL") failed parts)"
echo
"$PYTHON" -B "$SWEEP_DIR/prior_brief.py"
