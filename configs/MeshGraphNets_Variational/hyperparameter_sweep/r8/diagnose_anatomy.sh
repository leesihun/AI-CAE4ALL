#!/usr/bin/env bash
# B300 round 8, node 135: the hybrid latent taken apart -- in one command:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/diagnose_anatomy.sh
#
# It goes to the background on its own, so the terminal can close. Forward
# passes only, no training. For each job of anatomy_brief.py's list (base,
# e360) it runs methods/MeshGraphNets_Variational/misc/latent_anatomy.py on:
#
#   ho  the held-out file (8 geometries x 256 FE draws), the final checkpoint
#   jt  the same, the joint-phase checkpoint (<model>.joint.pth), when kept
#   ts  the training file's 15 test geometries x 64 draws
#   tr  16 of its 120 training geometries x 64 draws
#
# Each part decodes swapped latents (which part, g or l, carries the draw),
# integrates the prior from controlled noise (g noise off, l noise off, the
# token's write-back off, a g x l noise grid), fits the decisive latent
# directions, and splits the prior's flow-matching error along them. The
# parts share the node's 8 cards (the ones with room), one part per card.
#
#   progress:  tail -f output/b300_r8/_diag/anatomy.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/anatomy_brief.py
#              (also the end of anatomy.out)
#   figures:   output/b300_r8/<job>/diag/latent_anatomy_anat_<part>_{out,lat}.png (stay on the node)
#   stop:      kill_all.sh (or kill the pid in output/b300_r8/anatomy_135.pid)
#
# Run it again after a stop: finished parts (their JSON exists) are skipped.
# ANAT_REDO=1 runs every part again.
#
# The script calls itself: ANAT_PHASE=run is the background driver.
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
LOG="$DIAG_DIR/anatomy.out"
FAIL="$DIAG_DIR/anatomy_fail.txt"
PID_FILE="$OUT_ROOT/anatomy_$NODE.pid"
GPUS="0 1 2 3 4 5 6 7"   # all 8 cards since 2026-11-09
MIN_FREE_MIB=60000       # a card with less free memory is busy
METHOD=methods/MeshGraphNets_Variational

pid_alive() { local p; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
cfg_path() { sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1; }

# ================================================================ the terminal part
if [ "${ANAT_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "diagnose_anatomy is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
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
    ( cd "$METHOD" && "$MGNV_PYTHON" -B misc/latent_anatomy.py --help 2>/dev/null ) | grep -q -- "--grad-draws" \
        || { echo "misc/latent_anatomy.py is missing or broken: git pull first"; exit 2; }
    n=0
    for job in $("$PYTHON" -B "$SWEEP_DIR/anatomy_brief.py" --list | tr -d '\r'); do
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
    ANAT_PHASE=run MGNV_PYTHON="$MGNV_PYTHON" nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (pid $!), $n jobs; the terminal can close."
    echo "  progress : tail -f $LOG"
    echo "  result   : python $SWEEP_DIR/anatomy_brief.py   (also the end of $LOG)"
    exit 0
fi

# ================================================================ the background driver
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
echo "=================================================================="
log "diagnose_anatomy started (host $(hostname), MGN-V python $MGNV_PYTHON)"
: > "$FAIL"

# One part: <job> <tag> <gpu> [extra args]; skipped when its JSON exists.
part() {
    local job="$1" tag="$2" gpu="$3"; shift 3
    local cfg="$SWEEP_DIR/config_infer_$job.txt"
    local out="$OUT_ROOT/$job/diag"
    local json="$out/latent_anatomy_anat_$tag.json"
    if [ -f "$json" ] && [ -z "${ANAT_REDO:-}" ]; then log "$job $tag: kept"; return 0; fi
    mkdir -p "$out"
    local plog="$out/anat_$tag.log"
    log "$job $tag on GPU $gpu"
    ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
        misc/latent_anatomy.py --config "../../$cfg" --gpu 0 --batch 16 \
        --out "../../$out" --tag "anat_$tag" "$@" ) > "$plog" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
        log "$job $tag FAILED (exit $rc): $plog"
        echo "FAIL $job $tag rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-100)" >> "$FAIL"
    else
        log "$job $tag done"
    fi
}

# The part list: "<job> <tag> [extra args]" per line.
PARTS=()
for job in $("$PYTHON" -B "$SWEEP_DIR/anatomy_brief.py" --list | tr -d '\r'); do
    cfg="$SWEEP_DIR/config_infer_$job.txt"
    if [ ! -f "$cfg" ] || [ ! -f "$METHOD/$(cfg_path "$cfg" modelpath)" ]; then
        log "skip $job: no config or checkpoint"   # anatomy_brief.py prints it as "<n> none"
        continue
    fi
    ck="$(cfg_path "$cfg" modelpath)"
    PARTS+=("$job ho")
    if [ -f "$METHOD/${ck%.pth}.joint.pth" ]; then
        PARTS+=("$job jt --modelpath ${ck%.pth}.joint.pth")
    else
        log "$job jt: no $METHOD/${ck%.pth}.joint.pth (the job kept no joint checkpoint)"
    fi
    PARTS+=("$job ts --split test")
    PARTS+=("$job tr --split train --n-groups 16")
done

# Cards with room; none free -> wait.
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
log "parts: ${#PARTS[@]}"
log "cards: ${FREE[*]}"

# Round-robin: card k runs parts k, k+C, ... one after the other.
C=${#FREE[@]}
for ((k = 0; k < C; k++)); do
    (
        for ((j = k; j < ${#PARTS[@]}; j += C)); do
            # shellcheck disable=SC2086
            set -- ${PARTS[$j]}
            job="$1" tag="$2"; shift 2
            part "$job" "$tag" "${FREE[$k]}" "$@"
        done
    ) &
done
wait
log "diagnose_anatomy done"

echo
"$PYTHON" -B "$SWEEP_DIR/anatomy_brief.py"
