#!/usr/bin/env bash
# B300 round 14, node 135: the cause diagnosis of every finished job, without being asked.
#
# run_135.sh starts it after the jobs have started; by hand (again) it is
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/diagnose_135.sh
#
# It goes to the background on its own and waits. Each time a job has its
# spread_metrics.json (trained and scored), it runs
# methods/MeshGraphNets_Variational/misc/posterior_vs_prior_groups.py three
# times on that job's final checkpoint (forward passes only, ~10-20 GB, minutes):
#
#   ho  the held-out file (8 geometries x 256 FE draws)
#   tr  16 of the 120 training geometries x 64 draws
#   ts  the test split of the training file: 15 geometries x 64 draws, never
#       trained on, drawn from the same ranges as the training geometries
#
# and then misc/prior_diagnosis.py on the same three sets (what is wrong with
# the prior, and how much: prior_diagnosis_<ho|tr|ts>.json; `jt` too where the
# run kept <checkpoint stem>.joint.pth, which a joint run does not).
#
# The base's three runs brief.py shows beside them (r8's hybrid_e360, r9's and
# r10's base; `brief.py --refs` lists them) are diagnosed as well: their first
# diagnosis exists (r9-r11), their prior diagnosis is run again where it has no
# midpoint test yet. They wait until a job of this round has finished or the
# runner is gone, so they never sit on a card beside trainings that still grow.
# brief.py turns the JSONs into one line per setting and one decision per
# setting. A job goes on a card of run_135.sh's GPUS with at least MIN_FREE_MIB
# free, one job per card at a time (a finished job has left its card).
#
# It stops by itself once the round's runner is gone and every finished job is
# diagnosed. A part that fails is not retried (DIAG_REDO=1 runs every part again);
# the failures are in _diag/diag_fail.txt and _diag/prior_fail.txt.
# At the end it draws one histogram per geometry (unseen_brief.py --figures:
# output/b300_r14/_diag/unseen_<setting>_<tr|ts|ho>.png, the FE results and each
# seed's prior draws) and prints brief.py's block, with the decision per setting.
#
#   progress:  tail -f output/b300_r14/_diag/diag.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/brief.py
#   stop:      kill_all.sh (or kill the pid in output/b300_r14/deep_135.pid)
#
# The script calls itself: DIAG_PHASE=run is the background driver.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
ROUND="$(basename "$SCRIPT_DIR")"   # r14
NODE=135
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES = nvidia-smi's index

OUT_ROOT="output/b300_$ROUND"
DIAG_DIR="$OUT_ROOT/_diag"
LOG="$DIAG_DIR/diag.out"
FAIL="$DIAG_DIR/diag_fail.txt"
PFAIL="$DIAG_DIR/prior_fail.txt"      # the same for the prior diagnosis; brief.py reads both
PID_FILE="$OUT_ROOT/deep_$NODE.pid"   # deep_*.pid: kill_all.sh stops it with the round
RUN_PID="$OUT_ROOT/run_$NODE.pid"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
POLL="${DIAG_POLL:-600}"
METHOD=methods/MeshGraphNets_Variational
PARTS="ho tr ts"
PRIOR_PARTS="ho tr ts jt"             # jt: only with a joint-phase checkpoint
PRIOR_BATCH="${PRIOR_BATCH:-8}"

pid_alive() { local p; [ -f "$1" ] || return 1; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
cfg_path() { sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1; }

# ================================================================ the terminal part
if [ "${DIAG_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "diagnose_135 is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
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
    grep -q "l_shuffle" "$METHOD/misc/posterior_vs_prior_groups.py" \
        || { echo "posterior_vs_prior_groups.py has no g_shuffle/l_shuffle: git pull first"; exit 2; }
    grep -q -- "--split" "$METHOD/misc/posterior_vs_prior_groups.py" \
        || { echo "posterior_vs_prior_groups.py has no --split: git pull first"; exit 2; }
    for f in "$METHOD/misc/prior_diagnosis.py" "$METHOD/misc/prior_verdict.py"; do
        [ -f "$f" ] || { echo "no $f: git pull first"; exit 2; }
    done
    grep -q "mid_out" "$METHOD/misc/prior_diagnosis.py" \
        || { echo "prior_diagnosis.py has no midpoint test (mid_out): git pull first"; exit 2; }
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
[ -n "${DIAG_REDO:-}" ] && { : > "$FAIL"; : > "$PFAIL"; }
touch "$FAIL" "$PFAIL"

# The jobs: name, infer config, diag dir. After them the base's runs of earlier rounds.
NAMES=(); CFGS=(); OUTS=()
while read -r node jid _rest; do
    [ "$node" = "$NODE" ] || continue
    NAMES+=("$jid"); CFGS+=("$SWEEP_DIR/config_infer_$jid.txt"); OUTS+=("$OUT_ROOT/$jid/diag")
done < <(grep -v '^[[:space:]]*#' "$SWEEP_DIR/jobs.txt" | tr -d '\r')
N_JOBS=${#NAMES[@]}
while read -r label cfg out; do
    [ -n "$label" ] && [ -n "$out" ] || continue
    NAMES+=("$label"); CFGS+=("$cfg"); OUTS+=("$out")
done < <("$PYTHON" -B "$SWEEP_DIR/brief.py" --refs 2>/dev/null | tr -d '\r')
log "$N_JOBS jobs, $((${#NAMES[@]} - N_JOBS)) runs of earlier rounds"

json_of() { echo "${OUTS[$1]}/posterior_vs_prior_groups_$2.json"; }
prior_json() { echo "${OUTS[$1]}/prior_diagnosis_$2.json"; }
ckpt_of() { echo "$METHOD/$(cfg_path "${CFGS[$1]}" modelpath)"; }
joint_of() { echo "${1%.*}.joint.pth"; }   # training_loop.bank_joint_checkpoint
prior_part() {   # <index> <tag>: jt needs the joint-phase checkpoint
    [ "$2" != "jt" ] || [ -f "$(joint_of "$(ckpt_of "$1")")" ]
}
prior_have() {   # <index> <tag>: the JSON is there, with the midpoint test (jt has none)
    local j; j="$(prior_json "$1" "$2")"
    [ -f "$j" ] || return 1
    [ "$2" = "jt" ] || grep -q '"mid_out"' "$j"
}
finished() {   # trained and scored (an earlier round's run: its checkpoint is there)
    local cfg="${CFGS[$1]}" ck out
    [ -f "$cfg" ] || return 1
    ck="$(ckpt_of "$1")"
    [ "$1" -ge "$N_JOBS" ] && { [ -f "$ck" ]; return; }
    out="$METHOD/$(cfg_path "$cfg" inference_output_dir)/spread_metrics.json"
    [ -f "$ck" ] && [ -f "$out" ] && [ "$out" -nt "$ck" ]
}
declare -A STARTED=()   # job index -> started in this pass: each job is diagnosed once
todo() {   # 0 when a part of job $1 still has to run
    local tag
    [ -n "${STARTED[$1]:-}" ] && return 1
    for tag in $PARTS; do
        [ -f "$(json_of "$1" "$tag")" ] && [ -z "${DIAG_REDO:-}" ] && continue
        grep -q "^FAIL ${NAMES[$1]} $tag " "$FAIL" && continue
        return 0
    done
    for tag in $PRIOR_PARTS; do
        prior_part "$1" "$tag" || continue
        prior_have "$1" "$tag" && [ -z "${DIAG_REDO:-}" ] && continue
        grep -q "^FAIL ${NAMES[$1]} $tag " "$PFAIL" && continue
        return 0
    done
    return 1
}

one_job() {   # <index> <gpu>: the parts in turn
    local i="$1" gpu="$2" tag json plog rc
    local job="${NAMES[$i]}" cfg="${CFGS[$i]}" out="${OUTS[$i]}"
    mkdir -p "$out"
    for tag in $PARTS; do
        json="$(json_of "$i" "$tag")"
        if [ -f "$json" ] && [ -z "${DIAG_REDO:-}" ]; then continue; fi
        grep -q "^FAIL $job $tag " "$FAIL" && continue
        plog="$out/diag_$tag.log"
        log "$job $tag on GPU $gpu"
        local extra=()
        [ "$tag" = "tr" ] && extra=(--split train --n-groups 16)
        [ "$tag" = "ts" ] && extra=(--split test)
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
    # what is wrong with the prior, on the same sets
    local ck; ck="$(ckpt_of "$i")"
    for tag in $PRIOR_PARTS; do
        prior_part "$i" "$tag" || continue
        if prior_have "$i" "$tag" && [ -z "${DIAG_REDO:-}" ]; then continue; fi
        grep -q "^FAIL $job $tag " "$PFAIL" && continue
        json="$(prior_json "$i" "$tag")"
        plog="$out/prior_$tag.log"
        log "$job prior $tag on GPU $gpu"
        rm -f "$json"
        extra=(--split heldout)
        [ "$tag" = "tr" ] && extra=(--split train --n-groups 16)
        [ "$tag" = "ts" ] && extra=(--split test)
        [ "$tag" = "jt" ] && extra=(--split heldout --modelpath "../../$(joint_of "$ck")" --train-ref 0 --no-raw)
        ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
            misc/prior_diagnosis.py --config "../../$cfg" --gpu 0 --batch "$PRIOR_BATCH" \
            --out "../../$out" --tag "$tag" "${extra[@]}" ) > "$plog" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$json" ]; then
            log "$job prior $tag FAILED (exit $rc): $plog"
            echo "FAIL $job $tag rc=$rc $(grep -v '^[[:space:]]*$' "$plog" | tail -1 | cut -c1-80)" >> "$PFAIL"
        else
            log "$job prior $tag done"
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
    if pid_alive "$RUN_PID"; then dead=0; else dead=$((dead + 1)); fi
    room=0   # the earlier rounds' runs: once a job of this round has finished, or the runner is gone
    [ "$dead" -ge 2 ] && room=1
    for ((i = 0; i < N_JOBS && room == 0; i++)); do finished "$i" && room=1; done
    pending=0
    for ((i = 0; i < ${#NAMES[@]}; i++)); do
        [ -n "${ACTIVE[$i]:-}" ] && continue
        [ "$i" -lt "$N_JOBS" ] || [ "$room" = 1 ] || continue
        finished "$i" && todo "$i" || continue
        pending=$((pending + 1))
        [ "${#FREE[@]}" -gt 0 ] || continue
        g="${FREE[0]}"; FREE=("${FREE[@]:1}")
        STARTED[$i]=1
        one_job "$i" "$g" &
        BUSY[$g]=$!; ACTIVE[$i]=$g
        pending=$((pending - 1))
    done
    # the runner gone twice in a row (not a restart) and nothing left: done
    if [ "$dead" -ge 2 ] && [ "${#ACTIVE[@]}" -eq 0 ] && [ "$pending" -eq 0 ]; then break; fi
    [ "$pending" -gt 0 ] && [ "${#ACTIVE[@]}" -eq 0 ] && [ "${#FREE[@]}" -eq 0 ] \
        && log "$pending jobs wait for a card of '$GPUS' with ${MIN_FREE_MIB} MiB free"
    sleep "$POLL"
done
log "diagnose_135 done ($(grep -c '^FAIL' "$FAIL") failed parts, $(grep -c '^FAIL' "$PFAIL") of the prior diagnosis)"
echo
# The MGN-V interpreter has matplotlib: one histogram per geometry, every setting and set.
"$MGNV_PYTHON" -B "$SWEEP_DIR/unseen_brief.py" --figures
echo
"$PYTHON" -B "$SWEEP_DIR/brief.py"
