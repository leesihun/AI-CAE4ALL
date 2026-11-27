#!/usr/bin/env bash
# Stop the r14 noise trainings (stochastic_mode noise) where they are and score
# the checkpoint they kept: each run's model.pth is its best validation CRPS so
# far (best_by crps, val_interval 10), so inference can start from it at once.
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/stop_noise_135.sh -n   # show only
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/stop_noise_135.sh      # stop + infer
#
# 1. finds each noise job's launcher (AI_CAE4ALL_main.py --config .../config_train_<job>.txt)
#    among your own processes and sends it SIGINT, by PID; the launcher passes it to the
#    trainer's process group (the trainer keeps model.pth and its .resume state) and
#    escalates to terminate/kill after 10 s / 15 s on its own. Every other job is untouched.
# 2. waits until those launchers and their trainers are gone (SIGTERM after 120 s, SIGKILL
#    after 20 s more, still by PID).
# 3. runs TRAIN=0 JOBS="<the noise jobs>" run_135.sh: inference on the kept model.pth, then
#    the W2 and per-geometry histogram plots. Beside a run_135.sh that is still going (the
#    other jobs) it starts as LANE=b, on the cards the stopped jobs had.
#
# The first run's log shows these jobs as "train FAILED (rc=130)": that is the stop, not an
# error. The learning rate was not fully annealed (cosine over 360 epochs). Running
# run_135.sh normally later resumes their training from the .resume state.
set -u

DRY=0
[ "${1:-}" = "-n" ] && DRY=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd -P)"
cd "$REPO" || exit 2
CFG_DIR="configs/MeshGraphNets_Variational/hyperparameter_sweep/r14"
OUT_ROOT="output/b300_r14"
STATUS_DIR="$OUT_ROOT/run_logs/135/status"
NOISE_JOBS="hybrid_noise_s42 hybrid_noise_s43 hybrid_noise_coarse_s42 hybrid_noise_coarse_s43 noise_global_s42 noise_global_s43"
ME="$(id -u)"

cfg_value() {   # file key -> value (first match, comments stripped)
    awk -v k="$2" '$1 == k { sub(/[#%].*/, ""); print $2; exit }' "$1"
}
alive() { kill -0 "$1" 2>/dev/null; }

PS_OUT="$(ps -u "$ME" -o pid=,args=)"
declare -a TODO=() LAUNCHERS=() TREE=() GPU_LIST=()
echo "== r14 noise jobs on $(hostname) =="
for jid in $NOISE_JOBS; do
    tcfg="$CFG_DIR/config_train_$jid.txt"; icfg="$CFG_DIR/config_infer_$jid.txt"
    ck="$(cfg_value "$tcfg" modelpath)"; ck="${ck#../../}"
    tlog="$(cfg_value "$tcfg" log_file_dir)"; tlog="${tlog#../../}"
    out="$(cfg_value "$icfg" inference_output_dir)"; out="${out#../../}/spread_metrics.json"
    ep="$( [ -f "$tlog" ] && grep -o 'Epoch [0-9]*' "$tlog" | tail -1 | tr -dc '0-9')"
    lp="$(printf '%s\n' "$PS_OUT" | awk -v f="config_train_$jid.txt" -v me="$$" '
        $1 != me && index($0, "AI_CAE4ALL_main.py") && index($0, f) { print $1 }')"
    if [ -f "$STATUS_DIR/$jid.trained" ] && [ -f "$out" ] && [ ! "$icfg" -nt "$out" ]; then
        echo "  $jid  finished normally (trained + scored): left alone"
        continue
    fi
    if [ -n "$lp" ]; then
        g="$(tr '\0' '\n' < "/proc/$(echo "$lp" | head -1)/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
        echo "  $jid  training, epoch ${ep:-?}/360, gpu ${g:-?}, launcher pid $(echo $lp)"
        for p in $lp; do
            LAUNCHERS+=("$p"); TREE+=("$p")
            for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do TREE+=("$c"); done
        done
        [ -n "$g" ] && GPU_LIST+=("$g")
    elif [ -f "$ck" ]; then
        echo "  $jid  not training (epoch ${ep:-?}/360 in its log), checkpoint present"
    else
        echo "  $jid  not training and no checkpoint ($ck): skipped"
        continue
    fi
    TODO+=("$jid")
done

[ "${#TODO[@]}" -gt 0 ] || { echo "Nothing to do."; exit 0; }
MAIN_PID="$(tr -dc '0-9' < "$OUT_ROOT/run_135.pid" 2>/dev/null)"
LANE_ARG=""
if [ -n "$MAIN_PID" ] && alive "$MAIN_PID"; then
    LANE_ARG="b"
    B_PID="$(tr -dc '0-9' < "$OUT_ROOT/run_135_b.pid" 2>/dev/null)"
    if [ -n "$B_PID" ] && alive "$B_PID"; then
        echo "A LANE=b run is already going (pid $B_PID): finish or stop it first."; exit 2
    fi
fi
GPUS_ARG=""
if [ "${#GPU_LIST[@]}" -gt 0 ]; then
    while [ "${#GPU_LIST[@]}" -lt "${#TODO[@]}" ]; do GPU_LIST+=("${GPU_LIST[0]}"); done
    GPUS_ARG="${GPU_LIST[*]}"
fi

echo
echo "Will stop (SIGINT): ${LAUNCHERS[*]:-nothing}"
echo "Then: ${LANE_ARG:+LANE=$LANE_ARG }TRAIN=0 ${GPUS_ARG:+GPUS=\"$GPUS_ARG\" }JOBS=\"${TODO[*]}\" bash $CFG_DIR/run_135.sh"
[ "$DRY" = "1" ] && { echo "(-n: nothing done)"; exit 0; }

# ---- 1-2. stop, then wait for the launchers and their trainers to be gone
if [ "${#LAUNCHERS[@]}" -gt 0 ]; then
    kill -INT "${LAUNCHERS[@]}" 2>/dev/null
    waited=0; sig_term=0
    while :; do
        left=(); for p in "${TREE[@]}"; do alive "$p" && left+=("$p"); done
        [ "${#left[@]}" -eq 0 ] && break
        if [ "$waited" -ge 140 ]; then
            echo "  still alive after ${waited}s, SIGKILL: ${left[*]}"; kill -KILL "${left[@]}" 2>/dev/null; sleep 3; break
        elif [ "$waited" -ge 120 ] && [ "$sig_term" = "0" ]; then
            echo "  still alive after ${waited}s, SIGTERM: ${left[*]}"; kill -TERM "${left[@]}" 2>/dev/null; sig_term=1
        fi
        sleep 5; waited=$((waited + 5))
    done
    left=(); for p in "${TREE[@]}"; do alive "$p" && left+=("$p"); done
    [ "${#left[@]}" -eq 0 ] || { echo "Could not stop: ${left[*]}. Nothing launched."; exit 1; }
    echo "  stopped after ~${waited}s"
fi
for jid in "${TODO[@]}"; do
    kept="$(grep -ho 'Kept checkpoint: epoch [0-9]*' "$OUT_ROOT/run_logs/135/$jid.train.out" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "  $jid  kept checkpoint epoch ${kept:-? (see $OUT_ROOT/run_logs/135/$jid.train.out)}"
done

# ---- 3. inference on the kept checkpoints (+ W2 / histograms)
echo
export TRAIN=0 JOBS="${TODO[*]}"
[ -n "$GPUS_ARG" ] && export GPUS="$GPUS_ARG"
[ -n "$LANE_ARG" ] && export LANE="$LANE_ARG"
bash "$CFG_DIR/run_135.sh"
