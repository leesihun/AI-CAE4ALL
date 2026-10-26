#!/usr/bin/env bash
# B300 placement sweep: one round (r6, r7, ...) on node 135's cards as one queue.
# Do not call this file directly; run, from the repo root or anywhere:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/run_135.sh r6
#
# <round>/jobs.txt is the job table (node, job, method, train config, infer
# config), written by make_sweep.py; <round>/README.md says what each job is
# for. Job k starts on the k-th slot (GPUS lists a card twice for two slots);
# each slot that frees up takes the next line. A train config of '-' is an
# inference-only job: it samples the checkpoint its infer config names. When a
# training job of the same table writes that checkpoint, it waits for it.
#
# No nohup: the checks (stages 1-3) run in the terminal, and when they pass the
# jobs go to the background on their own, so the terminal can close.
#   progress:  tail -f output/b300_<round>/run_135.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/brief.py <round>
# The run log ends with the block between "==== TYPE THIS BACK ====" and
# "==== END ====" (brief.py); brief.py prints the same block at any time.
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; the MGN-V interpreter
#                   imports torch / torch_geometric / h5py and runs a bf16
#                   matmul and a scatter on GPU 0; the checkout has split_group_attr.
#   2. data         every dataset a config names exists (exact case), and so
#                   does the checkpoint of every inference-only job; the
#                   held-out file is the 256-draw one (draws_per_shape 256,
#                   2048 scenes, 8 geometries), never the old 64-draw file.
#   3. --check      the launcher's full preflight on every config of the round,
#                   8 at a time (inference configs without the path layer while
#                   their checkpoint does not exist yet).
#   Any failure in 1-3 aborts before a GPU-hour is spent, says why, and
#   prints a one-line block to type back.
#   4. jobs         per job: train (none for a '-' job), then its inference (4 draws
#                   per held-out scene = 1024 per geometry -> spread_metrics.json,
#                   one histogram per geometry), then ../plots.py (loss curves and
#                   W2 against the 256 FE results; never fails the job).
#   5. summary      ok / the failed stage per job, ../plots.py over the round
#                   (noise_floor.png), then the brief.py block.
#
# A job that already finished is skipped: one whose training wrote
# status/<job>.trained (and whose train config is not newer) and whose
# spread_metrics.json is newer than its infer config. So after a stop or a
# crash, run the same command again: finished jobs are skipped, unfinished
# trainings continue from their resume state (resume_training True).
#
# Status for brief.py: output/b300_<round>/run_logs/<node>/status/
# <job>.{state,err,trained,train_start,train_end,infer_start,infer_end,inferfail},
# _start, _end, _abort. Stdout per stage: run_logs/<node>/<job>.<train|infer|plots>.out.
#
# Overrides, in front of the command (GPUS="0 2 3" bash run_135.sh r6):
#   PYTHON (launcher), MGNV_PYTHON (default from ai_cae4all.local.toml),
#   GPUS (run_135.sh: 0 2 3 4 5 6 7 0 2 3 4 5 6 7, a card listed twice runs two jobs at once;
#   slot s takes job s first, so list every card once before any twice),
#   THREADS_PER_JOB=4, PROBE_TIMEOUT=900, PREFLIGHT=1, TRAIN=1, INFER=1,
#   DETACH=1 (0 keeps the jobs in this terminal; the default when stdout is
#   not a terminal), JOBS="a b" (only these, in this order, run again even if done),
#   REDO=1 (every job again, from scratch), LANE=b (a second run of the SAME round
#   beside one still going: its own run_<node>_b.out/.pid, and it leaves the
#   hierarchy cache and the run hours of the first alone).
# Another round beside a running one needs no LANE (its output root differs),
# only disjoint cards:  GPUS="5 6 7" bash run_135.sh r7
# Inference-only rerun:  TRAIN=0 JOBS="..." bash run_135.sh r6
# End the queue early:   touch output/b300_<round>/stop_135   (running jobs finish,
#                        no new job starts; the next run deletes it)
#
# The script calls itself: SWEEP_PHASE=run is the background driver (stages 4-5).
set -uo pipefail

NODE="${1:-}"
ROUND="${2:-}"
case "$NODE" in
    135) ;;
    *) echo "usage: bash run_135.sh <round>   (run_node.sh takes the node and the round: 135 r6)"; exit 2 ;;
esac
PHASE="${SWEEP_PHASE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
case "$ROUND" in
    r[0-9]|r[0-9][0-9]) ;;
    *) echo "usage: bash run_135.sh <round>, e.g. r6 (rounds here: $(cd "$SCRIPT_DIR" && ls -d r*/ 2>/dev/null | tr -d / | tr '\n' ' '))"; exit 2 ;;
esac
[ -f "$SCRIPT_DIR/$ROUND/jobs.txt" ] || {
    echo "$ROUND has no jobs.txt: generate it first, python $SCRIPT_DIR/make_sweep.py"; exit 2; }
TAG="$(echo "$ROUND" | tr '[:lower:]' '[:upper:]')"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 7+ trainers per node, each with 4 torch threads + 4 loader workers. Without
# the caps every pool starts one thread per core and the CPU sits at 100%
# while the GPUs wait (round 1).
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
JOB_TABLE="$SWEEP_DIR/$ROUND/jobs.txt"

GPUS="${GPUS:-}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
REDO="${REDO:-0}"

OUT_ROOT="output/b300_$ROUND"
LOG_ROOT="$OUT_ROOT/run_logs/$NODE"
STATUS_DIR="$LOG_ROOT/status"
LANE="${LANE:-}"
export LANE
RUN_LOG="$OUT_ROOT/run_$NODE${LANE:+_$LANE}.out"
PID_FILE="$OUT_ROOT/run_$NODE${LANE:+_$LANE}.pid"
mkdir -p "$LOG_ROOT" "$STATUS_DIR"

TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
export SWEEP_TS="$TS"
MASTER_LOG="$LOG_ROOT/node_$TS.out"

# ---------------------------------------------------------------- job table
JOB_IDS=(); JOB_METHOD=(); JOB_TRAIN=(); JOB_INFER=()
while read -r n jid meth tcfg icfg; do
    case "$n" in ''|'#'*) continue ;; esac
    [ "$n" = "$NODE" ] || continue
    JOB_IDS+=("$jid"); JOB_METHOD+=("$meth"); JOB_TRAIN+=("$tcfg"); JOB_INFER+=("$icfg")
done < "$JOB_TABLE"
NAMED=""   # jobs named in JOBS run again even when done
if [ -n "${JOBS:-}" ]; then   # a subset, in the order given
    ids=(); ms=(); trs=(); ifs=()
    for j in $JOBS; do
        found=""; for i in "${!JOB_IDS[@]}"; do [ "${JOB_IDS[$i]}" = "$j" ] && found="$i"; done
        [ -n "$found" ] || { echo "JOBS entry '$j' is not a job of $ROUND (jobs: ${JOB_IDS[*]})."; exit 2; }
        ids+=("${JOB_IDS[$found]}"); ms+=("${JOB_METHOD[$found]}")
        trs+=("${JOB_TRAIN[$found]}"); ifs+=("${JOB_INFER[$found]}")
    done
    JOB_IDS=("${ids[@]}"); JOB_METHOD=("${ms[@]}"); JOB_TRAIN=("${trs[@]}"); JOB_INFER=("${ifs[@]}")
    NAMED=" $JOBS "
fi
NJ=${#JOB_IDS[@]}
METHODS="$(printf '%s\n' "${JOB_METHOD[@]}" | sort -u | tr '\n' ' ')"

method_info() {   # m -> "model_id spec_id method_dir label"
    case "$1" in
        mgnv) echo "meshgraphnets-v meshgraphnets_variational methods/MeshGraphNets_Variational MGN-V" ;;
        *)    echo "" ;;
    esac
}

cfg_value() {   # cfg key -> value (first token, comment stripped)
    sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#,]*\).*/\1/p" "$1" | head -1
}

producer_of() {   # infer cfg -> index of this table's training job that writes its checkpoint, or ""
    local ck k; ck="$(cfg_value "$1" modelpath)"
    for k in "${!JOB_IDS[@]}"; do
        [ "${JOB_TRAIN[$k]}" != "-" ] && [ "$(cfg_value "${JOB_TRAIN[$k]}" modelpath)" = "$ck" ] && { echo "$k"; return; }
    done
}

# need_train / need_infer: i -> 0 (yes) / 1 (no). A finished job is skipped.
need_train() {
    local i="$1" jid="${JOB_IDS[$1]}" tcfg="${JOB_TRAIN[$1]}" ck
    [ "$TRAIN" = "1" ] && [ "$tcfg" != "-" ] || return 1
    [ "$REDO" = "1" ] && return 0
    case "$NAMED" in *" $jid "*) return 0 ;; esac
    ck="$(cfg_value "$tcfg" modelpath)"; ck="${ck#../../}"
    [ -f "$STATUS_DIR/$jid.trained" ] && [ -f "$ck" ] && [ ! "$tcfg" -nt "$STATUS_DIR/$jid.trained" ] && return 1
    return 0
}
need_infer() {
    local i="$1" jid="${JOB_IDS[$1]}" icfg="${JOB_INFER[$1]}" out
    [ "$INFER" = "1" ] || return 1
    need_train "$i" && return 0
    [ "$REDO" = "1" ] && return 0
    case "$NAMED" in *" $jid "*) return 0 ;; esac
    out="$(cfg_value "$icfg" inference_output_dir)"; out="${out#../../}/spread_metrics.json"
    [ -f "$out" ] && [ ! -f "$STATUS_DIR/$jid.inferfail" ] && [ ! "$icfg" -nt "$out" ] && return 1
    return 0
}

# ======================================================= the background driver
if [ "$PHASE" = "run" ]; then
    exec > >(tee -a "$MASTER_LOG") 2>&1
    echo "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    GPU_ARR=($GPUS)
    NG=${#GPU_ARR[@]}

    RUN_IDX=()
    for i in "${!JOB_IDS[@]}"; do
        if need_train "$i" || need_infer "$i"; then RUN_IDX+=("$i"); fi
    done
    NR=${#RUN_IDX[@]}
    echo "=================================================================="
    echo " $(date '+%F %T')  B300 placement sweep $ROUND, node $NODE  (run $TS, host $(hostname))"
    echo " $NR of $NJ jobs to run on GPUs $GPUS; CPU threads per job: torch $OMP_NUM_THREADS, BLAS $OPENBLAS_NUM_THREADS ($(nproc) CPUs)"
    echo "=================================================================="

    # One hierarchy cache per coarsening, shared by the round's trainings (the
    # cache module locks the build). A cache left by an aborted run may hold a
    # half-written variant; start clean. A LANE run starts beside a running
    # one, whose jobs may have it open.
    if [ "$TRAIN" = "1" ] && [ -z "$LANE" ]; then
        for d in "$OUT_ROOT"/mscache/mgnv_*; do
            [ -e "$d" ] || continue
            echo "  removing the leftover hierarchy cache $d"
            rm -rf "${d:?}"
        done
    fi

    first_error() {   # log -> one short line naming what went wrong
        local line
        line="$(grep -aE '(Error|Exception|ERROR|FAILED|Killed|out of memory)' "$1" 2>/dev/null \
                | grep -avE '^\s*(File |raise |  )' | tail -1)"
        [ -n "$line" ] || line="$(grep -av '^\s*$' "$1" 2>/dev/null | tail -1)"
        printf '%s' "$line" | tr -s ' \t' ' ' | cut -c1-100
    }

    run_job() {   # gpu index
        local gpu="$1" i="$2"
        local jid="${JOB_IDS[$i]}" tcfg="${JOB_TRAIN[$i]}" icfg="${JOB_INFER[$i]}"
        local py="$MGNV_PYTHON"
        local st="$STATUS_DIR/$jid" ck log marker out_dir rc do_train=0 do_infer=0
        need_train "$i" && do_train=1
        need_infer "$i" && do_infer=1
        rm -f "$st.state" "$st.err"
        if [ "$tcfg" = "-" ]; then   # inference only: the checkpoint is another run's
            ck="$(cfg_value "$icfg" modelpath)"; ck="${ck#../../}"
            local pk pst; pk="$(producer_of "$icfg")"
            if [ -n "$pk" ] && [ "$do_infer" = "1" ]; then   # trained in this queue: wait for it
                local pj="${JOB_IDS[$pk]}" waited=0
                while :; do
                    pst="$(cat "$STATUS_DIR/$pj.state" 2>/dev/null)"
                    case "$pst" in fail*) break ;; esac
                    case "$pst" in queued|train|"") ;; *) [ -f "$STATUS_DIR/$pj.trained" ] && [ -f "$ck" ] && break ;; esac
                    [ -e "$OUT_ROOT/stop_$NODE" ] && break
                    [ "$waited" = "0" ] && echo "  [gpu $gpu] $jid  waits for $pj's checkpoint"
                    echo "wait" > "$st.state"; waited=1
                    sleep 300
                done
                if [ ! -f "$STATUS_DIR/$pj.trained" ] || [ ! -f "$ck" ]; then
                    echo "fail train no-checkpoint" > "$st.state"
                    echo "$pj did not produce $ck (${pst:-no state})" > "$st.err"
                    echo "  [gpu $gpu] $jid  $pj did not produce $ck"
                    return
                fi
            fi
        else
            ck="$(cfg_value "$tcfg" modelpath)"; ck="${ck#../../}"
        fi

        if [ "$do_train" = "1" ]; then
            rm -f "$st.train_start" "$st.train_end" "$st.trained" "$st.inferfail"
            log="$LOG_ROOT/$jid.train.out"
            marker="$LOG_ROOT/$jid.train.marker"
            : > "$marker"
            echo "train" > "$st.state"
            date +%s > "$st.train_start"
            echo "  [gpu $gpu] $jid  train start  $(date '+%m-%d %H:%M')"
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$py" --config "$tcfg" > "$log" 2>&1
            rc=$?
            date +%s > "$st.train_end"
            # A stale checkpoint must never flow into inference after a failed run.
            if [ "$rc" != "0" ] || [ ! -f "$ck" ] || [ ! "$ck" -nt "$marker" ]; then
                [ "$rc" = "0" ] && rc="no-fresh-checkpoint"
                echo "fail train rc=$rc" > "$st.state"
                first_error "$log" > "$st.err"
                echo "  [gpu $gpu] $jid  train FAILED (rc=$rc) -- $log"
                tail -15 "$log" | sed "s/^/      [$jid] /"
                return
            fi
            date +%s > "$st.trained"
            echo "  [gpu $gpu] $jid  train done   $(date '+%m-%d %H:%M')"
        elif [ ! -f "$ck" ]; then
            echo "fail train no-checkpoint" > "$st.state"
            echo "no training here, and $ck does not exist" > "$st.err"
            echo "  [gpu $gpu] $jid  no training here, and $ck does not exist"
            return
        fi

        if [ "$do_infer" = "1" ]; then
            rm -f "$st.inferfail" "$st.infer_end"
            out_dir="$(cfg_value "$icfg" inference_output_dir)"; out_dir="${out_dir#../../}"
            log="$LOG_ROOT/$jid.infer.out"
            marker="$LOG_ROOT/$jid.infer.marker"
            : > "$marker"
            echo "infer" > "$st.state"
            date +%s > "$st.infer_start"
            echo "  [gpu $gpu] $jid  infer start  $(date '+%m-%d %H:%M')  ($ck)"
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$py" --config "$icfg" > "$log" 2>&1
            rc=$?
            date +%s > "$st.infer_end"
            if [ "$rc" != "0" ] || [ ! -f "$out_dir/spread_metrics.json" ] \
                    || [ ! "$out_dir/spread_metrics.json" -nt "$marker" ]; then
                [ "$rc" = "0" ] && rc="no-fresh-metrics"
                echo "infer rc=$rc" > "$st.inferfail"
                first_error "$log" > "$st.err"
                echo "  [gpu $gpu] $jid  infer FAILED (rc=$rc) -- $log"
                tail -8 "$log" | sed "s/^/      [$jid] /"
            else
                echo "  [gpu $gpu] $jid  infer done   $(date '+%m-%d %H:%M')"
            fi
        fi
        # Loss curves, W2 against the FE results, per-geometry histograms, in the run dir.
        CUDA_VISIBLE_DEVICES="$gpu" "$py" -B "$SCRIPT_DIR/../plots.py" "$SCRIPT_DIR/$ROUND" "$jid" \
            > "$LOG_ROOT/$jid.plots.out" 2>&1 \
            || echo "  [gpu $gpu] $jid  plots failed, the job still counts -- $LOG_ROOT/$jid.plots.out"
        echo "done" > "$st.state"
    }

    # ---------------------------------------------------------------- 4. jobs
    QUEUE_DIR="$LOG_ROOT/queue_$TS"
    mkdir -p "$QUEUE_DIR"
    echo "---- 4. jobs: $NR over $NG GPU slot(s) -------------------------------"
    echo "  stdout: $LOG_ROOT/<job>.<train|infer>.out   status: $STATUS_DIR/"
    for i in "${!JOB_IDS[@]}"; do
        case " ${RUN_IDX[*]} " in
            *" $i "*) echo "queued" > "$STATUS_DIR/${JOB_IDS[$i]}.state" ;;
            *) echo "  ${JOB_IDS[$i]}  already done, skipped (REDO=1 or JOBS=${JOB_IDS[$i]} runs it again)" ;;
        esac
    done
    # Run hours (brief.py) count from the round's first training run.
    if [ "$TRAIN" = "1" ] && [ -z "$LANE" ]; then
        { [ ! -f "$STATUS_DIR/_start" ] || [ "$REDO" = "1" ]; } && date +%s > "$STATUS_DIR/_start"
        rm -f "$STATUS_DIR/_end"
    fi
    # The first round is dealt before any card starts (slot s takes job s), so
    # a job that fails in seconds cannot let its card steal the next card's job.
    # After that a slot takes the next job nobody has claimed, unless
    # output/b300_<round>/stop_<node> exists (touch it to end the queue early).
    for ((s = 0; s < NG && s < NR; s++)); do mkdir "$QUEUE_DIR/claim_$s"; done
    pids=""
    for ((s = 0; s < NG; s++)); do
        gpu="${GPU_ARR[$s]}"
        (
            [ "$s" -lt "$NR" ] && run_job "$gpu" "${RUN_IDX[$s]}"
            for k in "${!RUN_IDX[@]}"; do
                [ -e "$OUT_ROOT/stop_$NODE" ] && break   # no new jobs; running ones finish
                mkdir "$QUEUE_DIR/claim_$k" 2>/dev/null || continue
                run_job "$gpu" "${RUN_IDX[$k]}"
            done
        ) &
        pids="$pids $!"
        sleep 2   # stagger the starts: 7 interpreters importing torch at once
    done
    for pid in $pids; do wait "$pid"; done
    [ "$TRAIN" = "1" ] && [ -z "$LANE" ] && date +%s > "$STATUS_DIR/_end"

    # ------------------------------------------------------------- 5. summary
    rc=0
    echo "=================================================================="
    echo " summary, $ROUND node $NODE"
    echo "=================================================================="
    for jid in "${JOB_IDS[@]}"; do
        state="$(cat "$STATUS_DIR/$jid.state" 2>/dev/null || echo 'did not run')"
        extra=""
        [ -f "$STATUS_DIR/$jid.inferfail" ] && extra="  failed inference: $(tr '\n' ' ' < "$STATUS_DIR/$jid.inferfail")"
        printf '  %-40s %s%s\n' "$jid" "$state" "$extra"
        { [ "$state" = "done" ] && [ -z "$extra" ]; } || rc=1
    done
    echo "Finished with rc=$rc   $(date '+%F %T')"
    echo
    # Every job's W2 (cached ones are reused) and the noise-floor figure.
    "$MGNV_PYTHON" -B "$SCRIPT_DIR/../plots.py" "$SCRIPT_DIR/$ROUND" --no-loss > "$LOG_ROOT/_plots_all.out" 2>&1 \
        || echo "  plots over the round failed (the jobs still count) -- $LOG_ROOT/_plots_all.out"
    "$PYTHON" -B "$SCRIPT_DIR/brief.py" "$ROUND"
    exit "$rc"
fi

# ============================================= the command: checks, then detach
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "$ROUND is already running on node $NODE here (pid $(cat "$PID_FILE")): tail -f $RUN_LOG"
    echo "If it is not, delete $PID_FILE and start again."
    exit 2
fi
rm -f "$STATUS_DIR/_abort"
exec > >(tee -a "$MASTER_LOG") 2>&1
if [ -e "$OUT_ROOT/stop_$NODE" ] && [ -z "$LANE" ]; then
    echo "removing $OUT_ROOT/stop_$NODE left by an earlier run (it would stop the queue from taking a further job)"
    rm -f "$OUT_ROOT/stop_$NODE"
fi

die() {
    echo
    echo "ABORT: $*"
    echo "Nothing was trained. Full log: $MASTER_LOG"
    printf '%s\n' "$*" | head -1 | cut -c1-80 > "$STATUS_DIR/_abort"
    echo "==== TYPE THIS BACK ===="
    echo "$TAG $NODE ABORT $(cat "$STATUS_DIR/_abort")"
    echo "==== END ===="
    exit 1
}

echo "=================================================================="
echo " B300 placement sweep $ROUND, node $NODE, checks   ($TS, host $(hostname))"
echo "=================================================================="
[ "$NJ" -ge 1 ] || die "$ROUND/jobs.txt has no job for node $NODE."

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
echo "  [$(date +%T)] launcher python '$PYTHON': import cae_suite"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."

PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
NGPU_SEEN=""
for m in $METHODS; do
    read -r model_id spec_id method_dir label <<< "$(method_info "$m")"
    [ -n "$model_id" ] || die "jobs.txt names an unknown method '$m' (this sweep runs MGN-V only)."
    var="$(echo "$m" | tr '[:lower:]' '[:upper:]')_PYTHON"
    py="${!var:-}"
    if [ -z "$py" ]; then
        py="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('$model_id', '$spec_id'))
")" || die "could not resolve the $label interpreter (ai_cae4all.local.toml); set $var."
    fi
    [ -x "$py" ] || command -v "$py" >/dev/null 2>&1 || die "$label interpreter '$py' does not exist."
    printf -v "$var" '%s' "$py"
    export "$var"

    echo "  [$(date +%T)] probe $label '$py' (gives up after ${PROBE_TIMEOUT}s):"
    out="$(env -u CUDA_VISIBLE_DEVICES timeout -k 30 "$PROBE_TIMEOUT" "$py" -c "
import sys, time
t0 = time.time()
def step(what):
    print(f'      [{time.time() - t0:6.1f}s] {what}', file=sys.stderr, flush=True)
step('import torch')
import torch
step('import torch_geometric, h5py')
import torch_geometric, h5py
from torch_geometric.utils import scatter
step('CUDA init: count the GPUs')
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
info = f'torch {torch.__version__} (CUDA {torch.version.cuda}), torch_geometric {torch_geometric.__version__}'
if n:
    major, minor = torch.cuda.get_device_capability(0)
    step(f'{n} x {torch.cuda.get_device_name(0)} (sm_{major}{minor}); kernels for: {\" \".join(torch.cuda.get_arch_list())}')
    step('bf16 matmul on GPU 0')
    x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
    (x @ x).float().sum().item()
    step('scatter on GPU 0')
    idx = torch.tensor([0, 0, 1, 1], device='cuda')
    scatter(torch.ones(4, 2, device='cuda'), idx, dim=0, reduce='sum').sum().item()
    info += f' | {torch.cuda.get_device_name(0)} sm_{major}{minor}, kernels run'
step('done')
print(n)
print(info)
")" && rc=0 || rc=$?
    case "$rc" in
        0) ;;
        124|137) die "the $label probe hung past ${PROBE_TIMEOUT}s in the last step printed above (slow filesystem: PROBE_TIMEOUT=1800; CUDA init: driver / nvidia-fabricmanager; bf16 matmul without sm_100 kernels: install a CUDA 12.8+ torch)." ;;
        *) die "$label interpreter '$py' cannot import torch / torch_geometric / h5py or run a CUDA kernel (traceback above)." ;;
    esac
    n="$(printf '%s\n' "$out" | sed -n 1p)"
    [ "${n:-0}" -ge 1 ] 2>/dev/null || die "$label interpreter '$py' sees no CUDA device."
    NGPU_SEEN="$n"
    printf '  %-15s : %s\n' "$label" "$py"
    echo "                    $(printf '%s\n' "$out" | sed -n 2p)"
    grep -q "split_group_attr" "$method_dir/general_modules/mesh_dataset.py" \
        || die "this checkout's $method_dir has no split_group_attr: the train/test split would leak geometries. Pull first."
done
MGNV_PYTHON="${MGNV_PYTHON:-}"

[ -n "$GPUS" ] || GPUS="$(seq -s ' ' 0 $(( NGPU_SEEN - 1 )))"
for g in $GPUS; do
    case "$g" in ''|*[!0-9]*) die "GPUS entry '$g' is not a GPU index." ;; esac
    [ "$g" -lt "$NGPU_SEEN" ] || die "GPU $g does not exist: this machine has $NGPU_SEEN GPU(s)."
done
export GPUS
echo "  commit          : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPUs            : $GPUS  ($NGPU_SEEN on this machine)"
echo "  jobs            : $NJ in $ROUND"
if command -v nvidia-smi >/dev/null 2>&1; then
    timeout -k 5 60 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /' || echo "      (nvidia-smi did not answer; skipped)"
fi

# -------------------------------------------------------------------- 2. data
echo "---- 2. data -----------------------------------------------------"
CFG_LIST="$LOG_ROOT/configs_$TS.txt"
HELDOUT_LIST="$LOG_ROOT/heldout_$TS.txt"
: > "$CFG_LIST"
INFER_ONLY=""   # infer configs of '-' jobs whose checkpoint must exist now
for i in "${!JOB_IDS[@]}"; do
    # A checkpoint trained by this table's own job is checked when it is written.
    [ "${JOB_TRAIN[$i]}" = "-" ] && [ -z "$(producer_of "${JOB_INFER[$i]}")" ] && INFER_ONLY="$INFER_ONLY ${JOB_INFER[$i]}"
    need_train "$i" && echo "${JOB_METHOD[$i]} ${JOB_TRAIN[$i]}" >> "$CFG_LIST"
    need_infer "$i" && echo "${JOB_METHOD[$i]} ${JOB_INFER[$i]}" >> "$CFG_LIST"
done
"$PYTHON" - "$CFG_LIST" "$HELDOUT_LIST" $INFER_ONLY <<'PY' || die "missing configs, dataset files or checkpoints (listed above)."
import sys
from pathlib import Path
from cae_suite.config_parser import parse_config

DIRS = {"mgnv": "methods/MeshGraphNets_Variational"}
missing_cfg, needed, heldout = [], {}, set()
for line in Path(sys.argv[1]).read_text().splitlines():
    m, cfg = line.split(None, 1)
    cfg = Path(cfg)
    if not cfg.is_file():
        missing_cfg.append(cfg)
        continue
    values = parse_config(cfg).values
    for key in ("dataset_dir", "infer_dataset", "eval_dataset"):
        v = values.get(key)
        if v:
            # Every path is spelled from the method dir (../../dataset/...).
            p = str((Path(DIRS[m]) / str(v)).resolve())
            needed.setdefault(p, cfg.name)
            if key != "dataset_dir" and values.get("mode") == "inference":
                heldout.add(p)

def exists_exact(p):
    p = Path(p)   # Linux opens only the case written in the config
    return p.is_file() and p.name in {c.name for c in p.parent.iterdir()}

missing = [(v, c) for v, c in sorted(needed.items()) if not exists_exact(v)]
missing_ck = []
for cfg in map(Path, sys.argv[3:]):
    if cfg.is_file():
        ck = (Path(DIRS["mgnv"]) / str(parse_config(cfg).values.get("modelpath"))).resolve()
        if not exists_exact(ck):
            missing_ck.append((ck, cfg.name))
for cfg in missing_cfg:
    print(f"  MISSING CONFIG  {cfg}")
for v, c in missing:
    print(f"  MISSING DATA    {v}   (first named by {c})")
for v, c in missing_ck:
    print(f"  MISSING CHECKPOINT  {v}   (inference-only job {c}; copy it from the round that trained it)")
if not missing_cfg and not missing and not missing_ck:
    print(f"  {len(needed)} dataset files present" + (f", {len(sys.argv) - 3} earlier checkpoints present" if len(sys.argv) > 3 else ""))
Path(sys.argv[2]).write_text("".join(f"{p}\n" for p in sorted(heldout)))
sys.exit(1 if (missing_cfg or missing or missing_ck) else 0)
PY
if [ -s "$HELDOUT_LIST" ]; then
    "$MGNV_PYTHON" - "$HELDOUT_LIST" <<'PY' || die "the held-out file is not the 256-draw one (above): copy ex3_shell_buckling_infer.h5 (1.57 GB, draws_per_shape 256) to dataset/probabilistic/."
import sys
import h5py
bad = False
for p in open(sys.argv[1]).read().split():
    with h5py.File(p, "r") as f:
        a = {k: f.attrs.get(k) for k in ("draws_per_shape", "num_samples", "n_shapes_packed")}
    ok = a["draws_per_shape"] == 256 and a["num_samples"] == 2048 and a["n_shapes_packed"] == 8
    print(f"  held-out {p}: draws_per_shape {a['draws_per_shape']}, num_samples {a['num_samples']}, "
          f"n_shapes_packed {a['n_shapes_packed']}  {'ok' if ok else 'WRONG FILE'}")
    if not ok:
        print("    the old file has 64 draws per geometry (num_samples 512, no draws_per_shape); this sweep "
              "scores 1024 draws against 256 FE results per geometry and needs the 256-draw file.")
    bad |= not ok
sys.exit(1 if bad else 0)
PY
fi

# ----------------------------------------------------------------- 3. --check
if [ "$PREFLIGHT" = "1" ] && [ -s "$CFG_LIST" ]; then
    echo "---- 3. launcher preflight (--check), $(wc -l < "$CFG_LIST") configs ----------"
    CHECK_DIR="$LOG_ROOT/check_$TS"
    mkdir -p "$CHECK_DIR"
    check_one() {   # "method cfg" -> "OK cfg" | "FAILED cfg"
        local cfg="${1#* }" extra="" log
        case "$(basename "$cfg")" in
            # The checkpoint does not exist until training ends; the real
            # launch re-runs the full preflight, path layer included.
            config_infer_*) [ "$TRAIN" = "1" ] && extra="--skip-filesystem-check" ;;
        esac
        case " $INFER_ONLY " in *" $cfg "*) extra="" ;; esac   # its checkpoint exists already
        log="$CHECK_DIR/$(basename "$cfg" .txt).log"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --python "$MGNV_PYTHON" --check $extra > "$log" 2>&1; then
            echo "OK $cfg"
        else
            echo "FAILED $cfg"
        fi
    }
    export -f check_one
    export PYTHON TRAIN CHECK_DIR INFER_ONLY MGNV_PYTHON
    tr '\n' '\0' < "$CFG_LIST" | xargs -0 -P 8 -I{} bash -c 'check_one "$1"' _ {} > "$CHECK_DIR/_results.txt"
    n_ok="$(grep -c '^OK ' "$CHECK_DIR/_results.txt")"
    echo "  $n_ok OK of $(wc -l < "$CFG_LIST")   (logs: $CHECK_DIR/)"
    bad="$(grep '^FAILED ' "$CHECK_DIR/_results.txt" | cut -d' ' -f2-)"
    if [ -n "$bad" ]; then
        for cfg in $bad; do
            echo "  FAILED $cfg"
            sed -n '/ERRORS/,$p' "$CHECK_DIR/$(basename "$cfg" .txt).log" | head -6 | sed 's/^/      /'
        done
        first="$(printf '%s\n' $bad | head -1)"
        err="$(grep -m1 -oE '\[[A-Z]+-[A-Z0-9-]+\].*' "$CHECK_DIR/$(basename "$first" .txt).log" | cut -c1-60)"
        die "preflight failed ($(printf '%s\n' $bad | wc -l | tr -d ' ')): $(basename "$first" .txt | sed 's/^config_//') $err"
    fi
fi
echo "  [$(date +%T)] $ROUND node $NODE checks passed"
if [ ! -s "$CFG_LIST" ]; then
    echo "  every job of $ROUND is done already (REDO=1 or JOBS=... runs one again)"
    "$PYTHON" -B "$SCRIPT_DIR/brief.py" "$ROUND"
    exit 0
fi

if [ -z "${DETACH:-}" ]; then
    if [ -t 1 ] || [ -t 0 ]; then DETACH=1; else DETACH=0; fi
fi
if [ "$DETACH" = "1" ]; then
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    SWEEP_PHASE=run nohup $SETSID bash "$SCRIPT" "$NODE" "$ROUND" >> "$RUN_LOG" 2>&1 < /dev/null &
    echo
    echo "Checks passed. The jobs now run in the background; this terminal can close."
    echo "  progress : tail -f $RUN_LOG"
    echo "  result   : python $SWEEP_DIR/brief.py $ROUND   (also the end of $RUN_LOG)"
    exit 0
fi
SWEEP_PHASE=run exec bash "$SCRIPT" "$NODE" "$ROUND" >> "$RUN_LOG" 2>&1
