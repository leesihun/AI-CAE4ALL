#!/usr/bin/env bash
# B300 round 4: 7 jobs on node 135's 7 cards, one per card (GPU1 is left
# out, see run_135.sh). Do not call this file directly; run, from this directory:
#
#   bash run_135.sh
#
# jobs.txt is the job table (node, job, method, train config, infer config);
# README.md says what each job changes and why. Job k starts on the k-th
# card. The 13 later jobs are commented out at the end of jobs.txt; with more
# active jobs than cards, each card that finishes takes the next one.
#
# No argument and no nohup: the checks (stages 1-3) run in the terminal, and
# when they pass the jobs go to the background on their own, so the terminal
# can close.
#   progress:  tail -f output/b300_r4/run_<node>.out
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py
# The run log ends with one block between "==== TYPE THIS BACK ====" and
# "==== END ====" (brief.py). brief.py prints the same block at any time.
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; each method's
#                   interpreter (MGN-V, SAR, LDGN) imports torch /
#                   torch_geometric / h5py and runs a bf16 matmul and a
#                   scatter on GPU 0; this checkout has split_group_attr.
#   2. data         every dataset a config names exists, exact case.
#   3. --check      the launcher's full preflight on all of the node's configs,
#                   8 at a time (inference configs without the path layer while
#                   their checkpoint does not exist yet).
#   Any failure in 1-3 aborts before a GPU-hour is spent, says why, and
#   prints a one-line block to type back.
#   4. jobs         per job: train, then its inference (64 samples per
#                   held-out geometry -> spread_metrics.json, one histogram
#                   per geometry), then ../plots.py (loss curves; never fails
#                   the job).
#   5. summary      ok / the failed stage per job, then the brief.py block.
#
# Status for brief.py: output/b300_r4/run_logs/<node>/status/
# <job>.{state,err,train_start,train_end,inferfail}, _start, _end, _abort.
# Stdout per stage: run_logs/<node>/<job>.<train|infer|plots>.out.
#
# Overrides, in front of the command (GPUS="0 2 3" bash run_135.sh):
#   PYTHON (launcher), MGNV_PYTHON, SAR_PYTHON, LDGN_PYTHON (method interpreters; default
#   from ai_cae4all.local.toml), GPUS (run_135.sh: 0 2 3 4 5 6 7),
#   THREADS_PER_JOB=4, PROBE_TIMEOUT=900, PREFLIGHT=1, TRAIN=1, INFER=1,
#   DETACH=1 (0 keeps the jobs in this terminal; the default when stdout is
#   not a terminal), JOBS="sar_wing_std ldgn_wing_std" (a subset, in this order),
#   LANE=b (a second run beside one still going: its own run_<node>_b.out/.pid,
#   and it leaves the hierarchy cache and the run hours of the first alone).
# Two jobs beside a running node:  LANE=b GPUS="0 2" JOBS="sar_wing_std ldgn_wing_std" bash run_135.sh
# Inference-only rerun:  TRAIN=0 bash run_135.sh
# One job again:         JOBS=hybrid_mmd bash run_135.sh
# A later job:           uncomment its line in jobs.txt, then JOBS=himgn_det bash run_135.sh
# End the queue early:   touch output/b300_r4/stop_135   (running jobs finish, no new
#                        job starts; the next run deletes it)
#
# The script calls itself: R4_PHASE=run is the background driver (stages 4-5).
set -uo pipefail

NODE="${1:-}"
case "$NODE" in
    135) ;;
    *) echo "usage: bash run_135.sh   (run_node.sh takes the node: 135)"; exit 2 ;;
esac
PHASE="${R4_PHASE:-}"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 8 trainers per node, each with 4 torch threads + 4 loader workers. Without
# the caps every pool starts one thread per core and the CPU sits at 100%
# while the GPUs wait (round 1).
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
JOB_TABLE="$SWEEP_DIR/jobs.txt"

GPUS="${GPUS:-}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"

OUT_ROOT="output/b300_r4"
LOG_ROOT="$OUT_ROOT/run_logs/$NODE"
STATUS_DIR="$LOG_ROOT/status"
LANE="${LANE:-}"
export LANE
RUN_LOG="$OUT_ROOT/run_$NODE${LANE:+_$LANE}.out"
PID_FILE="$OUT_ROOT/run_$NODE${LANE:+_$LANE}.pid"
mkdir -p "$LOG_ROOT" "$STATUS_DIR"

TS="${R4_TS:-$(date +%Y%m%d_%H%M%S)}"
export R4_TS="$TS"
MASTER_LOG="$LOG_ROOT/node_$TS.out"

# ---------------------------------------------------------------- job table
JOB_IDS=(); JOB_METHOD=(); JOB_TRAIN=(); JOB_INFER=()
while read -r n jid meth tcfg icfg; do
    case "$n" in ''|'#'*) continue ;; esac
    [ "$n" = "$NODE" ] || continue
    JOB_IDS+=("$jid"); JOB_METHOD+=("$meth"); JOB_TRAIN+=("$tcfg"); JOB_INFER+=("$icfg")
done < "$JOB_TABLE"
if [ -n "${JOBS:-}" ]; then   # a subset, in the order given
    ids=(); ms=(); trs=(); ifs=()
    for j in $JOBS; do
        found=""; for i in "${!JOB_IDS[@]}"; do [ "${JOB_IDS[$i]}" = "$j" ] && found="$i"; done
        [ -n "$found" ] || { echo "JOBS entry '$j' is not a job of node $NODE (jobs: ${JOB_IDS[*]}; a later job is a commented line in jobs.txt: uncomment it first)."; exit 2; }
        ids+=("${JOB_IDS[$found]}"); ms+=("${JOB_METHOD[$found]}")
        trs+=("${JOB_TRAIN[$found]}"); ifs+=("${JOB_INFER[$found]}")
    done
    JOB_IDS=("${ids[@]}"); JOB_METHOD=("${ms[@]}"); JOB_TRAIN=("${trs[@]}"); JOB_INFER=("${ifs[@]}")
fi
NJ=${#JOB_IDS[@]}
METHODS="$(printf '%s\n' "${JOB_METHOD[@]}" | sort -u | tr '\n' ' ')"

method_info() {   # m -> "model_id spec_id method_dir label"
    case "$1" in
        mgnv) echo "meshgraphnets-v meshgraphnets_variational methods/MeshGraphNets_Variational MGN-V" ;;
        sar)  echo "sar sar methods/SAR SAR" ;;
        ldgn) echo "ldgn ldgn methods/LDGN LDGN" ;;
        *)    echo "" ;;
    esac
}
method_python() {   # m -> the interpreter chosen in stage 1
    case "$1" in mgnv) echo "$MGNV_PYTHON" ;; sar) echo "$SAR_PYTHON" ;; ldgn) echo "$LDGN_PYTHON" ;; esac
}

cfg_value() {   # cfg key -> value (first token, comment stripped)
    sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#,]*\).*/\1/p" "$1" | head -1
}

# ======================================================= the background driver
if [ "$PHASE" = "run" ]; then
    exec > >(tee -a "$MASTER_LOG") 2>&1
    echo "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    GPU_ARR=($GPUS)
    NG=${#GPU_ARR[@]}
    echo "=================================================================="
    echo " $(date '+%F %T')  B300 round 4, node $NODE  (run $TS, host $(hostname))"
    echo " $NJ jobs on GPUs $GPUS; CPU threads per job: torch $OMP_NUM_THREADS, BLAS $OPENBLAS_NUM_THREADS ($(nproc) CPUs)"
    echo "=================================================================="

    # The hierarchy cache is run-private (one per node and coarsening). A cache
    # left by an aborted run may hold a half-written variant; start clean.
    # A LANE run starts beside a running one, whose jobs may have it open.
    if [ "$TRAIN" = "1" ] && [ -z "$LANE" ]; then
        for d in "$OUT_ROOT"/mscache/mgnv_*; do
            [ -e "$d" ] || continue
            echo "  removing the leftover hierarchy cache $d"
            rm -rf "$d"
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
        local py; py="$(method_python "${JOB_METHOD[$i]}")"
        local st="$STATUS_DIR/$jid" ck log marker out_dir rc
        rm -f "$st.state" "$st.err" "$st.inferfail"
        ck="$(cfg_value "$tcfg" modelpath)"; ck="${ck#../../}"

        if [ "$TRAIN" = "1" ]; then
            rm -f "$st.train_start" "$st.train_end"
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
            echo "  [gpu $gpu] $jid  train done   $(date '+%m-%d %H:%M')"
        elif [ ! -f "$ck" ]; then
            echo "fail train no-checkpoint" > "$st.state"
            echo "TRAIN=0 but $ck does not exist" > "$st.err"
            echo "  [gpu $gpu] $jid  TRAIN=0 but $ck does not exist"
            return
        fi

        if [ "$INFER" = "1" ]; then
            out_dir="$(cfg_value "$icfg" inference_output_dir)"; out_dir="${out_dir#../../}"
            log="$LOG_ROOT/$jid.infer.out"
            marker="$LOG_ROOT/$jid.infer.marker"
            : > "$marker"
            echo "infer" > "$st.state"
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$py" --config "$icfg" > "$log" 2>&1
            rc=$?
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
        # Loss curves (and any missing per-geometry histogram) as PNGs in the run dir.
        CUDA_VISIBLE_DEVICES="$gpu" "$py" -B "$SCRIPT_DIR/../plots.py" "$SCRIPT_DIR" "$jid"             > "$LOG_ROOT/$jid.plots.out" 2>&1             || echo "  [gpu $gpu] $jid  plots failed, the job still counts -- $LOG_ROOT/$jid.plots.out"
        echo "done" > "$st.state"
    }

    # ---------------------------------------------------------------- 4. jobs
    QUEUE_DIR="$LOG_ROOT/queue_$TS"
    mkdir -p "$QUEUE_DIR"
    echo "---- 4. jobs: $NJ over $NG GPU(s) ----------------------------------"
    echo "  stdout: $LOG_ROOT/<job>.<train|infer>.out   status: $STATUS_DIR/"
    for i in "${!JOB_IDS[@]}"; do echo "queued" > "$STATUS_DIR/${JOB_IDS[$i]}.state"; done
    # Run hours (brief.py) count the training run; an inference-only rerun keeps them.
    if [ "$TRAIN" = "1" ] && [ -z "$LANE" ]; then date +%s > "$STATUS_DIR/_start"; rm -f "$STATUS_DIR/_end"; fi
    # The first round is dealt before any card starts (card s takes job s), so
    # a job that fails in seconds cannot let its card steal the next card's job.
    # After that a card takes the next job nobody has claimed, unless
    # output/b300_r4/stop_<node> exists (touch it to end the queue early).
    for ((s = 0; s < NG && s < NJ; s++)); do mkdir "$QUEUE_DIR/claim_$s"; done
    pids=""
    for ((s = 0; s < NG; s++)); do
        gpu="${GPU_ARR[$s]}"
        (
            [ "$s" -lt "$NJ" ] && run_job "$gpu" "$s"
            for i in "${!JOB_IDS[@]}"; do
                [ -e "$OUT_ROOT/stop_$NODE" ] && break   # no new jobs; running ones finish
                mkdir "$QUEUE_DIR/claim_$i" 2>/dev/null || continue
                run_job "$gpu" "$i"
            done
        ) &
        pids="$pids $!"
        sleep 2   # stagger the starts: 8 interpreters importing torch at once
    done
    for pid in $pids; do wait "$pid"; done
    [ "$TRAIN" = "1" ] && [ -z "$LANE" ] && date +%s > "$STATUS_DIR/_end"

    # ------------------------------------------------------------- 5. summary
    rc=0
    echo "=================================================================="
    echo " summary, node $NODE"
    echo "=================================================================="
    for jid in "${JOB_IDS[@]}"; do
        state="$(cat "$STATUS_DIR/$jid.state" 2>/dev/null || echo 'did not run')"
        extra=""
        [ -f "$STATUS_DIR/$jid.inferfail" ] && extra="  failed inference: $(tr '\n' ' ' < "$STATUS_DIR/$jid.inferfail")"
        printf '  %-18s %s%s\n' "$jid" "$state" "$extra"
        { [ "$state" = "done" ] && [ -z "$extra" ]; } || rc=1
    done
    echo "Finished with rc=$rc   $(date '+%F %T')"
    echo
    "$PYTHON" -B "$SCRIPT_DIR/brief.py" "$NODE"
    exit "$rc"
fi

# ============================================= the command: checks, then detach
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Node $NODE is already running here (pid $(cat "$PID_FILE")): tail -f $RUN_LOG"
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
    echo "R4 $NODE ABORT $(cat "$STATUS_DIR/_abort")"
    echo "==== END ===="
    exit 1
}

echo "=================================================================="
echo " B300 round 4, node $NODE, checks   ($TS, host $(hostname))"
echo "=================================================================="
[ "$NJ" -ge 1 ] || die "jobs.txt has no job for node $NODE."

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
echo "  [$(date +%T)] launcher python '$PYTHON': import cae_suite"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."

PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
NGPU_SEEN=""
for m in $METHODS; do
    read -r model_id spec_id method_dir label <<< "$(method_info "$m")"
    [ -n "$model_id" ] || die "jobs.txt names an unknown method '$m'."
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
MGNV_PYTHON="${MGNV_PYTHON:-}"; SAR_PYTHON="${SAR_PYTHON:-}"; LDGN_PYTHON="${LDGN_PYTHON:-}"

[ -n "$GPUS" ] || GPUS="$(seq -s ' ' 0 $(( NGPU_SEEN - 1 )))"
for g in $GPUS; do
    case "$g" in ''|*[!0-9]*) die "GPUS entry '$g' is not a GPU index." ;; esac
    [ "$g" -lt "$NGPU_SEEN" ] || die "GPU $g does not exist: this machine has $NGPU_SEEN GPU(s)."
done
export GPUS
echo "  commit          : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPUs            : $GPUS  ($NGPU_SEEN on this machine)"
echo "  jobs            : $NJ  (${JOB_IDS[*]})"
if command -v nvidia-smi >/dev/null 2>&1; then
    timeout -k 5 60 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /' || echo "      (nvidia-smi did not answer; skipped)"
fi

# -------------------------------------------------------------------- 2. data
echo "---- 2. data -----------------------------------------------------"
CFG_LIST="$LOG_ROOT/configs_$TS.txt"
: > "$CFG_LIST"
for i in "${!JOB_IDS[@]}"; do
    [ "$TRAIN" = "1" ] && echo "${JOB_METHOD[$i]} ${JOB_TRAIN[$i]}" >> "$CFG_LIST"
    [ "$INFER" = "1" ] && echo "${JOB_METHOD[$i]} ${JOB_INFER[$i]}" >> "$CFG_LIST"
done
"$PYTHON" - "$CFG_LIST" <<'PY' || die "missing configs or dataset files (listed above)."
import sys
from pathlib import Path
from cae_suite.config_parser import parse_config

DIRS = {"mgnv": "methods/MeshGraphNets_Variational", "sar": "methods/SAR", "ldgn": "methods/LDGN"}
missing_cfg, needed = [], {}
for line in Path(sys.argv[1]).read_text().splitlines():
    m, cfg = line.split(None, 1)
    cfg = Path(cfg)
    if not cfg.is_file():
        missing_cfg.append(cfg)
        continue
    for key in ("dataset_dir", "infer_dataset", "eval_dataset"):
        v = parse_config(cfg).values.get(key)
        if v:
            # Every path is spelled from the method dir (../../dataset/...).
            needed.setdefault(str((Path(DIRS[m]) / str(v)).resolve()), cfg.name)

def exists_exact(p):
    p = Path(p)   # Linux opens only the case written in the config
    return p.is_file() and p.name in {c.name for c in p.parent.iterdir()}

missing = [(v, c) for v, c in sorted(needed.items()) if not exists_exact(v)]
for cfg in missing_cfg:
    print(f"  MISSING CONFIG  {cfg}")
for v, c in missing:
    print(f"  MISSING DATA    {v}   (first named by {c})")
if not missing_cfg and not missing:
    print(f"  {len(needed)} dataset files present")
sys.exit(1 if (missing_cfg or missing) else 0)
PY

# ----------------------------------------------------------------- 3. --check
if [ "$PREFLIGHT" = "1" ] && { [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; }; then
    echo "---- 3. launcher preflight (--check), $(wc -l < "$CFG_LIST") configs ----------"
    CHECK_DIR="$LOG_ROOT/check_$TS"
    mkdir -p "$CHECK_DIR"
    check_one() {   # "method cfg" -> "OK cfg" | "FAILED cfg"
        local m="${1%% *}" cfg="${1#* }" extra="" log py
        case "$m" in mgnv) py="$MGNV_PYTHON" ;; sar) py="$SAR_PYTHON" ;; ldgn) py="$LDGN_PYTHON" ;; esac
        case "$(basename "$cfg")" in
            # The checkpoint does not exist until training ends; the real
            # launch re-runs the full preflight, path layer included.
            config_infer_*) [ "$TRAIN" = "1" ] && extra="--skip-filesystem-check" ;;
        esac
        log="$CHECK_DIR/$(basename "$cfg" .txt).log"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --python "$py" --check $extra > "$log" 2>&1; then
            echo "OK $cfg"
        else
            echo "FAILED $cfg"
        fi
    }
    export -f check_one
    export PYTHON TRAIN CHECK_DIR
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
echo "  [$(date +%T)] node $NODE checks passed"
{ [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; } || exit 0

if [ -z "${DETACH:-}" ]; then
    if [ -t 1 ] || [ -t 0 ]; then DETACH=1; else DETACH=0; fi
fi
if [ "$DETACH" = "1" ]; then
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    R4_PHASE=run nohup $SETSID bash "$SCRIPT" "$NODE" >> "$RUN_LOG" 2>&1 < /dev/null &
    echo
    echo "Checks passed. The jobs now run in the background; this terminal can close."
    echo "  progress : tail -f $RUN_LOG"
    echo "  result   : python $SWEEP_DIR/brief.py   (also the end of $RUN_LOG)"
    exit 0
fi
R4_PHASE=run exec bash "$SCRIPT" "$NODE" >> "$RUN_LOG" 2>&1
