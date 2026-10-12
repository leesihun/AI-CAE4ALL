#!/usr/bin/env bash
# B300 round 2 of this method's sweep. One 8-GPU node per method, both studies
# at once, so every card holds one SAOI job and one shell-buckling job:
#
#   node 1:  cd configs/HI_MGNFlow/hyperparameter_sweep && ./run_sweep.sh
#   node 2:  cd configs/MeshGraphNets_Variational/hyperparameter_sweep && ./run_sweep.sh
#
# The same file sits, byte for byte, in both directories, and the directory it
# runs from picks the method. No argument and no nohup: the checks (stages 1-3
# below, both studies) run in the terminal, and when they pass the jobs go to
# the background on their own, so the terminal can close.
#   progress:  tail -f output/b300_r2/run_sweep_<chi|mgnv>.out
#   result:    python configs/<method>/hyperparameter_sweep/brief.py
# The run log ends with one block between "==== TYPE THIS BACK ====" and
# "==== END ====" (brief.py): both studies of this method, short enough to
# retype. brief.py prints the same block at any time while the run goes.
# `./run_sweep.sh SAOI` or `./run_sweep.sh probabilistic` runs one study only.
#
# THE JOBS (README.md has the questions; every config header names its change)
#   SAOI           arm x board half, configs in SAOI/:
#     c_smp c_grp c_grp_rep c_nopart  x {bot,top}    cHI-MGNflow
#     v_smp v_grp v_grp_rep v_nopart  x {bot,top}    MeshGraphNets-V
#   probabilistic  one job per arm (ex3.h5 -> ex3_infer.h5), configs in probabilistic/:
#     c_full_g512 c_full c_full_rep c_full_b4 c_full_g256 c_full_w128 c_full_short c_tok
#     v_base v_rep v_mp8 v_mp8_pmin01 v_mp8_aux100 v_mp8_mmd10 v_mp12 v_mp8_w256
#   Job k of each study runs on card k. A slot that finishes takes the next
#   unclaimed job of its own study, so fewer GPUs also work.
#
# ORDER OF WORK, per study
#   1. environment  launcher python imports cae_suite; this method's
#                   interpreter imports torch / torch_geometric / h5py and runs
#                   a bf16 matmul and a scatter on this GPU; this checkout has
#                   split_group_attr (and prior_global, for cHI-MGNflow).
#   2. data         (SAOI) SAOI/saoi_heldout.py: geometry labels -> the
#                   *_bygeom.h5 training copies, held-out part files and this
#                   method's configs for them; then every dataset a config
#                   names exists, exact case.
#   3. --check      the launcher's full preflight on every config, 8 at a time
#                   (inference configs without the path layer while their
#                   checkpoint does not exist yet).
#   Any failure in 1-3 aborts before a GPU-hour is spent, says why, and prints
#   a one-line block to type back.
#   4. jobs         per job: train, then every inference config of that job
#                   (official eval parts first, then held-out parts). A failed
#                   inference does not stop the next one.
#   5. summary      ok / the failed stage per job, with its log tail.
#
# Status for brief.py: output/b300_r2/<saoi|ex3>/run_logs/<chi|mgnv>/status/
# <job>.{state,err,train_start,train_end,inferfail} and _abort. Stdout per
# stage: run_logs/<chi|mgnv>/<job>.<stage>.out; each study's whole log:
# run_logs/<chi|mgnv>/sweep_<time>.out.
#
# Useful overrides, in front of the command (GPUS="0 1 2 3" ./run_sweep.sh):
#   PYTHON (launcher), METHOD_PYTHON (this method), GPUS (default: every GPU
#   visible), JOBS_PER_GPU=1 (per study), THREADS_PER_JOB=4,
#   PROBE_TIMEOUT=900, PREFLIGHT=1, TRAIN=1, INFER=1,
#   DETACH=1 (0 keeps the jobs in this terminal; the default when stdout is
#   not a terminal, e.g. under nohup),
#   JOBS="c_grp.bot c_grp.top" (a subset of one study, in this order)
# Inference-only rerun:  TRAIN=0 ./run_sweep.sh probabilistic
# One job again:         JOBS=c_grp.bot ./run_sweep.sh SAOI
#
# The script calls itself: R2_PHASE=check runs stages 1-3 of one study in the
# terminal, R2_PHASE=run is the background driver, and R2_PHASE=jobs runs
# stages 1, 2, 4 and 5 of one study under it.
set -uo pipefail

case "${1:-}" in
    "")                     SWEEP="";              NODE="" ;;
    SAOI|saoi)              SWEEP="SAOI";          NODE="saoi" ;;
    probabilistic|ex3)      SWEEP="probabilistic"; NODE="ex3" ;;
    *) echo "usage: ./run_sweep.sh [SAOI|probabilistic]   (no argument: both studies)"; exit 2 ;;
esac
PHASE="${R2_PHASE:-}"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 8 trainers per study, both studies at once, each with 4 torch threads + 4
# loader workers = 128 threads on a 128-core node. Without the caps every pool
# starts one thread per core and the CPU sits at 100% while the GPUs wait
# (round 1, two sweeps per node).
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

case "$(basename "$(dirname "$SCRIPT_DIR")")" in
    HI_MGNFlow)
        M="chi"; LABEL="cHI-MGNflow"; MODEL_ID="chi-mgnflow"; SPEC_ID="chi-mgnflow"
        METHOD_DIR="methods/HI_MGNFlow"
        SAOI_ARMS="c_smp c_grp c_grp_rep c_nopart"
        EX3_ARMS="c_full_g512 c_full c_full_rep c_full_b4 c_full_g256 c_full_w128 c_full_short c_tok" ;;
    MeshGraphNets_Variational)
        M="mgnv"; LABEL="MGN-V"; MODEL_ID="meshgraphnets-v"; SPEC_ID="meshgraphnets_variational"
        METHOD_DIR="methods/MeshGraphNets_Variational"
        SAOI_ARMS="v_smp v_grp v_grp_rep v_nopart"
        EX3_ARMS="v_base v_rep v_mp8 v_mp8_pmin01 v_mp8_aux100 v_mp8_mmd10 v_mp12 v_mp8_w256" ;;
    *) echo "run_sweep.sh must sit in configs/HI_MGNFlow/ or configs/MeshGraphNets_Variational/hyperparameter_sweep/"; exit 2 ;;
esac

GPUS="${GPUS:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"

# ---------------------------------------------- the command, and the driver
if [ "$PHASE" != "check" ] && [ "$PHASE" != "jobs" ]; then
    STUDIES="${SWEEP:-SAOI probabilistic}"
    RUN_LOG="output/b300_r2/run_sweep_$M.out"
    PID_FILE="output/b300_r2/run_sweep_$M.pid"
    if [ -z "$PHASE" ]; then
        if [ -n "${JOBS:-}" ] && [ -z "$SWEEP" ]; then
            echo "JOBS names jobs of one study: JOBS=\"$JOBS\" ./run_sweep.sh SAOI|probabilistic"
            exit 2
        fi
        mkdir -p output/b300_r2
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "A $LABEL sweep is already running here (pid $(cat "$PID_FILE")): tail -f $RUN_LOG"
            echo "If it is not, delete $PID_FILE and start again."
            exit 2
        fi
        R2_TS="$(date +%Y%m%d_%H%M%S)"
        export R2_TS
        for st in $STUDIES; do
            R2_PHASE=check bash "$SCRIPT" "$st" || exit 1
        done
        { [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; } || exit 0
        if [ -z "${DETACH:-}" ]; then
            if [ -t 1 ]; then DETACH=1; else DETACH=0; fi
        fi
        if [ "$DETACH" = "1" ]; then
            SETSID=""
            command -v setsid >/dev/null 2>&1 && SETSID="setsid"
            R2_PHASE=run nohup $SETSID bash "$SCRIPT" $SWEEP >> "$RUN_LOG" 2>&1 < /dev/null &
            echo
            echo "Checks passed. The jobs now run in the background; this terminal can close."
            echo "  progress : tail -f $RUN_LOG"
            echo "  result   : python ${SCRIPT_DIR#"$REPO_ROOT"/}/brief.py   (also the end of $RUN_LOG)"
            exit 0
        fi
        R2_PHASE=run exec bash "$SCRIPT" $SWEEP
    fi
    # R2_PHASE=run: the studies' jobs at once, then one block for all of them.
    echo "$$" > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    echo "=================================================================="
    echo " $(date '+%F %T')  $LABEL  ${SWEEP:-SAOI + probabilistic}  (run $R2_TS, host $(hostname))"
    echo "=================================================================="
    pids=""
    for st in $STUDIES; do
        if [ "$st" = "SAOI" ]; then tag="saoi"; else tag="ex3"; fi
        ( R2_PHASE=jobs PREFLIGHT=0 bash "$SCRIPT" "$st" 2>&1 | sed -u "s/^/[$tag] /"
          exit "${PIPESTATUS[0]}" ) &
        pids="$pids $!"
        sleep 1
    done
    rc=0
    for pid in $pids; do wait "$pid" || rc=1; done
    echo "Finished with rc=$rc   $(date '+%F %T')"
    echo
    "$PYTHON" -B "$SCRIPT_DIR/brief.py" $SWEEP
    exit "$rc"
fi

# ------------------------------------------------- one study: check or jobs
if [ "$NODE" = "saoi" ]; then ARMS="$SAOI_ARMS"; else ARMS="$EX3_ARMS"; fi
CFG_DIR="configs/$(basename "$(dirname "$SCRIPT_DIR")")/hyperparameter_sweep/$SWEEP"
if [ "$NODE" = "saoi" ]; then HALVES="bot top"; else HALVES=""; fi
OUT_ROOT="output/b300_r2/$NODE"
LOG_ROOT="$OUT_ROOT/run_logs/$M"
STATUS_DIR="$LOG_ROOT/status"
HELDOUT_CFG="output/b300_r2/saoi/heldout/configs/$M"

TS="${R2_TS:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT" "$STATUS_DIR"
rm -f "$STATUS_DIR/_abort"
MASTER_LOG="$LOG_ROOT/sweep_$TS.out"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo
    echo "ABORT: $*"
    echo "Nothing was trained. Full log: $MASTER_LOG"
    printf '%s\n' "$*" | head -1 | cut -c1-80 > "$STATUS_DIR/_abort"
    if [ "$PHASE" = "check" ]; then
        echo "==== TYPE THIS BACK ===="
        echo "R2 $NODE $M ABORT $(cat "$STATUS_DIR/_abort")"
        echo "==== END ===="
    fi
    exit 1
}

echo "=================================================================="
echo " B300 round 2, $SWEEP, $LABEL, $PHASE   ($TS, host $(hostname))"
echo " CPU threads per job: torch $OMP_NUM_THREADS, BLAS $OPENBLAS_NUM_THREADS ($(nproc) CPUs)"
echo "=================================================================="

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
echo "  [$(date +%T)] launcher python '$PYTHON': import cae_suite"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."

if [ -z "${METHOD_PYTHON:-}" ]; then
    METHOD_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('$MODEL_ID', '$SPEC_ID'))
")" || die "could not resolve the $MODEL_ID interpreter (ai_cae4all.local.toml); set METHOD_PYTHON."
fi
[ -x "$METHOD_PYTHON" ] || command -v "$METHOD_PYTHON" >/dev/null 2>&1 \
    || die "$LABEL interpreter '$METHOD_PYTHON' does not exist."

PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
echo "  [$(date +%T)] probe '$METHOD_PYTHON' (gives up after ${PROBE_TIMEOUT}s):"
out="$(env -u CUDA_VISIBLE_DEVICES timeout -k 30 "$PROBE_TIMEOUT" "$METHOD_PYTHON" -c "
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
    124|137) die "the $LABEL probe hung past ${PROBE_TIMEOUT}s in the last step printed above (slow filesystem: PROBE_TIMEOUT=1800; CUDA init: driver / nvidia-fabricmanager; bf16 matmul without sm_100 kernels: install a CUDA 12.8+ torch)." ;;
    *) die "$LABEL interpreter '$METHOD_PYTHON' cannot import torch / torch_geometric / h5py or run a CUDA kernel (traceback above)." ;;
esac
NGPU_SEEN="$(printf '%s\n' "$out" | sed -n 1p)"
METHOD_INFO="$(printf '%s\n' "$out" | sed -n 2p)"
[ "${NGPU_SEEN:-0}" -ge 1 ] 2>/dev/null || die "$LABEL interpreter '$METHOD_PYTHON' sees no CUDA device."

[ -n "$GPUS" ] || GPUS="$(seq -s ' ' 0 $(( NGPU_SEEN - 1 )))"
for g in $GPUS; do
    case "$g" in ''|*[!0-9]*) die "GPUS entry '$g' is not a GPU index." ;; esac
    [ "$g" -lt "$NGPU_SEEN" ] || die "GPU $g does not exist: this machine has $NGPU_SEEN GPU(s)."
done
case "$JOBS_PER_GPU" in ''|*[!0-9]*|0) die "JOBS_PER_GPU='$JOBS_PER_GPU' is not a positive integer." ;; esac
GPU_ARR=($GPUS)
NG=${#GPU_ARR[@]}

if [ "$M" = "chi" ]; then
    grep -q "PRIOR_GLOBAL_MODES" "$METHOD_DIR/model/autoencoder.py" \
        || die "this checkout has no prior_global ($METHOD_DIR/model/autoencoder.py): every token arm would train without it. Pull first."
fi
grep -q "split_group_attr" "$METHOD_DIR/general_modules/mesh_dataset.py" \
    || die "this checkout's $METHOD_DIR has no split_group_attr: the grouped arms would split by sample. Pull first."

echo "  launcher python : $(command -v "$PYTHON" || echo "$PYTHON")"
printf '  %-15s : %s\n' "$LABEL" "$METHOD_PYTHON"
echo "                    $METHOD_INFO"
echo "  commit          : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPUs            : $GPUS  x $JOBS_PER_GPU jobs each  ($NG of $NGPU_SEEN on this machine)"
if [ "$PHASE" = "check" ] && command -v nvidia-smi >/dev/null 2>&1; then
    timeout -k 5 60 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /' || echo "      (nvidia-smi did not answer; skipped)"
fi

# ---------------------------------------------------------------- job table
# One job = one training config + its inference configs, in ARMS order: slot
# i of the first round takes job i, and slot order is card-major per round,
# so job k lands on card k next to the other study's job k.
JOB_IDS=(); JOB_TRAIN=(); JOB_INFER=()
add_job() {   # jid train_cfg infer_cfg...
    JOB_IDS+=("$1"); JOB_TRAIN+=("$2"); shift 2; JOB_INFER+=("$*")
}
for arm in $ARMS; do
    if [ -n "$HALVES" ]; then
        for half in $HALVES; do
            add_job "$arm.$half" "$CFG_DIR/config_train_${arm}_${half}.txt" \
                "$CFG_DIR/config_infer_${arm}_${half}_s26fe_main.txt" \
                "$CFG_DIR/config_infer_${arm}_${half}_s26fe_sec.txt" \
                "$CFG_DIR/config_infer_${arm}_${half}_sm_l345u_main.txt" \
                "@heldout:${arm}_${half}"   # expanded after saoi_heldout.py has written them
        done
    else
        add_job "$arm" "$CFG_DIR/config_train_${arm}.txt" "$CFG_DIR/config_infer_${arm}.txt"
    fi
done
if [ -n "${JOBS:-}" ]; then   # a subset, in the order given
    keep=(); for j in $JOBS; do
        found=""; for i in "${!JOB_IDS[@]}"; do [ "${JOB_IDS[$i]}" = "$j" ] && found="$i"; done
        [ -n "$found" ] || die "JOBS entry '$j' is not a $LABEL job of $SWEEP (names: ${JOB_IDS[*]})."
        keep+=("$found")
    done
    ids=(); trs=(); ifs=()
    for i in "${keep[@]}"; do ids+=("${JOB_IDS[$i]}"); trs+=("${JOB_TRAIN[$i]}"); ifs+=("${JOB_INFER[$i]}"); done
    JOB_IDS=("${ids[@]}"); JOB_TRAIN=("${trs[@]}"); JOB_INFER=("${ifs[@]}")
fi
NJ=${#JOB_IDS[@]}
echo "  jobs            : $NJ  (${JOB_IDS[*]})"

# -------------------------------------------------------------------- 2. data
echo "---- 2. data -----------------------------------------------------"
if [ "$NODE" = "saoi" ]; then
    echo "  [$(date +%T)] SAOI/saoi_heldout.py (geometry labels, held-out parts, $LABEL configs for them)"
    "$METHOD_PYTHON" -B "$SCRIPT_DIR/SAOI/saoi_heldout.py" --method "$M" | sed 's/^/    /'
    [ "${PIPESTATUS[0]}" = "0" ] || die "saoi_heldout.py failed (output above)."
    for i in "${!JOB_IDS[@]}"; do
        list=""
        for c in ${JOB_INFER[$i]}; do
            case "$c" in
                @heldout:*) for h in "$HELDOUT_CFG"/config_infer_"${c#@heldout:}"_test*.txt; do
                                [ -f "$h" ] && list="$list $h"
                            done ;;
                *) list="$list $c" ;;
            esac
        done
        JOB_INFER[$i]="${list# }"
    done
fi
CFG_LIST="$LOG_ROOT/configs_$TS.txt"
: > "$CFG_LIST"
for i in "${!JOB_IDS[@]}"; do
    [ "$TRAIN" = "1" ] && echo "${JOB_TRAIN[$i]}" >> "$CFG_LIST"
    if [ "$INFER" = "1" ]; then for c in ${JOB_INFER[$i]}; do echo "$c" >> "$CFG_LIST"; done; fi
done
"$PYTHON" - "$CFG_LIST" "$METHOD_DIR" <<'PY' || die "missing configs or dataset files (listed above)."
import sys
from pathlib import Path
from cae_suite.config_parser import parse_config

missing_cfg, needed = [], {}
for line in Path(sys.argv[1]).read_text().splitlines():
    cfg = Path(line)
    if not cfg.is_file():
        missing_cfg.append(cfg)
        continue
    for key in ("dataset_dir", "infer_dataset", "eval_dataset"):
        v = parse_config(cfg).values.get(key)
        if v:
            # Every path is spelled from the method dir (../../dataset/...).
            needed.setdefault(str((Path(sys.argv[2]) / str(v)).resolve()), cfg.name)

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
    check_one() {   # cfg -> "OK cfg" | "FAILED cfg"
        local cfg="$1" extra="" log
        case "$(basename "$cfg")" in
            # The checkpoint does not exist until training ends; the real
            # launch re-runs the full preflight, path layer included.
            config_infer_*) [ "$TRAIN" = "1" ] && extra="--skip-filesystem-check" ;;
        esac
        log="$CHECK_DIR/$(basename "$cfg" .txt).log"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --python "$METHOD_PYTHON" --check $extra > "$log" 2>&1; then
            echo "OK $cfg"
        else
            echo "FAILED $cfg"
        fi
    }
    export -f check_one
    export PYTHON METHOD_PYTHON TRAIN CHECK_DIR
    xargs -P 8 -I{} bash -c 'check_one "$@"' _ {} < "$CFG_LIST" > "$CHECK_DIR/_results.txt"
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
if [ "$PHASE" = "check" ]; then
    echo "  [$(date +%T)] $SWEEP checks passed"
    exit 0
fi

# The hierarchy caches are sweep-private, one per study, method and coarsening
# (<saoi|ex3>/mscache/<chi|mgnv>_<coarsening>). A cache left by an aborted run
# may hold a half-written variant; start clean.
if [ "$TRAIN" = "1" ]; then
    for d in "$OUT_ROOT/mscache/${M}"_*; do
        [ -e "$d" ] || continue
        echo "  removing the leftover hierarchy cache $d"
        rm -rf "$d"
    done
fi

# -------------------------------------------------------------------- 4. jobs
QUEUE_DIR="$LOG_ROOT/queue_$TS"
mkdir -p "$QUEUE_DIR"

cfg_value() {   # cfg key -> value (first token, comment stripped)
    sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1
}
first_error() {   # log -> one short line naming what went wrong
    local line
    line="$(grep -aE '(Error|Exception|ERROR|FAILED|Killed|out of memory)' "$1" 2>/dev/null \
            | grep -avE '^\s*(File |raise |  )' | tail -1)"
    [ -n "$line" ] || line="$(grep -av '^\s*$' "$1" 2>/dev/null | tail -1)"
    printf '%s' "$line" | tr -s ' \t' ' ' | cut -c1-100
}

run_job() {   # slot gpu index
    local slot="$1" gpu="$2" i="$3"
    local jid="${JOB_IDS[$i]}" tcfg="${JOB_TRAIN[$i]}"
    local st="$STATUS_DIR/$jid" ck log marker cfg tag out_dir rc ninf=0 nfail=0
    rm -f "$st.state" "$st.err" "$st.inferfail"
    ck="$(cfg_value "$tcfg" modelpath)"; ck="${ck#../../}"

    if [ "$TRAIN" = "1" ]; then
        rm -f "$st.train_start" "$st.train_end"
        log="$LOG_ROOT/$jid.train.out"
        marker="$LOG_ROOT/$jid.train.marker"
        : > "$marker"
        echo "train" > "$st.state"
        date +%s > "$st.train_start"
        echo "  [gpu $gpu/$slot] $jid  train start  $(date '+%m-%d %H:%M')"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" --config "$tcfg" > "$log" 2>&1
        rc=$?
        date +%s > "$st.train_end"
        # A stale checkpoint must never flow into inference after a failed run.
        if [ "$rc" != "0" ] || [ ! -f "$ck" ] || [ ! "$ck" -nt "$marker" ]; then
            [ "$rc" = "0" ] && rc="no-fresh-checkpoint"
            echo "fail train rc=$rc" > "$st.state"
            first_error "$log" > "$st.err"
            echo "  [gpu $gpu/$slot] $jid  train FAILED (rc=$rc) -- $log"
            tail -15 "$log" | sed "s/^/      [$jid] /"
            return
        fi
        echo "  [gpu $gpu/$slot] $jid  train done   $(date '+%m-%d %H:%M')"
    elif [ ! -f "$ck" ]; then
        echo "fail train no-checkpoint" > "$st.state"
        echo "TRAIN=0 but $ck does not exist" > "$st.err"
        echo "  [gpu $gpu/$slot] $jid  TRAIN=0 but $ck does not exist"
        return
    fi

    if [ "$INFER" = "1" ]; then
        for cfg in ${JOB_INFER[$i]}; do
            ninf=$((ninf + 1))
            out_dir="$(cfg_value "$cfg" inference_output_dir)"; out_dir="${out_dir#../../}"
            tag="$(basename "$out_dir")"
            [ "$NODE" = "ex3" ] && tag="infer"
            log="$LOG_ROOT/$jid.infer_$tag.out"
            marker="$LOG_ROOT/$jid.infer_$tag.marker"
            : > "$marker"
            echo "infer $tag" > "$st.state"
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" --config "$cfg" > "$log" 2>&1
            rc=$?
            if [ "$rc" != "0" ] || [ ! -f "$out_dir/spread_metrics.json" ] \
                    || [ ! "$out_dir/spread_metrics.json" -nt "$marker" ]; then
                [ "$rc" = "0" ] && rc="no-fresh-metrics"
                nfail=$((nfail + 1))
                echo "$tag rc=$rc" >> "$st.inferfail"
                [ -s "$st.err" ] || first_error "$log" > "$st.err"
                echo "  [gpu $gpu/$slot] $jid  infer $tag FAILED (rc=$rc) -- $log"
                tail -8 "$log" | sed "s/^/      [$jid] /"
            fi
        done
        echo "  [gpu $gpu/$slot] $jid  $((ninf - nfail))/$ninf inferences ok   $(date '+%m-%d %H:%M')"
    fi
    echo "done" > "$st.state"
}

rc=0
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    NSLOT=$((NG * JOBS_PER_GPU))
    echo "---- 4. jobs: $NJ over $NG GPU(s) x $JOBS_PER_GPU = $NSLOT slots ----------------"
    echo "  stdout: $LOG_ROOT/<job>.<stage>.out   status: $STATUS_DIR/"
    for i in "${!JOB_IDS[@]}"; do echo "queued" > "$STATUS_DIR/${JOB_IDS[$i]}.state"; done
    # Run hours (brief.py) count the training run; an inference-only rerun keeps them.
    if [ "$TRAIN" = "1" ]; then date +%s > "$STATUS_DIR/_start"; rm -f "$STATUS_DIR/_end"; fi
    # The first round is dealt before any slot starts (slot s takes job s), so
    # a job that fails in seconds cannot let its slot steal the next card's job
    # and break the card pairing. After that a slot takes the next job nobody
    # has claimed.
    for ((s = 0; s < NSLOT && s < NJ; s++)); do mkdir "$QUEUE_DIR/claim_$s"; done
    pids=""
    for ((s = 0; s < NSLOT; s++)); do
        gpu="${GPU_ARR[$((s % NG))]}"
        slot="s$((s / NG))"
        (
            [ "$s" -lt "$NJ" ] && run_job "$slot" "$gpu" "$s"
            for i in "${!JOB_IDS[@]}"; do
                mkdir "$QUEUE_DIR/claim_$i" 2>/dev/null || continue
                run_job "$slot" "$gpu" "$i"
            done
        ) &
        pids="$pids $!"
        sleep 2   # stagger the starts: 16 interpreters importing torch at once per node
    done
    for pid in $pids; do wait "$pid"; done
    [ "$TRAIN" = "1" ] && date +%s > "$STATUS_DIR/_end"

    # ------------------------------------------------------------- 5. summary
    echo "=================================================================="
    echo " summary, $SWEEP"
    echo "=================================================================="
    for jid in "${JOB_IDS[@]}"; do
        state="$(cat "$STATUS_DIR/$jid.state" 2>/dev/null || echo 'did not run')"
        extra=""
        [ -f "$STATUS_DIR/$jid.inferfail" ] && extra="  failed inference: $(tr '\n' ' ' < "$STATUS_DIR/$jid.inferfail")"
        printf '  %-18s %s%s\n' "$jid" "$state" "$extra"
        { [ "$state" = "done" ] && [ -z "$extra" ]; } || rc=1
    done
fi
echo "$SWEEP finished with rc=$rc   (log: $MASTER_LOG)"
exit "$rc"
