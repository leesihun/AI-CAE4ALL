#!/usr/bin/env bash
# B300 round 2: one 8-GPU node, 16 jobs, two per card. NODE picks the half of
# the study this node runs; the other node runs the other half.
#
#   NODE=saoi nohup bash configs/campaigns/b300_r2/run_node.sh > output/b300_r2_saoi.out 2>&1 &
#   NODE=ex3  nohup bash configs/campaigns/b300_r2/run_node.sh > output/b300_r2_ex3.out  2>&1 &
#
# The last thing the run prints is a block of a few short lines between
# "==== TYPE THIS BACK ====" and "==== END ====" (brief.py). It is the whole
# result in a form short enough to retype; `python configs/campaigns/b300_r2/brief.py
# saoi|ex3` prints the same block at any time while the run goes.
#
# THE JOBS (README.md has the questions; every config header names its change)
#   NODE=saoi  arm x board half; cHI-MGNflow on the first slot of each card,
#              MGN-V on the second:
#     c_smp c_grp c_grp_rep c_nopart  x {bot,top}    cHI-MGNflow
#     v_smp v_grp v_grp_rep v_nopart  x {bot,top}    MeshGraphNets-V
#   NODE=ex3   one job per arm (shell buckling, ex3.h5 -> ex3_infer.h5):
#     c_full_g512 c_full c_full_rep c_full_b4 c_full_g256 c_full_w128 c_full_short c_tok
#     v_base v_rep v_mp8 v_mp8_pmin01 v_mp8_aux100 v_mp8_mmd10 v_mp12 v_mp8_w256
#   Card k runs cHI job k (heaviest first) next to MGN-V job k (lightest
#   first), so the two biggest jobs never share a card. A slot that finishes
#   takes the next unclaimed job, so fewer GPUs also work.
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; both method interpreters
#                   import torch / torch_geometric / h5py and run a bf16 matmul
#                   and a scatter on this GPU; this checkout has prior_global
#                   and split_group_attr.
#   2. data         (saoi) saoi_heldout.py: geometry labels -> the *_bygeom.h5
#                   training copies, held-out part files and their configs;
#                   then every dataset a config names exists, exact case.
#   3. --check      the launcher's full preflight on every config, 16 at a time
#                   (inference configs without the path layer while their
#                   checkpoint does not exist yet).
#   Any failure in 1-3 aborts before a GPU-hour is spent, and says why.
#   4. jobs         per job: train, then every inference config of that job
#                   (official eval parts first, then held-out parts). A failed
#                   inference does not stop the next one.
#   5. summary      ok / the failed stage per job, with its log tail; then the
#                   type-back block.
#
# Status for brief.py: output/b300_r2/<node>/run_logs/status/<job>.{state,err,
# train_start,train_end,inferfail}.  Stdout per stage: run_logs/<job>.<stage>.out.
#
# Useful overrides:
#   PYTHON, CHI_PYTHON, MGNV_PYTHON, GPUS (default: every GPU visible),
#   JOBS_PER_GPU=2, JOBS="c_grp.bot v_grp.top ..." (a subset, in this order),
#   THREADS_PER_JOB=4, PROBE_TIMEOUT=900, PREFLIGHT=1, TRAIN=1, INFER=1
# Inference-only rerun:  NODE=ex3 TRAIN=0 bash configs/campaigns/b300_r2/run_node.sh
# One job again:         NODE=saoi JOBS=c_grp.bot bash configs/campaigns/b300_r2/run_node.sh
set -uo pipefail

NODE="${NODE:-}"
case "$NODE" in
    saoi|ex3) ;;
    *) echo "Set NODE=saoi or NODE=ex3 (one per 8-GPU node)."; exit 2 ;;
esac

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 16 trainers x (4 torch threads + 4 loader workers) = 128 threads on a
# 128-core node. Without the caps every pool starts one thread per core and
# the CPU sits at 100% while the GPUs wait (round 1, two sweeps per node).
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CHI_DIR="methods/HI_MGNFlow"
MGNV_DIR="methods/MeshGraphNets_Variational"
if [ "$NODE" = "saoi" ]; then
    CHI_CFG="configs/HI_MGNFlow/hyperparameter_sweep/b300_r2"
    MGNV_CFG="configs/MeshGraphNets_Variational/hyperparameter_sweep/b300_r2"
    CHI_ARMS="c_smp c_grp c_grp_rep c_nopart"
    MGNV_ARMS="v_smp v_grp v_grp_rep v_nopart"
    HALVES="bot top"
else
    CHI_CFG="configs/HI_MGNFlow/probabilistic/ex3/b300_r2"
    MGNV_CFG="configs/MeshGraphNets_Variational/probabilistic/ex3/b300_r2"
    CHI_ARMS="c_full_g512 c_full c_full_rep c_full_b4 c_full_g256 c_full_w128 c_full_short c_tok"
    MGNV_ARMS="v_base v_rep v_mp8 v_mp8_pmin01 v_mp8_aux100 v_mp8_mmd10 v_mp12 v_mp8_w256"
    HALVES=""
fi
OUT_ROOT="output/b300_r2/$NODE"
LOG_ROOT="$OUT_ROOT/run_logs"
STATUS_DIR="$LOG_ROOT/status"
HELDOUT_CFG="output/b300_r2/saoi/heldout/configs"

GPUS="${GPUS:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT" "$STATUS_DIR"
MASTER_LOG="$LOG_ROOT/node_$TS.out"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo
    echo "ABORT: $*"
    echo "Nothing was trained. Full log: $MASTER_LOG"
    echo "==== TYPE THIS BACK ===="
    echo "R2 $NODE ABORT $(printf '%s' "$*" | head -1 | cut -c1-70)"
    echo "==== END ===="
    exit 1
}

echo "=================================================================="
echo " B300 round 2, node $NODE   ($TS, host $(hostname))"
echo " CPU threads per job: torch $OMP_NUM_THREADS, BLAS $OPENBLAS_NUM_THREADS ($(nproc) CPUs)"
echo "=================================================================="

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
echo "  [$(date +%T)] launcher python '$PYTHON': import cae_suite"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."

resolve() {   # model id, spec id
    "$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('$1', '$2'))
"
}
CHI_PYTHON="${CHI_PYTHON:-$(resolve chi-mgnflow chi-mgnflow)}" \
    || die "could not resolve the chi-mgnflow interpreter (ai_cae4all.local.toml); set CHI_PYTHON."
MGNV_PYTHON="${MGNV_PYTHON:-$(resolve meshgraphnets-v meshgraphnets_variational)}" \
    || die "could not resolve the meshgraphnets-v interpreter (ai_cae4all.local.toml); set MGNV_PYTHON."

PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
probe() {   # interpreter -> prints "<gpu count>\n<info>"; step lines on stderr
    env -u CUDA_VISIBLE_DEVICES timeout -k 30 "$PROBE_TIMEOUT" "$1" -c "
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
"
}
NGPU_SEEN=""
for which in CHI MGNV; do
    py_var="${which}_PYTHON"; py="${!py_var}"
    [ -x "$py" ] || command -v "$py" >/dev/null 2>&1 || die "$which interpreter '$py' does not exist."
    if [ "$which" = "MGNV" ] && [ "$py" = "$CHI_PYTHON" ]; then
        echo "  MGN-V uses the same interpreter; probed once"
        MGNV_INFO="$CHI_INFO"
        continue
    fi
    echo "  [$(date +%T)] probe $which '$py' (gives up after ${PROBE_TIMEOUT}s):"
    out="$(probe "$py")" && rc=0 || rc=$?
    case "$rc" in
        0) ;;
        124|137) die "the $which probe hung past ${PROBE_TIMEOUT}s in the last step printed above (slow filesystem: PROBE_TIMEOUT=1800; CUDA init: driver / nvidia-fabricmanager; bf16 matmul without sm_100 kernels: install a CUDA 12.8+ torch)." ;;
        *) die "$which interpreter '$py' cannot import torch / torch_geometric / h5py or run a CUDA kernel (traceback above)." ;;
    esac
    n="$(printf '%s\n' "$out" | sed -n 1p)"
    printf -v "${which}_INFO" '%s' "$(printf '%s\n' "$out" | sed -n 2p)"
    [ "${n:-0}" -ge 1 ] 2>/dev/null || die "$which interpreter '$py' sees no CUDA device."
    NGPU_SEEN="$n"
done

[ -n "$GPUS" ] || GPUS="$(seq -s ' ' 0 $(( NGPU_SEEN - 1 )))"
for g in $GPUS; do
    case "$g" in ''|*[!0-9]*) die "GPUS entry '$g' is not a GPU index." ;; esac
    [ "$g" -lt "$NGPU_SEEN" ] || die "GPU $g does not exist: this machine has $NGPU_SEEN GPU(s)."
done
case "$JOBS_PER_GPU" in ''|*[!0-9]*|0) die "JOBS_PER_GPU='$JOBS_PER_GPU' is not a positive integer." ;; esac
GPU_ARR=($GPUS)
NG=${#GPU_ARR[@]}

grep -q "PRIOR_GLOBAL_MODES" "$CHI_DIR/model/autoencoder.py" \
    || die "this checkout has no prior_global ($CHI_DIR/model/autoencoder.py): every token arm would train without it. Pull first."
for d in "$CHI_DIR" "$MGNV_DIR"; do
    grep -q "split_group_attr" "$d/general_modules/mesh_dataset.py" \
        || die "this checkout's $d has no split_group_attr: the grouped arms would split by sample. Pull first."
done

echo "  launcher python : $(command -v "$PYTHON" || echo "$PYTHON")"
echo "  cHI-MGNflow     : $CHI_PYTHON"
echo "                    $CHI_INFO"
echo "  MGN-V           : $MGNV_PYTHON"
echo "                    $MGNV_INFO"
echo "  commit          : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPUs            : $GPUS  x $JOBS_PER_GPU jobs each  ($NG of $NGPU_SEEN on this machine)"
if command -v nvidia-smi >/dev/null 2>&1; then
    timeout -k 5 60 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /' || echo "      (nvidia-smi did not answer; skipped)"
fi

# ---------------------------------------------------------------- job table
# One job = one training config + its inference configs. cHI jobs first
# (heaviest first), MGN-V after (lightest first): slot i of the first round
# takes job i, and slot order is card-major per round, so card k pairs cHI
# job k with MGN-V job k.
JOB_IDS=(); JOB_PY=(); JOB_TRAIN=(); JOB_INFER=()
add_job() {   # jid python train_cfg infer_cfg...
    JOB_IDS+=("$1"); JOB_PY+=("$2"); JOB_TRAIN+=("$3"); shift 3; JOB_INFER+=("$*")
}
for method in CHI MGNV; do
    arms_var="${method}_ARMS"; cfg_var="${method}_CFG"; py_var="${method}_PYTHON"
    for arm in ${!arms_var}; do
        if [ -n "$HALVES" ]; then
            for half in $HALVES; do
                add_job "$arm.$half" "${!py_var}" "${!cfg_var}/config_train_${arm}_${half}.txt" \
                    "${!cfg_var}/config_infer_${arm}_${half}_s26fe_main.txt" \
                    "${!cfg_var}/config_infer_${arm}_${half}_s26fe_sec.txt" \
                    "${!cfg_var}/config_infer_${arm}_${half}_sm_l345u_main.txt" \
                    "@heldout:${arm}_${half}"   # expanded after saoi_heldout.py has written them
            done
        else
            add_job "$arm" "${!py_var}" "${!cfg_var}/config_train_${arm}.txt" \
                "${!cfg_var}/config_infer_${arm}.txt"
        fi
    done
done
if [ -n "${JOBS:-}" ]; then   # a subset, in the order given
    keep=(); for j in $JOBS; do
        found=""; for i in "${!JOB_IDS[@]}"; do [ "${JOB_IDS[$i]}" = "$j" ] && found="$i"; done
        [ -n "$found" ] || die "JOBS entry '$j' is not a job on node $NODE (names: ${JOB_IDS[*]})."
        keep+=("$found")
    done
    ids=(); pys=(); trs=(); ifs=()
    for i in "${keep[@]}"; do ids+=("${JOB_IDS[$i]}"); pys+=("${JOB_PY[$i]}"); trs+=("${JOB_TRAIN[$i]}"); ifs+=("${JOB_INFER[$i]}"); done
    JOB_IDS=("${ids[@]}"); JOB_PY=("${pys[@]}"); JOB_TRAIN=("${trs[@]}"); JOB_INFER=("${ifs[@]}")
fi
NJ=${#JOB_IDS[@]}
echo "  jobs            : $NJ  (${JOB_IDS[*]})"

# -------------------------------------------------------------------- 2. data
echo "---- 2. data -----------------------------------------------------"
if [ "$NODE" = "saoi" ]; then
    echo "  [$(date +%T)] saoi_heldout.py (geometry labels, held-out parts, their configs)"
    "$CHI_PYTHON" -B "$SCRIPT_DIR/saoi_heldout.py" | sed 's/^/    /'
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
"$PYTHON" - "$CFG_LIST" <<'PY' || die "missing configs or dataset files (listed above)."
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
            needed.setdefault(str((Path("methods/HI_MGNFlow") / str(v)).resolve()), cfg.name)

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
        local cfg="$1" py="$CHI_PYTHON" extra="" log
        grep -qiE '^model[[:space:]]+meshgraphnets-v([[:space:]]|$)' "$cfg" && py="$MGNV_PYTHON"
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
    export PYTHON CHI_PYTHON MGNV_PYTHON TRAIN CHECK_DIR
    xargs -P 16 -I{} bash -c 'check_one "$@"' _ {} < "$CFG_LIST" > "$CHECK_DIR/_results.txt"
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
        die "preflight failed for $(printf '%s\n' $bad | wc -l) config(s): $(basename "$first") $err"
    fi
fi

# The hierarchy caches are campaign-private (one per method and coarsening).
# A cache left by an aborted run may hold a half-written variant; start clean.
if [ "$TRAIN" = "1" ] && [ -d "$OUT_ROOT/mscache" ]; then
    echo "  removing the leftover hierarchy cache $OUT_ROOT/mscache"
    rm -rf "$OUT_ROOT/mscache"
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
    local jid="${JOB_IDS[$i]}" py="${JOB_PY[$i]}" tcfg="${JOB_TRAIN[$i]}"
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
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$py" --config "$tcfg" > "$log" 2>&1
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
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$py" --config "$cfg" > "$log" 2>&1
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
    # and break the heavy/light pairing. After that a slot takes the next job
    # nobody has claimed.
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
        sleep 2   # stagger the starts: 16 interpreters importing torch at once
    done
    for pid in $pids; do wait "$pid"; done
    [ "$TRAIN" = "1" ] && date +%s > "$STATUS_DIR/_end"

    # ------------------------------------------------------------- 5. summary
    echo "=================================================================="
    echo " summary"
    echo "=================================================================="
    for jid in "${JOB_IDS[@]}"; do
        state="$(cat "$STATUS_DIR/$jid.state" 2>/dev/null || echo 'did not run')"
        extra=""
        [ -f "$STATUS_DIR/$jid.inferfail" ] && extra="  failed inference: $(tr '\n' ' ' < "$STATUS_DIR/$jid.inferfail")"
        printf '  %-18s %s%s\n' "$jid" "$state" "$extra"
        { [ "$state" = "done" ] && [ -z "$extra" ]; } || rc=1
    done
fi
echo "Finished with rc=$rc   (log: $MASTER_LOG)"
echo
"$PYTHON" -B "$SCRIPT_DIR/brief.py" "$NODE"
exit "$rc"
