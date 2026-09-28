#!/usr/bin/env bash
# MeshGraphNets-V SAOI hyperparameter sweep: train -> infer -> diagnose -> report.
#
#   nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
#       > output/meshgraphnets-v/run_sweep.out 2>&1 &
#
# The script also tees everything it prints into
#   output/meshgraphnets-v/saoi_sweep/run_logs/sweep_<timestamp>.out
# so the run leaves a log even when it was started without a redirect.
#
# Eight arms, each one config key away from `base` (the SAOI_run recipe
# verbatim), on BOTH halves of the SAOI board: 16 jobs. `seed` changes only
# training_seed; it is the noise floor analyze_sweep.py measures every other arm
# against. Each half keeps its own output root so arm names stay plain:
#   bot -> output/.../saoi_sweep/       top -> output/.../saoi_sweep_top/
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; the MGN-V interpreter
#                   imports torch / torch_geometric / h5py and sees CUDA; every
#                   index in GPUS exists on THIS machine.
#   2. datasets     every dataset_dir / infer_dataset / eval_dataset named by a
#                   selected config exists, with the case written in the config.
#   3. eval pairs   SAOI_run/check_eval_inputs.py: each _infer_ file is paired
#                   with its own _compare_ file.
#   4. --check      the launcher's full preflight on all 16 train configs.
#   Any failure in 1-4 aborts before a single GPU-hour is spent, and says why.
#   5. queue        16 jobs (half x arm) over the GPUs, one job per card at a
#                   time. A job is the whole chain for one arm on one half:
#                   train -> 3 inference runs -> 3 posterior_vs_prior dumps. A
#                   card that finishes early takes the next unclaimed job, so
#                   4 cards and 8 cards both work without re-dealing anything.
#   6. report       analyze_sweep.py once per half.
#   7. summary      one line per job; each failure prints the tail of its log.
#
# One card per job: Batch_size 16 is per-rank, so two-card DDP would double the
# global batch and break comparability with base and SAOI_run.
# CUDA_VISIBLE_DEVICES does the assignment, so every config asks for device 0.
#
# Useful overrides:
#   PYTHON, METHOD_PYTHON, GPUS (default: every GPU the MGN-V interpreter sees),
#   ARMS, REPORT_ARMS, INFER_TAGS,
#   EVAL_PREFLIGHT=1, PREFLIGHT=1, TRAIN=1, INFER=1, POSTERIOR_VS_PRIOR=1, REPORT=1
# Report-only rerun:  TRAIN=0 INFER=0 POSTERIOR_VS_PRIOR=0 bash .../run_sweep.sh
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# GPU numbers mean what nvidia-smi prints.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
CFG_REL="configs/MeshGraphNets_Variational/hyperparameter_sweep"
METHOD_DIR="methods/MeshGraphNets_Variational"

ARMS="${ARMS:-base seed zdim4 pmin30 arecon aux100 fmmom g2e}"
REPORT_ARMS="${REPORT_ARMS:-$ARMS}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
GPUS="${GPUS:-}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
POSTERIOR_VS_PRIOR="${POSTERIOR_VS_PRIOR:-1}"
REPORT="${REPORT:-1}"
HALVES="bot top"

out_root() {
    case "$1" in
        bot) echo "output/meshgraphnets-v/saoi_sweep" ;;
        top) echo "output/meshgraphnets-v/saoi_sweep_top" ;;
    esac
}
cfg_suffix() { [ "$1" = "top" ] && echo "_top" || echo ""; }

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="$(out_root bot)/run_logs"
mkdir -p "$MASTER_DIR" "$(out_root top)/run_logs"
MASTER_LOG="$MASTER_DIR/sweep_$TS.out"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo
    echo "ABORT: $*"
    echo "Nothing was trained. Full log: $MASTER_LOG"
    exit 1
}

echo "=================================================================="
echo " MGN-V SAOI hyperparameter sweep   ($TS, host $(hostname))"
echo "=================================================================="

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Run from the repo's launcher env or set PYTHON=/path/to/python."

if [ -z "${METHOD_PYTHON:-}" ]; then
    METHOD_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets-v'))
")" || die "could not resolve the meshgraphnets-v interpreter from ai_cae4all.local.toml; set METHOD_PYTHON=/path/to/python."
fi
[ -x "$METHOD_PYTHON" ] || command -v "$METHOD_PYTHON" >/dev/null 2>&1 \
    || die "MGN-V interpreter '$METHOD_PYTHON' does not exist."

PROBE_ERR="$MASTER_DIR/env_probe_$TS.err"
NGPU_SEEN="$(env -u CUDA_VISIBLE_DEVICES "$METHOD_PYTHON" -c "
import torch, torch_geometric, h5py
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
" 2>"$PROBE_ERR")" || {
    tail -8 "$PROBE_ERR" | sed 's/^/      /'
    die "'$METHOD_PYTHON' cannot import torch / torch_geometric / h5py (this is the env the training runs in)."
}
[ "${NGPU_SEEN:-0}" -ge 1 ] 2>/dev/null \
    || die "'$METHOD_PYTHON' sees no CUDA device."

if [ -z "$GPUS" ]; then
    GPUS="$(seq -s ' ' 0 $(( NGPU_SEEN - 1 )))"
fi
for g in $GPUS; do
    case "$g" in ''|*[!0-9]*) die "GPUS entry '$g' is not a GPU index." ;; esac
    [ "$g" -lt "$NGPU_SEEN" ] \
        || die "GPU $g does not exist here: this machine has $NGPU_SEEN GPU(s), indices 0..$(( NGPU_SEEN - 1 )). Fix GPUS=\"...\"."
done
GPU_ARR=($GPUS)
NG=${#GPU_ARR[@]}

echo "  launcher python : $(command -v "$PYTHON" || echo "$PYTHON")"
echo "  MGN-V python    : $METHOD_PYTHON"
echo "  GPUs            : $GPUS   ($NG of $NGPU_SEEN on this machine)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /'
fi
echo "  arms (train)    : $ARMS"
echo "  arms (report)   : $REPORT_ARMS"
echo "  halves          : $HALVES"
echo "  eval sets       : $INFER_TAGS"

# --------------------------------------------------------------- 2. datasets
echo "---- 2. datasets -------------------------------------------------"
CFG_LIST="$(mktemp)"
for half in $HALVES; do
    sfx="$(cfg_suffix "$half")"
    for arm in $ARMS; do
        echo "$CFG_REL/config_train_${arm}${sfx}.txt" >> "$CFG_LIST"
        if [ "$INFER" = "1" ] || [ "$POSTERIOR_VS_PRIOR" = "1" ]; then
            for tag in $INFER_TAGS; do
                echo "$CFG_REL/config_infer_${arm}_${tag}${sfx}.txt" >> "$CFG_LIST"
            done
        fi
    done
done
"$PYTHON" - "$CFG_LIST" "$METHOD_DIR" <<'PY' || die "missing configs or dataset files (listed above)."
import sys
from pathlib import Path
from cae_suite.config_parser import parse_config

cfg_list, method_dir = Path(sys.argv[1]), Path(sys.argv[2])
missing_cfg, needed = [], {}
for line in cfg_list.read_text().splitlines():
    cfg = Path(line)
    if not cfg.is_file():
        missing_cfg.append(cfg)
        continue
    values = parse_config(cfg).values
    for key in ("dataset_dir", "infer_dataset", "eval_dataset"):
        if values.get(key):
            needed.setdefault(str(values[key]), cfg.name)

def exists_exact(p):
    # Exact-case check: Linux opens only the case written in the config.
    p = p.resolve()
    return p.is_file() and p.name in {c.name for c in p.parent.iterdir()}

missing = [(v, c) for v, c in sorted(needed.items())
           if not exists_exact(method_dir / v)]
for cfg in missing_cfg:
    print(f"  MISSING CONFIG  {cfg}")
for v, c in missing:
    print(f"  MISSING DATA    {(method_dir / v).resolve()}   (first named by {c})")
if not missing_cfg and not missing:
    print(f"  {len(needed)} dataset files present")
sys.exit(1 if (missing_cfg or missing) else 0)
PY
rm -f "$CFG_LIST"

# -------------------------------------------------------------- 3. eval pairs
# The _compare_ file must describe the same part as its _infer_ file, or every
# sd_ratio is a comparison between two geometries. Silent, so checked up front.
if [ "$EVAL_PREFLIGHT" = "1" ] && [ "$INFER" = "1" ]; then
    echo "---- 3. infer/compare pairs --------------------------------------"
    CHECKER="$CFG_DIR/../SAOI_run/check_eval_inputs.py"
    EVAL_CHECK_LOG="$MASTER_DIR/eval_check_$TS.log"
    [ -f "$CHECKER" ] || die "$CHECKER not found."
    if "$METHOD_PYTHON" "$CHECKER" --config-dir "$CFG_DIR" > "$EVAL_CHECK_LOG" 2>&1; then
        echo "  OK"
    else
        tail -15 "$EVAL_CHECK_LOG" | sed 's/^/      /'
        die "infer/compare pairing check failed -- $EVAL_CHECK_LOG"
    fi
fi

# ----------------------------------------------------------------- 4. --check
if [ "$PREFLIGHT" = "1" ] && [ "$TRAIN" = "1" ]; then
    echo "---- 4. launcher preflight (--check) -----------------------------"
    bad=""
    for half in $HALVES; do
        sfx="$(cfg_suffix "$half")"
        LOG_ROOT="$(out_root "$half")/run_logs"
        for arm in $ARMS; do
            log="$LOG_ROOT/${arm}.check.log"
            if "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_train_${arm}${sfx}.txt" \
                    --check > "$log" 2>&1; then
                echo "  $half/$arm  OK"
            else
                echo "  $half/$arm  FAILED -- $log"
                sed -n '/ERRORS/,$p' "$log" | head -8 | sed 's/^/      /'
                bad="$bad $half/$arm"
            fi
        done
    done
    [ -z "$bad" ] || die "launcher preflight failed for:$bad"
fi

# ------------------------------------------------------------------- 5. queue
# job = "<half>:<arm>". Claimed with mkdir, which is atomic, so two cards can
# never take the same job.
JOBS=""
for half in $HALVES; do
    for arm in $ARMS; do JOBS="$JOBS $half:$arm"; done
done
QUEUE_DIR="$MASTER_DIR/queue_$TS"
mkdir -p "$QUEUE_DIR"

# Writes "$QUEUE_DIR/<half>_<arm>.status" = "ok" | "<stage> FAILED <log>".
run_job() {
    local gpu="$1" half="$2" arm="$3"
    local OUT_ROOT LOG_ROOT DIAG_ROOT sfx st log tag cfg ck
    OUT_ROOT="$(out_root "$half")"
    LOG_ROOT="$OUT_ROOT/run_logs"
    DIAG_ROOT="$OUT_ROOT/diag"
    sfx="$(cfg_suffix "$half")"
    st="$QUEUE_DIR/${half}_${arm}.status"
    mkdir -p "$LOG_ROOT" "$DIAG_ROOT"
    ck="$OUT_ROOT/${arm}.pth"

    fail() {
        echo "$1 FAILED $2" > "$st"
        echo "  [gpu $gpu] $half/$arm  $1 FAILED -- $2"
        tail -20 "$2" 2>/dev/null | sed "s/^/      [$half\/$arm] /"
    }

    if [ "$TRAIN" = "1" ]; then
        log="$OUT_ROOT/${arm}.log"
        : > "$LOG_ROOT/${arm}.launch.marker"
        echo "  [gpu $gpu] $half/$arm  train start  $(date '+%m-%d %H:%M')"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py \
                --config "$CFG_DIR/config_train_${arm}${sfx}.txt" > "$log" 2>&1; then
            fail train "$log"; return
        fi
        # A stale checkpoint must never flow into inference after a failed run.
        if [ ! -f "$ck" ] || [ ! "$ck" -nt "$LOG_ROOT/${arm}.launch.marker" ]; then
            fail "train (no fresh $ck)" "$log"; return
        fi
        echo "  [gpu $gpu] $half/$arm  train done   $(date '+%m-%d %H:%M')"
    elif [ ! -f "$ck" ]; then
        if [ "$INFER" = "1" ] || [ "$POSTERIOR_VS_PRIOR" = "1" ]; then
            echo "train SKIPPED, no checkpoint $ck" > "$st"
            echo "  [gpu $gpu] $half/$arm  TRAIN=0 but $ck does not exist"
            return
        fi
    fi

    # One inference pass per eval set yields the whole inflation curve:
    # latent_inflation is a list and rollout.py tags each draw with its lam.
    if [ "$INFER" = "1" ]; then
        for tag in $INFER_TAGS; do
            log="$LOG_ROOT/${arm}.infer_${tag}.log"
            cfg="$CFG_DIR/config_infer_${arm}_${tag}${sfx}.txt"
            if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py \
                    --config "$cfg" > "$log" 2>&1; then
                fail "infer $tag" "$log"; return
            fi
            if [ ! -f "$OUT_ROOT/infer/$arm/$tag/spread_values.npz" ]; then
                fail "infer $tag (no spread_values.npz)" "$log"; return
            fi
        done
        echo "  [gpu $gpu] $half/$arm  infer done"
    fi

    # truth / posterior_mean / posterior_sample / prior for one part: says
    # whether the missing width is lost in the prior or in the decoder.
    # posterior_vs_prior.py chdirs to the method root, so paths are ../../-relative.
    if [ "$POSTERIOR_VS_PRIOR" = "1" ]; then
        for tag in $INFER_TAGS; do
            log="$LOG_ROOT/${arm}.posterior_vs_prior_${tag}.log"
            if ! CUDA_VISIBLE_DEVICES="$gpu" "$METHOD_PYTHON" \
                    "$METHOD_DIR/misc/posterior_vs_prior.py" \
                    --config "../../$CFG_REL/config_infer_${arm}_${tag}${sfx}.txt" \
                    --tag "${arm}_${tag}" \
                    --out "../../$DIAG_ROOT" \
                    --gpu 0 > "$log" 2>&1; then
                fail "posterior vs. prior $tag" "$log"; return
            fi
        done
        echo "  [gpu $gpu] $half/$arm  posterior vs. prior done"
    fi
    echo "ok" > "$st"
}

if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ] || [ "$POSTERIOR_VS_PRIOR" = "1" ]; then
    echo "---- 5. jobs: $(echo $JOBS | wc -w) over $NG GPU(s) -------------------------"
    echo "  per-job logs: <out_root>/<arm>.log, <out_root>/run_logs/<arm>.{infer,posterior_vs_prior}_<tag>.log"
    pids=""
    for gpu in "${GPU_ARR[@]}"; do
        (
            for job in $JOBS; do
                mkdir "$QUEUE_DIR/claim_${job/:/_}" 2>/dev/null || continue
                run_job "$gpu" "${job%%:*}" "${job##*:}"
            done
        ) &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid"; done
fi

# ------------------------------------------------------------------ 6. report
if [ "$REPORT" = "1" ]; then
    for half in $HALVES; do
        OUT_ROOT="$(out_root "$half")"
        echo "---- 6. report ($half, arms: $REPORT_ARMS) -----------------------"
        "$PYTHON" "$CFG_DIR/analyze_sweep.py" \
            --out-root "$OUT_ROOT" \
            --arms "$REPORT_ARMS" \
            --tags "$INFER_TAGS" \
            | tee "$OUT_ROOT/run_logs/report.txt"
    done
fi

# ----------------------------------------------------------------- 7. summary
rc=0
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ] || [ "$POSTERIOR_VS_PRIOR" = "1" ]; then
    echo "=================================================================="
    echo " summary"
    echo "=================================================================="
    for job in $JOBS; do
        st="$QUEUE_DIR/${job/:/_}.status"
        if [ -f "$st" ] && [ "$(cat "$st")" = "ok" ]; then
            printf '  %-14s ok\n' "$job"
        else
            printf '  %-14s %s\n' "$job" "$( [ -f "$st" ] && cat "$st" || echo 'did not finish')"
            rc=1
        fi
    done
fi
echo "Finished with rc=$rc   (log: $MASTER_LOG)"
exit "$rc"
