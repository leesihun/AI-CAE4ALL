#!/usr/bin/env bash
# cHI-MGNflow SAOI parametric study for one 8-GPU B300 node:
# 8 arms x 2 board halves, one GPU per arm, train -> infer (3 eval sets) -> report.
#
#   nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh \
#       > output/chi-mgnflow/saoi_b300_sweep.out 2>&1 &
#
# The script also tees everything it prints into
#   output/chi-mgnflow/saoi_b300_sweep/run_logs/sweep_<timestamp>.out
# so the run leaves a log even when it was started without a redirect.
#
# THE ARMS (README.md says what each one asks; each config's header names its change)
#   ctrl       prior_global none                           the SAOI_run recipe
#   tok        prior_global token                          the global-token prior
#   tok_seed   tok repeated unchanged                      run-to-run noise floor
#   tok_recon  tok + best_by recon                         checkpoint selection
#   tok_kl     tok + ae_kl_weight 1e-3                     smoother latent
#   tok_c8     tok + latent_ch 8                           wider latent
#   tok_g50    tok + voronoi_clusters 1000, 50 + latent_ch 8
#                                                          same latent, half the coarse nodes
#   tok_w256   tok + latent_dim 256                        wider compressor and prior
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; the cHI-MGNflow interpreter
#                   imports torch / torch_geometric / h5py and runs a CUDA kernel;
#                   every index in GPUS exists on THIS machine; this checkout knows
#                   prior_global (an older one would warn and then silently
#                   train every token arm as ctrl).
#   2. datasets     every dataset_dir / infer_dataset / eval_dataset named by a
#                   selected config exists, with the case written in the config;
#                   then SAOI_run/check_eval_inputs.py proves each _infer_ file
#                   and its _compare_ file describe the same part.
#   3. --check      the launcher's full preflight on every train config, and on
#                   every infer config without the path layer (its checkpoint
#                   does not exist until training ends). One arm per process,
#                   all arms at once.
#   Any failure in 1-3 aborts before a single GPU-hour is spent, and says why.
#   4. jobs         one card per arm. A card takes the next unclaimed arm in
#                   ARMS order and runs its two halves side by side
#                   (PARALLEL_HALVES=1). Each half is one lane: train, then the
#                   three eval sets in series. Fewer than 8 GPUs also works.
#   5. report       per arm x half x eval set: the GT-vs-generated spread scores
#                   from spread_values.npz, then one ranked row per arm, plus
#                   one histogram grid over every cell.
#   6. summary      ok or the failed stage per lane; a failure prints its log
#                   tail. The sweep's own hierarchy caches are removed.
#
# One card per arm keeps the two halves of an arm on the same hardware, so a
# timing difference between arms is the arm's. bot and top are separate
# datasets and separate models; they share nothing but the card.
#
# Useful overrides:
#   PYTHON, METHOD_PYTHON, GPUS (default: every GPU the interpreter sees), ARMS,
#   PROBE_TIMEOUT=900 (seconds the stage-1 environment probe may take),
#   THREADS_PER_JOB=4 (each trainer's torch CPU threads; BLAS stays at 1),
#   HALVES, INFER_TAGS, PARALLEL_HALVES=1, KEEP_CACHE=0,
#   PREFLIGHT=1, EVAL_PREFLIGHT=1, TRAIN=1, INFER=1, REPORT=1
# Report-only rerun:   TRAIN=0 INFER=0 bash .../run_sweep.sh
# Re-score one arm:    ARMS=tok TRAIN=0 bash .../run_sweep.sh
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# GPU numbers mean what nvidia-smi prints.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# CPU threads. Left alone, every trainer's torch pool and every loader worker's
# numpy BLAS pool start about one thread per core, and idle BLAS threads spin
# rather than sleep. With 16 trainers and 64 loader workers per sweep (two
# sweeps per node), the CPU reads 100% while the GPUs wait for batches. torch
# already runs each loader worker on one thread. These cap the rest: a
# trainer's own torch pool to THREADS_PER_JOB, and BLAS to one thread
# everywhere. A worker's matmul is one sample's 3x3 rotation; a trainer's
# matmuls run on the GPU.
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
CFG_REL="configs/HI_MGNFlow/hyperparameter_sweep"
METHOD_DIR="methods/HI_MGNFlow"
EVAL_CHECK="configs/HI_MGNFlow/SAOI_run/check_eval_inputs.py"
# Must match the configs' ../../output/chi-mgnflow/saoi_b300_sweep (cwd = METHOD_DIR).
OUT_ROOT="output/chi-mgnflow/saoi_b300_sweep"
LOG_ROOT="$OUT_ROOT/run_logs"

ARMS="${ARMS:-tok_w256 tok_c8 tok_g50 ctrl tok tok_seed tok_recon tok_kl}"
HALVES="${HALVES:-bot top}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
GPUS="${GPUS:-}"
PARALLEL_HALVES="${PARALLEL_HALVES:-1}"
KEEP_CACHE="${KEEP_CACHE:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
REPORT="${REPORT:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/sweep_$TS.out"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo
    echo "ABORT: $*"
    echo "Nothing was trained. Full log: $MASTER_LOG"
    exit 1
}

echo "=================================================================="
echo " cHI-MGNflow SAOI B300 sweep   ($TS, host $(hostname))"
echo " CPU threads per job: torch $OMP_NUM_THREADS, BLAS $OPENBLAS_NUM_THREADS ($(nproc) CPUs)"
echo "=================================================================="

# ------------------------------------------------------------ 1. environment
echo "---- 1. environment ----------------------------------------------"
# Every step prints its name before it starts, so a stall names itself.
echo "  [$(date +%T)] launcher python '$PYTHON': import cae_suite"
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Run from the repo's launcher env or set PYTHON=/path/to/python."

if [ -z "${METHOD_PYTHON:-}" ]; then
    echo "  [$(date +%T)] resolve the chi-mgnflow interpreter from ai_cae4all.local.toml"
    METHOD_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('chi-mgnflow', 'chi-mgnflow'))
")" || die "could not resolve the chi-mgnflow interpreter from ai_cae4all.local.toml; set METHOD_PYTHON=/path/to/python."
fi
[ -x "$METHOD_PYTHON" ] || command -v "$METHOD_PYTHON" >/dev/null 2>&1 \
    || die "cHI-MGNflow interpreter '$METHOD_PYTHON' does not exist."

# The probe's step lines and any traceback go straight into this log as they
# happen (stderr); only its two result lines are captured (stdout).
PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
echo "  [$(date +%T)] probe '$METHOD_PYTHON' (gives up after ${PROBE_TIMEOUT}s):"
PROBE_OUT="$(env -u CUDA_VISIBLE_DEVICES timeout -k 30 "$PROBE_TIMEOUT" "$METHOD_PYTHON" -c "
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
    archs = ' '.join(torch.cuda.get_arch_list())
    step(f'{n} x {torch.cuda.get_device_name(0)} (sm_{major}{minor}); this torch has kernels for: {archs}')
    # A torch wheel built without this card's arch imports fine and counts the
    # card, then either dies on the first kernel ('no kernel image is
    # available') or, when it ships PTX, JIT-compiles every kernel it touches.
    # Run the ops the trainer leans on: a bf16 matmul (use_amp) and a scatter.
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
")" && probe_rc=0 || probe_rc=$?
case "$probe_rc" in
    0) ;;
    124|137)
        die "the probe was still running after ${PROBE_TIMEOUT}s and was stopped. It hung in the last step printed above:
       import torch / torch_geometric -> the env sits on a slow (network) filesystem; PROBE_TIMEOUT=1800 waits longer.
       CUDA init                      -> driver side: nvidia-smi, persistence mode, and on NVSwitch nodes
                                         'systemctl status nvidia-fabricmanager'.
       bf16 matmul, and the kernel list has no sm_100 / sm_103 -> this torch has no Blackwell kernels and is
                                         JIT-compiling PTX; install a CUDA 12.8+ build of torch and matching PyG wheels." ;;
    *)
        die "'$METHOD_PYTHON' cannot import torch / torch_geometric / h5py, or cannot run a CUDA kernel on this GPU (traceback above; this is the env the training runs in)." ;;
esac
NGPU_SEEN="$(printf '%s\n' "$PROBE_OUT" | sed -n 1p)"
TORCH_INFO="$(printf '%s\n' "$PROBE_OUT" | sed -n 2p)"
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

grep -q "PRIOR_GLOBAL_MODES" "$METHOD_DIR/model/autoencoder.py" \
    || die "this checkout has no prior_global (methods/HI_MGNFlow/model/autoencoder.py); pull the commit that adds it, or every tok arm trains the ctrl prior."

echo "  launcher python : $(command -v "$PYTHON" || echo "$PYTHON")"
echo "  method python   : $METHOD_PYTHON"
echo "                    $TORCH_INFO"
echo "  commit          : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPUs            : $GPUS   ($NG of $NGPU_SEEN on this machine)"
if command -v nvidia-smi >/dev/null 2>&1; then
    timeout -k 5 60 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/      /' \
        || echo "      (nvidia-smi failed or did not answer within 60 s; skipped)"
fi
echo "  arms            : $ARMS"
echo "  halves          : $HALVES   (side by side on the arm's card: PARALLEL_HALVES=$PARALLEL_HALVES)"
echo "  eval sets       : $INFER_TAGS"
echo "  output          : $OUT_ROOT/<arm>/<half>/{model.pth,train.log,infer/<eval set>/}"

# --------------------------------------------------------------- 2. datasets
echo "---- 2. datasets -------------------------------------------------"
CFG_LIST="$(mktemp)"
for arm in $ARMS; do
    for half in $HALVES; do
        echo "$CFG_REL/config_train_${arm}_${half}.txt" >> "$CFG_LIST"
        for tag in $INFER_TAGS; do
            echo "$CFG_REL/config_infer_${arm}_${half}_${tag}.txt" >> "$CFG_LIST"
        done
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

# The _compare_ file must describe the same part as its _infer_ file, or the
# histogram compares two different parts and reads as a spread defect. The
# check reads every config_infer_*.txt here; the 48 share 6 distinct pairs.
if [ "$EVAL_PREFLIGHT" = "1" ] && [ "$INFER" = "1" ]; then
    if "$METHOD_PYTHON" "$EVAL_CHECK" --config-dir "$CFG_DIR" \
            > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
        echo "  infer/compare pairs OK"
    else
        grep -E "FAIL|^ +- " "$LOG_ROOT/eval_inputs.check.log" | head -12 | sed 's/^/      /'
        die "inference data preflight failed -- $LOG_ROOT/eval_inputs.check.log"
    fi
fi

# ----------------------------------------------------------------- 3. --check
if [ "$PREFLIGHT" = "1" ] && { [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; }; then
    echo "---- 3. launcher preflight (--check) -----------------------------"
    CHECK_DIR="$LOG_ROOT/checks"
    mkdir -p "$CHECK_DIR"
    # The config stems one arm launches, e.g. train_tok_bot, infer_tok_bot_s26fe_main.
    stems_for() {
        local arm="$1" half tag
        for half in $HALVES; do
            if [ "$TRAIN" = "1" ]; then echo "train_${arm}_${half}"; fi
            if [ "$INFER" = "1" ]; then
                for tag in $INFER_TAGS; do echo "infer_${arm}_${half}_${tag}"; done
            fi
        done
    }
    # One subshell per arm, so 8 arms cost about the time of one.
    check_arm() {
        local stem extra
        for stem in $(stems_for "$1"); do
            rm -f "$CHECK_DIR/${stem}.rc"
            extra=""
            # Checkpoint is produced by the train step; the real launch
            # re-runs the full preflight including the path layer.
            case "$stem" in infer_*) [ "$TRAIN" = "1" ] && extra="--skip-filesystem-check" ;; esac
            "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_${stem}.txt" \
                --python "$METHOD_PYTHON" --check $extra > "$CHECK_DIR/${stem}.log" 2>&1
            echo $? > "$CHECK_DIR/${stem}.rc"
        done
    }
    pids=""
    for arm in $ARMS; do
        check_arm "$arm" &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid"; done

    bad=""
    for arm in $ARMS; do
        nok=0
        for stem in $(stems_for "$arm"); do
            if [ "$(cat "$CHECK_DIR/${stem}.rc" 2>/dev/null)" = "0" ]; then
                nok=$(( nok + 1 ))
            else
                echo "  $stem  FAILED -- $CHECK_DIR/${stem}.log"
                sed -n '/ERRORS/,$p' "$CHECK_DIR/${stem}.log" | head -8 | sed 's/^/      /'
                bad="$bad $stem"
            fi
        done
        echo "  $arm  $nok config(s) OK"
    done
    [ -z "$bad" ] || die "launcher preflight failed for:$bad"
fi

# -------------------------------------------------------------------- 4. jobs
# Claimed with mkdir, which is atomic, so two cards never take the same arm.
QUEUE_DIR="$LOG_ROOT/queue_$TS"
mkdir -p "$QUEUE_DIR"

# One lane = one board half of one arm: train, then every eval set.
# Writes "$QUEUE_DIR/<arm>_<half>.status" = "ok" | "<stage> FAILED <log>; ...".
run_lane() {
    local gpu="$1" arm="$2" half="$3"
    local lane="${arm}_${half}" dir="$OUT_ROOT/$arm/$half"
    local st="$QUEUE_DIR/${arm}_${half}.status" ck="$dir/model.pth"
    local log marker metrics tag t0 failed=""
    mkdir -p "$dir"

    note_fail() {
        failed="$failed${failed:+; }$1 FAILED $2"
        echo "  [gpu $gpu] $lane  $1 FAILED -- $2"
        tail -20 "$2" 2>/dev/null | sed "s/^/      [$lane] /"
    }

    if [ "$TRAIN" = "1" ]; then
        log="$LOG_ROOT/${lane}.train.out"
        marker="$LOG_ROOT/${lane}.train.marker"
        : > "$marker"
        t0=$(date +%s)
        echo "  [gpu $gpu] $lane  train start  $(date '+%m-%d %H:%M')"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" \
                --config "$CFG_DIR/config_train_${lane}.txt" > "$log" 2>&1; then
            note_fail train "$log"; echo "$failed" > "$st"; return
        fi
        # A stale checkpoint must never flow into inference after a failed run.
        if [ ! -f "$ck" ] || [ ! "$ck" -nt "$marker" ]; then
            note_fail "train (no fresh $ck)" "$log"; echo "$failed" > "$st"; return
        fi
        echo $(( $(date +%s) - t0 )) > "$LOG_ROOT/${lane}.train.seconds"
        echo "  [gpu $gpu] $lane  train done   $(date '+%m-%d %H:%M')"
    elif [ "$INFER" = "1" ] && [ ! -f "$ck" ]; then
        echo "train SKIPPED, no checkpoint $ck" > "$st"
        echo "  [gpu $gpu] $lane  TRAIN=0 but $ck does not exist"
        return
    fi

    if [ "$INFER" = "1" ]; then
        for tag in $INFER_TAGS; do
            log="$LOG_ROOT/${lane}.infer_${tag}.out"
            marker="$LOG_ROOT/${lane}.infer_${tag}.marker"
            metrics="$dir/infer/$tag/spread_values.npz"
            : > "$marker"
            echo "  [gpu $gpu] $lane  infer $tag start  $(date '+%m-%d %H:%M')"
            if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" \
                    --config "$CFG_DIR/config_infer_${lane}_${tag}.txt" > "$log" 2>&1; then
                note_fail "infer $tag" "$log"; continue
            fi
            # rollout.py catches a histogram error and still exits 0.
            if [ ! -f "$metrics" ] || [ ! "$metrics" -nt "$marker" ]; then
                note_fail "infer $tag (no fresh spread_values.npz)" "$log"; continue
            fi
            echo "  [gpu $gpu] $lane  infer $tag done   $(date '+%m-%d %H:%M')"
        done
    fi
    echo "${failed:-ok}" > "$st"
}

run_arm() {
    local gpu="$1" arm="$2" half lpids=""
    for half in $HALVES; do
        if [ "$PARALLEL_HALVES" = "1" ]; then
            run_lane "$gpu" "$arm" "$half" &
            lpids="$lpids $!"
        else
            run_lane "$gpu" "$arm" "$half"
        fi
    done
    for pid in $lpids; do wait "$pid"; done
}

if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    echo "---- 4. jobs: $(echo $ARMS | wc -w) arms x $(echo $HALVES | wc -w) halves over $NG GPU(s) ------------"
    echo "  stdout per lane: $LOG_ROOT/<arm>_<half>.{train,infer_<eval set>}.out"
    echo "  trainer log:     $OUT_ROOT/<arm>/<half>/train.log"
    pids=""
    for gpu in "${GPU_ARR[@]}"; do
        (
            for arm in $ARMS; do
                mkdir "$QUEUE_DIR/claim_$arm" 2>/dev/null || continue
                run_arm "$gpu" "$arm"
            done
        ) &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid"; done
fi

# ------------------------------------------------------------------ 5. report
# Each eval set is one part (the _infer_ file) against its 125 simulated
# realizations (the _compare_ file). Their sample IDs differ, so rollout.py
# pairs no scenes and writes only the pooled scores; the calibration of the
# draws is computed here from spread_values.npz instead. Every draw and every
# truth describe the same part, so the pooled ranks ARE the calibration.
if [ "$REPORT" = "1" ]; then
    echo "---- 5. report ---------------------------------------------------"
    "$METHOD_PYTHON" - "$OUT_ROOT" "$HALVES" "$INFER_TAGS" $ARMS <<'PY' | tee "$LOG_ROOT/report.txt"
import math
import re
import sys
from pathlib import Path

import numpy as np

out_root = Path(sys.argv[1])
halves, tags, arms = sys.argv[2].split(), sys.argv[3].split(), sys.argv[4:]
logs = out_root / 'run_logs'
EPOCH = re.compile(r'\[Prior\] Elapsed: ([\d.]+)s Epoch (\d+) .*?'
                   r'CRPS ([\d.e+-]+) spread ([\d.]+)')
KEPT = re.compile(r'\[Prior\] stage \w+\. Kept checkpoint: epoch (\d+)')


def scores(gt, gen):
    """Pooled scores of the generated spread values against the realizations."""
    sd = float(gt.std()) or 1.0
    q = np.linspace(0.0, 1.0, 1001)
    s = {'sd_ratio': float(gen.std()) / sd,
         'w1': float(np.abs(np.quantile(gen, q) - np.quantile(gt, q)).mean()) / sd,
         'dmean': (float(gen.mean()) - float(gt.mean())) / sd}
    g = np.sort(gen)
    n = g.size
    # CRPS of each truth against the full ensemble: E|X-y| - E|X-X'|/2,
    # the second term from the sorted draws in O(n).
    pair = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * g)) / (n * n)
    s['crps'] = float(np.mean([np.abs(g - y).mean() for y in gt]) - pair) / sd
    # Rank of each truth among the draws, ties split evenly.
    r = np.sort((np.searchsorted(g, gt, 'left') + np.searchsorted(g, gt, 'right')) / (2.0 * n))
    m = r.size
    s['pit_ks'] = float(np.max(np.abs(np.arange(1, m + 1) / m - r)))
    s['tails'] = float(((r < 0.02) | (r > 0.98)).mean())
    return s


def num(v, w, p=3):
    return f'{v:>{w}.{p}f}' if isinstance(v, float) and math.isfinite(v) else f'{"-":>{w}}'


cells, values = {}, {}
for arm in arms:
    for half in halves:
        for tag in tags:
            npz = out_root / arm / half / 'infer' / tag / 'spread_values.npz'
            if npz.is_file():
                with np.load(npz) as z:
                    gt, gen = np.asarray(z['gt'], float), np.asarray(z['gen'], float)
                values[arm, half, tag] = (gt, gen)
                cells[arm, half, tag] = scores(gt, gen)

print('SAOI: one part per eval set, 2000 generated draws vs 125 simulated realizations;')
print('statistic = z_disp peak-to-valley at the final step, every score divided by sd(truth).')
print('targets: sd_ratio 1, W1 0, dmean 0, CRPS low, PIT tails 0.04, pit_ks 0.')
print()
head = (f"{'arm':<10}{'half':<5}{'eval set':<15}{'sd_ratio':>9}{'W1':>7}"
        f"{'dmean':>8}{'CRPS':>7}{'tails':>7}{'pit_ks':>8}")
print(head)
print('-' * len(head))
for arm in arms:
    for half in halves:
        for tag in tags:
            s = cells.get((arm, half, tag), {})
            print(f'{arm:<10}{half:<5}{tag:<15}{num(s.get("sd_ratio"), 9)}{num(s.get("w1"), 7)}'
                  f'{num(s.get("dmean"), 8)}{num(s.get("crps"), 7)}{num(s.get("tails"), 7, 2)}'
                  f'{num(s.get("pit_ks"), 8)}')
    print()


def train_info(arm, half):
    """Kept prior epoch, its validation CRPS/spread, and training hours."""
    kept, crps, spread, hours = '-', None, None, None
    out = logs / f'{arm}_{half}.train.out'
    if out.is_file():
        k = KEPT.findall(out.read_text(errors='replace'))
        kept = k[-1] if k else '-'
    log = out_root / arm / half / 'train.log'
    if log.is_file() and kept != '-':
        rows = {int(e): (float(c), float(s)) for _, e, c, s in EPOCH.findall(log.read_text(errors='replace'))}
        crps, spread = rows.get(int(kept), (None, None))
    sec = logs / f'{arm}_{half}.train.seconds'
    if sec.is_file():
        hours = int(sec.read_text().strip() or 0) / 3600
    return kept, crps, spread, hours


n_cells = len(halves) * len(tags)
rank = []
for arm in arms:
    got = [cells[arm, h, t] for h in halves for t in tags if (arm, h, t) in cells]
    if not got:
        rank.append((math.inf, arm, None))
        continue
    agg = {k: float(np.mean([g[k] for g in got])) for k in ('w1', 'crps', 'tails')}
    agg['logsd'] = float(np.mean([abs(math.log(max(g['sd_ratio'], 1e-12))) for g in got]))
    agg['absmean'] = float(np.mean([abs(g['dmean']) for g in got]))
    agg['n'] = len(got)
    rank.append((agg['w1'], arm, agg))
# An arm missing cells averages over different eval sets; it ranks after the
# complete ones instead of beside them.
rank.sort(key=lambda r: (r[2] is None or r[2]['n'] < n_cells, r[0]))

print(f'ARMS, ranked by mean W1 over the {n_cells} cells (lower is better; incomplete arms last)')
print('|log sd|: 0 = right width; |dmean|: bias magnitude. kept/val CRPS/spread per half.')
head = (f"{'arm':<10}{'cells':>6}{'W1':>7}{'|log sd|':>9}{'|dmean|':>8}{'CRPS':>7}{'tails':>7}"
        + ''.join(f"  {h + ' kept':>8}{'valCRPS':>9}{'sprd':>6}{'h':>5}" for h in halves))
print(head)
print('-' * len(head))
for _, arm, a in rank:
    row = f'{arm:<10}'
    if a is None:
        row += f'{0:>6}' + f'{"-":>7}{"-":>9}{"-":>8}{"-":>7}{"-":>7}'
    else:
        row += (f'{a["n"]:>6}{num(a["w1"], 7)}{num(a["logsd"], 9)}{num(a["absmean"], 8)}'
                f'{num(a["crps"], 7)}{num(a["tails"], 7, 2)}')
    for h in halves:
        kept, c, s, hrs = train_info(arm, h)
        row += (f'  {kept:>8}' + (f'{c:>9.2e}' if c is not None else f'{"-":>9}')
                + (f'{s:>6.2f}' if s is not None else f'{"-":>6}')
                + (f'{hrs:>5.1f}' if hrs is not None else f'{"-":>5}'))
    print(row)
by = {arm: a for _, arm, a in rank}
if by.get('tok') and by.get('tok_seed'):
    print()
    print(f'noise floor: |tok - tok_seed| = {abs(by["tok"]["w1"] - by["tok_seed"]["w1"]):.3f} in mean W1; '
          'a smaller gap between two arms is not a result.')

# One grid: arms down, cells across; truth vs generated on each cell's own axis.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as exc:  # the table above stands without the picture
    print(f'\n(no histogram grid: matplotlib unavailable: {exc})')
    sys.exit(0)
if not values:
    sys.exit(0)
cols = [(h, t) for h in halves for t in tags]
fig, axes = plt.subplots(len(arms), len(cols), figsize=(2.6 * len(cols), 1.7 * len(arms) + 0.6),
                         squeeze=False, sharex='col')
for j, (h, t) in enumerate(cols):
    pooled = [v for (a, hh, tt), (g, x) in values.items() if (hh, tt) == (h, t) for v in (g, x)]
    if pooled:
        lo, hi = np.quantile(np.concatenate(pooled), [0.005, 0.995])
        bins = np.linspace(lo, hi, 41)
    for i, arm in enumerate(arms):
        ax = axes[i, j]
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        if (arm, h, t) in values:
            gt, gen = values[arm, h, t]
            ax.hist(gt, bins=bins, density=True, color='#9aa3ad', label='FEA truth (125)')
            ax.hist(gen, bins=bins, density=True, histtype='step', linewidth=1.6,
                    color='#1f6fd1', label='generated (2000)')
            s = cells[arm, h, t]
            ax.text(0.98, 0.95, f'sd {s["sd_ratio"]:.2f}\nW1 {s["w1"]:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, color='#333333')
        else:
            ax.text(0.5, 0.5, 'no result', transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='#888888')
        if i == 0:
            ax.set_title(f'{h} / {t}', fontsize=9)
        if j == 0:
            ax.set_ylabel(arm, fontsize=9, rotation=0, ha='right', va='center')
handles = next((ax.get_legend_handles_labels() for ax in axes.flat
                if ax.get_legend_handles_labels()[0]), ([], []))
fig.legend(*handles, loc='upper right', ncol=2, fontsize=8, frameon=False)
fig.suptitle('SAOI z_disp peak-to-valley: FEA realizations vs generated draws',
             fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.97))
png = logs / 'histograms.png'
fig.savefig(png, dpi=130)
print(f'\nhistogram grid: {png}')
print(f'per cell:       {out_root}/<arm>/<half>/infer/<eval set>/histogram_compare.png')
PY
fi

# ----------------------------------------------------------------- 6. summary
rc=0
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    echo "=================================================================="
    echo " summary"
    echo "=================================================================="
    for arm in $ARMS; do
        for half in $HALVES; do
            st="$QUEUE_DIR/${arm}_${half}.status"
            if [ -f "$st" ] && [ "$(cat "$st")" = "ok" ]; then
                printf '  %-14s ok\n' "${arm}_${half}"
            else
                printf '  %-14s %s\n' "${arm}_${half}" "$( [ -f "$st" ] && cat "$st" || echo 'did not finish')"
                rc=1
            fi
        done
    done
fi

# The hierarchy cache is derived data, rebuilt in minutes by the next run.
# hierarchy_cache_keep False already deletes it when the last reader closes;
# this sweeps what a killed job left behind. Results are never touched.
if [ "$TRAIN" = "1" ] && [ "$KEEP_CACHE" != "1" ] && [ -d "$OUT_ROOT/mscache" ]; then
    rm -f "$OUT_ROOT"/mscache/*/*.mscache.*.h5 "$OUT_ROOT"/mscache/*/*.mscache.*.h5.tmp.*
    rmdir "$OUT_ROOT"/mscache/* "$OUT_ROOT/mscache" 2>/dev/null
    echo "  hierarchy caches removed ($OUT_ROOT/mscache; KEEP_CACHE=1 keeps them)"
fi
echo "Finished with rc=$rc   (log: $MASTER_LOG)"
exit "$rc"
