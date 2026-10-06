#!/usr/bin/env bash
# cHI-MGNflow ex3 (shell buckling) parametric sweep for one 8-GPU B300 node:
# 8 arms, one GPU each, train -> infer -> report.
#
#   nohup bash configs/HI_MGNFlow/probabilistic/ex3/b300_sweep/run_sweep.sh \
#       > output/chi-mgnflow/run_sweep.out 2>&1 &
#
# The script also tees everything it prints into
#   output/chi-mgnflow/ex3_b300_sweep/run_logs/sweep_<timestamp>.out
# so the run leaves a log even when it was started without a redirect.
#
# THE ARMS (README.md says what each one asks; each config's header names its change)
#   ctrl      prior_global none                          the current ex3 recipe
#   tok       prior_global token                         the global-token prior
#   tok_seed  tok repeated unchanged                     run-to-run noise floor
#   tok_b8    tok + prior_blocks 8                       deeper prior
#   tok_w256  tok + latent_dim 256                       wider AE and prior
#   tok_long  tok + ae_epochs 120 + training_epochs 240  twice the epochs
#   tok_kl    tok + ae_kl_weight 1e-4                    smoother AE latent
#   tok_g64   tok + voronoi_clusters 4096, 64 + latent_ch 16
#                                                        same latent, 4x fewer coarse nodes
#
# ORDER OF WORK
#   1. environment  launcher python imports cae_suite; the cHI-MGNflow interpreter
#                   imports torch / torch_geometric / h5py and sees CUDA; every
#                   index in GPUS exists on THIS machine; this checkout knows
#                   prior_global (an older one would warn and then silently
#                   train every token arm as ctrl).
#   2. datasets     every dataset_dir / infer_dataset / eval_dataset named by a
#                   selected config exists, with the case written in the config.
#   3. --check      the launcher's full preflight on every train config, and on
#                   every infer config without the path layer (its checkpoint
#                   does not exist until training ends).
#   Any failure in 1-3 aborts before a single GPU-hour is spent, and says why.
#   4. jobs         one job per arm: train, then infer, on one card. A card
#                   takes the next unclaimed arm in ARMS order (longest first),
#                   so fewer than 8 GPUs also works without re-dealing anything.
#   5. report       one row per arm: held-out max|u| ensemble scores from
#                   infer/spread_metrics.json + the kept prior epoch and hours.
#   6. summary      ok or the failed stage per arm; a failure prints its log tail.
#
# One card per arm: batch_size is per rank, so two-card DDP would double the
# global batch and break comparability. CUDA_VISIBLE_DEVICES does the
# assignment, so every config asks for device 0. The 8 arms share one
# hierarchy cache next to the dataset (tok_g64 builds its own); its lock lets
# one job build while the others wait.
#
# Useful overrides:
#   PYTHON, METHOD_PYTHON, GPUS (default: every GPU the interpreter sees), ARMS,
#   PROBE_TIMEOUT=900 (seconds the stage-1 environment probe may take),
#   PREFLIGHT=1, TRAIN=1, INFER=1, REPORT=1
# Report-only rerun:   TRAIN=0 INFER=0 bash .../run_sweep.sh
# Re-score one arm:    ARMS=tok TRAIN=0 bash .../run_sweep.sh
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# GPU numbers mean what nvidia-smi prints.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
CFG_REL="configs/HI_MGNFlow/probabilistic/ex3/b300_sweep"
METHOD_DIR="methods/HI_MGNFlow"
# Must match the configs' ../../output/chi-mgnflow/ex3_b300_sweep (cwd = METHOD_DIR).
OUT_ROOT="output/chi-mgnflow/ex3_b300_sweep"
LOG_ROOT="$OUT_ROOT/run_logs"

ARMS="${ARMS:-tok_w256 tok_long tok_b8 tok_g64 ctrl tok tok_seed tok_kl}"
GPUS="${GPUS:-}"
PREFLIGHT="${PREFLIGHT:-1}"
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
echo " cHI-MGNflow ex3 B300 sweep   ($TS, host $(hostname))"
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
echo "  output          : $OUT_ROOT/<arm>/{model.pth,train.log,infer/}"

# --------------------------------------------------------------- 2. datasets
echo "---- 2. datasets -------------------------------------------------"
CFG_LIST="$(mktemp)"
for arm in $ARMS; do
    echo "$CFG_REL/config_train_${arm}.txt" >> "$CFG_LIST"
    echo "$CFG_REL/config_infer_${arm}.txt" >> "$CFG_LIST"
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

# ----------------------------------------------------------------- 3. --check
if [ "$PREFLIGHT" = "1" ] && { [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; }; then
    echo "---- 3. launcher preflight (--check) -----------------------------"
    bad=""
    for arm in $ARMS; do
        for kind in train infer; do
            [ "$kind" = "train" ] && [ "$TRAIN" != "1" ] && continue
            [ "$kind" = "infer" ] && [ "$INFER" != "1" ] && continue
            log="$LOG_ROOT/${arm}.${kind}.check.log"
            extra=""
            # Checkpoint is produced by the train step; the real launch re-runs
            # the full preflight including the path layer.
            [ "$kind" = "infer" ] && [ "$TRAIN" = "1" ] && extra="--skip-filesystem-check"
            if "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_${kind}_${arm}.txt" \
                    --python "$METHOD_PYTHON" --check $extra > "$log" 2>&1; then
                echo "  $arm/$kind  OK"
            else
                echo "  $arm/$kind  FAILED -- $log"
                sed -n '/ERRORS/,$p' "$log" | head -8 | sed 's/^/      /'
                bad="$bad $arm/$kind"
            fi
        done
    done
    [ -z "$bad" ] || die "launcher preflight failed for:$bad"
fi

# -------------------------------------------------------------------- 4. jobs
# Claimed with mkdir, which is atomic, so two cards never take the same arm.
QUEUE_DIR="$LOG_ROOT/queue_$TS"
mkdir -p "$QUEUE_DIR"

# Writes "$QUEUE_DIR/<arm>.status" = "ok" | "<stage> FAILED <log>".
run_job() {
    local gpu="$1" arm="$2"
    local st ck metrics log marker
    st="$QUEUE_DIR/${arm}.status"
    ck="$OUT_ROOT/$arm/model.pth"
    metrics="$OUT_ROOT/$arm/infer/spread_metrics.json"
    mkdir -p "$OUT_ROOT/$arm"

    fail() {
        echo "$1 FAILED $2" > "$st"
        echo "  [gpu $gpu] $arm  $1 FAILED -- $2"
        tail -20 "$2" 2>/dev/null | sed "s/^/      [$arm] /"
    }

    if [ "$TRAIN" = "1" ]; then
        log="$LOG_ROOT/${arm}.train.out"
        marker="$LOG_ROOT/${arm}.train.marker"
        : > "$marker"
        echo "  [gpu $gpu] $arm  train start  $(date '+%m-%d %H:%M')"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" \
                --config "$CFG_DIR/config_train_${arm}.txt" > "$log" 2>&1; then
            fail train "$log"; return
        fi
        # A stale checkpoint must never flow into inference after a failed run.
        if [ ! -f "$ck" ] || [ ! "$ck" -nt "$marker" ]; then
            fail "train (no fresh $ck)" "$log"; return
        fi
        echo "  [gpu $gpu] $arm  train done   $(date '+%m-%d %H:%M')"
    elif [ "$INFER" = "1" ] && [ ! -f "$ck" ]; then
        echo "train SKIPPED, no checkpoint $ck" > "$st"
        echo "  [gpu $gpu] $arm  TRAIN=0 but $ck does not exist"
        return
    fi

    if [ "$INFER" = "1" ]; then
        log="$LOG_ROOT/${arm}.infer.out"
        marker="$LOG_ROOT/${arm}.infer.marker"
        : > "$marker"
        echo "  [gpu $gpu] $arm  infer start  $(date '+%m-%d %H:%M')"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" \
                --config "$CFG_DIR/config_infer_${arm}.txt" > "$log" 2>&1; then
            fail infer "$log"; return
        fi
        if [ ! -f "$metrics" ] || [ ! "$metrics" -nt "$marker" ]; then
            fail "infer (no fresh spread_metrics.json)" "$log"; return
        fi
        echo "  [gpu $gpu] $arm  infer done   $(date '+%m-%d %H:%M')"
    fi
    echo "ok" > "$st"
}

if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    echo "---- 4. jobs: $(echo $ARMS | wc -w) arms over $NG GPU(s) ----------------------------"
    echo "  stdout per arm: $LOG_ROOT/<arm>.{train,infer}.out;  trainer log: $OUT_ROOT/<arm>/train.log"
    pids=""
    for gpu in "${GPU_ARR[@]}"; do
        (
            for arm in $ARMS; do
                mkdir "$QUEUE_DIR/claim_$arm" 2>/dev/null || continue
                run_job "$gpu" "$arm"
            done
        ) &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid"; done
fi

# ------------------------------------------------------------------ 5. report
if [ "$REPORT" = "1" ]; then
    echo "---- 5. report ---------------------------------------------------"
    "$PYTHON" - "$OUT_ROOT" $ARMS <<'PY' | tee "$LOG_ROOT/report.txt"
import json
import re
import sys
from pathlib import Path

out_root, arms = Path(sys.argv[1]), sys.argv[2:]
EPOCH = re.compile(r'\[Prior\] Elapsed: ([\d.]+)s Epoch (\d+) .*?'
                   r'CRPS ([\d.e+-]+) spread ([\d.]+)')
ELAPSED = re.compile(r'\[(?:AE|Prior)\] Elapsed: ([\d.]+)s')

print('held-out ex3_infer (8 unseen geometries x 64 draws), statistic max|u|;')
print('targets: sd_ratio 1, w1_norm 0, crps_norm 0, spread_skill 1, pit_ks 0.')
print('val CRPS/spread = the validation-split sampled readout at the kept prior epoch.')
print()
head = (f"{'arm':<10}{'sd_ratio':>9}{'w1_norm':>9}{'crps':>8}{'skill':>8}"
        f"{'pit_ks':>8}  {'kept ep':>7}{'val CRPS':>10}{'spread':>8}{'hours':>7}")
print(head)
print('-' * len(head))
for arm in arms:
    m_path = out_root / arm / 'infer' / 'spread_metrics.json'
    log = out_root / arm / 'train.log'
    m = json.loads(m_path.read_text()) if m_path.is_file() else {}
    text = log.read_text(errors='replace') if log.is_file() else ''
    rows = {int(e): (float(c), float(s)) for _, e, c, s in EPOCH.findall(text)}
    hours = ELAPSED.findall(text)
    hours = f'{float(hours[-1]) / 3600:.1f}' if hours else '-'
    kept = '-'
    out = out_root / 'run_logs' / f'{arm}.train.out'
    if out.is_file():
        k = re.findall(r'\[Prior\] stage \w+\. Kept checkpoint: epoch (\d+)',
                       out.read_text(errors='replace'))
        kept = k[-1] if k else '-'
    crps, spread = rows.get(int(kept), ('-', '-')) if kept != '-' else ('-', '-')

    def f(key, w, p=3):
        v = m.get(key)
        return f'{v:>{w}.{p}f}' if isinstance(v, (int, float)) else f'{"-":>{w}}'

    vc = f'{crps:>10.3e}' if crps != '-' else f'{"-":>10}'
    vs = f'{spread:>8.3f}' if spread != '-' else f'{"-":>8}'
    print(f'{arm:<10}{f("sd_ratio", 9)}{f("w1_norm", 9)}{f("crps_norm", 8)}'
          f'{f("spread_skill", 8)}{f("pit_ks", 8)}  {kept:>7}{vc}{vs}{hours:>7}')
print()
print(f'histograms: {out_root}/<arm>/infer/histogram_compare.png')
PY
fi

# ----------------------------------------------------------------- 6. summary
rc=0
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    echo "=================================================================="
    echo " summary"
    echo "=================================================================="
    for arm in $ARMS; do
        st="$QUEUE_DIR/${arm}.status"
        if [ -f "$st" ] && [ "$(cat "$st")" = "ok" ]; then
            printf '  %-10s ok\n' "$arm"
        else
            printf '  %-10s %s\n' "$arm" "$( [ -f "$st" ] && cat "$st" || echo 'did not finish')"
            rc=1
        fi
    done
fi
echo "Finished with rc=$rc   (log: $MASTER_LOG)"
exit "$rc"
