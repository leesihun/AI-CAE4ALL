#!/usr/bin/env bash
# SAOI train-part check: does the prior reproduce a part it TRAINED on?
#
#   bash configs/HI_MGNFlow/hyperparameter_sweep/train_parts/run_train_parts.sh
#
# Runs on the models run_sweep.sh trained; it trains nothing. The sweep scores
# each arm on three parts no model saw. This scores the same arms on parts from
# their own training file, where generalization cannot be the excuse: the draws
# must match the scatter of that part's realizations.
#
# ORDER OF WORK
#   1. data     make_train_parts.py groups saoi_train_<half>.h5 into parts
#               (identical coordinates + mesh + condition rows), writes the
#               census, and cuts the two largest parts of each half into
#               dataset/SAOI/train_part<k>_{infer,compare}_<half>.h5.
#   2. pairs    SAOI_run/check_eval_inputs.py proves each _infer_ file and its
#               _compare_ file describe the same part.
#   3. jobs     one inference per arm x half x part, on a model.pth the sweep
#               finished training, spread over GPUS. Each takes minutes.
#   4. report   the train parts beside the sweep's unseen eval sets for the same
#               arm and half, scored exactly as run_sweep.sh scores them; the
#               validation [FlowDiag] ratio at the kept epoch; one histogram grid.
#
# Everything lands in output/chi-mgnflow/saoi_b300_sweep/:
#   train_parts/{census_<half>.json, report.txt, histograms.png, run_logs/}
#   <arm>/<half>/infer/train_part<k>/{spread_values.npz, histogram_compare.png}
#
# Useful overrides:
#   PYTHON, METHOD_PYTHON, GPUS (default: every GPU nvidia-smi lists), ARMS,
#   HALVES, PARTS="1 2", MAKE=1, INFER=1, REPORT=1, ALLOW_UNFINISHED=0,
#   MAKE_ARGS (extra make_train_parts.py flags, e.g. "--decimals 3")
# Census only:        MAKE_ARGS=--census-only INFER=0 REPORT=0 bash .../run_train_parts.sh
# Report-only rerun:  MAKE=0 INFER=0 bash .../run_train_parts.sh
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Same CPU caps as run_sweep.sh, for the same reason.
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
EVAL_CHECK="configs/HI_MGNFlow/SAOI_run/check_eval_inputs.py"
# Must match the configs' ../../output/chi-mgnflow/saoi_b300_sweep (cwd = methods/HI_MGNFlow).
OUT_ROOT="output/chi-mgnflow/saoi_b300_sweep"
SWEEP_LOGS="$OUT_ROOT/run_logs"
KIT_OUT="$OUT_ROOT/train_parts"
LOG_ROOT="$KIT_OUT/run_logs"

ARMS="${ARMS:-ctrl tok tok_seed tok_recon tok_kl tok_c8 tok_g50 tok_w256}"
HALVES="${HALVES:-bot top}"
PARTS="${PARTS:-1 2}"
# The sweep's own eval sets, read from the sweep's results for the comparison.
UNSEEN_TAGS="${UNSEEN_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
GPUS="${GPUS:-}"
MAKE="${MAKE:-1}"
INFER="${INFER:-1}"
REPORT="${REPORT:-1}"
ALLOW_UNFINISHED="${ALLOW_UNFINISHED:-0}"
MAKE_ARGS="${MAKE_ARGS:-}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/train_parts_$TS.out"
exec > >(tee -a "$MASTER_LOG") 2>&1

die() {
    echo
    echo "ABORT: $*"
    echo "Full log: $MASTER_LOG"
    exit 1
}

echo "=================================================================="
echo " cHI-MGNflow SAOI train-part check   ($TS, host $(hostname))"
echo "=================================================================="

# ------------------------------------------------------------ environment
"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || die "launcher python '$PYTHON' cannot import cae_suite. Run from the repo's launcher env or set PYTHON=/path/to/python."
if [ -z "${METHOD_PYTHON:-}" ]; then
    METHOD_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('chi-mgnflow', 'chi-mgnflow'))
")" || die "could not resolve the chi-mgnflow interpreter from ai_cae4all.local.toml; set METHOD_PYTHON=/path/to/python."
fi
[ -x "$METHOD_PYTHON" ] || command -v "$METHOD_PYTHON" >/dev/null 2>&1 \
    || die "cHI-MGNflow interpreter '$METHOD_PYTHON' does not exist."
"$METHOD_PYTHON" -c "import h5py, numpy" 2>/dev/null \
    || die "'$METHOD_PYTHON' cannot import h5py / numpy."

if [ -z "$GPUS" ]; then
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
    GPUS="$(seq -s ' ' 0 $(( ${n:-1} > 0 ? ${n:-1} - 1 : 0 )))"
fi
GPU_ARR=($GPUS)

echo "  method python : $METHOD_PYTHON"
echo "  commit        : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  arms          : $ARMS"
echo "  halves        : $HALVES     parts: $PARTS     GPUs: $GPUS"

# ------------------------------------------------------------------ 1. data
if [ "$MAKE" = "1" ]; then
    echo "---- 1. census and part files ------------------------------------"
    # Parts are cut largest-first, so PARTS="2" still needs the two largest.
    n_parts="$(printf '%s\n' $PARTS | sort -n | tail -1)"
    # shellcheck disable=SC2086
    "$METHOD_PYTHON" "$SCRIPT_DIR/make_train_parts.py" --halves $HALVES \
            --parts "$n_parts" $MAKE_ARGS \
        || die "make_train_parts.py could not cut $n_parts parts per half (census above)."
fi

# ----------------------------------------------------------------- 2. pairs
if [ "$INFER" = "1" ]; then
    echo "---- 2. infer/compare pairs --------------------------------------"
    # check_eval_inputs.py reads every config_infer_*.txt in a directory; give it
    # only the selected ones. Their paths are relative to the method repo, so a
    # copy elsewhere names the same files.
    SEL_DIR="$(mktemp -d)"
    for arm in $ARMS; do for half in $HALVES; do for k in $PARTS; do
        cp "$CFG_DIR/config_infer_${arm}_${half}_part${k}.txt" "$SEL_DIR/" \
            || die "no config for arm $arm, half $half, part $k in $CFG_DIR"
    done; done; done
    if "$METHOD_PYTHON" "$EVAL_CHECK" --config-dir "$SEL_DIR" \
            > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
        echo "  infer/compare pairs OK"
    else
        grep -E "FAIL|^ +- " "$LOG_ROOT/eval_inputs.check.log" | head -12 | sed 's/^/      /'
        rm -rf "$SEL_DIR"
        die "inference data preflight failed -- $LOG_ROOT/eval_inputs.check.log"
    fi
    rm -rf "$SEL_DIR"
fi

# ------------------------------------------------------------------ 3. jobs
QUEUE_DIR="$LOG_ROOT/queue_$TS"
mkdir -p "$QUEUE_DIR"
JOBS=()
if [ "$INFER" = "1" ]; then
    echo "---- 3. inference ------------------------------------------------"
    for arm in $ARMS; do for half in $HALVES; do
        ck="$OUT_ROOT/$arm/$half/model.pth"
        done_mark="$SWEEP_LOGS/${arm}_${half}.train.seconds"
        if [ ! -f "$ck" ]; then
            echo "  skip $arm/$half: no checkpoint $ck"
            continue
        fi
        # run_sweep.sh writes .train.seconds when training ends, and model.pth
        # at every new best before that: a newer checkpoint is a run in progress.
        if [ "$ALLOW_UNFINISHED" != "1" ] && { [ ! -f "$done_mark" ] || [ "$ck" -nt "$done_mark" ]; }; then
            echo "  skip $arm/$half: training not finished ($done_mark absent or older than model.pth;"
            echo "       ALLOW_UNFINISHED=1 scores the checkpoint as it stands)"
            continue
        fi
        for k in $PARTS; do JOBS+=("$arm $half $k"); done
    done; done
    echo "  ${#JOBS[@]} inference job(s) over ${#GPU_ARR[@]} GPU(s); stdout per job: $LOG_ROOT/<arm>_<half>_part<k>.infer.out"
fi

run_job() {
    local gpu="$1" arm="$2" half="$3" k="$4"
    local stem="${arm}_${half}_part${k}"
    local log="$LOG_ROOT/${stem}.infer.out" marker="$LOG_ROOT/${stem}.infer.marker"
    local npz="$OUT_ROOT/$arm/$half/infer/train_part$k/spread_values.npz"
    local st="$QUEUE_DIR/${stem}.status"
    : > "$marker"
    echo "  [gpu $gpu] $stem  start  $(date '+%m-%d %H:%M')"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py --python "$METHOD_PYTHON" \
            --config "$CFG_DIR/config_infer_${stem}.txt" > "$log" 2>&1; then
        echo "FAILED $log" > "$st"
        echo "  [gpu $gpu] $stem  FAILED -- $log"
        tail -20 "$log" | sed "s/^/      [$stem] /"
        return
    fi
    # rollout.py catches a histogram error and still exits 0.
    if [ ! -f "$npz" ] || [ ! "$npz" -nt "$marker" ]; then
        echo "FAILED (no fresh spread_values.npz) $log" > "$st"
        echo "  [gpu $gpu] $stem  FAILED: no fresh $npz -- $log"
        tail -20 "$log" | sed "s/^/      [$stem] /"
        return
    fi
    echo ok > "$st"
    echo "  [gpu $gpu] $stem  done   $(date '+%m-%d %H:%M')"
}

if [ "${#JOBS[@]}" -gt 0 ]; then
    pids=""
    for gpu in "${GPU_ARR[@]}"; do
        (
            for i in "${!JOBS[@]}"; do
                # mkdir is atomic: two cards never take the same job.
                mkdir "$QUEUE_DIR/claim_$i" 2>/dev/null || continue
                run_job "$gpu" ${JOBS[$i]}
            done
        ) &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid"; done
fi

# ---------------------------------------------------------------- 4. report
if [ "$REPORT" = "1" ]; then
    echo "---- 4. report ---------------------------------------------------"
    "$METHOD_PYTHON" - "$OUT_ROOT" "$HALVES" "$PARTS" "$UNSEEN_TAGS" $ARMS <<'PY' | tee "$KIT_OUT/report.txt"
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

out_root = Path(sys.argv[1])
halves, parts, unseen, arms = sys.argv[2].split(), sys.argv[3].split(), sys.argv[4].split(), sys.argv[5:]
kit = out_root / 'train_parts'
sweep_logs = out_root / 'run_logs'
sets = [f'train_part{k}' for k in parts] + unseen

FLOWDIAG = re.compile(r'\[FlowDiag\] crps=(\S+)\s+det\(1fwd\) mse=(\S+)\s+1-draw mse=(\S+)\s+spread/gt=(\S+)')
PRIOR_EPOCH = re.compile(r'\[Prior\] Epoch (\d+)/')
KEPT = re.compile(r'\[Prior\] stage \w+\. Kept checkpoint: epoch (\d+)')


def scores(gt, gen):
    """run_sweep.sh's scores, unchanged, so the two reports compare directly."""
    sd = float(gt.std()) or 1.0
    q = np.linspace(0.0, 1.0, 1001)
    s = {'sd_ratio': float(gen.std()) / sd,
         'w1': float(np.abs(np.quantile(gen, q) - np.quantile(gt, q)).mean()) / sd,
         'dmean': (float(gen.mean()) - float(gt.mean())) / sd}
    g = np.sort(gen)
    n = g.size
    pair = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * g)) / (n * n)
    s['crps'] = float(np.mean([np.abs(g - y).mean() for y in gt]) - pair) / sd
    r = np.sort((np.searchsorted(g, gt, 'left') + np.searchsorted(g, gt, 'right')) / (2.0 * n))
    m = r.size
    s['pit_ks'] = float(np.max(np.abs(np.arange(1, m + 1) / m - r)))
    s['tails'] = float(((r < 0.02) | (r > 0.98)).mean())
    return s


def num(v, w, p=3):
    return f'{v:>{w}.{p}f}' if isinstance(v, float) and math.isfinite(v) else f'{"-":>{w}}'


def flowdiag(arm, half):
    """{epoch: 1-draw mse / det mse} from the sweep's stdout, and the kept epoch.

    Each [FlowDiag] line is printed by the validation pass of the epoch whose
    '[Prior] Epoch N/' line follows it.
    """
    out = sweep_logs / f'{arm}_{half}.train.out'
    if not out.is_file():
        return {}, None
    ratios, pending, kept = {}, None, None
    for line in out.read_text(errors='replace').splitlines():
        m = FLOWDIAG.search(line)
        if m:
            det, one = float(m.group(2)), float(m.group(3))
            pending = one / det if det > 0 else math.nan
            continue
        m = PRIOR_EPOCH.search(line)
        if m and pending is not None:
            ratios[int(m.group(1))] = pending
            pending = None
            continue
        m = KEPT.search(line)
        if m:
            kept = int(m.group(1))
    return ratios, kept


# ---- census
print('SAOI train-part check: generated z_disp peak-to-valley vs the FEA realizations of one part.')
print('train_part<k> = a part from the model\'s own training file; the other sets are the sweep\'s unseen parts.')
print('scores as in run_sweep.sh, divided by sd(truth): sd_ratio 1, W1 0, dmean 0, tails 0.04, pit_ks 0.')
print()
for half in halves:
    path = kit / f'census_{half}.json'
    if not path.is_file():
        print(f'census {half}: missing ({path})')
        continue
    c = json.loads(path.read_text())
    p = c['parts']
    sizes = np.array([q['size'] for q in p])
    multi = [q for q in p if q['size'] > 1]
    within = float(np.median([q['pv_sd'] for q in multi])) if multi else math.nan
    between = float(np.std([q['pv_mean'] for q in p], ddof=1)) if len(p) > 1 else math.nan
    print(f'census {half}: {c["samples"]} samples = {len(p)} parts, realizations per part '
          f'{sizes.min()}..{int(np.median(sizes))}..{sizes.max()} (min..median..max)')
    print(f'  PV scatter: within a part (median sd) {within:.4g}  vs  between part means (sd) {between:.4g}')
    for k in parts:
        i = int(k) - 1
        if i < len(p):
            q = p[i]
            print(f'  train_part{k}: {q["size"]} realizations, PV {q["pv_mean"]:.4g} +- {q["pv_sd"]:.4g}, '
                  f'conditions: {"; ".join(q["conditions"])}')
print()

# ---- per-cell table
cells, values = {}, {}
for arm in arms:
    for half in halves:
        for s in sets:
            npz = out_root / arm / half / 'infer' / s / 'spread_values.npz'
            if npz.is_file():
                with np.load(npz) as z:
                    gt, gen = np.asarray(z['gt'], float), np.asarray(z['gen'], float)
                values[arm, half, s] = (gt, gen)
                cells[arm, half, s] = scores(gt, gen)

head = (f"{'arm':<10}{'half':<5}{'set':<15}{'truth':>6}{'draws':>6}{'sd_ratio':>9}{'W1':>7}"
        f"{'dmean':>8}{'CRPS':>7}{'tails':>7}{'pit_ks':>8}")
print(head)
print('-' * len(head))
for arm in arms:
    for half in halves:
        for s in sets:
            v = values.get((arm, half, s))
            c = cells.get((arm, half, s), {})
            counts = f'{v[0].size:>6}{v[1].size:>6}' if v else f'{"-":>6}{"-":>6}'
            print(f'{arm:<10}{half:<5}{s:<15}{counts}{num(c.get("sd_ratio"), 9)}{num(c.get("w1"), 7)}'
                  f'{num(c.get("dmean"), 8)}{num(c.get("crps"), 7)}{num(c.get("tails"), 7, 2)}'
                  f'{num(c.get("pit_ks"), 8)}')
    print()

# ---- per-arm: seen vs unseen, and the training-time calibration readout
print('PER ARM: train parts vs unseen parts (mean over halves and sets), and the validation')
print('[FlowDiag] ratio 1-draw MSE / det MSE at the kept epoch (median of the last 5 validations).')
print('The validation split holds other realizations of TRAINING parts, node by node:')
print('  ~2 = draws scatter as widely as the realizations; ~1 = draws sit on the mean;')
print('  an sd ratio of 0.5 reads ~1.25. A mean that misses the part pulls it toward 1 too.')
head = (f"{'arm':<10}{'train W1':>9}{'|log sd|':>9}{'unseen W1':>10}{'|log sd|':>9}"
        + ''.join(f"  {h + ' kept':>9}{'ratio':>7}{'last5':>7}" for h in halves))
print(head)
print('-' * len(head))


def agg(arm, which):
    got = [cells[arm, h, s] for h in halves for s in which if (arm, h, s) in cells]
    if not got:
        return math.nan, math.nan
    return (float(np.mean([g['w1'] for g in got])),
            float(np.mean([abs(math.log(max(g['sd_ratio'], 1e-12))) for g in got])))


for arm in arms:
    tw, tl = agg(arm, sets[:len(parts)])
    uw, ul = agg(arm, unseen)
    row = f'{arm:<10}{num(tw, 9)}{num(tl, 9)}{num(uw, 10)}{num(ul, 9)}'
    for h in halves:
        ratios, kept = flowdiag(arm, h)
        at = ratios.get(kept, math.nan) if kept is not None else math.nan
        tail = [ratios[e] for e in sorted(ratios)[-5:]]
        last5 = float(np.median(tail)) if tail else math.nan
        row += f'  {(str(kept) if kept is not None else "-"):>9}{num(at, 7, 2)}{num(last5, 7, 2)}'
    print(row)

# ---- histogram grid: arms down, half x set across
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f'\n(no histogram grid: matplotlib unavailable: {exc})')
    sys.exit(0)
rows = [a for a in arms if any((a, h, s) in values for h in halves for s in sets)]
if not rows:
    sys.exit(0)
cols = [(h, s) for h in halves for s in sets]
fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 1.6 * len(rows) + 0.8),
                         squeeze=False, sharex='col')
for j, (h, s) in enumerate(cols):
    pooled = [v for (a, hh, ss), (g, x) in values.items() if (hh, ss) == (h, s) for v in (g, x)]
    bins = 40
    if pooled:
        lo, hi = np.quantile(np.concatenate(pooled), [0.005, 0.995])
        # A training part has tens of realizations: 40 bins would draw them as spikes.
        n_truth = min(g.size for (a, hh, ss), (g, x) in values.items() if (hh, ss) == (h, s))
        bins = np.linspace(lo, hi, int(np.clip(2 * np.sqrt(n_truth), 8, 40)) + 1)
    for i, arm in enumerate(rows):
        ax = axes[i, j]
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        if (arm, h, s) in values:
            gt, gen = values[arm, h, s]
            ax.hist(gt, bins=bins, density=True, color='#9aa3ad', label='FEA realizations')
            ax.hist(gen, bins=bins, density=True, histtype='step', linewidth=1.6,
                    color='#1f6fd1', label='generated draws')
            c = cells[arm, h, s]
            ax.text(0.98, 0.95, f'sd {c["sd_ratio"]:.2f}\nW1 {c["w1"]:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, color='#333333')
        else:
            ax.text(0.5, 0.5, 'no result', transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='#888888')
        if i == 0:
            seen = s.startswith('train_part')
            n = next((values[a, h, s][0].size for a in rows if (a, h, s) in values), None)
            ax.set_title(f'{h} / {s}' + (f' (n={n})' if n else '') + ('\nTRAINING part' if seen else '\nunseen part'),
                         fontsize=8, color='#111111' if seen else '#555555')
        if j == 0:
            ax.set_ylabel(arm, fontsize=9, rotation=0, ha='right', va='center')
handles = next((ax.get_legend_handles_labels() for ax in axes.flat
                if ax.get_legend_handles_labels()[0]), ([], []))
fig.legend(*handles, loc='upper right', ncol=2, fontsize=8, frameon=False)
fig.suptitle('SAOI z_disp peak-to-valley per part: training parts vs unseen parts',
             fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96))
png = kit / 'histograms.png'
fig.savefig(png, dpi=130)
print(f'\nhistogram grid: {png}')
print(f'per cell:       {out_root}/<arm>/<half>/infer/train_part<k>/histogram_compare.png')
PY
fi

# --------------------------------------------------------------- summary
rc=0
if [ "${#JOBS[@]}" -gt 0 ]; then
    echo "=================================================================="
    for i in "${!JOBS[@]}"; do
        set -- ${JOBS[$i]}
        st="$QUEUE_DIR/${1}_${2}_part${3}.status"
        if [ -f "$st" ] && [ "$(cat "$st")" = "ok" ]; then
            printf '  %-22s ok\n' "${1}_${2}_part${3}"
        else
            printf '  %-22s %s\n' "${1}_${2}_part${3}" "$( [ -f "$st" ] && cat "$st" || echo 'did not finish')"
            rc=1
        fi
    done
fi
echo "Finished with rc=$rc   (log: $MASTER_LOG)"
exit "$rc"
