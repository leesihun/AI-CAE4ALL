#!/usr/bin/env bash
# cHI-MGNflow SAOI: train, then infer.
#
#   nohup bash configs/HI_MGNFlow/SAOI_run/run_all.sh \
#       > output/chi-mgnflow/saoi_run/run.out 2>&1 &
#
# One invocation validates every input, trains both board halves from scratch
# and runs inference on each evaluation set.
#
# Inference is also what draws the figure: rollout.py writes the GT-vs-
# generated spread histogram, spread_values.npz and the machine-readable
# [SPREAD] log lines itself when make_histogram is on, which the infer configs
# set. There is no separate figure, diagnostic or ranking stage.
#
# The arms are the two board halves, bot and top. They are different datasets,
# not variants of one recipe -- the recipe is fixed (see README.md). Each half
# gets its own GPU and the two halves run in parallel; the three evaluation
# sets of one half run in series on that half's card.
#
# Each half trains BOTH stages in one process: ae_epochs of the compressor,
# then training_epochs of the flow-matching prior against it, frozen. There is
# no resume, so a killed run restarts from zero -- one launch is one complete
# model, and the marker check below refuses to infer from anything else.
#
# Useful overrides:
#   PYTHON, ARMS, INFER_TAGS, STAGGER
#   PREFLIGHT=1, STRICT_PREFLIGHT=1, EVAL_PREFLIGHT=1, TRAIN=1, INFER=1
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/chi-mgnflow/saoi_run"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"

ARMS="${ARMS:-bot top}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
STAGGER="${STAGGER:-10}"

mkdir -p "$LOG_ROOT"
rc=0
SKIPPED=""

echo "=================================================================="
echo " cHI-MGNflow SAOI: train -> infer -> histogram"
echo "=================================================================="
echo "  arms      : $ARMS"
echo "  eval sets : $INFER_TAGS"
echo "  output    : $OUT_ROOT"

# The _compare_ file must describe the same part as its _infer_ file, or the
# histogram compares two different geometries and reads as a spread defect.
if [ "$EVAL_PREFLIGHT" = "1" ] && [ "$INFER" = "1" ]; then
    echo "---- same-condition data preflight ------------------------------"
    if ! "$PYTHON" "$CFG_DIR/check_eval_inputs.py" --config-dir "$CFG_DIR" \
            > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
        echo "Inference data preflight FAILED -- $LOG_ROOT/eval_inputs.check.log" >&2
        exit 1
    fi
    echo "  infer/compare pairs OK"
fi

if [ "$PREFLIGHT" = "1" ] && [ "$TRAIN" = "1" ]; then
    echo "---- model preflight --------------------------------------------"
    ok=""
    for arm in $ARMS; do
        cfg="$CFG_DIR/config_train_${arm}.txt"
        if [ ! -f "$cfg" ]; then
            echo "  arm $arm  MISSING CONFIG"
            SKIPPED="$SKIPPED $arm(config)"
            continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check \
                > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  arm $arm  OK"
            ok="$ok $arm"
        else
            echo "  arm $arm  FAILED -- $LOG_ROOT/${arm}.check.log"
            sed -n '/ERRORS/,$p' "$LOG_ROOT/${arm}.check.log" | head -8 | sed 's/^/      /'
            SKIPPED="$SKIPPED $arm(preflight)"
        fi
    done
    if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
        echo "STRICT_PREFLIGHT=1; no partial experiment will be launched." >&2
        exit 1
    fi
    ARMS="$(echo "$ok" | xargs)"
    [ -n "$ARMS" ] || { echo "Every arm failed preflight." >&2; exit 1; }
fi

if [ "$TRAIN" = "1" ]; then
    echo "---- from-scratch training (compressor, then prior) -------------"
    pids=""
    for arm in $ARMS; do
        marker="$LOG_ROOT/${arm}.launch.marker"
        : > "$marker"
        "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_train_${arm}.txt" \
            > "$OUT_ROOT/${arm}.log" 2>&1 &
        pid=$!
        pids="$pids $pid"
        echo "  arm $arm launched (pid $pid)"
        sleep "$STAGGER"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done

    # A stale or absent checkpoint must never flow into inference after a
    # failed fresh run. The marker was created immediately before launch.
    completed=""
    for arm in $ARMS; do
        checkpoint="$OUT_ROOT/${arm}.pth"
        marker="$LOG_ROOT/${arm}.launch.marker"
        if [ -f "$checkpoint" ] && [ "$checkpoint" -nt "$marker" ]; then
            completed="$completed $arm"
            echo "  arm $arm complete -> $checkpoint"
        else
            echo "  arm $arm FAILED to produce a fresh checkpoint"
            SKIPPED="$SKIPPED $arm(train)"
            rc=1
        fi
    done
    ARMS="$(echo "$completed" | xargs)"
    [ -n "$ARMS" ] || { echo "No fresh checkpoint was produced." >&2; exit 1; }
fi

# Each inference run writes its own rollout metrics, spread_values.npz and the
# GT-vs-generated histogram PNG under $OUT_ROOT/infer/<arm>/<eval set>.
run_infer_arm() {
    arm="$1"
    [ -f "$OUT_ROOT/${arm}.pth" ] || return 1
    arm_rc=0
    for tag in $INFER_TAGS; do
        cfg="$CFG_DIR/config_infer_${arm}_${tag}.txt"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" \
                > "$LOG_ROOT/${arm}.infer_${tag}.log" 2>&1; then
            echo "  arm $arm $tag done"
        else
            echo "  arm $arm $tag FAILED -- $LOG_ROOT/${arm}.infer_${tag}.log"
            arm_rc=1
        fi
    done
    return "$arm_rc"
}

if [ "$INFER" = "1" ]; then
    echo "---- parallel inference + spread histograms ----------------------"
    pids=""
    for arm in $ARMS; do
        run_infer_arm "$arm" &
        pids="$pids $!"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done
    echo "  histograms under $OUT_ROOT/infer"
fi

# hierarchy_cache_keep True leaves each half's multiscale cache beside its
# training HDF5 so a rerun does not rebuild it. Nothing deletes it for you.
echo "  hierarchy caches kept: rm -f dataset/SAOI/saoi_train_*.mscache.*.h5"

if [ -n "$SKIPPED" ]; then
    echo "SKIPPED -- campaign incomplete:$SKIPPED" >&2
fi
echo "Finished with rc=$rc"
exit "$rc"
