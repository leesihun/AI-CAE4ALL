#!/usr/bin/env bash
#
# cHI-MGNflow SAOI sweep: train, infer, score and write the document.
#
#   nohup bash configs/HI_MGNFlow/SAOI_sweep/run_sweep.sh \
#       > output/chi-mgnflow/saoi_sweep/run.out 2>&1 &
#   tail -f output/chi-mgnflow/saoi_sweep/run.out
#
# EIGHT ARMS, ONE PER CARD 0-7: a 2^3 FULL factorial over section x learningr
# x capacity, so no effect is confounded with another. Arms 1..4 are bot,
# 5..8 are top; within each, lr1/lr3 x k0/k1. Each arm pins its own card in
# its config, so this script only launches them and never sets
# CUDA_VISIBLE_DEVICES.
#
# 2000 epochs at a measured ~113 s/epoch is ~63 h for a k0 arm; the four k1
# arms cost roughly 1.5x per epoch and land about a day later. The budget is
# held identical on purpose -- a capacity comparison at a different budget
# answers a different question. There is no resume and
# cosine_T0 = epochs - warmup, so a killed arm restarts from zero.
#
# READ THE MIDDLE OF THE RUN: eta_min is 1e-8, so every curve flattens at the
# end whether or not it converged. The report computes the fractional gain in
# det(1fwd) mse through the middle half, which is where it is decided.
#
# A FAILING ARM IS DROPPED, NOT FATAL. One bad config or dead card says nothing
# about the other seven, and losing a night to it is the worse outcome.
# Everything dropped is collected in SKIPPED and reprinted at the end, so a
# shortened sweep cannot be mistaken for a complete one. STRICT_PREFLIGHT=1
# restores abort-on-any-failure.
#
# Regenerate the configs with gen_configs.py; do not hand-edit them.
#
# Environment overrides:
#   PYTHON            interpreter (default: python)
#   ARMS              arms to run (default: all 8)
#   INFER_TAGS        eval sets (default: s26fe_main s26fe_sec sm_l345u_main)
#   PREFLIGHT         1 = --check every arm before launching any (default)
#   STRICT_PREFLIGHT  1 = abort the batch if anything fails preflight
#   TRAIN / INFER     1 = run that stage (default 1 each)
#   SCORE             1 = rank the arms and write the document (default 1)
#   STAGGER           seconds between arm launches (default 10)
#   LOG_ROOT          transcript directory
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1          # else an arm's log sits dead while it loads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/chi-mgnflow/saoi_sweep"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"
DOC="docs/research/SAOI_SWEEP_FLOW.md"

# Printed verbatim by gen_configs.py, so the two cannot drift silently.
ARMS="${ARMS:-1 2 3 4 5 6 7 8}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-0}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
SCORE="${SCORE:-1}"
STAGGER="${STAGGER:-10}"

mkdir -p "$LOG_ROOT"
rc=0
SKIPPED=""

echo "=================================================================="
echo " cHI-MGNflow SAOI sweep -- 2^3 full factorial (section x lr x capacity)"
echo "=================================================================="
echo "  arms      : $ARMS"
echo "  eval sets : $INFER_TAGS"
echo "  configs   : $CFG_DIR"
echo "  output    : $OUT_ROOT"
echo ""

# ---- preflight, before committing GPU-days ---------------------------------
if [ "$PREFLIGHT" = "1" ] && [ "$TRAIN" = "1" ]; then
    echo "---- preflight ---------------------------------------------------"
    ok=""
    for arm in $ARMS; do
        cfg="$CFG_DIR/config_train_${arm}.txt"
        if [ ! -f "$cfg" ]; then
            echo "  arm $arm  MISSING CONFIG -- run: $PYTHON $CFG_DIR/gen_configs.py"
            SKIPPED="$SKIPPED $arm"
            continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check \
                > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  arm $arm  OK"
            ok="$ok $arm"
        else
            echo "  arm $arm  FAILED -- see $LOG_ROOT/${arm}.check.log"
            sed -n '/ERRORS/,$p' "$LOG_ROOT/${arm}.check.log" | head -6 | sed 's/^/      /'
            SKIPPED="$SKIPPED $arm"
        fi
    done
    if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
        echo "STRICT_PREFLIGHT=1 and$SKIPPED failed -- aborting." >&2
        exit 1
    fi
    ARMS="$(echo "$ok" | xargs)"
    if [ -z "$ARMS" ]; then
        echo ""
        echo "Every arm failed preflight -- nothing left to run." >&2
        exit 1
    fi
    echo ""
fi

# ---- train, staggered ------------------------------------------------------
if [ "$TRAIN" = "1" ]; then
    echo "---- training ----------------------------------------------------"
    pids=""
    for arm in $ARMS; do
        "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_train_${arm}.txt" \
            > "$OUT_ROOT/${arm}.log" 2>&1 &
        pids="$pids $!"
        echo "  arm $arm launched (pid $!)  -> $OUT_ROOT/${arm}.log"
        sleep "$STAGGER"
    done
    echo "  waiting for $(echo "$ARMS" | wc -w) arm(s)..."
    for pid in $pids; do
        wait "$pid" || rc=1
    done
    echo "  training finished (rc=$rc)"
    echo ""
fi

# ---- inference: every arm against every eval set ---------------------------
if [ "$INFER" = "1" ]; then
    echo "---- inference ---------------------------------------------------"
    for arm in $ARMS; do
        if [ ! -f "$OUT_ROOT/${arm}.pth" ]; then
            echo "  arm $arm  no checkpoint -- skipped"
            SKIPPED="$SKIPPED $arm(infer)"
            continue
        fi
        for tag in $INFER_TAGS; do
            cfg="$CFG_DIR/config_infer_${arm}_${tag}.txt"
            [ -f "$cfg" ] || { echo "  arm $arm $tag  no config -- skipped"; continue; }
            if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" \
                    > "$LOG_ROOT/${arm}.infer_${tag}.log" 2>&1; then
                echo "  arm $arm  $tag  done"
            else
                echo "  arm $arm  $tag  FAILED -- $LOG_ROOT/${arm}.infer_${tag}.log"
                rc=1
            fi
        done
    done
    echo ""
fi

# ---- score + document ------------------------------------------------------
if [ "$SCORE" = "1" ]; then
    echo "================= RANKING ================="
    "$PYTHON" configs/campaigns/rank_arms.py "$OUT_ROOT/infer" 2>&1 \
        | tee "$LOG_ROOT/rank_arms.log"
    echo "==========================================="
    echo ""
    if "$PYTHON" configs/campaigns/write_report.py \
            --kind flow --axis learningr \
            --label "cHI-MGNflow SAOI sweep (section x lr x capacity)" \
            --infer "$OUT_ROOT/infer" --logs "$OUT_ROOT" --configs "$CFG_DIR" \
            --arms "$ARMS" --out "$DOC" > "$LOG_ROOT/write_report.log" 2>&1; then
        echo "Document : $DOC"
        echo "Ranking  : $LOG_ROOT/rank_arms.log"
    else
        echo "Report FAILED -- see $LOG_ROOT/write_report.log" >&2
        rc=1
    fi
else
    echo "SCORE=0 -- skipped. Run it later with:"
    echo "  SCORE=1 TRAIN=0 INFER=0 bash $0"
fi

if [ -n "$SKIPPED" ]; then
    echo ""
    echo "SKIPPED -- THIS SWEEP IS INCOMPLETE. Dropped, not run:$SKIPPED"
    echo "Re-run just those once fixed, e.g.:  ARMS=\"3 7\" bash $0"
fi

echo ""
echo "THEN DELETE THE SHARED CACHE (the configs set hierarchy_cache_keep True):"
echo "  rm dataset/SAOI/saoi_train_{bot,top}.mscache.*.h5"
exit $rc
