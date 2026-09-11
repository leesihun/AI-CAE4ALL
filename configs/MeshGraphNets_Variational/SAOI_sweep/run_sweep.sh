#!/usr/bin/env bash
#
# MeshGraphNets-V SAOI sweep: train, infer, score and write the document.
#
#   nohup bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh \
#       > output/meshgraphnets-v/saoi_sweep/run.out 2>&1 &
#   tail -f output/meshgraphnets-v/saoi_sweep/run.out
#
# EIGHT ARMS, ONE PER CARD 0-7: a 2^3 FULL factorial over the PRIOR --
# section x prior fit (joint | frozen tail with latent standardization) x
# prior trunk (small | large) -- so no effect is confounded with another. The
# decoder is common to the posterior and prior paths, and it is the prior path
# that comes out ~2x too narrow, so the prior is what this sweep varies;
# beta_aux is held at 10 everywhere. Arms 1..4 are bot, 5..8 are top; each
# arm pins its own card in its config, so this script only launches them and
# never sets CUDA_VISIBLE_DEVICES.
#
# ~1000 epochs at a measured ~346 s/epoch is about four days per arm, all
# eight in parallel. There is no resume and cosine_T0 = epochs - warmup, so a
# killed arm restarts from zero.
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
#   INFER_TAGS        eval sets (default: s26fe_main s26fe_sec sm_l345u)
#   PREFLIGHT         1 = --check every arm before launching any (default)
#   STRICT_PREFLIGHT  1 = abort the batch if anything fails preflight
#   TRAIN / INFER     1 = run that stage (default 1 each)
#   SCORE             1 = rank the arms and write the document (default 1)
#   DET               1 = also run each arm's deterministic control after its
#                     inference (one draw per scene, prior at temperature ~0);
#                     its offset from the truths is pure bias (default 1)
#   STAGGER           seconds between arm launches (default 10)
#   LOG_ROOT          transcript directory
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1          # else an arm's log sits dead while it loads

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/meshgraphnets-v/saoi_sweep"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"
DOC="docs/research/SAOI_SWEEP_MGNV.md"

# Printed verbatim by gen_configs.py, so the two cannot drift silently.
ARMS="${ARMS:-1 2 3 4 5 6 7 8}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-0}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
SCORE="${SCORE:-1}"
DET="${DET:-1}"                    # deterministic control after each inference
STAGGER="${STAGGER:-10}"

mkdir -p "$LOG_ROOT"
rc=0
SKIPPED=""

echo "=================================================================="
echo " MeshGraphNets-V SAOI sweep -- 2^3 over the prior: section x fit x trunk"
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
            # Deterministic control: one draw per scene, sampling noise off. Its
            # offset from the truths is pure BIAS, which is what tells a shrunk
            # conditional mean (no width fix can repair it) apart from an
            # over-tight ensemble. One forward per scene, so it always runs.
            if [ "$DET" = "1" ] && [ -f "$CFG_DIR/config_infer_${arm}_${tag}_det.txt" ]; then
                if "$PYTHON" AI_CAE4ALL_main.py \
                        --config "$CFG_DIR/config_infer_${arm}_${tag}_det.txt" \
                        > "$LOG_ROOT/${arm}.infer_${tag}_det.log" 2>&1; then
                    echo "  arm $arm  $tag  det control done"
                else
                    echo "  arm $arm  $tag  det control FAILED -- $LOG_ROOT/${arm}.infer_${tag}_det.log"
                    rc=1
                fi
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
            --kind mgnv --axis prior_freeze_epoch \
            --label "MeshGraphNets-V SAOI sweep (prior: section x fit x trunk)" \
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
