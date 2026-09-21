#!/usr/bin/env bash
# MeshGraphNets-V SAOI hyperparameter sweep: train -> infer -> diagnose -> report.
#
#   nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
#       > output/meshgraphnets-v/run_sweep.out 2>&1 &
#
# Eight arms, each one config key away from `base`, which is itself the
# SAOI_run recipe verbatim. One of the eight is `seed`, which changes only
# training_seed: it measures the noise floor, and analyze_sweep.py refuses to
# call anything an effect unless it clears that floor on all three eval sets.
#
# One invocation always runs BOTH halves of the SAOI board, one after the
# other: `bot` then `top`, unconditionally -- there is no switch to run only
# one. They are two fully independent datasets, checkpoints and reports --
# nothing is shared between them -- so each half gets its own output root
# precisely so arm names stay plain (`base`, `seed`, ...) in both and
# analyze_sweep.py needs no changes:
#   bot -> output/.../saoi_sweep/       top -> output/.../saoi_sweep_top/
# They run sequentially, never concurrently, because both default to the same
# eight cards and each arm needs a card to itself (see SCHEDULING below) --
# running them at once would silently double up two arms per card. A hard
# failure in one half (preflight or training) aborts only that half; the run
# still attempts the other.
#
# `mmd10` is benched: its config still exists on both halves and still runs
# standalone (`ARMS=mmd10`), but it is not in the default roster. `arecon`
# (alpha_recon 1000 -> 100) took its slot instead -- see "The eight arms" in
# README.md for why.
#
# WHAT IT PRODUCES, in order, per half:
#   TRAIN     8 checkpoints under that half's $OUT_ROOT
#   INFER     24 runs (8 arms x 3 eval sets), each writing spread_values.npz
#             with the whole latent_inflation curve tagged in `gen_lam`
#   PVP       24 posterior_vs_prior JSON dumps -- the truth / posterior_mean /
#             posterior_sample / prior decomposition that says whether the
#             missing width is lost in the prior or in the decoder. This is the
#             measurement the sweep exists to condition on; an arm ranking
#             without it cannot tell a better prior from a better decoder.
#   REPORT    one table, written by analyze_sweep.py
# Both halves together (the default): 16 checkpoints, 48 infer runs, 48 PVP
# dumps, 2 reports.
#
# SCHEDULING. Eight arms divide evenly across eight cards: round-robin dealing
# puts exactly one arm per lane, one wave, no idle cards, in EITHER half.
# CUDA_VISIBLE_DEVICES does the assignment, so every config asks for local
# device 0 and nothing in the configs has to know about the machine. Running
# both halves is therefore two such waves back to back, not four.
# Batch_size 16 is per-rank, so one card per arm is required: two-card DDP
# would make the global batch 32 and break comparability with base and with
# SAOI_run.
#
# Useful overrides:
#   PYTHON, METHOD_PYTHON, GPUS, ARMS, REPORT_ARMS, INFER_TAGS
#   EVAL_PREFLIGHT=1, PREFLIGHT=1, STRICT_PREFLIGHT=1, TRAIN=1, INFER=1, PVP=1, REPORT=1
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"

# mmd10 is benched (see README.md, "The eight arms"): its config still runs
# standalone via ARMS=mmd10 on either half, but is not in the default roster.
ARMS_DEFAULT="${ARMS:-base seed zdim8 zdim4 pmin15 pmin30 g2e arecon}"
REPORT_ARMS_DEFAULT="${REPORT_ARMS:-$ARMS_DEFAULT}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
PVP="${PVP:-1}"
REPORT="${REPORT:-1}"

# posterior_vs_prior.py imports the method package directly, so it needs the
# MeshGraphNets_Variational interpreter rather than the launcher one. Ask the
# launcher settings for it; fall back to $PYTHON if that cannot be read.
if [ -z "${METHOD_PYTHON:-}" ]; then
    METHOD_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets-v'))
" 2>/dev/null)"
    [ -n "$METHOD_PYTHON" ] || METHOD_PYTHON="$PYTHON"
fi

GPU_ARR=($GPUS)
NG=${#GPU_ARR[@]}

echo "=================================================================="
echo " MGN-V SAOI hyperparameter sweep"
echo "=================================================================="
echo "  arms (train)  : $ARMS_DEFAULT"
echo "  arms (report) : $REPORT_ARMS_DEFAULT"
echo "  eval sets     : $INFER_TAGS"
echo "  gpus          : $GPUS  ($NG lanes)"
echo "  method python : $METHOD_PYTHON"

# Deal the arms onto lanes. Arms cost within a few percent of each other (same
# epochs, same data, same architecture bar one key), so round-robin is balanced
# without needing a work queue.
deal_lanes() {
    LANE=()
    local i=0 a L
    for a in $1; do
        L=$(( i % NG ))
        LANE[$L]="${LANE[$L]:-} $a"
        i=$(( i + 1 ))
    done
    for L in $(seq 0 $(( NG - 1 ))); do
        echo "  lane ${GPU_ARR[$L]} :${LANE[$L]:-  (idle)}"
    done
}

# ------------------------------------------------------------ eval preflight
# The _compare_ file must describe the same part as its _infer_ file, or every
# sd_ratio in the report is a comparison between two different geometries and
# reads as a spread defect. SAOI_run owns the checker; it globs config_infer_*
# out of whatever --config-dir it is given, so it is reused rather than copied.
# Both halves' infer configs live in this same directory, so one pass here
# covers both -- run up front: this is a multi-wave training campaign and the
# failure is silent, so finding it afterwards costs days.
if [ "$EVAL_PREFLIGHT" = "1" ] && [ "$INFER" = "1" ]; then
    echo "---- same-condition data preflight -------------------------------"
    CHECKER="$CFG_DIR/../SAOI_run/check_eval_inputs.py"
    EVAL_CHECK_LOG="$(mktemp)"
    if [ ! -f "$CHECKER" ]; then
        echo "  SKIPPED: $CHECKER not found"
    elif "$METHOD_PYTHON" "$CHECKER" --config-dir "$CFG_DIR" \
            > "$EVAL_CHECK_LOG" 2>&1; then
        echo "  infer/compare pairs OK"
    else
        echo "  FAILED -- $EVAL_CHECK_LOG" >&2
        tail -12 "$EVAL_CHECK_LOG" | sed 's/^/      /' >&2
        exit 1
    fi
fi

rc=0
ALL_SKIPPED=""

# ============================================================= per-half run
# Everything below is local to one half: reassigning ARMS/REPORT_ARMS here
# each call means a preflight or training failure on `bot` can never shrink
# the roster `top` starts from, or vice versa.
run_half() {
    local HALF="$1"
    local OUT_ROOT CFG_SUFFIX LOG_ROOT DIAG_ROOT
    local ARMS="$ARMS_DEFAULT" REPORT_ARMS="$REPORT_ARMS_DEFAULT"
    local SKIPPED="" half_rc=0

    case "$HALF" in
        bot) OUT_ROOT="output/meshgraphnets-v/saoi_sweep";     CFG_SUFFIX="" ;;
        top) OUT_ROOT="output/meshgraphnets-v/saoi_sweep_top"; CFG_SUFFIX="_top" ;;
        *)   echo "internal error: bad half '$HALF'" >&2; return 1 ;;
    esac
    LOG_ROOT="$OUT_ROOT/run_logs"
    DIAG_ROOT="$OUT_ROOT/diag"
    mkdir -p "$LOG_ROOT" "$DIAG_ROOT"

    echo "=================================================================="
    echo " half: $HALF  ->  $OUT_ROOT"
    echo "=================================================================="

    # -------------------------------------------------------------- preflight
    if [ "$PREFLIGHT" = "1" ]; then
        echo "---- preflight ---------------------------------------------------"
        local ok="" arm cfg
        for arm in $ARMS; do
            cfg="$CFG_DIR/config_train_${arm}${CFG_SUFFIX}.txt"
            if [ ! -f "$cfg" ]; then
                echo "  arm $arm  MISSING CONFIG"
                SKIPPED="$SKIPPED $arm(config)[$HALF]"
                continue
            fi
            if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check \
                    > "$LOG_ROOT/${arm}.check.log" 2>&1; then
                echo "  arm $arm  OK"
                ok="$ok $arm"
            else
                echo "  arm $arm  FAILED -- $LOG_ROOT/${arm}.check.log"
                sed -n '/ERRORS/,$p' "$LOG_ROOT/${arm}.check.log" | head -6 | sed 's/^/      /'
                SKIPPED="$SKIPPED $arm(preflight)[$HALF]"
            fi
        done
        if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
            echo "STRICT_PREFLIGHT=1; a partial sweep has no usable baseline." >&2
            ALL_SKIPPED="$ALL_SKIPPED$SKIPPED"
            echo "---- half $HALF ABORTED (preflight) -------------------------------"
            return 1
        fi
        ARMS="$(echo "$ok" | xargs)"
        if [ -z "$ARMS" ]; then
            echo "Every arm failed preflight." >&2
            ALL_SKIPPED="$ALL_SKIPPED$SKIPPED"
            echo "---- half $HALF ABORTED (preflight) -------------------------------"
            return 1
        fi
    fi

    # `base` and `seed` are the measurement, not two of eight results: without both
    # in the REPORT roster there is no noise floor and every other number is
    # unreadable. Checked against REPORT_ARMS, not ARMS, so a report-only rerun
    # (TRAIN=0, REPORT_ARMS trimmed) is still checked against what it will print.
    case " $REPORT_ARMS " in
        *" base "*) ;;
        *) echo "WARNING: arm 'base' is not in this report -- nothing to compare against." >&2 ;;
    esac
    case " $REPORT_ARMS " in
        *" seed "*) ;;
        *) echo "WARNING: arm 'seed' is not in this report -- no noise floor, so analyze_sweep.py" >&2
           echo "         cannot separate a real effect from single-seed scatter." >&2 ;;
    esac

    # -------------------------------------------------------------------- train
    if [ "$TRAIN" = "1" ]; then
        echo "---- from-scratch training (1000 epochs, freeze at 700) ----------"
        deal_lanes "$ARMS"
        local arm
        for arm in $ARMS; do : > "$LOG_ROOT/${arm}.launch.marker"; done
        local pids="" L gpu
        for L in $(seq 0 $(( NG - 1 ))); do
            [ -n "${LANE[$L]:-}" ] || continue
            (
                gpu="${GPU_ARR[$L]}"
                for arm in ${LANE[$L]}; do
                    echo "  [gpu $gpu] train $arm"
                    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py \
                        --config "$CFG_DIR/config_train_${arm}${CFG_SUFFIX}.txt" \
                        > "$OUT_ROOT/${arm}.log" 2>&1 \
                        || echo "  [gpu $gpu] train $arm FAILED -- $OUT_ROOT/${arm}.log"
                done
            ) &
            pids="$pids $!"
        done
        local pid
        for pid in $pids; do wait "$pid" || half_rc=1; done

        # A stale checkpoint must never flow into inference after a failed run.
        local completed="" ck
        for arm in $ARMS; do
            ck="$OUT_ROOT/${arm}.pth"
            if [ -f "$ck" ] && [ "$ck" -nt "$LOG_ROOT/${arm}.launch.marker" ]; then
                completed="$completed $arm"
                echo "  arm $arm complete -> $ck"
            else
                echo "  arm $arm produced NO fresh checkpoint"
                SKIPPED="$SKIPPED $arm(train)[$HALF]"
                half_rc=1
            fi
        done
        ARMS="$(echo "$completed" | xargs)"
        if [ -z "$ARMS" ]; then
            echo "No fresh checkpoint was produced." >&2
            ALL_SKIPPED="$ALL_SKIPPED$SKIPPED"
            echo "---- half $HALF ABORTED (train) ------------------------------------"
            return 1
        fi
    fi

    # ---------------------------------------------------------------- inference
    # One pass per arm per eval set yields the entire inflation curve, because
    # latent_inflation is a list and rollout.py tags every draw with the lam that
    # produced it. Do not split this into one run per lam.
    if [ "$INFER" = "1" ]; then
        echo "---- inference + inflation curve ---------------------------------"
        deal_lanes "$ARMS"
        local pids="" L gpu arm tag cfg
        for L in $(seq 0 $(( NG - 1 ))); do
            [ -n "${LANE[$L]:-}" ] || continue
            (
                gpu="${GPU_ARR[$L]}"
                for arm in ${LANE[$L]}; do
                    for tag in $INFER_TAGS; do
                        cfg="$CFG_DIR/config_infer_${arm}_${tag}${CFG_SUFFIX}.txt"
                        if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" AI_CAE4ALL_main.py \
                                --config "$cfg" \
                                > "$LOG_ROOT/${arm}.infer_${tag}.log" 2>&1; then
                            echo "  [gpu $gpu] infer $arm $tag done"
                        else
                            echo "  [gpu $gpu] infer $arm $tag FAILED -- $LOG_ROOT/${arm}.infer_${tag}.log"
                        fi
                    done
                done
            ) &
            pids="$pids $!"
        done
        local pid
        for pid in $pids; do wait "$pid" || half_rc=1; done
    fi

    # ---------------------------------------------------------------------- PVP
    # Nothing is trained here. Each run decodes four ensembles for ONE part --
    # truth, posterior_mean, posterior_sample, prior -- and dumps both the spread
    # table and the latent PCA comparison to JSON.
    if [ "$PVP" = "1" ]; then
        echo "---- posterior vs prior decomposition ----------------------------"
        deal_lanes "$ARMS"
        local pids="" L gpu arm tag
        for L in $(seq 0 $(( NG - 1 ))); do
            [ -n "${LANE[$L]:-}" ] || continue
            (
                gpu="${GPU_ARR[$L]}"
                for arm in ${LANE[$L]}; do
                    for tag in $INFER_TAGS; do
                        # posterior_vs_prior.py chdirs to the method root at import,
                        # so --config is resolved from there, exactly like the paths
                        # inside the config itself.
                        if CUDA_VISIBLE_DEVICES="$gpu" "$METHOD_PYTHON" \
                                methods/MeshGraphNets_Variational/misc/posterior_vs_prior.py \
                                --config "../../configs/MeshGraphNets_Variational/hyperparameter_sweep/config_infer_${arm}_${tag}${CFG_SUFFIX}.txt" \
                                --tag "${arm}_${tag}" \
                                --out "../../$DIAG_ROOT" \
                                --gpu 0 \
                                > "$LOG_ROOT/${arm}.pvp_${tag}.log" 2>&1; then
                            echo "  [gpu $gpu] pvp $arm $tag done"
                        else
                            echo "  [gpu $gpu] pvp $arm $tag FAILED -- $LOG_ROOT/${arm}.pvp_${tag}.log"
                        fi
                    done
                done
            ) &
            pids="$pids $!"
        done
        local pid
        for pid in $pids; do wait "$pid" || half_rc=1; done
    fi

    # ------------------------------------------------------------------- report
    if [ "$REPORT" = "1" ]; then
        echo "---- report (half: $HALF, arms: $REPORT_ARMS) ---------------------"
        "$PYTHON" "$CFG_DIR/analyze_sweep.py" \
            --out-root "$OUT_ROOT" \
            --arms "$REPORT_ARMS" \
            --tags "$INFER_TAGS" \
            | tee "$LOG_ROOT/report.txt"
    fi

    if [ -n "$SKIPPED" ]; then
        ALL_SKIPPED="$ALL_SKIPPED$SKIPPED"
    fi
    echo "---- half $HALF finished (rc=$half_rc) -----------------------------"
    return "$half_rc"
}

for h in bot top; do
    run_half "$h" || rc=1
done

if [ -n "$ALL_SKIPPED" ]; then
    echo "SKIPPED -- sweep incomplete:$ALL_SKIPPED" >&2
fi
echo "Finished with rc=$rc"
exit "$rc"
