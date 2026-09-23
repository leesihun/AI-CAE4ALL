#!/usr/bin/env bash
# MeshGraphNets-V SAOI hyperparameter sweep 2: train -> infer -> diagnose -> report.
#
#   nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep2.sh \
#       > output/meshgraphnets-v/run_sweep2.out 2>&1 &
#
# Runs ALONGSIDE run_sweep.sh (sweep1), on the same cards. The
# only results that count are sweep1's and this one's; the older SAOI sweeps,
# the FM-v2 ablation and sweep3 (beta_aux, no dump ever recovered) are treated
# as nonexistent, so nothing here is excluded or justified by them.
#
# Eight arms, each one config key away from sweep1's `base`:
#   aux0 aux3 aux30 aux100   beta_aux 10 -> 0 / 3 / 30 / 100   (the PV term)
#   mmd10 mmd100             lambda_mmd 1 -> 10 / 100           (encoder only)
#   vel512                   prior_velocity_hidden_dim 256 -> 512 (prior side)
#   fmmom                    prior_fm_moments False -> True       (prior side)
# At prior_grad_to_encoder 0, Adam makes sweep1's arecon (alpha_recon 1000 ->
# 100) the same thing as lambda_mmd x10 plus beta_aux x10, so base / mmd10 /
# aux100 / arecon form a 2x2 factorial that splits arecon into its two parts.
# sweep1 has no prior-side arm; vel512 and fmmom are that half of the picture.
#
# NO base/seed OF ITS OWN. Every arm writes into sweep1's output roots
# (saoi_sweep/ and saoi_sweep_top/) under its own name, so sweep1's base and
# seed are the reference and the noise floor for this sweep too, and
# analyze_sweep.py reads both sweeps side by side with no changes. Nothing
# here overwrites a sweep1 file: the arm names are disjoint, and the report
# goes to run_logs/report_sweep2.txt, not report.txt.
#
# If sweep1's base/seed are not finished when this sweep's report stage runs,
# the report marks them `incomplete`; re-read from disk afterwards with
#   TRAIN=0 INFER=0 PVP=0 bash .../run_sweep2.sh
#
# GPUS defaults to 0..7, the same eight cards as sweep1: each card then
# carries one sweep1 arm and one sweep2 arm side by side. Same halves, lanes,
# stale-checkpoint guard and stage switches as run_sweep.sh.
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

ARMS_DEFAULT="${ARMS:-aux0 aux3 aux30 aux100 mmd10 mmd100 vel512 fmmom}"
# base/seed/arecon are trained by sweep1 into the same roots; the report
# reads them from there.
REPORT_ARMS_DEFAULT="${REPORT_ARMS:-base seed arecon $ARMS_DEFAULT}"
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
echo " MGN-V SAOI hyperparameter sweep 2 (alongside sweep1)"
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
# Same checker as run_sweep.sh: the _compare_ file must describe the same part
# as its _infer_ file, or every sd_ratio is a comparison between two geometries.
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
BASELINE_MISSING=""

# ============================================================= per-half run
run_half() {
    local HALF="$1"
    local OUT_ROOT CFG_SUFFIX LOG_ROOT DIAG_ROOT
    local ARMS="$ARMS_DEFAULT" REPORT_ARMS="$REPORT_ARMS_DEFAULT"
    local SKIPPED="" half_rc=0

    # The SAME roots as sweep1: its base/seed are this sweep's reference.
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
            echo "STRICT_PREFLIGHT=1; fix the failing arm or drop it from ARMS." >&2
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
    # One pass per arm per eval set yields the entire inflation curve.
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
                        # so --config is resolved from there.
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
        # base/seed belong to sweep1. Say so if they are not on disk yet, rather
        # than let the floor silently go missing.
        local ref t
        for ref in base seed; do
            case " $REPORT_ARMS " in *" $ref "*) ;; *) continue ;; esac
            for t in $INFER_TAGS; do
                if [ ! -f "$OUT_ROOT/infer/$ref/$t/spread_values.npz" ] \
                        || [ ! -f "$DIAG_ROOT/posterior_vs_prior_${ref}_${t}.json" ]; then
                    BASELINE_MISSING="$BASELINE_MISSING $ref[$HALF]"
                    break
                fi
            done
        done
        echo "---- report (half: $HALF, arms: $REPORT_ARMS) ---------------------"
        "$PYTHON" "$CFG_DIR/analyze_sweep.py" \
            --out-root "$OUT_ROOT" \
            --arms "$REPORT_ARMS" \
            --tags "$INFER_TAGS" \
            | tee "$LOG_ROOT/report_sweep2.txt"
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
if [ -n "$BASELINE_MISSING" ]; then
    echo "sweep1 reference not on disk yet:$BASELINE_MISSING" >&2
    echo "  once run_sweep.sh has finished, re-read with:" >&2
    echo "  TRAIN=0 INFER=0 PVP=0 bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep2.sh" >&2
fi
echo "Finished with rc=$rc"
exit "$rc"
