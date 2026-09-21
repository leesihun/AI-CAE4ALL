#!/usr/bin/env bash
# cHI-MGNflow SAOI hyperparameter sweep -- runs the whole campaign unattended.
#
#   bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh
#   bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh all
#
# With no argument (or `all`), this runs Phase A, selects and promotes ONE
# compressor per half automatically (select_ae.py), then Phase B and Phase C
# for whichever halves came out ready, then prints ae_report.py and
# rank_arms.py at the end. No human step in between.
#
# SELECTION IS NOT "TAKE THE BEST RECON". Phase B fans four prior arms out
# from ONE frozen compressor per half, so something has to decide which
# Phase-A arm that is. That decision is not a minimum over a column: the four
# compressors differ in latent_ch, so the best reconstruction loss belongs to
# the largest one almost by definition, and picking it blindly buys capacity
# the prior then has to model for no reason. select_ae.py applies a knee +
# ceiling-gate policy instead -- see its own docstring for the exact rule, and
# `PROMOTE_BOT=<arm>` / `PROMOTE_TOP=<arm>` below to override it.
#
# Each phase runs eight jobs on eight cards, bot on 0-3 and top on 4-7; the
# assignment lives in each config's gpu_ids and this script only reports it.
#
# Phase A and Phase B reuse the same cards, which is why their log_file_dir
# values sit in different subdirectories: the periodic dumps land under
# <log_dir>/test/<gpu_ids>/<epoch>/, so a shared log_dir would have Phase B
# overwriting Phase A's reconstructions on every card.
#
# There is no resume anywhere: build_optimizer_scheduler sets cosine_T0 to
# training_epochs - warmup_epochs, so a killed job restarts from zero and one
# launch is one complete model. The marker check below refuses to promote or
# infer from a checkpoint older than its own launch.
#
# To run one phase by hand instead (e.g. to inspect ae_report.py's numbers
# yourself before committing to a compressor):
#
#   bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh A
#   python configs/HI_MGNFlow/hyperparameter_sweep/ae_report.py
#   python configs/HI_MGNFlow/hyperparameter_sweep/ae_ceiling_check.py --half bot --arm c8
#   PROMOTE_BOT=c8 PROMOTE_TOP=c8 bash .../run_sweep.sh promote
#   bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh B
#   bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh C
#   python configs/HI_MGNFlow/hyperparameter_sweep/rank_arms.py
#
# Useful overrides:
#   PYTHON, HALVES, AE_ARMS, PRIOR_ARMS, INFER_TAGS, STAGGER
#   PREFLIGHT=1, STRICT_PREFLIGHT=1, EVAL_PREFLIGHT=1
#   PROMOTE_BOT=<arm>, PROMOTE_TOP=<arm>   (phase `promote`, and `all`'s
#                                           auto-selection, which else runs
#                                           select_ae.py's own policy)
set -uo pipefail

PHASE="${1:-all}"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/chi-mgnflow/saoi_sweep"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"

HALVES="${HALVES:-bot top}"
AE_ARMS="${AE_ARMS:-c4 c8 c16 c8kl}"
PRIOR_ARMS="${PRIOR_ARMS:-p4u p8u p4x p8x}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u_main}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
# Four arms of a half share one multiscale cache; the first to arrive builds it
# under an exclusive lock while the rest block. Staggering only keeps them from
# piling onto the lock at once -- correctness does not depend on it.
STAGGER="${STAGGER:-20}"

rc=0
SKIPPED=""

usage() {
    cat >&2 <<'USAGE'
usage: run_sweep.sh [all|A|promote|B|C]

  all      (default, i.e. no argument) The whole campaign, unattended:
           A -> auto-select+promote per half -> B -> C -> reports. A half
           that select_ae.py cannot clear the ceiling gate for is promoted
           anyway (least-bad arm) but skipped for B/C; see NEEDS_ATTENTION
           in its output. Override the pick with PROMOTE_BOT=<arm> and/or
           PROMOTE_TOP=<arm>.
  A        Phase A only -- train 8 compressors (mode train_ae), 4 per half.
           Arms c4 / c8 / c16 vary latent_ch; c8kl raises ae_kl_weight.
  promote  Run select_ae.py's auto-selection for each half by hand (or force
           one arm via PROMOTE_BOT=<arm> / PROMOTE_TOP=<arm>), without also
           running B/C.
  B        Phase B only -- train 8 flow priors (mode train_prior) against the
           frozen promoted compressors. Arms vary prior_blocks (4/8) and
           flow_loss_weighting (uniform/x0).
  C        Phase C only -- 24 inference runs (8 arms x 3 eval parts). Each
           one writes its own spread histogram and spread_values.npz.

`all` prints ae_report.py and rank_arms.py itself; after a manual A or C you
can run them yourself (ae_ceiling_check.py too, after A).
USAGE
}

# --- shared helpers ------------------------------------------------------

gpu_of() {
    # Reported only; the config is what actually selects the card.
    awk '$1 == "gpu_ids" { print $2; exit }' "$1" 2>/dev/null || echo "?"
}

preflight_one() {
    # preflight_one <config> <label> -> 0 if the launcher validates it
    cfg="$1"
    label="$2"
    if [ ! -f "$cfg" ]; then
        echo "  $label  MISSING CONFIG ($cfg)"
        SKIPPED="$SKIPPED $label(config)"
        return 1
    fi
    if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check \
            > "$LOG_ROOT/${label}.check.log" 2>&1; then
        echo "  $label  OK  (gpu $(gpu_of "$cfg"))"
        return 0
    fi
    echo "  $label  FAILED -- $LOG_ROOT/${label}.check.log"
    sed -n '/ERRORS/,$p' "$LOG_ROOT/${label}.check.log" | head -8 | sed 's/^/      /'
    SKIPPED="$SKIPPED $label(preflight)"
    return 1
}

# --- phase A / B ---------------------------------------------------------

run_train_phase() {
    # run_train_phase <kind: ae|prior> <arms>
    kind="$1"
    arms="$2"

    if [ "$kind" = "prior" ]; then
        # A missing ae.pth fails preflight anyway, but the reason is worth
        # spelling out before eight cards are committed.
        for half in $HALVES; do
            frozen="$OUT_ROOT/${half}.ae.pth"
            if [ ! -f "$frozen" ]; then
                echo "No promoted compressor for '$half': $frozen" >&2
                echo "Run phase A, read ae_report.py, then:" >&2
                echo "  PROMOTE_$(echo "$half" | tr '[:lower:]' '[:upper:]')=<arm> \\" >&2
                echo "  bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh promote" >&2
                return 1
            fi
            echo "  $half frozen compressor: $frozen ($(date -r "$frozen" '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'mtime?'))"
        done
    fi

    selected=""
    if [ "$PREFLIGHT" = "1" ]; then
        echo "---- preflight ---------------------------------------------------"
        for half in $HALVES; do
            for arm in $arms; do
                label="${half}.${kind}_${arm}"
                if preflight_one "$CFG_DIR/config_train_${kind}_${half}_${arm}.txt" "$label"; then
                    selected="$selected ${half}:${arm}"
                fi
            done
        done
        if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
            echo "STRICT_PREFLIGHT=1; no partial phase will be launched." >&2
            return 1
        fi
    else
        for half in $HALVES; do
            for arm in $arms; do selected="$selected ${half}:${arm}"; done
        done
    fi
    [ -n "$selected" ] || { echo "Every arm failed preflight." >&2; return 1; }

    echo "---- launching ---------------------------------------------------"
    pids=""
    for pair in $selected; do
        half="${pair%%:*}"
        arm="${pair##*:}"
        label="${half}.${kind}_${arm}"
        marker="$LOG_ROOT/${label}.launch.marker"
        : > "$marker"
        # stdout goes somewhere OTHER than log_file_dir: the trainer owns
        # $OUT_ROOT/$kind/$label.log and would interleave with the shell's
        # redirect if both pointed at one file.
        "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_train_${kind}_${half}_${arm}.txt" \
            > "$LOG_ROOT/${label}.out" 2>&1 &
        pid=$!
        pids="$pids $pid"
        echo "  $label launched (pid $pid)"
        sleep "$STAGGER"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done

    echo "---- results -----------------------------------------------------"
    for pair in $selected; do
        half="${pair%%:*}"
        arm="${pair##*:}"
        label="${half}.${kind}_${arm}"
        checkpoint="$OUT_ROOT/${label}.pth"
        marker="$LOG_ROOT/${label}.launch.marker"
        if [ -f "$checkpoint" ] && [ "$checkpoint" -nt "$marker" ]; then
            echo "  $label complete -> $checkpoint"
        else
            echo "  $label FAILED to produce a fresh checkpoint -- $LOG_ROOT/${label}.out"
            SKIPPED="$SKIPPED $label(train)"
            rc=1
        fi
    done
    return 0
}

# --- promote (auto-select, or a forced arm via PROMOTE_BOT/PROMOTE_TOP) --

select_and_promote_half() {
    # select_and_promote_half <half> -> select_ae.py's own exit code:
    # 0 promoted and clear to proceed; 3 promoted but NEEDS_ATTENTION (skip
    # B/C for this half); 1 could not decide or promote_ae.py rejected it.
    half="$1"
    case "$half" in
        bot) force_arm="${PROMOTE_BOT:-}" ;;
        top) force_arm="${PROMOTE_TOP:-}" ;;
        *)   force_arm="" ;;
    esac
    if [ -n "$force_arm" ]; then
        checkpoint="$OUT_ROOT/${half}.ae_${force_arm}.pth"
        marker="$LOG_ROOT/${half}.ae_${force_arm}.launch.marker"
        if [ -f "$marker" ] && [ ! "$checkpoint" -nt "$marker" ]; then
            echo "  $half: $checkpoint is older than its own launch marker -- that" >&2
            echo "         run did not finish. Not promoting." >&2
            return 1
        fi
    fi
    if [ -n "$force_arm" ]; then
        echo "---- select_ae $half (forced: $force_arm) ----"
    else
        echo "---- select_ae $half (auto) ----"
    fi
    args=(--half "$half" --out-root "$OUT_ROOT" --config-dir "$CFG_DIR")
    [ -n "$force_arm" ] && args+=(--force-arm "$force_arm")
    "$PYTHON" "$CFG_DIR/select_ae.py" "${args[@]}"
}

run_promote() {
    any=0
    for half in $HALVES; do
        any=1
        if select_and_promote_half "$half"; then
            :
        else
            code=$?
            if [ "$code" = "3" ]; then
                echo "  $half: NEEDS_ATTENTION -- promoted the least-bad arm anyway (see above)." >&2
            else
                SKIPPED="$SKIPPED ${half}.promote"
                rc=1
            fi
        fi
    done
    if [ "$any" = "0" ]; then
        echo "HALVES is empty; nothing to promote." >&2
        return 1
    fi
    return 0
}

# --- phase C -------------------------------------------------------------

run_infer_arm() {
    half="$1"
    arm="$2"
    arm_rc=0
    for tag in $INFER_TAGS; do
        label="${half}.prior_${arm}.infer_${tag}"
        cfg="$CFG_DIR/config_infer_${half}_${arm}_${tag}.txt"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" \
                > "$LOG_ROOT/${label}.out" 2>&1; then
            echo "  $label done"
        else
            echo "  $label FAILED -- $LOG_ROOT/${label}.out"
            arm_rc=1
        fi
    done
    return "$arm_rc"
}

run_phase_c() {
    # An _infer_/_compare_ mismatch compares two different parts and reads as
    # a spread defect, so the pairing is verified before any GPU work.
    if [ "$EVAL_PREFLIGHT" = "1" ]; then
        echo "---- same-condition data preflight -------------------------------"
        if ! "$PYTHON" "$CFG_DIR/check_eval_inputs.py" --config-dir "$CFG_DIR" \
                > "$LOG_ROOT/eval_inputs.check.log" 2>&1; then
            echo "Inference data preflight FAILED -- $LOG_ROOT/eval_inputs.check.log" >&2
            return 1
        fi
        echo "  infer/compare pairs OK"
    fi

    selected=""
    if [ "$PREFLIGHT" = "1" ]; then
        echo "---- preflight ---------------------------------------------------"
        for half in $HALVES; do
            for arm in $PRIOR_ARMS; do
                ok=1
                for tag in $INFER_TAGS; do
                    label="${half}.prior_${arm}.infer_${tag}"
                    preflight_one "$CFG_DIR/config_infer_${half}_${arm}_${tag}.txt" \
                        "$label" || ok=0
                done
                [ "$ok" = "1" ] && selected="$selected ${half}:${arm}"
            done
        done
        if [ -n "$SKIPPED" ] && [ "$STRICT_PREFLIGHT" = "1" ]; then
            echo "STRICT_PREFLIGHT=1; no partial phase will be launched." >&2
            return 1
        fi
    else
        for half in $HALVES; do
            for arm in $PRIOR_ARMS; do selected="$selected ${half}:${arm}"; done
        done
    fi
    [ -n "$selected" ] || { echo "Every arm failed preflight." >&2; return 1; }

    # One card per (half, arm); that arm's three eval parts run in series on it.
    echo "---- inference + spread histograms -------------------------------"
    pids=""
    for pair in $selected; do
        run_infer_arm "${pair%%:*}" "${pair##*:}" &
        pids="$pids $!"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done
    echo "  histograms and spread_values.npz under $OUT_ROOT/infer"
    return 0
}

# --- all (the default: unattended end to end) -----------------------------

run_all() {
    echo "==== Phase A: train compressors ($HALVES) =========================="
    run_train_phase ae "$AE_ARMS" || rc=1

    echo
    echo "==== auto-select + promote (per half) ==============================="
    READY=""
    for half in $HALVES; do
        select_log="$LOG_ROOT/${half}.select_ae.log"
        if select_and_promote_half "$half" 2>&1 | tee "$select_log"; then
            READY="$READY $half"
        else
            code=$?
            if [ "$code" = "3" ]; then
                echo "  $half: NEEDS_ATTENTION -- promoted the least-bad arm, but" >&2
                echo "        Phase B/C will be SKIPPED for $half (see $select_log)" >&2
                SKIPPED="$SKIPPED ${half}(needs_attention)"
            else
                echo "  $half: select_ae.py failed (rc=$code) -- Phase B/C will be" >&2
                echo "        SKIPPED for $half (see $select_log)" >&2
                SKIPPED="$SKIPPED ${half}(select_ae)"
                rc=1
            fi
        fi
    done

    if [ -z "$READY" ]; then
        echo
        echo "No half is ready for Phase B/C -- stopping here." >&2
        return 1
    fi
    if [ "$READY" != "$HALVES" ]; then
        echo "  proceeding to Phase B/C with:$READY"
    fi

    echo
    echo "==== Phase B: train flow priors ($READY) ============================"
    HALVES="$READY" run_train_phase prior "$PRIOR_ARMS" || rc=1

    echo
    echo "==== Phase C: inference + spread histograms ($READY) ================"
    HALVES="$READY" run_phase_c || rc=1

    echo
    echo "==== Phase A convergence report ======================================"
    "$PYTHON" "$CFG_DIR/ae_report.py" --out-root "$OUT_ROOT" 2>&1 \
        | tee "$LOG_ROOT/ae_report.txt"

    echo
    echo "==== Phase C ranking =================================================="
    "$PYTHON" "$CFG_DIR/rank_arms.py" --out-root "$OUT_ROOT" 2>&1 \
        | tee "$OUT_ROOT/RESULTS.txt"
    echo
    echo "Full results also saved to $OUT_ROOT/RESULTS.txt"

    return 0
}

# --- dispatch ------------------------------------------------------------

case "$PHASE" in
    all|ALL) ;;
    A|a) ;;
    B|b) ;;
    C|c) ;;
    promote|PROMOTE) ;;
    -h|--help|help) usage; exit 2 ;;
    *) echo "unknown phase: $PHASE" >&2; usage; exit 2 ;;
esac

mkdir -p "$LOG_ROOT" "$OUT_ROOT/ae" "$OUT_ROOT/prior" "$OUT_ROOT/infer"

echo "=================================================================="
echo " cHI-MGNflow SAOI sweep -- phase $PHASE"
echo "=================================================================="
echo "  halves : $HALVES"
echo "  output : $OUT_ROOT"

case "$PHASE" in
    all|ALL)
        run_all || rc=1
        ;;
    A|a)
        echo "  arms   : $AE_ARMS   (stage 1, the compressor/VAE)"
        run_train_phase ae "$AE_ARMS" || rc=1
        echo
        echo "Next: read the curves, then choose ONE compressor per half."
        echo "  python configs/HI_MGNFlow/hyperparameter_sweep/ae_report.py"
        echo "  python configs/HI_MGNFlow/hyperparameter_sweep/ae_ceiling_check.py --half bot --arm <arm>"
        ;;
    promote|PROMOTE)
        run_promote || rc=1
        ;;
    B|b)
        echo "  arms   : $PRIOR_ARMS   (stage 2, the flow prior)"
        run_train_phase prior "$PRIOR_ARMS" || rc=1
        ;;
    C|c)
        echo "  arms   : $PRIOR_ARMS"
        echo "  parts  : $INFER_TAGS"
        run_phase_c || rc=1
        echo
        echo "Next: python configs/HI_MGNFlow/hyperparameter_sweep/rank_arms.py"
        ;;
esac

# hierarchy_cache_keep True leaves each half's multiscale cache beside its
# training HDF5 so the next phase does not rebuild it. Nothing deletes it for
# you, and it is worth keeping until Phase B is done.
case "$PHASE" in
    all|ALL|A|a|B|b)
        echo "  hierarchy caches kept: rm -f dataset/SAOI/saoi_train_*.mscache.*.h5"
        ;;
esac

if [ -n "$SKIPPED" ]; then
    echo "SKIPPED -- phase incomplete:$SKIPPED" >&2
fi
echo "Finished phase $PHASE with rc=$rc"
exit "$rc"
