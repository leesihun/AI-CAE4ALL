#!/usr/bin/env bash
# Complete from-scratch MeshGraphNets-V SAOI FM-v2 campaign.
#
#   nohup bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh \
#       > output/meshgraphnets-v/saoi_fm_v2_sweep/run.out 2>&1 &
#
# One invocation generates configs, validates every input, trains eight arms,
# runs stochastic and deterministic inference, builds the same-condition
# GT -> posterior -> prior figures, ranks the arms and writes the report.
#
# Arms 1..4 are bot P0/P1/P2/P3; arms 5..8 are top P0/P1/P2/P3.
# Training and both inference stages run one arm per GPU in parallel.
#
# Useful overrides:
#   PYTHON, ARMS, INFER_TAGS, STAGGER
#   GENERATE=1, PREFLIGHT=1, STRICT_PREFLIGHT=1, EVAL_PREFLIGHT=1
#   TRAIN=1, INFER=1, DET=1, PVP=1, SCORE=1
#   PVP_N=500, PVP_CHUNK=16, PVP_FORCE=0
set -uo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="$SCRIPT_DIR"
OUT_ROOT="output/meshgraphnets-v/saoi_fm_v2_sweep"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/run_logs}"
DOC="docs/research/SAOI_FM_V2_SWEEP_MGNV.md"

ARMS="${ARMS:-1 2 3 4 5 6 7 8}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u}"
GENERATE="${GENERATE:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-1}"
EVAL_PREFLIGHT="${EVAL_PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
DET="${DET:-1}"
PVP="${PVP:-1}"
SCORE="${SCORE:-1}"
PVP_N="${PVP_N:-500}"
PVP_CHUNK="${PVP_CHUNK:-16}"
PVP_FORCE="${PVP_FORCE:-0}"
STAGGER="${STAGGER:-10}"

mkdir -p "$LOG_ROOT"
rc=0
SKIPPED=""

echo "=================================================================="
echo " MeshGraphNets-V SAOI FM-v2: P0 -> P1 -> P2 -> P3, bot and top"
echo "=================================================================="
echo "  arms      : $ARMS"
echo "  eval sets : $INFER_TAGS"
echo "  output    : $OUT_ROOT"

# Always derive configs from the production profiles immediately before a run.
if [ "$GENERATE" = "1" ]; then
    echo "---- generating configs -----------------------------------------"
    if ! "$PYTHON" "$CFG_DIR/gen_configs.py" > "$LOG_ROOT/generate.log" 2>&1; then
        echo "Config generation FAILED -- $LOG_ROOT/generate.log" >&2
        exit 1
    fi
    echo "  generated from SAOI_all_input"
fi

# The eval pair is common to all arms. A mismatch invalidates the entire
# posterior/prior comparison, so this check is intentionally fail-fast.
if [ "$EVAL_PREFLIGHT" = "1" ] && { [ "$INFER" = "1" ] || [ "$PVP" = "1" ]; }; then
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
    echo "---- from-scratch training --------------------------------------"
    pids=""
    for arm in $ARMS; do
        marker="$LOG_ROOT/${arm}.launch.marker"
        : > "$marker"
        "$PYTHON" AI_CAE4ALL_main.py --config "$CFG_DIR/config_train_${arm}.txt" \
            > "$OUT_ROOT/${arm}.log" 2>&1 &
        pid=$!
        pids="$pids $pid"
        echo "  arm $arm launched (pid $pid, GPU $((arm - 1)))"
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

run_infer_arm() {
    arm="$1"
    [ -f "$OUT_ROOT/${arm}.pth" ] || return 1
    arm_rc=0
    for tag in $INFER_TAGS; do
        cfg="$CFG_DIR/config_infer_${arm}_${tag}.txt"
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" \
                > "$LOG_ROOT/${arm}.infer_${tag}.log" 2>&1; then
            echo "  arm $arm $tag stochastic done"
        else
            echo "  arm $arm $tag stochastic FAILED"
            arm_rc=1
        fi
        if [ "$DET" = "1" ]; then
            if "$PYTHON" AI_CAE4ALL_main.py \
                    --config "$CFG_DIR/config_infer_${arm}_${tag}_det.txt" \
                    > "$LOG_ROOT/${arm}.infer_${tag}_det.log" 2>&1; then
                echo "  arm $arm $tag deterministic done"
            else
                echo "  arm $arm $tag deterministic FAILED"
                arm_rc=1
            fi
        fi
    done
    return "$arm_rc"
}

if [ "$INFER" = "1" ]; then
    echo "---- parallel inference -----------------------------------------"
    pids=""
    for arm in $ARMS; do
        run_infer_arm "$arm" &
        pids="$pids $!"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done
fi

run_pvp_arm() {
    arm="$1"
    arm_rc=0
    checkpoint="$OUT_ROOT/${arm}.pth"
    [ -f "$checkpoint" ] || return 1
    pvp_script="methods/MeshGraphNets_Variational/misc/posterior_vs_prior.py"
    for tag in $INFER_TAGS; do
        cfg="$CFG_DIR/config_infer_${arm}_${tag}.txt"
        diag_dir="$OUT_ROOT/diag/$arm/$tag"
        diag_tag="arm_${arm}_${tag}"
        base="$diag_dir/posterior_vs_prior_${diag_tag}"
        if [ "$PVP_FORCE" != "1" ] \
                && [ -f "$base.json" ] && [ -f "$base.png" ] \
                && [ -f "$base.pdf" ] && [ -f "$base.svg" ] \
                && [ "$base.json" -nt "$checkpoint" ] \
                && [ "$base.json" -nt "$cfg" ] \
                && [ "$base.json" -nt "$pvp_script" ]; then
            echo "  arm $arm $tag posterior/prior diagnostic reused"
            continue
        fi
        mkdir -p "$diag_dir"
        if "$PYTHON" "$pvp_script" --config "$cfg" --tag "$diag_tag" \
                --n-prior "$PVP_N" --chunk "$PVP_CHUNK" --out "$diag_dir" \
                > "$LOG_ROOT/${arm}.posterior_vs_prior_${tag}.log" 2>&1; then
            echo "  arm $arm $tag GT/posterior/prior done"
        else
            echo "  arm $arm $tag GT/posterior/prior FAILED"
            arm_rc=1
        fi
    done
    return "$arm_rc"
}

if [ "$PVP" = "1" ]; then
    echo "---- parallel GT -> posterior -> FM-prior diagnosis ------------"
    pids=""
    for arm in $ARMS; do
        run_pvp_arm "$arm" &
        pids="$pids $!"
    done
    for pid in $pids; do
        wait "$pid" || rc=1
    done
fi

if [ "$SCORE" = "1" ]; then
    echo "================= RANKING ================="
    "$PYTHON" configs/campaigns/rank_arms.py "$OUT_ROOT/infer" 2>&1 \
        | tee "$LOG_ROOT/rank_arms.log" || rc=1
    echo "==========================================="
    if "$PYTHON" configs/campaigns/write_report.py \
            --kind mgnv --axis fm_variant \
            --label "MeshGraphNets-V SAOI FM-v2 (P0 baseline, P1 width, P2 residual FiLM, P3 conditional moments)" \
            --infer "$OUT_ROOT/infer" --logs "$OUT_ROOT" --configs "$CFG_DIR" \
            --diag "$OUT_ROOT/diag" --arms "$ARMS" --out "$DOC" \
            > "$LOG_ROOT/write_report.log" 2>&1; then
        echo "Report: $DOC"
    else
        echo "Report FAILED -- $LOG_ROOT/write_report.log" >&2
        rc=1
    fi
fi

if [ -n "$SKIPPED" ]; then
    echo "SKIPPED -- campaign incomplete:$SKIPPED" >&2
fi
echo "Finished with rc=$rc"
exit "$rc"
