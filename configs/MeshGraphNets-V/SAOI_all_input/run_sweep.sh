#!/usr/bin/env bash
# One-click runner for the SAOI "all input" TOP/BOT production run.
#
# This is NOT a hyperparameter sweep (unlike the sibling SAOI_sweep2/
# SAOI_sweep3 directories, which searched over training recipes). Every config
# here shares ONE fixed recipe; the only thing that varies is:
#
#   TRAIN  config_train_top.txt / config_train_bot.txt
#          Same hyperparameters, disjoint mesh halves (saoi_train_top.h5 vs
#          saoi_train_bot.h5), disjoint GPUs (top: 4-7, bot: 0-3), disjoint
#          checkpoints (saoi_top.pth / saoi_bot.pth). Nothing is shared between
#          them -- including the multiscale cache, since each reads a
#          different source HDF5 -- so both launch together with no lock
#          contention (unlike SAOI_sweep2/3, where N arms shared one dataset
#          and had to warm one shared cache first).
#
#   INFER  6 configs = 3 held-out extrapolation test geometries
#          (S26FE-MAIN, S26FE-SEC, SM-L345U-MAIN) x 2 sections (top/bot).
#          Each pins one GPU (0-5) and points at the matching section's
#          checkpoint, so all 6 run concurrently once training is done.
#
# "Which is better" therefore means: does the TOP or BOT section model
# generalize better, and on which held-out geometry -- not which
# hyperparameter cell wins. score_generalization.py answers that from the
# rollout output vs each held-out geometry's ground truth.
#
# use_vae is True and the 6 infer configs carry num_vae_samples 5000 (train
# configs carry 10000, unused in train mode): inference draws that many
# stochastic latent samples PER held-out geometry and writes one HDF5 per
# draw (preflight already flags this as MGNV-SAMPLES-WORKLOAD). Fine for a
# final production sampling run, far too slow/disk-heavy for a first
# correctness/ranking pass across 6 arms at once. Override it for THIS run
# only (checked-in configs are never modified) with:
#   INFER_VAE_SAMPLES=50 bash run_sweep.sh
#
# THIS IS A MULTI-HOUR/MULTI-DAY RUN. Start it detached:
#   nohup bash configs/MeshGraphNets-V/SAOI_all_input/run_sweep.sh > run_sweep.out 2>&1 &
#   tail -f run_sweep.out
#   tail -f outputs/saoi_all_input/run_logs/train_top.log      # watch one arm
#
# Environment overrides:
#   PYTHON             interpreter (default: python)
#   LOG_ROOT            transcript directory (default: outputs/saoi_all_input/run_logs)
#   PREFLIGHT           1 = --check every arm before launching anything (default); 0 = skip
#   TRAIN               1 = run the two training arms (default); 0 = skip (reuse existing checkpoints)
#   INFER               1 = run the six inference arms (default); 0 = skip
#   INFER_VAE_SAMPLES   override num_vae_samples in the 6 infer configs for this run only (unset = use checked-in value, currently 5000)
#   SCORE               1 = run score_generalization.py once inference ends (default); 0 = skip
#
# Usage:
#   bash configs/MeshGraphNets-V/SAOI_all_input/run_sweep.sh
#   TRAIN=0 bash .../run_sweep.sh                       # only re-run inference + scoring
#   INFER_VAE_SAMPLES=50 bash .../run_sweep.sh          # quick ranking pass

# NOT `set -e`: one arm failing must not kill the rest of the batch.
set -uo pipefail

PYTHON="${PYTHON:-python}"
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
INFER_VAE_SAMPLES="${INFER_VAE_SAMPLES:-}"
SCORE="${SCORE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="configs/MeshGraphNets-V/SAOI_all_input"
LOG_ROOT="${LOG_ROOT:-outputs/saoi_all_input/run_logs}"
TMP_CFG_DIR="outputs/saoi_all_input/tmp_configs"

TRAIN_ARMS="train_top train_bot"
INFER_ARMS="infer_s26fe_main_top infer_s26fe_main_bot \
infer_s26fe_sec_top infer_s26fe_sec_bot \
infer_sm_l345u_main_top infer_sm_l345u_main_bot"

mkdir -p "$LOG_ROOT"

cfg_for()  { echo "$CFG_DIR/config_${1}.txt"; }
log_for()  { echo "$LOG_ROOT/${1}.log"; }

# Checkpoint each infer arm depends on ("infer_s26fe_main_top" -> top's ckpt).
ckpt_for() {
    case "$1" in
        *_top) echo "output/meshgraphnets-v/saoi_all/saoi_top.pth" ;;
        *_bot) echo "output/meshgraphnets-v/saoi_all/saoi_bot.pth" ;;
    esac
}

run_arm() {
    local arm=$1 cfg log rc
    cfg="$2"
    log="$(log_for "$arm")"
    if [ ! -f "$cfg" ]; then
        echo "[$arm] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    # `$?` after a false `if` with no `else` is the if-statement's own status
    # (0), not the command's -- capture rc directly instead.
    "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[$arm] DONE"
        return 0
    fi
    echo "[$arm] FAILED (exit $rc) -- see $log" >&2
    return 1
}

echo "SAOI all-input -- TOP/BOT production train + held-out generalization check"
echo "  REPO_ROOT = $REPO_ROOT"
echo "  PYTHON    = $PYTHON"
echo "  LOG_ROOT  = $LOG_ROOT"
echo ""

# ---- Preflight every arm before committing GPU-hours -----------------------
if [ "$PREFLIGHT" = "1" ]; then
    echo "Preflight (--check) on every arm..."
    pf_bad=0
    for arm in $TRAIN_ARMS $INFER_ARMS; do
        cfg="$(cfg_for "$arm")"
        if [ ! -f "$cfg" ]; then
            echo "  $arm  MISSING CONFIG" >&2; pf_bad=1; continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  $arm  ok"
        else
            echo "  $arm  FAILED -- see $LOG_ROOT/${arm}.check.log" >&2; pf_bad=1
        fi
    done
    if [ "$pf_bad" != "0" ]; then
        echo "" >&2
        echo "Preflight failed. Nothing launched. Fix the configs and re-run," >&2
        echo "or set PREFLIGHT=0 to launch anyway." >&2
        exit 2
    fi
    echo "All arms validated."
    echo ""
fi

started=$(date +%s)
train_rc=0

# ---- Phase 1: train TOP and BOT concurrently --------------------------------
# Different source HDF5 -> different multiscale-cache file each -> no shared
# lock, so (unlike SAOI_sweep2/3) both can launch at once with no warm-up step.
if [ "$TRAIN" = "1" ]; then
    echo "Launching training arms: $TRAIN_ARMS"
    pids=(); names=()
    for arm in $TRAIN_ARMS; do
        run_arm "$arm" "$(cfg_for "$arm")" &
        pids+=("$!"); names+=("$arm")
        echo "  launched $arm (pid $!)"
    done
    for k in "${!pids[@]}"; do
        if ! wait "${pids[$k]}"; then
            echo "${names[$k]} exited non-zero" >&2
            train_rc=1
        fi
    done
    echo "Training phase finished in $(( ($(date +%s) - started) / 60 ))m (rc=$train_rc)."
    echo ""
else
    echo "TRAIN=0 -- skipping training, reusing existing checkpoints."
    echo ""
fi

# ---- Phase 2: infer on the 3 held-out geometries x 2 sections --------------
infer_rc=0
if [ "$INFER" = "1" ]; then
    active_infer_arms=""
    for arm in $INFER_ARMS; do
        ckpt="$(ckpt_for "$arm")"
        if [ ! -f "$ckpt" ]; then
            echo "[$arm] SKIP: checkpoint not found ($ckpt) -- its training arm did not finish" >&2
            continue
        fi
        active_infer_arms="$active_infer_arms $arm"
    done

    if [ -z "${active_infer_arms// /}" ]; then
        echo "No infer arm has a usable checkpoint -- skipping inference." >&2
        infer_rc=1
    else
        # INFER_VAE_SAMPLES: materialize throwaway config copies with
        # num_vae_samples patched, rather than editing the checked-in configs.
        if [ -n "$INFER_VAE_SAMPLES" ]; then
            echo "Overriding num_vae_samples -> $INFER_VAE_SAMPLES for this run (temp configs in $TMP_CFG_DIR)"
            mkdir -p "$TMP_CFG_DIR"
        fi

        echo "Launching inference arms:$active_infer_arms"
        pids=(); names=()
        for arm in $active_infer_arms; do
            cfg="$(cfg_for "$arm")"
            if [ -n "$INFER_VAE_SAMPLES" ]; then
                tmp_cfg="$TMP_CFG_DIR/config_${arm}.txt"
                sed -E "s/^num_vae_samples([[:space:]]+)[0-9]+/num_vae_samples\\1${INFER_VAE_SAMPLES}/" \
                    "$cfg" > "$tmp_cfg"
                cfg="$tmp_cfg"
            fi
            run_arm "$arm" "$cfg" &
            pids+=("$!"); names+=("$arm")
            echo "  launched $arm (pid $!)"
        done
        for k in "${!pids[@]}"; do
            if ! wait "${pids[$k]}"; then
                echo "${names[$k]} exited non-zero" >&2
                infer_rc=1
            fi
        done
        echo "Inference phase finished (rc=$infer_rc)."
        echo ""
    fi
else
    echo "INFER=0 -- skipping inference."
    echo ""
fi

ended=$(date +%s)
echo "Total wall time: $(( (ended - started) / 3600 ))h $(( ((ended - started) % 3600) / 60 ))m"
echo ""
echo "Transcripts : $LOG_ROOT/<arm>.log"
echo "Checkpoints : output/meshgraphnets-v/saoi_all/saoi_{top,bot}.pth"
echo "Rollouts    : output/meshgraphnets-v/saoi_all/infer_<test_set>_<section>/"
echo ""

# ---- Phase 3: score TOP vs BOT on each held-out geometry -------------------
rc=$(( train_rc || infer_rc ))
REPORT="outputs/saoi_all_input/generalization_report.md"
if [ "$SCORE" = "1" ]; then
    echo "Scoring TOP vs BOT generalization against the 3 held-out geometries..."
    if "$PYTHON" "$CFG_DIR/score_generalization.py" \
            --out-dir outputs/saoi_all_input \
            > "$LOG_ROOT/score_generalization.log" 2>&1; then
        echo "Scoring complete."
    else
        echo "Scoring FAILED (exit $?) -- see $LOG_ROOT/score_generalization.log" >&2
        rc=1
    fi
    echo ""
    if [ -f "$REPORT" ]; then
        echo "================= RESULTS ================="
        cat "$REPORT"
        echo "==========================================="
        echo ""
        echo "Report   : $REPORT      <-- paste this file to Claude"
        echo "Raw JSON : outputs/saoi_all_input/generalization_results.json"
    fi
else
    echo "SCORE=0 -- skipped. Run it later with:"
    echo "  $PYTHON $CFG_DIR/score_generalization.py"
fi

exit $rc
