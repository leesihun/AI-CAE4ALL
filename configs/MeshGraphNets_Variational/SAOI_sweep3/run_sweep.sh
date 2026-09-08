#!/usr/bin/env bash
# One-click runner for the SAOI WAVE 3 sweep: a 2^(4-1) resolution-IV half
# fraction, 8 arms, ONE PER GPU.
#
# Defining relation I = ABCD: the fourth factor is D = A xor B xor C. The 4
# main effects are clean; the 6 two-factor interactions come in 3 CONFOUNDED
# PAIRS (AB=CD, AC=BD, AD=BC) -- a large one cannot be attributed to a single
# pair without a follow-up run.
#
#   A  z_conditioning         cc  concat (legacy fuser) | ad  adaln (AdaLN-Zero)
#   B  prior_grad_to_encoder  g0  detached (no CVAE rate term) | g1  end-to-end
#   C  capacity               c0  128 / mp 4,6,8,6,4  | c1  128 / mp 6,8,12,8,6
#   D  regularizer scale      r001 (lambda_mmd 1, prior_nll_weight 1) |
#                             r100 (lambda_mmd 100, prior_nll_weight 100)
#                             GENERATED: D = A xor B xor C
#
# `vae_latent_dim` is FIXED at 16 in every arm -- dropped from an earlier
# 5-factor / 16-arm version of this grid. See gen_sweep_configs.py for why.
#
# Arm names encode the cell: <cc|ad>_<g0|g1>_<c0|c1>_<r001|r100>, and the
# files are config_train_<arm>.txt / config_infer_<arm>_<tag>.txt.
# Regenerate the configs with gen_sweep_configs.py; do not hand-edit them.
#
# EIGHT arms, ONE PER GPU (0-7) -- no card sharing, so there is no VRAM
# co-residency exposure. Each arm pins its GPU in its own config (gpu_ids), so
# this script only launches them; it does not set CUDA_VISIBLE_DEVICES.
#
# Budget: 500 epochs at a measured ~576 s/epoch is ~3.3 days per arm. This
# replaced an earlier 2000-epoch / 16-arm / 2-per-GPU design: 24h of wall time
# only reached epoch 150 on that budget (multiple GPU-weeks to finish), and
# two-per-GPU sharing was a real OOM risk on the wider capacity level.
# **THIS IS A BUDGET-LIMITED COMPARISON, NOT A CONVERGED ONE** -- read the
# report that way, and see the twin cHI-MGNflow sweep (SAOI_sweepB), which
# made the identical trade for the identical reason.
#
# WATCH THE FIRST EPOCHS' tqdm postfix: `mmd` AND `fm_p` against `total`.
# alpha_recon is 1000 while both regularizers sit at ~1 in r001, so each is
# ~0.1% of the objective there -- that half of the grid is the "regularizers
# effectively off" control, and the g0/g1 contrast can only show force in the
# r100 half. If BOTH r001 and r100 look negligible the axis has to move UP
# (1000/10000), not sideways.
#
# THIS IS A MULTI-DAY RUN. Start it detached:
#   nohup bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep.sh > sweep.out 2>&1 &
#   tail -f sweep.out
#   tail -f output/meshgraphnets-v/saoi_sweep3/run_logs/3.log   # watch one arm
#
# Multiscale cache: all 8 arms hash to ONE cache file (none of the swept keys
# are part of the coarsening signature), and every arm is launched straight
# away. Coordination is left entirely to general_modules/multiscale_cache.py,
# which does it in the process that can actually see the answer: an exclusive
# O_CREAT|O_EXCL lock means exactly one builds while the rest poll every 3 s and
# return the instant the file is valid (default wait 10 h, stale lock reclaimed
# at 6 h).
#
# This script used to gate the other 7 behind a warm-up arm, which meant
# GUESSING the cache file's path from the shell. It guessed wrong twice -- the
# dataset directory's case (invisible on Windows, fatal on Linux) and the
# digest, which changes on nearly every run because the signature hashes the
# source HDF5's mtime and write_preprocessing_to_hdf5 bumps it. Either way the
# gate never released and 7 of 8 GPUs idled to the timeout. Idle GPU time is the
# same either way; the difference is that nothing can strand them now.
#
# The configs set hierarchy_cache_keep True so no finishing arm deletes the
# cache out from under the others. DELETE IT MANUALLY when the sweep is done:
#   rm dataset/SAOI/saoi_train_bot.mscache.*.h5
#
# NOTE ON THE SHARED DATASET FILE: every arm's setup phase re-derives and
# rewrites normalization stats into saoi_train_bot.h5 itself (unconditionally,
# every run -- there is no "already present" guard). With 8 arms launched
# seconds apart against the SAME file, one arm's writer can collide with
# another's reader and raise "Unable to synchronously open file"
# (HDF5_USE_FILE_LOCKING=FALSE is set repo-wide for NFS compatibility, which
# removes HDF5's own guard against exactly this). general_modules/mesh_dataset.py
# now retries that specific open with backoff, so a transient collision no
# longer kills the arm -- if one still fails outright, just relaunch it:
#   ARMS="<the one arm>" PREFLIGHT=0 TRAIN=1 INFER=0 SCORE=0 bash .../run_sweep.sh
#
# Environment overrides:
#   PROBE         1 = time 3 epochs and size training_epochs to BUDGET_H
#                 before launching (default); 0 = use the config as generated
#   BUDGET_H      wall-clock target in hours for the sized run (default: 16)
#   HEADROOM      fraction of BUDGET_H kept back (default: 0.90)
#   PYTHON        interpreter (default: python)
#   LOG_ROOT      transcript directory (default: output/meshgraphnets-v/saoi_sweep3/run_logs)
#   ARMS          space-separated arm names (default: all 8)
#   PREFLIGHT     1 = --check every arm before launching any (default); 0 = skip.
#                 A failing arm is DROPPED; the rest still run.
#   STRICT_PREFLIGHT 1 = abort the batch if anything fails preflight
#                 (default 0 = drop the failures, run the remainder)
#   TRAIN         1 = train (default). 0 = SKIP training and go straight to
#                 inference + scoring on checkpoints that already exist.
#   STAGGER       seconds between arm launches (default: 10)
#   INFL          1 = run the latent-inflation sweep configs (MGN-V only):
#                 one pass per arm over lam in {1,1.5,2,2.5,3}, output under
#                 infer/infl/, and the lam ranking printed at the end.
#   DET           1 = run the deterministic-control inference configs
#                 (one draw, no sampling noise) into output .../infer/det/;
#                 default 0 = the multi-draw run
#   INFER         1 = run each arm's inference configs after training (default)
#   INFER_TAGS    eval sets to infer (default: s26fe_main s26fe_sec sm_l345u)
#   SCORE         1 = run score_sweep.py when training ends (default); 0 = skip
#   SCORE_K       draws per geometry for the rank histogram (default: 50)
#   SCORE_SPLIT   split to score (default: test)
#   SCORE_SAMPLERS  eval_distribution.py samplers; EMPTY by default,
#                 which skips that (slow, serial, GPU) stage. Set to
#                 "prior normal" to include the rank histograms.
#
# Usage:
#   bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep.sh
#   ARMS="3 1" bash .../run_sweep.sh   # subset
#   PREFLIGHT=0 bash .../run_sweep.sh                          # skip validation
#   TRAIN=0 bash .../run_sweep.sh                              # infer + score only

# NOT `set -e`: per-arm failures are collected so one bad arm cannot kill the batch.
set -uo pipefail

PYTHON="${PYTHON:-python}"

# Every launch below redirects stdout to a log file, and Python block-buffers
# (~8 KB) when stdout is not a TTY. The effect is that an arm prints "Starting
# distributed training ..." and then the log sits dead for many minutes while
# the workers load the dataset and build the coarsening hierarchy -- it looks
# hung when it is fine. Unbuffer so `tail -f` reflects real progress; the
# per-line flush cost is nothing next to an epoch.
export PYTHONUNBUFFERED=1
PREFLIGHT="${PREFLIGHT:-1}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
INFER_TAGS="${INFER_TAGS:-s26fe_main s26fe_sec sm_l345u}"
SCORE="${SCORE:-1}"
SCORE_K="${SCORE_K:-50}"
SCORE_SPLIT="${SCORE_SPLIT:-test}"
# Empty = skip eval_distribution.py (see the SCORE stage). Set to
# "prior normal" to add the rank histograms, at GPU-hours per arm.
SCORE_SAMPLERS="${SCORE_SAMPLERS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Absolute, derived from the script itself: the sweep folder can be renamed
# or copied for a wave 4 without editing anything here.
CFG_DIR="$SCRIPT_DIR"
LOG_ROOT="${LOG_ROOT:-output/meshgraphnets-v/saoi_sweep3/run_logs}"

# Must match gen_sweep_configs.arms() exactly -- it prints this line, so if the
# generator changes, re-paste its ARMS= output here rather than hand-editing.
# The peak-to-valley run, not the 8-arm factorial. Every factor that grid
# swept moved W1/sd by 0.001-0.054 against a 0.10 arm-to-arm range, so none
# of them was worth a GPU; beta_aux is the only live axis, and pvA (0) is
# the control that separates the new term from the deleted aux head.
# Regenerate with: gen_sweep_configs.py --pv   (the old grid: no flag)
DEFAULT_ARMS="pvA pvB"
ARMS="${ARMS:-$DEFAULT_ARMS}"
STAGGER="${STAGGER:-10}"   # seconds between arm launches
# DET=1 runs the deterministic-control inference configs instead of the
# multi-draw ones: one draw per scene with the sampling noise off, into a
# separate det/ output subtree. It reads the SAME checkpoints, so it needs no
# training, and it is what tells a shrunk conditional mean (a regression
# failure) apart from an over-tight ensemble (a sampler failure).
DET="${DET:-0}"
# INFL=1 runs the latent-inflation sweep configs instead: one pass per arm
# cycling lam in {1,1.5,2,2.5,3} across draw batches, dumps tagged by lam,
# into infer/infl/. The SCORE stage then ranks the lam values in place with
# configs/campaigns/rank_arms.py -- score_sweep.py is not used there because
# its report is keyed by arm and would collapse an arm's five lam variants
# into a single row.
INFL="${INFL:-0}"
if [ "$DET" = "1" ]; then
    CFG_SUFFIX="_det"
    DET_FLAG="--det"
elif [ "$INFL" = "1" ]; then
    CFG_SUFFIX="_infl"
    DET_FLAG=""
else
    CFG_SUFFIX=""
    DET_FLAG=""
fi

mkdir -p "$LOG_ROOT"

cfg_for()  { echo "$CFG_DIR/config_train_${1}.txt"; }
inf_cfg_for() { echo "$CFG_DIR/config_infer_${1}_${2}${CFG_SUFFIX}.txt"; }
log_for()  { echo "$LOG_ROOT/${1}.log"; }

run_arm() {
    local arm=$1 cfg log rc
    cfg="$(cfg_for "$arm")"
    log="$(log_for "$arm")"
    if [ ! -f "$cfg" ]; then
        echo "[$arm] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    # Capture the status directly. `$?` read after an `if` whose condition was
    # false and which has no else branch is the status of the *if statement*
    # (0), not of the command -- it always printed "exit 0" for a failed arm.
    "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[$arm] DONE"
        return 0
    fi
    echo "[$arm] FAILED (exit $rc) -- see $log" >&2
    return 1
}

echo "SAOI peak-to-valley run -- beta_aux 0 (control) vs 100, 3 GPUs each"
echo "  REPO_ROOT = $REPO_ROOT"
echo "  PYTHON    = $PYTHON"
echo "  LOG_ROOT  = $LOG_ROOT"
echo "  ARMS      = $(echo "$ARMS" | wc -w) arms"
echo ""

# ---- Probe: size the epoch budget before committing the GPUs ---------------
# Neither tree can resume, and cosine_T0 = training_epochs - warmup, so the
# epoch count has to be right before the first real step -- a wrong one is a
# full restart, not an early stop. Three epochs are timed and BUDGET_H divided
# by the measured cost.
#
# The measurement deliberately OVER-estimates: the probe's wall time carries
# process start, dataset load, the multiscale cache build and one validation, while the real
# run validates every 30. So the derived budget errs toward finishing early,
# which is the only safe direction when the alternative is losing the run.
#
# It also warms the shared cache, so the real arms do not queue behind whichever
# one would have built it.
PROBE="${PROBE:-1}"
BUDGET_H="${BUDGET_H:-16}"
HEADROOM="${HEADROOM:-0.90}"

if [ "$PROBE" = "1" ] && [ "$TRAIN" = "1" ]; then
    USABLE=$("$PYTHON" -c "print(int($BUDGET_H * 3600 * $HEADROOM))")
    PROBE_CFG="$CFG_DIR/config_train_pvA_probe.txt"
    PROBE_LOG="$LOG_ROOT/pvA_probe.stdout"
    echo "---- PROBE: 3 epochs to size a ${BUDGET_H}h budget (${USABLE}s usable) ----"
    if [ ! -f "$PROBE_CFG" ]; then
        echo "  missing $PROBE_CFG -- run: $PYTHON $CFG_DIR/gen_sweep_configs.py --pv" >&2
        exit 1
    fi
    T0=$(date +%s)
    PYTHONUNBUFFERED=1 "$PYTHON" "$REPO_ROOT/AI_CAE4ALL_main.py" \
        --config "$PROBE_CFG" 2>&1 | tee "$PROBE_LOG"
    PROBE_RC=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - T0 ))
    if [ "$PROBE_RC" != "0" ]; then
        echo "  PROBE FAILED (exit $PROBE_RC) -- nothing launched. Log: $PROBE_LOG" >&2
        exit "$PROBE_RC"
    fi
    DERIVED=$("$PYTHON" -c "print(max(1, int($USABLE / ($ELAPSED / 3.0))))")
    echo ""
    echo "  ${ELAPSED}s / 3 epochs  ->  $(( ELAPSED / 3 ))s per epoch (over-estimate)"
    echo "  -> PV_EPOCHS=$DERIVED for ${BUDGET_H}h"
    # What weight would put the peak-to-valley term at 30% of the objective.
    # Reported, NOT applied: pvB's beta_aux is a chosen value, and the point of
    # the control arm is that the comparison is between two fixed weights.
    if [ -f configs/campaigns/pv_weight.py ]; then
        "$PYTHON" configs/campaigns/pv_weight.py "$PROBE_LOG" \
            --config "$PROBE_CFG" --fraction 0.30 >/dev/null 2>"$LOG_ROOT/pv_weight.txt"
        [ -s "$LOG_ROOT/pv_weight.txt" ] && sed 's/^/  /' "$LOG_ROOT/pv_weight.txt"
    fi
    env PV_EPOCHS="$DERIVED" "$PYTHON" "$CFG_DIR/gen_sweep_configs.py" --pv || exit 1
    echo ""
fi

# ---- Preflight every arm before committing GPU-days ------------------------
# A failing arm is DROPPED, not fatal. One arm's bad config or dead card says
# nothing about the other seven, and losing a whole night to it is the worse
# outcome. Everything dropped lands in SKIPPED and is reprinted at the end, so a
# shortened sweep cannot be mistaken for a complete one.
# STRICT_PREFLIGHT=1 restores abort-on-any-failure.
SKIPPED=""
# Only when training. With TRAIN=0 the checkpoints already exist and the train
# configs are irrelevant -- validating them would drop an arm (and with it that
# arm's inference) for a reason inference does not care about.
if [ "$PREFLIGHT" = "1" ] && [ "$TRAIN" = "1" ]; then
    echo "Preflight (--check) on every arm..."
    ok_arms=""
    for arm in $ARMS; do
        cfg="$(cfg_for "$arm")"
        if [ ! -f "$cfg" ]; then
            echo "  $arm  MISSING CONFIG ($cfg)" >&2
            SKIPPED="$SKIPPED@  $arm (train)  no config: $cfg"
            continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  $arm  ok"
            ok_arms="$ok_arms $arm"
        else
            echo "  $arm  FAILED -- see $LOG_ROOT/${arm}.check.log" >&2
            SKIPPED="$SKIPPED@  $arm (train)  preflight failed: $LOG_ROOT/${arm}.check.log"
        fi
    done

    ARMS="$(echo $ok_arms)"
    if [ -z "$ARMS" ]; then
        echo "" >&2
        echo "Every arm failed preflight -- nothing left to run." >&2
        echo "If they all report MISSING CONFIG the generated configs are stale" >&2
        echo "or absent; regenerate them first:" >&2
        echo "  python $CFG_DIR/gen_sweep_configs.py --pv" >&2
        exit 2
    fi
    if [ -n "$SKIPPED" ] && [ "${STRICT_PREFLIGHT:-0}" = "1" ]; then
        echo "" >&2
        echo "STRICT_PREFLIGHT=1 and something failed -- nothing launched." >&2
        echo "Skipped:" >&2
        echo "$SKIPPED" | tr '@' '\n' >&2
        exit 2
    fi
    echo "$(echo "$ARMS" | wc -w) arm(s) validated: $ARMS"
    echo ""
fi

# Valid train configs say NOTHING about whether the histogram will have ground
# truth. rollout.py skips it silently when `eval_dataset` is absent, and draws
# against the wrong column when it points at the rollout's own input instead of
# the `_compare_` file. Reported but NOT fatal: it costs the histogram, not the
# run. Checked on EVERY preflighted run, not just inference ones: catching a
# broken ground-truth path before a multi-day training batch is the whole point,
# and a TRAIN=0 run needs it too.
if [ "$PREFLIGHT" = "1" ]; then
    echo "Preflight (inference + histogram inputs)..."
    if ! "$PYTHON" "$CFG_DIR/check_eval_inputs.py"; then
        echo "  ^ the warpage histogram will be wrong or missing for the" >&2
        echo "    datasets named above. Everything else continues." >&2
        SKIPPED="$SKIPPED@  (histogram)  eval inputs failed check_eval_inputs.py"
    fi
    echo ""
fi

started=$(date +%s)
pids=(); names=()
rc=0

if [ "$TRAIN" != "1" ]; then
    echo "TRAIN=0 -- skipping training; using the checkpoints already on disk."
    echo "Arms with no checkpoint will fail their inference preflight and be"
    echo "reported, without stopping the rest."
    echo ""
else

# ---- Launch every arm, staggered -------------------------------------------
# All arms go up together even on a cold cache. multiscale_cache.ensure_cache
# already coordinates the build across processes -- an exclusive O_CREAT|O_EXCL
# lock means exactly one builds while the rest poll every 3 s and return the
# instant it is valid (default wait 10 h, stale lock reclaimed at 6 h). A
# shell-level gate added nothing on top of that and was the fragile half: it had
# to guess the cache file's path, and it guessed wrong twice -- the dataset
# directory's case, and the digest, which changes on nearly every run because
# the signature hashes the dataset's mtime and training writes normalization
# stats back into that same file. Either way it never released and seven of
# eight GPUs idled to the timeout.
#
# STAGGER seconds apart so eight processes do not open the same HDF5 and claim
# GPU memory in the same instant.
for arm in $ARMS; do
    run_arm "$arm" &
    pids+=("$!"); names+=("$arm")
    echo "  launched $arm (pid $!)"
    sleep "$STAGGER"
done

echo ""
echo "$(echo "${names[*]}" | wc -w) arms running. Waiting..."

for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
        echo "${names[$k]} exited non-zero" >&2
        rc=1
    fi
done

ended=$(date +%s)
echo ""
echo "Training finished in $(( (ended - started) / 3600 ))h $(( ((ended - started) % 3600) / 60 ))m (rc=$rc)."

fi   # TRAIN
echo ""
echo "Transcripts : $LOG_ROOT/<arm>.log"
echo "Checkpoints : output/meshgraphnets-v/saoi_sweep3/<arm>.pth"
echo ""

# ---- Inference: every arm against every held-out eval set -------------------
# One arm per GPU, so all 8 run concurrently; each arm's 3 eval sets run
# sequentially on the GPU it trained on. The generated configs set
# save_rollouts False: no trajectory HDF5s are written (scene x draws would be
# thousands of files across the grid). What each run leaves behind is
# histogram_compare.png and spread_values.npz -- the GT vs generated z_disp
# spread (max - min per realization) that score_sweep.py then tabulates and
# overlays for all 8 arms on one axis.
#
# An arm whose training failed has no checkpoint; its inference preflights as a
# missing-input error, is logged, and does not stop the others.
run_infer_arm() {
    local arm=$1 tag cfg log rc=0 irc
    for tag in $INFER_TAGS; do
        cfg="$(inf_cfg_for "$arm" "$tag")"
        log="$LOG_ROOT/${arm}.infer_${tag}${CFG_SUFFIX}.log"
        if [ ! -f "$cfg" ]; then
            echo "[$arm/$tag] SKIP: no config ($cfg)" >&2
            continue
        fi
        # Dropped by the inference preflight -- launching it would only
        # reproduce the same error after paying for the process start.
        case " $INFER_SKIP " in
            *" $arm/$tag "*) echo "[$arm/$tag] SKIP: failed preflight" >&2; continue ;;
        esac
        "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" > "$log" 2>&1
        irc=$?
        if [ "$irc" -ne 0 ]; then
            echo "[$arm/$tag] INFER FAILED (exit $irc) -- see $log" >&2
            rc=1
        fi
    done
    return $rc
}

if [ "$INFER" = "1" ]; then
    # Second gate: the checkpoints exist only now, so this is the first moment
    # the inference configs can be validated in full. Without it a bad infer
    # config is discovered after training has already been thrown away for the
    # night, with nobody watching.
    # Same rule as the training preflight: a failing (arm, tag) is dropped, not
    # fatal. An arm whose training died has no checkpoint, so it ALWAYS fails
    # here -- aborting on that would let one dead card cost the entire inference
    # phase for the seven arms that trained fine.
    INFER_SKIP=""
    if [ "$PREFLIGHT" = "1" ]; then
        echo "Preflight (--check) on every inference config..."
        inf_ok=0
        inf_bad=0
        for arm in $ARMS; do
            for tag in $INFER_TAGS; do
                icfg="$(inf_cfg_for "$arm" "$tag")"
                if [ ! -f "$icfg" ]; then
                    echo "  $arm/$tag  MISSING CONFIG ($icfg)" >&2
                    INFER_SKIP="$INFER_SKIP $arm/$tag"
                    SKIPPED="$SKIPPED@  $arm/$tag (infer)  no config"
                    inf_bad=$((inf_bad + 1))
                    continue
                fi
                if "$PYTHON" AI_CAE4ALL_main.py --config "$icfg" --check > "$LOG_ROOT/${arm}.${tag}${CFG_SUFFIX}.check.log" 2>&1; then
                    inf_ok=$((inf_ok + 1))
                else
                    echo "  $arm/$tag  FAILED -- see $LOG_ROOT/${arm}.${tag}.check.log" >&2
                    INFER_SKIP="$INFER_SKIP $arm/$tag"
                    SKIPPED="$SKIPPED@  $arm/$tag (infer)  preflight failed: $LOG_ROOT/${arm}.${tag}.check.log"
                    inf_bad=$((inf_bad + 1))
                fi
            done
        done
        echo "  $inf_ok runnable, $inf_bad skipped."
        if [ "$inf_ok" = "0" ]; then
            echo "  Nothing to infer -- every config failed. Checkpoints untouched." >&2
        fi
        echo ""
    fi

    n_tags=$(echo "$INFER_TAGS" | wc -w)
    n_arms=$(echo "$ARMS" | wc -w)
    n_skip=$(echo "$INFER_SKIP" | wc -w)
    echo "Inference: $n_arms arms x $n_tags eval sets = $(( n_arms * n_tags - n_skip )) runs ($n_skip skipped)."
    inf_started=$(date +%s)
    inf_pids=(); inf_names=()
    for arm in $ARMS; do
        run_infer_arm "$arm" &
        inf_pids+=("$!"); inf_names+=("$arm")
        sleep 2
    done
    for k in "${!inf_pids[@]}"; do
        if ! wait "${inf_pids[$k]}"; then
            echo "${inf_names[$k]}: at least one eval set failed" >&2
            rc=1
        fi
    done
    inf_ended=$(date +%s)
    echo "Inference finished in $(( (inf_ended - inf_started) / 60 ))m."
    echo "  Histograms : output/meshgraphnets-v/saoi_sweep3/infer/<arm>/<tag>/histogram_compare.png"
    echo ""
else
    echo "INFER=0 -- skipped. The warpage histogram table will be empty."
    echo ""
fi

# ---- Score the grid --------------------------------------------------------
# The training loss measures the POSTERIOR path; generation is what inference
# uses, so the sweep is decided by CRPS + wild rate + rank calibration. Runs
# even when rc != 0 so a partially-failed batch still yields a report for the
# arms that did finish (score_sweep.py skips arms with no checkpoint).
# Scoring reads and writes the DET subtree when DET=1, so a control run
# cannot overwrite the multi-draw report with one-draw numbers.
if [ "$DET" = "1" ]; then SCORE_OUT="output/meshgraphnets-v/saoi_sweep3/det"; else SCORE_OUT="output/meshgraphnets-v/saoi_sweep3"; fi
REPORT="$SCORE_OUT/sweep_results.md"
if [ "$SCORE" = "1" ] && [ "$INFL" = "1" ]; then
    # Inflation run: rank the lam values, in place.
    INFL_DIR="output/meshgraphnets-v/saoi_sweep3/infer/infl"
    echo "================= INFLATION RANKING ================="
    "$PYTHON" configs/campaigns/rank_arms.py "$INFL_DIR" 2>&1 | tee "$LOG_ROOT/rank_arms_infl.log"
    echo "===================================================="
    echo ""
    echo "Pick the lam whose sd_ratio is nearest 1 on s26fe_main, then read that"
    echo "same lam on the other two eval sets in the per-set table above."
    echo "Near 1 on all three = shape was right, only the scale was off: calibrated."
    echo ""
    echo "Saved   : $LOG_ROOT/rank_arms_infl.log"
    echo "Dumps   : $INFL_DIR/<arm>/<eval set>/spread_values.npz"
elif [ "$SCORE" = "1" ]; then
    # SCORE_SAMPLERS empty (the default) skips eval_distribution.py
    # entirely: it is a serial, per-arm, per-sampler GPU job with an
    # hour-long timeout per rung of a five-step OOM ladder, so eight
    # arms can run for GPU-days with seven cards idle. The warpage and
    # per-scene tables are read from the spread dumps and cost seconds.
    if [ "$DET" = "1" ]; then
        # One draw per scene: rank histograms measure nothing here.
        SCORE_SAMPLERS=""
    fi
    if [ -n "$SCORE_SAMPLERS" ]; then
        echo "Scoring the grid (eval_distribution.py per arm: $SCORE_SAMPLERS -- SLOW, serial)..."
    else
        echo "Scoring the grid (warpage + per-scene tables; eval_distribution.py off)..."
        echo "  set SCORE_SAMPLERS=\"prior normal\" to add the rank histograms (GPU-hours)."
    fi
    if "$PYTHON" "$CFG_DIR/score_sweep.py" \
            --arms $ARMS \
            ${DET_FLAG} \
            --split "$SCORE_SPLIT" \
            --k "$SCORE_K" \
            --samplers $SCORE_SAMPLERS \
            --python "$PYTHON" \
            --out-dir "$SCORE_OUT" \
            --run-logs "$LOG_ROOT" \
            > "$LOG_ROOT/score_sweep${CFG_SUFFIX}.log" 2>&1; then
        echo "Scoring complete."
    else
        echo "Scoring FAILED (exit $?) -- see $LOG_ROOT/score_sweep${CFG_SUFFIX}.log" >&2
        rc=1
    fi
    echo ""
    if [ -f "$REPORT" ]; then
        echo "================= RESULTS ================="
        cat "$REPORT"
        echo "==========================================="
        echo ""
        echo "Report   : $REPORT      <-- paste this file to Claude"
        echo "Raw JSON : $SCORE_OUT/sweep_results.json"
    fi
else
    echo "SCORE=0 -- skipped. Run it later with:"
    echo "  $PYTHON $CFG_DIR/score_sweep.py --split $SCORE_SPLIT --k $SCORE_K --run-logs $LOG_ROOT"
fi

echo ""
if [ -n "$SKIPPED" ]; then
    echo ""
    echo "SKIPPED -- THIS SWEEP IS INCOMPLETE. These were dropped, not run:"
    echo "$SKIPPED" | tr '@' '\n'
    echo ""
    echo "Re-run just those once fixed, e.g.:  ARMS=\"3 7\" bash $0"
fi

echo ""
echo "THEN DELETE THE SHARED CACHE (configs set hierarchy_cache_keep True):"
# Derived from the config rather than hardcoded: the cache lives beside the
# dataset file (multiscale_cache.cache_path_for), and a hardcoded copy of that
# path is what drifted out of sync before.
_ds_hint="$(sed -n 's/^dataset_dir[[:space:]]\{1,\}//p' "$(cfg_for "$(echo "$ARMS" | awk '{print $1}')")" 2>/dev/null | head -1)"
_ds_hint="${_ds_hint%%#*}"
_ds_hint="$(echo "$_ds_hint" | sed 's/[[:space:]]*$//')"
_ds_hint="${_ds_hint#../../}"
echo "  rm ${_ds_hint%.h5}.mscache.*.h5"
exit $rc
