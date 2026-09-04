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
#   PYTHON        interpreter (default: python)
#   LOG_ROOT      transcript directory (default: output/meshgraphnets-v/saoi_sweep3/run_logs)
#   ARMS          space-separated arm names (default: all 8)
#   PREFLIGHT     1 = --check every arm before launching any (default); 0 = skip
#   TRAIN         1 = train (default). 0 = SKIP training and go straight to
#                 inference + scoring on checkpoints that already exist.
#   STAGGER       seconds between arm launches (default: 10)
#   INFER         1 = run each arm's inference configs after training (default)
#   INFER_TAGS    eval sets to infer (default: s26fe_main s26fe_sec sm_l345u)
#   SCORE         1 = run score_sweep.py when training ends (default); 0 = skip
#   SCORE_K       draws per geometry for the rank histogram (default: 50)
#   SCORE_SPLIT   split to score (default: test)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Absolute, derived from the script itself: the sweep folder can be renamed
# or copied for a wave 4 without editing anything here.
CFG_DIR="$SCRIPT_DIR"
LOG_ROOT="${LOG_ROOT:-output/meshgraphnets-v/saoi_sweep3/run_logs}"

# Must match gen_sweep_configs.arms() exactly -- it prints this line, so if the
# generator changes, re-paste its ARMS= output here rather than hand-editing.
DEFAULT_ARMS="1 2 3 4 5 6 7 8"
ARMS="${ARMS:-$DEFAULT_ARMS}"
STAGGER="${STAGGER:-10}"   # seconds between arm launches

mkdir -p "$LOG_ROOT"

cfg_for()  { echo "$CFG_DIR/config_train_${1}.txt"; }
inf_cfg_for() { echo "$CFG_DIR/config_infer_${1}_${2}.txt"; }
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

echo "SAOI wave 3 -- 2^(4-1) resolution IV: zcond x rate coupling x capacity x reg"
echo "  REPO_ROOT = $REPO_ROOT"
echo "  PYTHON    = $PYTHON"
echo "  LOG_ROOT  = $LOG_ROOT"
echo "  ARMS      = $(echo "$ARMS" | wc -w) arms"
echo ""

# ---- Preflight every arm before committing GPU-days ------------------------
if [ "$PREFLIGHT" = "1" ]; then
    echo "Preflight (--check) on every arm..."
    pf_bad=0
    for arm in $ARMS; do
        cfg="$(cfg_for "$arm")"
        if [ ! -f "$cfg" ]; then
            echo "  $arm  MISSING CONFIG ($cfg)" >&2; pf_bad=1; continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  $arm  ok"
        else
            echo "  $arm  FAILED -- see $LOG_ROOT/${arm}.check.log" >&2; pf_bad=1
        fi
    done

    # Valid train configs say NOTHING about whether the histogram will have
    # ground truth. rollout.py skips it silently when `eval_dataset` is absent,
    # and draws against the wrong column when it points at the rollout's own
    # input instead of the `_compare_` file. Both have happened, neither raises,
    # and the sweep still reports a clean run. Catch it here -- before the
    # GPU-days, not on Monday morning.
    echo ""
    echo "Preflight (inference + histogram inputs)..."
    if ! "$PYTHON" "$CFG_DIR/check_eval_inputs.py"; then
        pf_bad=1
    fi

    if [ "$pf_bad" != "0" ]; then
        echo "" >&2
        echo "Preflight failed. Nothing launched. Fix the configs and re-run," >&2
        echo "or set PREFLIGHT=0 to launch anyway." >&2
        echo "If every arm reports MISSING CONFIG, the generated configs are" >&2
        echo "stale or absent -- regenerate them first:" >&2
        echo "  python configs/MeshGraphNets_Variational/SAOI_sweep3/gen_sweep_configs.py" >&2
        exit 2
    fi
    echo "All arms validated."
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
        log="$LOG_ROOT/${arm}.infer_${tag}.log"
        if [ ! -f "$cfg" ]; then
            echo "[$arm/$tag] SKIP: no config ($cfg)" >&2
            continue
        fi
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
    if [ "$PREFLIGHT" = "1" ]; then
        echo "Preflight (--check) on every inference config..."
        inf_bad=0
        for arm in $ARMS; do
            for tag in $INFER_TAGS; do
                icfg="$(inf_cfg_for "$arm" "$tag")"
                if [ ! -f "$icfg" ]; then
                    echo "  $arm/$tag  MISSING CONFIG ($icfg)" >&2; inf_bad=1; continue
                fi
                if ! "$PYTHON" AI_CAE4ALL_main.py --config "$icfg" --check                         > "$LOG_ROOT/${arm}.${tag}.check.log" 2>&1; then
                    echo "  $arm/$tag  FAILED -- see $LOG_ROOT/${arm}.${tag}.check.log" >&2
                    inf_bad=1
                fi
            done
        done
        if [ "$inf_bad" != "0" ]; then
            echo "" >&2
            echo "Inference preflight failed. No inference launched; the trained" >&2
            echo "checkpoints are untouched. Fix the configs and re-run with" >&2
            echo "  TRAIN=0 INFER=1 SCORE=1 bash $0" >&2
            exit 2
        fi
        echo "All inference configs validated."
        echo ""
    fi

    n_tags=$(echo "$INFER_TAGS" | wc -w)
    n_arms=$(echo "$ARMS" | wc -w)
    echo "Inference: $n_arms arms x $n_tags eval sets = $(( n_arms * n_tags )) runs."
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
REPORT="output/meshgraphnets-v/saoi_sweep3/sweep_results.md"
if [ "$SCORE" = "1" ]; then
    echo "Scoring the grid (this runs eval_distribution.py per arm, both samplers)..."
    if "$PYTHON" "$CFG_DIR/score_sweep.py" \
            --arms $ARMS \
            --split "$SCORE_SPLIT" \
            --k "$SCORE_K" \
            --python "$PYTHON" \
            --out-dir output/meshgraphnets-v/saoi_sweep3 \
            --run-logs "$LOG_ROOT" \
            > "$LOG_ROOT/score_sweep.log" 2>&1; then
        echo "Scoring complete."
    else
        echo "Scoring FAILED (exit $?) -- see $LOG_ROOT/score_sweep.log" >&2
        rc=1
    fi
    echo ""
    if [ -f "$REPORT" ]; then
        echo "================= RESULTS ================="
        cat "$REPORT"
        echo "==========================================="
        echo ""
        echo "Report   : $REPORT      <-- paste this file to Claude"
        echo "Raw JSON : output/meshgraphnets-v/saoi_sweep3/sweep_results.json"
    fi
else
    echo "SCORE=0 -- skipped. Run it later with:"
    echo "  $PYTHON $CFG_DIR/score_sweep.py --split $SCORE_SPLIT --k $SCORE_K --run-logs $LOG_ROOT"
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
