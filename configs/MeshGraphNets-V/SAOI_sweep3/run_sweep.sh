#!/usr/bin/env bash
# One-click runner for the SAOI WAVE 3 sweep: a 2^4 full factorial, 16 arms.
#
# A 2^(5-1) RESOLUTION-V HALF FRACTION, not a full factorial: the fifth factor
# is E = A xor B xor C xor D (defining relation I = ABCDE). All 5 main effects
# and all 10 two-factor interactions are clean; only 3-factor and higher alias.
#
#   A  z_conditioning         cc  concat (legacy fuser) | ad  adaln (AdaLN-Zero)
#   B  prior_grad_to_encoder  g0  detached (no CVAE rate term) | g1  end-to-end
#   C  vae_latent_dim         z16 | z64
#   D  capacity               c0  Latent_dim 128 / mp 4,6,8,6,4  |
#                             c1  Latent_dim 192 / mp 6,8,12,8,6 (+VAE/prior depth)
#   E  regularizer scale      r001 (lambda_mmd 1, prior_nll_weight 1) |
#                             r100 (lambda_mmd 100, prior_nll_weight 100)
#
# Arm names encode the cell: <cc|ad>_<g0|g1>_<z16|z64>_<c0|c1>_<r001|r100>, and
# the files are config_train_<arm>.txt / config_infer_<arm>_<tag>.txt.
# Regenerate the configs with gen_sweep_configs.py; do not hand-edit them.
#
# 16 arms, TWO per GPU across GPUs 0-7, paired by complementing the four FREE
# factors, so every GPU carries one c0 + one c1 and VRAM stays balanced. Each arm
# pins its GPU in its own config (gpu_ids), so this script only launches them;
# it does not set CUDA_VISIBLE_DEVICES.
#
# WATCH THE `VRAM peak=` LINE OF A c1 ARM. Two arms share each GPU at the full
# production Batch_size 16, so ONE CARD MUST HOLD BOTH, and c1 is ~1.5x the width
# with 40 processor blocks instead of 28. If the pair does not fit, either set
# Batch_size 8 in gen_sweep_configs.py and regenerate (the design survives, MMD's
# sample count halves), or shrink the c1 level. Splitting into two waves of 8
# does NOT work by name prefix -- the first 8 arms are all `cc`.
#
# ALSO WATCH THE FIRST EPOCHS' tqdm postfix: `mmd` AND `fm_p` against `total`.
# alpha_recon is 1000 while both regularizers sit at ~1 in r001, so each is
# ~0.1% of the objective there -- that half of the grid is the "regularizers
# effectively off" control, and the g0/g1 contrast can only show force in the
# r100 half. If BOTH r001 and r100 look negligible the axis has to move UP
# (1000/10000), not sideways.
#
# THIS IS A MULTI-DAY RUN. Start it detached:
#   nohup bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh > sweep.out 2>&1 &
#   tail -f sweep.out
#   tail -f outputs/saoi_sweep3/run_logs/ad_g1_z16_c1_r100.log   # watch one arm
#
# Multiscale cache: all 16 arms hash to ONE cache file (none of the swept keys
# are part of the coarsening signature). An
# exclusive O_EXCL lock in general_modules/multiscale_cache.py lets exactly one
# process build it while the rest poll. Rather than have 15 jobs idle through a
# potentially hours-long build, this script launches ONE arm first and waits for
# the cache to appear before launching the other 15 — and aborts the whole
# batch if that first arm dies, so a config error costs minutes, not days.
#
# DELETE ANY LEFTOVER CACHE BEFORE STARTING: cache_ready() only globs the file
# name, so a stale cache from a previous run makes this script skip the warm-up
# and launch all 16 straight into a cache MISS (the signature pins the source
# HDF5's mtime, which write_preprocessing_to_hdf5 bumps every run).
#
# The configs set hierarchy_cache_keep True so no finishing arm deletes the
# cache out from under the others. DELETE IT MANUALLY when the sweep is done:
#   rm dataset/saoi/saoi_train_bot.mscache.*.h5
#
# Environment overrides:
#   PYTHON        interpreter (default: python)
#   LOG_ROOT      transcript directory (default: outputs/saoi_sweep3/run_logs)
#   ARMS          space-separated arm names (default: all 16)
#   PREFLIGHT     1 = --check every arm before launching any (default); 0 = skip
#   WARM_TIMEOUT  seconds to wait for the shared cache (default: 21600 = 6h)
#   INFER         1 = run each arm's inference configs after training (default)
#   INFER_TAGS    eval sets to infer (default: s26fe_main s26fe_sec sm_l345u)
#   SCORE         1 = run score_sweep.py when training ends (default); 0 = skip
#   SCORE_K       draws per geometry for the rank histogram (default: 50)
#   SCORE_SPLIT   split to score (default: test)
#
# Usage:
#   bash configs/MeshGraphNets-V/SAOI_sweep3/run_sweep.sh
#   ARMS="ad_g1_z16_c1_r100 cc_g0_z16_c0_r001" bash .../run_sweep.sh  # subset
#   PREFLIGHT=0 bash .../run_sweep.sh                       # skip validation

# NOT `set -e`: per-arm failures are collected so one bad arm cannot kill the batch.
set -uo pipefail

PYTHON="${PYTHON:-python}"
PREFLIGHT="${PREFLIGHT:-1}"
WARM_TIMEOUT="${WARM_TIMEOUT:-21600}"
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
LOG_ROOT="${LOG_ROOT:-outputs/saoi_sweep3/run_logs}"
CACHE_GLOB="dataset/saoi/saoi_train_bot.mscache.*.h5"

# Kept in the generator's emission order (bit order A P Z M); gen_sweep_configs.py
# prints this exact line so the two can never drift.
DEFAULT_ARMS="\
cc_g0_z16_c0_r001 cc_g0_z16_c1_r100 \
cc_g0_z64_c0_r100 cc_g0_z64_c1_r001 \
cc_g1_z16_c0_r100 cc_g1_z16_c1_r001 \
cc_g1_z64_c0_r001 cc_g1_z64_c1_r100 \
ad_g0_z16_c0_r100 ad_g0_z16_c1_r001 \
ad_g0_z64_c0_r001 ad_g0_z64_c1_r100 \
ad_g1_z16_c0_r001 ad_g1_z16_c1_r100 \
ad_g1_z64_c0_r100 ad_g1_z64_c1_r001"
ARMS="${ARMS:-$DEFAULT_ARMS}"

mkdir -p "$LOG_ROOT"

cfg_for()  { echo "$CFG_DIR/config_train_${1}.txt"; }
inf_cfg_for() { echo "$CFG_DIR/config_infer_${1}_${2}.txt"; }
log_for()  { echo "$LOG_ROOT/${1}.log"; }
cache_ready() { compgen -G "$CACHE_GLOB" > /dev/null 2>&1; }

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
    echo "[$arm] FAILED (exit $rc) — see $log" >&2
    return 1
}

echo "SAOI wave 3 -- 2^(5-1) res-V: zcond x rate coupling x latent x capacity x reg"
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
            echo "  $arm  MISSING CONFIG" >&2; pf_bad=1; continue
        fi
        if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" --check > "$LOG_ROOT/${arm}.check.log" 2>&1; then
            echo "  $arm  ok"
        else
            echo "  $arm  FAILED — see $LOG_ROOT/${arm}.check.log" >&2; pf_bad=1
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
pids=(); names=()

# ---- Warm the shared multiscale cache with a single arm --------------------
# Word-split into an array: `cut -d' ' -f2-` echoes the WHOLE line back when it
# finds no delimiter, so a single-arm ARMS would have launched that one arm
# twice -- two jobs writing the same checkpoint and log.
read -r -a arm_list <<< "$ARMS"
first_arm="${arm_list[0]}"
rest_arms="${arm_list[*]:1}"

if cache_ready; then
    echo "Multiscale cache already present — launching all arms at once."
    rest_arms="$ARMS"
else
    echo "Cold cache. Launching $first_arm alone to build it (timeout ${WARM_TIMEOUT}s)..."
    run_arm "$first_arm" & warm_pid=$!
    pids+=("$warm_pid"); names+=("$first_arm")

    deadline=$(( $(date +%s) + WARM_TIMEOUT ))
    while :; do
        if cache_ready; then
            echo "Cache is ready after $(( $(date +%s) - started ))s."
            break
        fi
        if ! kill -0 "$warm_pid" 2>/dev/null; then
            echo "" >&2
            echo "$first_arm exited before the cache appeared — aborting the batch." >&2
            echo "See $(log_for "$first_arm")" >&2
            exit 3
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "" >&2
            echo "Timed out waiting ${WARM_TIMEOUT}s for the cache. $first_arm is still" >&2
            echo "running (pid $warm_pid); raise WARM_TIMEOUT or launch the rest by hand." >&2
            exit 4
        fi
        sleep 30
    done
fi

# ---- Launch the remaining arms ---------------------------------------------
for arm in $rest_arms; do
    run_arm "$arm" &
    pids+=("$!"); names+=("$arm")
    echo "  launched $arm (pid $!)"
    sleep 3
done

echo ""
echo "$(echo "${names[*]}" | wc -w) arms running. Waiting..."

rc=0
for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
        echo "${names[$k]} exited non-zero" >&2
        rc=1
    fi
done

ended=$(date +%s)
echo ""
echo "Training finished in $(( (ended - started) / 3600 ))h $(( ((ended - started) % 3600) / 60 ))m (rc=$rc)."
echo ""
echo "Transcripts : $LOG_ROOT/<arm>.log"
echo "Checkpoints : output/meshgraphnets-v/saoi_sweep3/<arm>.pth"
echo ""

# ---- Inference: every arm against every held-out eval set -------------------
# Each arm's eval sets run SEQUENTIALLY on the GPU it trained on, and the arms
# run concurrently, so the same two-per-card packing applies as in training.
# The generated configs set save_rollouts False: no trajectory HDF5s are written
# (scene x draws would be tens of thousands of files across the grid). What each
# run leaves behind is histogram_compare.png and spread_values.npz -- the GT vs
# generated z_disp spread (max - min per realization) that score_sweep.py then
# tabulates and overlays for all 16 arms on one axis.
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
REPORT="outputs/saoi_sweep3/sweep_results.md"
if [ "$SCORE" = "1" ]; then
    echo "Scoring the grid (this runs eval_distribution.py per arm, both samplers)..."
    if "$PYTHON" "$CFG_DIR/score_sweep.py" \
            --arms $ARMS \
            --split "$SCORE_SPLIT" \
            --k "$SCORE_K" \
            --python "$PYTHON" \
            --out-dir outputs/saoi_sweep3 \
            --run-logs "$LOG_ROOT" \
            > "$LOG_ROOT/score_sweep.log" 2>&1; then
        echo "Scoring complete."
    else
        echo "Scoring FAILED (exit $?) — see $LOG_ROOT/score_sweep.log" >&2
        rc=1
    fi
    echo ""
    if [ -f "$REPORT" ]; then
        echo "================= RESULTS ================="
        cat "$REPORT"
        echo "==========================================="
        echo ""
        echo "Report   : $REPORT      <-- paste this file to Claude"
        echo "Raw JSON : outputs/saoi_sweep3/sweep_results.json"
    fi
else
    echo "SCORE=0 — skipped. Run it later with:"
    echo "  $PYTHON $CFG_DIR/score_sweep.py --split $SCORE_SPLIT --k $SCORE_K --run-logs $LOG_ROOT"
fi

echo ""
echo "THEN DELETE THE SHARED CACHE (configs set hierarchy_cache_keep True):"
echo "  rm $CACHE_GLOB"
exit $rc
