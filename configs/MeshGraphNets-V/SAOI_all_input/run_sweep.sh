#!/usr/bin/env bash
# One-click runner for the SAOI 16-arm latent x regularization sweep.
#
#   AXIS 1  vae_latent_dim  {128, 64, 16, 8}
#   AXIS 2  alpha_recon:lambda_mmd  {1000:0.1, 1000:1, 100:1, 10:1}
#
# 16 arms, 2 per GPU across GPUs 0-7. Each arm pins its GPU in its own config
# (gpu_ids), so this script only has to launch them; it does not set
# CUDA_VISIBLE_DEVICES.
#
# THIS IS A MULTI-DAY RUN. Start it detached:
#   nohup bash configs/MeshGraphNets-V/SAOI_all_input/run_sweep.sh > sweep.out 2>&1 &
#   tail -f sweep.out
#   tail -f outputs/saoi_sweep/run_logs/sweep_z8_r1.log     # watch one arm
#
# Multiscale cache: all 16 arms hash to ONE cache file (vae_latent_dim /
# alpha_recon / lambda_mmd are not part of the coarsening signature). An
# exclusive O_EXCL lock in general_modules/multiscale_cache.py lets exactly one
# process build it while the rest poll. Rather than have 15 jobs idle through a
# potentially hours-long build, this script launches ONE arm first and waits for
# the cache to appear before launching the other 15 — and aborts the whole
# batch if that first arm dies, so a config error costs minutes, not days.
#
# The configs set hierarchy_cache_keep True so no finishing arm deletes the
# cache out from under the others. DELETE IT MANUALLY when the sweep is done:
#   rm dataset/saoi/saoi_train_top.mscache.*.h5
#
# Environment overrides:
#   PYTHON        interpreter (default: python)
#   LOG_ROOT      transcript directory (default: outputs/saoi_sweep/run_logs)
#   ARMS          space-separated arm names (default: all 16)
#   PREFLIGHT     1 = --check every arm before launching any (default); 0 = skip
#   WARM_TIMEOUT  seconds to wait for the shared cache (default: 21600 = 6h)
#   SCORE         1 = run score_sweep.py when training ends (default); 0 = skip
#   SCORE_K       draws per geometry for the rank histogram (default: 50)
#   SCORE_SPLIT   split to score (default: test)
#
# Usage:
#   bash configs/MeshGraphNets-V/SAOI_all_input/run_sweep.sh
#   ARMS="sweep_z8_r1 sweep_z8_r4" bash .../run_sweep.sh    # subset
#   PREFLIGHT=0 bash .../run_sweep.sh                       # skip validation

# NOT `set -e`: per-arm failures are collected so one bad arm cannot kill the batch.
set -uo pipefail

PYTHON="${PYTHON:-python}"
PREFLIGHT="${PREFLIGHT:-1}"
WARM_TIMEOUT="${WARM_TIMEOUT:-21600}"
SCORE="${SCORE:-1}"
SCORE_K="${SCORE_K:-50}"
SCORE_SPLIT="${SCORE_SPLIT:-test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CFG_DIR="configs/MeshGraphNets-V/SAOI_all_input"
LOG_ROOT="${LOG_ROOT:-outputs/saoi_sweep/run_logs}"
CACHE_GLOB="dataset/saoi/saoi_train_top.mscache.*.h5"

DEFAULT_ARMS="\
sweep_z128_r1 sweep_z128_r2 sweep_z128_r3 sweep_z128_r4 \
sweep_z64_r1 sweep_z64_r2 sweep_z64_r3 sweep_z64_r4 \
sweep_z16_r1 sweep_z16_r2 sweep_z16_r3 sweep_z16_r4 \
sweep_z8_r1 sweep_z8_r2 sweep_z8_r3 sweep_z8_r4"
ARMS="${ARMS:-$DEFAULT_ARMS}"

mkdir -p "$LOG_ROOT"

cfg_for()  { echo "$CFG_DIR/config_${1}.txt"; }
log_for()  { echo "$LOG_ROOT/${1}.log"; }
cache_ready() { compgen -G "$CACHE_GLOB" > /dev/null 2>&1; }

run_arm() {
    local arm=$1 cfg log
    cfg="$(cfg_for "$arm")"
    log="$(log_for "$arm")"
    if [ ! -f "$cfg" ]; then
        echo "[$arm] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    if "$PYTHON" AI_CAE4ALL_main.py --config "$cfg" > "$log" 2>&1; then
        echo "[$arm] DONE"
        return 0
    fi
    echo "[$arm] FAILED (exit $?) — see $log" >&2
    return 1
}

echo "SAOI latent x regularization sweep"
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
first_arm="$(echo "$ARMS" | awk '{print $1}')"
rest_arms="$(echo "$ARMS" | cut -d' ' -f2-)"

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
echo "Checkpoints : output/meshgraphnets-v/saoi_sweep/<arm>.pth"
echo ""

# ---- Score the grid --------------------------------------------------------
# The training loss measures the POSTERIOR path; generation is what inference
# uses, so the sweep is decided by CRPS + wild rate + rank calibration. Runs
# even when rc != 0 so a partially-failed batch still yields a report for the
# arms that did finish (score_sweep.py skips arms with no checkpoint).
REPORT="outputs/saoi_sweep/sweep_results.md"
if [ "$SCORE" = "1" ]; then
    echo "Scoring the grid (this runs eval_distribution.py per arm, both samplers)..."
    if "$PYTHON" "$CFG_DIR/score_sweep.py" \
            --arms $ARMS \
            --split "$SCORE_SPLIT" \
            --k "$SCORE_K" \
            --python "$PYTHON" \
            --out-dir outputs/saoi_sweep \
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
        echo "Raw JSON : outputs/saoi_sweep/sweep_results.json"
    fi
else
    echo "SCORE=0 — skipped. Run it later with:"
    echo "  $PYTHON $CFG_DIR/score_sweep.py --split $SCORE_SPLIT --k $SCORE_K"
fi

echo ""
echo "THEN DELETE THE SHARED CACHE (configs set hierarchy_cache_keep True):"
echo "  rm $CACHE_GLOB"
exit $rc
