#!/usr/bin/env bash
# Shared 8-lane campaign runner for the dataset/method baseline matrix.
#
# Driven by configs/run_all_135.sh and configs/run_all_136.sh, which set
# MACHINE and source this file. Each machine owns half the examples and runs
# the same eight lanes over them; `gpu_ids` in every config names its home lane
# (see LANE_GPU in generate.py).
#
#   lane 0  deeponet + fno        lane 4  meshgraphnets
#   lane 1  point_deeponet        lane 5  himgn
#   lane 2  (no home arms)        lane 6  himgn_v + lsh_vae
#   lane 3  transolver3           lane 7  chi_mgnflow + sdfflow
#
# Behaviour
#   - Waits, polling hourly, until all eight GPUs read 0% utilisation and no
#     compute processes are running, so a job already on the box finishes first.
#   - One worker per physical GPU. A worker drains its own lane in example
#     order, running config_train then config_infer for each arm, and then
#     steals whatever is left in the shared queue.
#   - A stolen arm keeps the `gpu_ids` written in its config; CUDA_VISIBLE_DEVICES
#     is permuted so that logical index lands on the worker's physical card.
#   - A failing arm never stops the worker or the other lanes. A worker that
#     loses its GPU, or fails three arms in a row, exits and leaves its
#     remaining work in the queue for the others.
#   - Re-running resumes: completed stages are skipped via markers under
#     output/dataset_matrix/_campaign/<machine>/done/.
#   - When the workers stop, score_spread.py tabulates every probabilistic arm
#     that produced spread_metrics.json and writes the cross-arm overlay; it can
#     also be run by hand at any time during the campaign.
#
# Environment overrides: PYTHON, SKIP_GPU_GATE=1, GATE_INTERVAL, DRY_RUN=1.

set -u
set -o pipefail

: "${MACHINE:?run_all_<machine>.sh must set MACHINE}"
: "${PYTHON:=python3}"
: "${GATE_INTERVAL:=3600}"
: "${SKIP_GPU_GATE:=0}"
: "${DRY_RUN:=0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MANIFEST="$ROOT/configs/campaigns/dataset_matrix/manifest.json"
STATE="$ROOT/output/dataset_matrix/_campaign/$MACHINE"
QUEUE="$STATE/queue.txt"
LOCK="$STATE/queue.lock"
LOGS="$STATE/logs"
DONE="$STATE/done"
NGPU=8

# Each machine owns a disjoint set of examples. The heavy slots (ex2 at 200k
# nodes x 50 steps, ex3_full, ex4 at 599 steps, ex6 at 400, ex7 at 1000 samples)
# are split across the two boxes rather than stacked on one.
case "$MACHINE" in
  135) EXAMPLES="deterministic/ex1 deterministic/ex2 deterministic/ex3_full deterministic/ex5 deterministic/ex8 deterministic/ex10 probabilistic/ex1" ;;
  136) EXAMPLES="deterministic/ex3_mid deterministic/ex4 deterministic/ex6 deterministic/ex7 deterministic/ex9 probabilistic/ex2 geometry_generation/ex1" ;;
  *) echo "Unknown MACHINE '$MACHINE' (expected 135 or 136)" >&2; exit 2 ;;
esac

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# --------------------------------------------------------------- queue build

build_queue() {
  mkdir -p "$STATE" "$LOGS" "$DONE"
  [ -f "$MANIFEST" ] || { echo "Missing $MANIFEST" >&2; exit 2; }
  if [ -s "$QUEUE" ]; then
    log "Resuming existing queue with $(wc -l < "$QUEUE") arm(s) left."
    return
  fi
  "$PYTHON" - "$MANIFEST" "$EXAMPLES" > "$QUEUE" <<'PY'
import json, sys

manifest, examples = sys.argv[1], set(sys.argv[2].split())
# Mirrors LANE_GPU in generate.py; the runner must agree with the configs.
lane = {'deeponet': 0, 'fno': 0, 'point_deeponet': 1, 'transolver3': 3,
        'meshgraphnets': 4, 'himgn': 5, 'himgn_v': 6, 'lsh_vae': 6,
        'chi_mgnflow': 7, 'sdfflow': 7}
pairs = json.load(open(manifest, encoding='utf-8'))['pairs']
rows = [p for p in pairs if f"{p['category']}/{p['example']}" in examples]
missing = examples - {f"{p['category']}/{p['example']}" for p in pairs}
if missing:
    sys.exit('Examples not present in the manifest: ' + ', '.join(sorted(missing)))
# Lane-major, example order preserved inside a lane.
rows.sort(key=lambda p: lane[p['method']])
for p in rows:
    print('|'.join((str(lane[p['method']]), p['method'], p['category'],
                    p['example'], p['train'], p['infer'])))
PY
  local rc=$?
  [ "$rc" -eq 0 ] || { echo "Queue build failed." >&2; rm -f "$QUEUE"; exit "$rc"; }
  log "Machine $MACHINE queue: $(wc -l < "$QUEUE") arm(s) over $(echo "$EXAMPLES" | wc -w) example(s)."
  awk -F'|' '{c[$1]++} END {for (l=0; l<8; l++) printf "  lane %d: %d arm(s)\n", l, c[l]+0}' "$QUEUE"
}

# flock is the normal path on Linux; the mkdir spin lock keeps the runner
# correct on a box (or a Git Bash shell) that does not ship util-linux.
HAVE_FLOCK=0
command -v flock >/dev/null 2>&1 && HAVE_FLOCK=1

lock_acquire() {
  if [ "$HAVE_FLOCK" -eq 1 ]; then exec 9>"$LOCK"; flock -x 9; return 0; fi
  until mkdir "$LOCK.d" 2>/dev/null; do sleep 0.2; done
}

lock_release() {
  if [ "$HAVE_FLOCK" -eq 1 ]; then flock -u 9; exec 9>&-; return 0; fi
  rmdir "$LOCK.d" 2>/dev/null
}

# Atomically take the first queue line for $1 (a lane number, or ANY).
claim() {
  local want="$1" line
  lock_acquire
  line=$(awk -F'|' -v w="$want" '(w == "ANY" || $1 == w) {print; exit}' "$QUEUE" 2>/dev/null)
  if [ -n "$line" ]; then
    grep -vxF -- "$line" "$QUEUE" > "$QUEUE.claim.tmp" 2>/dev/null
    mv "$QUEUE.claim.tmp" "$QUEUE"
  fi
  lock_release
  printf '%s' "$line"
}

# ------------------------------------------------------------------ gpu gate

gpu_alive() { nvidia-smi -i "$1" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; }

wait_for_idle_gpus() {
  if [ "$SKIP_GPU_GATE" = "1" ]; then
    log "SKIP_GPU_GATE=1 - starting without waiting for the box to drain."
    return 0
  fi
  while true; do
    local utils count busy apps
    utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$utils" ]; then
      log "nvidia-smi returned nothing; retrying in ${GATE_INTERVAL}s."
      sleep "$GATE_INTERVAL"; continue
    fi
    count=$(printf '%s\n' "$utils" | wc -l)
    busy=$(printf '%s\n' "$utils" | awk '$1 > 0 {n++} END {print n+0}')
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
    if [ "$count" -eq "$NGPU" ] && [ "$busy" -eq 0 ] && [ "$apps" -eq 0 ]; then
      log "All $NGPU GPUs idle and no compute processes - starting."
      return 0
    fi
    log "Box still busy (${count} GPU(s) visible, ${busy} above 0%, ${apps} compute process(es)); re-checking in ${GATE_INTERVAL}s."
    sleep "$GATE_INTERVAL"
  done
}

# -------------------------------------------------------------------- worker

# Make the config's own gpu_ids resolve to this worker's physical card by
# swapping those two positions in the visible-device list. When the arm runs on
# its home lane the list is the identity and nothing is remapped.
visible_for() {
  local home="$1" phys="$2" i out=()
  for ((i = 0; i < NGPU; i++)); do out+=("$i"); done
  out[home]=$phys
  out[phys]=$home
  local IFS=,
  printf '%s' "${out[*]}"
}

run_stage() {  # home_lane phys_gpu marker config
  local home="$1" phys="$2" marker="$3" config="$4"
  if [ -e "$DONE/$marker" ]; then
    log "  skip $marker (already done)"
    return 0
  fi
  local vis; vis=$(visible_for "$home" "$phys")
  if [ "$DRY_RUN" = "1" ]; then
    log "  DRY_RUN CUDA_VISIBLE_DEVICES=$vis $PYTHON AI_CAE4ALL_main.py --config $config"
    return 0
  fi
  log "  run $marker (CUDA_VISIBLE_DEVICES=$vis)"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES="$vis" "$PYTHON" AI_CAE4ALL_main.py --config "$config" )
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    : > "$DONE/$marker"
  else
    log "  FAILED $marker (exit $rc)"
  fi
  return "$rc"
}

worker() {
  local phys="$1"
  local worklog="$LOGS/gpu${phys}.log"
  exec >>"$worklog" 2>&1
  log "=== worker on physical GPU $phys started (machine $MACHINE) ==="
  local strikes=0 own_drained=0 line
  while true; do
    if ! gpu_alive "$phys"; then
      log "GPU $phys no longer reports to nvidia-smi - worker exiting, its work stays queued."
      return 1
    fi
    if [ "$own_drained" -eq 0 ]; then
      line=$(claim "$phys")
      if [ -z "$line" ]; then
        own_drained=1
        log "Lane $phys drained; switching to stealing from the shared queue."
        continue
      fi
    else
      line=$(claim ANY)
      [ -n "$line" ] || { log "Queue empty - worker on GPU $phys done."; return 0; }
    fi
    IFS='|' read -r home method category example train infer <<<"$line"
    local tag="$category/$example/$method"
    if [ "$home" != "$phys" ]; then
      log "STOLEN $tag (home lane $home) onto physical GPU $phys"
    else
      log "$tag"
    fi
    local key="${category}__${example}__${method}"
    if run_stage "$home" "$phys" "${key}__train" "$train"; then
      if run_stage "$home" "$phys" "${key}__infer" "$infer"; then
        strikes=0
      else
        strikes=$((strikes + 1))
      fi
    else
      log "  skipping inference for $tag - training did not produce a checkpoint."
      strikes=$((strikes + 1))
    fi
    if [ "$strikes" -ge 3 ]; then
      log "Three consecutive failures on GPU $phys - worker exiting so the other lanes take over."
      return 1
    fi
  done
}

# ---------------------------------------------------------------------- main

PIDS=()
cleanup() {
  log "Interrupted - stopping workers."
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  exit 130
}
trap cleanup INT TERM

# The probabilistic arms score themselves at the end of inference (one
# spread_metrics.json per arm); this collects them into one table and one
# overlay per example. Informational: its exit code never decides the
# campaign's, since each machine owns only one probabilistic example and the
# other machine's arm is legitimately absent here.
score_report() {
  local script="$ROOT/configs/campaigns/dataset_matrix/score_spread.py"
  [ -f "$script" ] || return 0
  log "Probabilistic spread scores:"
  ( cd "$ROOT" && "$PYTHON" "$script"       --csv "output/dataset_matrix/_campaign/$MACHINE/spread_scores.csv" ) || true
}

main() {
  build_queue
  [ "$HAVE_FLOCK" -eq 1 ] || log "NOTE: flock not found; using the mkdir fallback lock for the shared queue."
  wait_for_idle_gpus
  log "Launching $NGPU workers; per-GPU logs under $LOGS/"
  local g
  for ((g = 0; g < NGPU; g++)); do
    worker "$g" &
    PIDS+=("$!")
  done
  wait
  score_report
  local left=0
  [ -f "$QUEUE" ] && left=$(grep -c . "$QUEUE")
  if [ "$left" -gt 0 ]; then
    log "All workers stopped with $left arm(s) still queued:"
    awk -F'|' '{printf "  lane %s  %s/%s/%s\n", $1, $3, $4, $2}' "$QUEUE"
    log "Re-run this script to pick them up (completed stages are skipped)."
    exit 1
  fi
  log "Machine $MACHINE: all arms completed."
}

main "$@"
