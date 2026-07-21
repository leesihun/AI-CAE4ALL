#!/usr/bin/env bash
# TRAINING runner for the ex2 Transolver capacity sweep (OFAT star, 8 cells).
#
# All 8 cells run CONCURRENTLY in a single wave: each is a 2-GPU DDP job whose
# gpu_ids are baked into its config, so 4 cells fill one 8-GPU B300 server.
# Cells 1-4 are assigned to server A, cells 5-8 to server B. Run once per server,
# on that server:
#
#   Server A:  SERVER=A bash run_train_all.sh
#   Server B:  SERVER=B bash run_train_all.sh
#
# Both can run simultaneously -- they are disjoint machines and there is only
# one wave, so there is nothing to sequence and nothing to wait between.
#
# Total time is set by the single slowest cell (cell 5, L10 C512 M128), since
# every cell has its own dedicated GPU pair.
#
# schedule.txt is written by gen_sweep_configs.py and is the single source of
# truth this script reads. Regenerate both together; never hand-edit one.
#
# Environment overrides:
#   SERVER     = A or B (REQUIRED unless CELLS is given)
#   CELLS      = explicit space-separated cell indices, bypassing SERVER. Only
#                use this if you have checked schedule.txt for GPU collisions
#                yourself -- the server split guarantees none.
#   PYTHON     = python interpreter (default: python)
#   LOG_ROOT   = transcript dir (default: <repo>/output/transolver/ex2_sweep/run_logs)
#   DRY_RUN    = 1 to print what would launch and exit
#
# Transcripts go to ${LOG_ROOT}/train_{n}.log (redirected, not tee'd -- 4
# concurrent DDP streams would interleave; use `tail -f` to watch one cell).
# The per-epoch train/valid log that summarize_sweep.py parses is written
# separately by the trainer to <repo>/output/transolver/ex2_sweep/train{n}_<tag>.log

set -uo pipefail

PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TRANSOLVER_DIR="$REPO_ROOT/Transolver"
SCHEDULE="$SCRIPT_DIR/schedule.txt"

LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/output/transolver/ex2_sweep/run_logs}"
mkdir -p "$LOG_ROOT"

if [ ! -f "$SCHEDULE" ]; then
    echo "ERROR: $SCHEDULE not found -- run 'python gen_sweep_configs.py' first." >&2
    exit 2
fi

# ---- resolve which cells to run ------------------------------------------
if [ -n "${CELLS:-}" ]; then
    TARGETS="$CELLS"
    LABEL="explicit CELLS=\"$CELLS\""
else
    if [ -z "${SERVER:-}" ]; then
        echo "ERROR: set SERVER=A or SERVER=B (or CELLS=\"...\")." >&2
        echo "Schedule:" >&2
        column -t "$SCHEDULE" >&2
        exit 2
    fi
    TARGETS="$(awk -v s="$SERVER" '!/^#/ && $1 == s { printf "%s ", $3 }' "$SCHEDULE")"
    LABEL="server $SERVER"
    if [ -z "$(echo "$TARGETS" | tr -d '[:space:]')" ]; then
        echo "ERROR: no cells scheduled for server '$SERVER' (expected A or B)." >&2
        echo "Schedule:" >&2
        column -t "$SCHEDULE" >&2
        exit 2
    fi
fi

echo "ex2 Transolver capacity sweep (OFAT star) -- training runner"
echo "  target   = $LABEL"
echo "  cells    = $TARGETS"
echo "  PYTHON   = $PYTHON"
echo "  LOG_ROOT = $LOG_ROOT"
echo ""
printf "%-6s %-26s %s\n" "cell" "config" "gpus"
for i in $TARGETS; do
    printf "%-6s %-26s %s\n" "$i" \
        "$(awk -v n="$i" '!/^#/ && $3 == n { print $4 }' "$SCHEDULE")" \
        "$(awk -v n="$i" '!/^#/ && $3 == n { print $2 }' "$SCHEDULE")"
done
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing launched."
    exit 0
fi

train_one() {
    local idx=$1
    local cfg="$SCRIPT_DIR/config_train${idx}.txt"
    local log="$LOG_ROOT/train_${idx}.log"
    if [ ! -f "$cfg" ]; then
        echo "[train$idx] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    echo "[train$idx] START  cfg=$cfg  -> $log"
    if (cd "$TRANSOLVER_DIR" && "$PYTHON" Transolver_main.py --config "$cfg") > "$log" 2>&1; then
        echo "[train$idx] DONE"
        return 0
    else
        echo "[train$idx] FAILED (exit $?) -- see $log" >&2
        return 1
    fi
}

started=$(date +%s)
rc=0
pids=()
idxs=()
for i in $TARGETS; do
    train_one "$i" &
    pids+=("$!")
    idxs+=("$i")
    echo "  launched train$i (pid $!)"
done
for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
        echo "train${idxs[$k]} exited non-zero" >&2
        rc=1
    fi
done

ended=$(date +%s)
echo ""
echo "$LABEL finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:    $LOG_ROOT/train_<n>.log"
echo "Per-epoch logs: $REPO_ROOT/output/transolver/ex2_sweep/train<n>_<tag>.log"
echo "Checkpoints:    $REPO_ROOT/output/transolver/ex2_sweep/transolver_train<n>_<tag>.pth"
echo "Analysis:       python summarize_sweep.py   (per-axis effects vs the anchor)"
exit $rc
