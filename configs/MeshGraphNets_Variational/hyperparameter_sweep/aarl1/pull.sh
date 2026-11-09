#!/usr/bin/env bash
# Pull this round's results from aarl into the local output/aarl1, on the LOCAL machine:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/pull.sh          # once
#   LOOP=1800 bash .../aarl1/pull.sh                                                  # every 30 min, until killed
#
# Copies every file under aarl's output/aarl1 changed since the last pull (by mtime, not size: a log
# rewritten to the same size is still copied), except the hierarchy cache and the resume states.
# Checkpoints (model.pth) come along. The time of the last pull is kept in output/aarl1/.last_pull.
set -u
HOST="${HOST:-root@100.91.92.59}"
REMOTE="${REMOTE:-/data/Lee/mgnv_hx3/AI-CAE4ALL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.." || exit 1
mkdir -p output/aarl1
STAMP=output/aarl1/.last_pull

pull_once() {
    local since now
    since="$(cat "$STAMP" 2>/dev/null || echo '1970-01-01 00:00:00')"
    now="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" 'date "+%F %T"')" || { echo "pull: ssh failed"; return 1; }
    # shellcheck disable=SC2029  # $since expands locally on purpose
    ssh -o BatchMode=yes "$HOST" "cd '$REMOTE' && [ -d output/aarl1 ] && \
        find output/aarl1 -path output/aarl1/mscache -prune -o -type f ! -name '*.resume' \
             ! -name '*.resume.tmp' -newermt '$since' -print0 | tar --null -T - -czf -" \
        | tar -xzf - || { echo "pull: transfer failed (kept the old stamp)"; return 1; }
    # 5 s back, so a file written during this pull is copied again next time
    date -d "$now 5 seconds ago" '+%F %T' > "$STAMP" 2>/dev/null || echo "$now" > "$STAMP"
    echo "pull: $(date '+%F %T') copied changes since $since"
}

if [ -n "${LOOP:-}" ]; then
    while true; do pull_once; sleep "$LOOP"; done
else
    pull_once
fi
