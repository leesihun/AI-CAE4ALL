#!/usr/bin/env bash
# B300 placement sweep, node 135: bash run_135.sh r6. See run_node.sh (order of work, overrides)
# and README.md (the plan). GPU1 is left out; each of the other 7 cards runs 2 jobs (14 slots,
# listed card by card twice so the first 7 jobs, the longest, land on 7 different cards).
GPUS="${GPUS:-0 2 3 4 5 6 7 0 2 3 4 5 6 7}" exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135 "$@"
