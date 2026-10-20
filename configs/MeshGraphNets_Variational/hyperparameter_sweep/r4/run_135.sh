#!/usr/bin/env bash
# B300 round 4, node 135: see run_node.sh (order of work, overrides) and README.md (18 jobs in a priority queue; GPU1 is left out, so the default cards are 0 2 3 4 5 6 7).
GPUS="${GPUS:-0 2 3 4 5 6 7}" exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135
