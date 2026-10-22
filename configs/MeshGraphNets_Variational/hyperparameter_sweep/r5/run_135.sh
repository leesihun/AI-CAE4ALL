#!/usr/bin/env bash
# B300 round 5, node 135: see run_node.sh (order of work, overrides) and README.md (13 jobs as one queue on 7 cards; GPU1 is left out, so the default cards are 0 2 3 4 5 6 7).
GPUS="${GPUS:-0 2 3 4 5 6 7}" exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135
