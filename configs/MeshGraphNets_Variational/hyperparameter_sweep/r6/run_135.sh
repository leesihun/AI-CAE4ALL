#!/usr/bin/env bash
# B300 round 6 (placement sweep), node 135: see run_node.sh (order of work, overrides) and README.md
# (98 jobs as one queue). GPU1 is left out; each of the other 7 cards runs 3 jobs (21 slots, listed card
# by card three times so the first 7 jobs, the longest, land on 7 different cards). 21 jobs x (2 torch
# threads + 4 DataLoader workers) ~ 126 of the 128 CPUs, hence THREADS_PER_JOB 2.
THREADS_PER_JOB="${THREADS_PER_JOB:-2}" \
GPUS="${GPUS:-0 2 3 4 5 6 7 0 2 3 4 5 6 7 0 2 3 4 5 6 7}" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135 "$@"
