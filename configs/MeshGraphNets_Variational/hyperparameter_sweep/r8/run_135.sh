#!/usr/bin/env bash
# B300 round 8 (large-batch HI-MGN-hybrid sweep), node 135: see run_node.sh (order of work, overrides)
# and README.md (the jobs as one queue). WRITTEN BY make_sweep.py --pack 1: 1 job(s) per card on
# GPUs 0, 2, 3, 4, 5, 6, 7 (GPU1 left out), 7 slots; each job gets 13 torch threads + 4
# DataLoader workers. To change the packing, regenerate with another --pack (running jobs resume).
THREADS_PER_JOB="${THREADS_PER_JOB:-13}" \
GPUS="${GPUS:-0 2 3 4 5 6 7}" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135 "$@"
