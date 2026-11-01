#!/usr/bin/env bash
# B300 round 7 (HI-MGN-hybrid sweep), node 135: see run_node.sh (order of work, overrides) and README.md
# (14 jobs as one queue). GPU1 is left out. GPUs 0, 2, 3 are listed once: the three 480-epoch jobs,
# first in jobs.txt, run there alone (no card sharing, no gradient checkpointing). GPUs 4-7 are listed
# twice and run two jobs each: 11 slots. 11 jobs x (6 torch threads + 4 DataLoader workers) = 110 of
# the 128 CPUs, hence THREADS_PER_JOB 6. Keep GPUS in step with make_sweep.py (it checks).
THREADS_PER_JOB="${THREADS_PER_JOB:-6}" \
GPUS="${GPUS:-0 2 3 4 5 6 7 4 5 6 7}" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 135 "$@"
