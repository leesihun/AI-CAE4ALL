#!/usr/bin/env bash
# aarl round 1 (HI-MGN-hybrid ex3 ablations and tuning): see run_node.sh (order of work, overrides),
# README.md (the design) and LOG.md (the journal). WRITTEN BY make_sweep.py --pack 2: 2 job(s) per
# card on GPUs 0, 1 only (aarl is shared: two cards at most), 4 slots; each job gets
# 6 torch threads + 4 DataLoader workers. Another --pack: regenerate (running jobs resume).
THREADS_PER_JOB="${THREADS_PER_JOB:-6}" \
GPUS="${GPUS:-0 1 0 1}" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" aarl "$@"
