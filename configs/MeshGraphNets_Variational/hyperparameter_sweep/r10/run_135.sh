#!/usr/bin/env bash
# B300 round 10 (HI-MGN-hybrid coarse prior against over-fitting, 3 seeds), node 135: see run_node.sh (order of work, overrides)
# and README.md (the jobs as one queue). WRITTEN BY make_sweep.py --pack 2: 2 job(s) per card on
# GPUs 0, 1, 2, 3, 4, 5, 6, 7, 16 slots; each job gets 3 torch threads + 4
# DataLoader workers. To change the packing, regenerate with another --pack (running jobs resume).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREADS_PER_JOB="${THREADS_PER_JOB:-3}" \
GPUS="${GPUS:-0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7}" \
    bash "$HERE/run_node.sh" 135 "$@" || exit $?
# The jobs have started: diagnose_135.sh waits in the background and diagnoses each job as it
# finishes (brief.py prints the result); a second start finds it running and does nothing.
if [ "${TRAIN:-1}${INFER:-1}" != "00" ]; then
    bash "$HERE/diagnose_135.sh"
fi
