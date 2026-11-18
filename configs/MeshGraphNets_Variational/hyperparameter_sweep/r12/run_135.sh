#!/usr/bin/env bash
# B300 round 12 (the prior trained with the encoder and the decoder to the end, its loss sent into the encoder; three changes on top), node 135: see run_node.sh (order of work, overrides)
# and README.md (the jobs as one queue). WRITTEN BY make_sweep.py --pack 1: 1 job(s) per card on
# GPUs 0, 1, 2, 3, 4, 5, 6, 7, 8 slots; each job gets 11 torch threads + 4
# DataLoader workers. To change the packing, regenerate with another --pack (running jobs resume).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREADS_PER_JOB="${THREADS_PER_JOB:-11}" \
GPUS="${GPUS:-0 1 2 3 4 5 6 7}" \
    bash "$HERE/run_node.sh" 135 "$@" || exit $?
# The jobs have started: diagnose_135.sh waits in the background, diagnoses each job as it
# finishes (brief.py prints the result) and draws the per-geometry histograms at the end;
# a second start finds it running and does nothing.
if [ "${TRAIN:-1}${INFER:-1}" != "00" ]; then
    bash "$HERE/diagnose_135.sh"
fi
