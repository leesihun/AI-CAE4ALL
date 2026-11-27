#!/usr/bin/env bash
# B300 round 15 (round 14's best setting, joint_tfk8_dec, on the augmented ex3 dataset: three seeds, a
# 450-epoch run, the per-sample KL beside it; the base again; stochastic_mode noise with the energy score),
# node 135: see run_node.sh (order of work, overrides) and README.md (the jobs as one queue).
# WRITTEN BY make_sweep.py: 14 jobs at once on GPUs 0, 1, 2, 3, 4, 5, 6, 7, the card of job k at position k of GPUS.
# Per card: 0: hybrid_joint_tfk8_dec_e450_s42; 1: hybrid_noise_energy_s42 + noise_global_energy_s42; 2: hybrid_joint_tfk8_s43 + hybrid_joint_tfk8_dec_kl10_s43; 3: hybrid_joint_tfk8_dec_e450_s43; 4: hybrid_base_s42 + hybrid_joint_tfk8_dec_s43; 5: hybrid_base_s43 + hybrid_joint_tfk8_dec_s44; 6: hybrid_base_s44 + hybrid_joint_tfk8_dec_kl10_s42; 7: hybrid_joint_tfk8_dec_s42 + hybrid_joint_tfk8_s42.
# Each job gets 4 torch threads + 4 DataLoader workers.
# MPS is OFF here, always (MPS=0 below is not an override): round 13 lost both seeds of two
# settings with two jobs per card under MPS. To change the layout, edit ORDER in make_sweep.py
# and regenerate (running jobs resume).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPS=0 \
THREADS_PER_JOB="${THREADS_PER_JOB:-4}" \
GPUS="${GPUS:-4 5 6 7 2 1 0 3 4 5 6 7 2 1}" \
    bash "$HERE/run_node.sh" 135 "$@" || exit $?
# The jobs have started: diagnose_135.sh waits in the background, diagnoses each job as it
# finishes (brief.py prints the result) and draws the per-geometry histograms at the end;
# a second start finds it running and does nothing.
if [ "${TRAIN:-1}${INFER:-1}" != "00" ]; then
    bash "$HERE/diagnose_135.sh"
fi
