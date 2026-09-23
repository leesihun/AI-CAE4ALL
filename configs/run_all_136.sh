#!/usr/bin/env bash
# Dataset/method baseline matrix - machine 136 (8 GPUs).
#
#   bash configs/run_all_136.sh              run it
#   DRY_RUN=1 SKIP_GPU_GATE=1 bash ...      print the plan and exit
#   SKIP_GPU_GATE=1 bash ...                start now, do not wait for the box
#
# The example split and the eight lanes live in run_matrix.sh.
set -u
MACHINE=136
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/campaigns/dataset_matrix/run_matrix.sh"
