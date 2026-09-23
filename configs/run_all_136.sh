#!/usr/bin/env bash
# dataset_matrix campaign, machine 136: every train+infer arm of this machine's
# half of configs/campaigns/dataset_matrix/manifest.json, on 8 GPUs.
#
#   bash configs/run_all_136.sh                 (run inside tmux, or under nohup)
#   DRY_RUN=1 bash configs/run_all_136.sh       print the plan, write nothing
#   CHECK=1   bash configs/run_all_136.sh       launcher --check on every stage still to run
#   SKIP_GPU_GATE=1 ...                         start without waiting for an idle box
#
# The launcher needs Python >= 3.10. Point each method at its venv in
# ai_cae4all.local.toml, or run from one venv that has everything; PYTHON=...
# picks the interpreter. See configs/campaigns/dataset_matrix/README.md.
here=$(cd "$(dirname "$0")" && pwd)
PY="${PYTHON:-$(command -v python3 || command -v python)}"
[ -n "$PY" ] || { echo "no python3 on PATH; set PYTHON=/path/to/python" >&2; exit 2; }
exec "$PY" "$here/campaigns/dataset_matrix/run_matrix.py" --machine 136 "$@"
