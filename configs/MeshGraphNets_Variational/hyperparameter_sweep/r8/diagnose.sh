#!/usr/bin/env bash
# Posterior vs. prior diagnosis of the finished hybrid checkpoints: where does
# the w2 above the floor (27) come from -- the encoder/decoder, or the prior?
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/diagnose.sh
#   GPU=1 bash .../r8/diagnose.sh                       # another card (default 0)
#   bash .../r8/diagnose.sh path/to/config_infer_X.txt  # other checkpoints
#
# Default targets: r6 hybrid_e240 (69), r7 hybrid_e480_bs16 (68) and
# hybrid_e240_bs8 (74). Each runs methods/MeshGraphNets_Variational/misc/
# posterior_vs_prior_groups.py on its INFERENCE config: per geometry it decodes
# the 256 FE draws three ways (posterior sample, posterior mean, prior sample)
# and scores each against the held-out half like plots.py. Forward passes only,
# no training; a few minutes per checkpoint. It writes
# output/b300_r*/<job>/diag/posterior_vs_prior_groups_<job>.{json,png}.
#
# Type back every block between ==== TYPE THIS BACK ==== and ==== END ====
# (all of them are repeated at the end).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP="configs/MeshGraphNets_Variational/hyperparameter_sweep"
die() { echo "diagnose: $*" >&2; exit 2; }

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
if [ $# -gt 0 ]; then
    CFGS=("$@")
else
    CFGS=("$SWEEP/r6/config_infer_hybrid_e240.txt"
          "$SWEEP/r7/config_infer_hybrid_e480_bs16.txt"
          "$SWEEP/r7/config_infer_hybrid_e240_bs8.txt")
fi

MGNV_PYTHON="${MGNV_PYTHON:-}"
if [ -z "$MGNV_PYTHON" ]; then
    MGNV_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets_variational'))
")" || die "could not resolve the MGN-V interpreter (ai_cae4all.local.toml); set MGNV_PYTHON."
fi
[ -x "$MGNV_PYTHON" ] || command -v "$MGNV_PYTHON" >/dev/null 2>&1 \
    || die "MGN-V interpreter '$MGNV_PYTHON' does not exist."

LOG_DIR="output/b300_r8/_diag"
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/type_back_$TS.txt"
: > "$SUMMARY"
echo "  MGN-V python : $MGNV_PYTHON"
echo "  GPU          : $GPU"
echo "  logs         : $LOG_DIR"

for cfg in "${CFGS[@]}"; do
    [ -f "$cfg" ] || { echo "  SKIP $cfg: no such config" | tee -a "$SUMMARY"; continue; }
    ck="$(sed -n 's/^modelpath[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p' "$cfg" | head -1)"
    ck_abs="methods/MeshGraphNets_Variational/$ck"
    [ -f "$ck_abs" ] || { echo "  SKIP $cfg: checkpoint $ck_abs not found" | tee -a "$SUMMARY"; continue; }
    name="$(basename "$(dirname "$cfg")")_$(basename "$cfg" .txt | sed 's/^config_infer_//')"
    log="$LOG_DIR/${name}_$TS.log"
    echo "  [$(date +%T)] $name -> $log"
    ( cd methods/MeshGraphNets_Variational \
      && CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B misc/posterior_vs_prior_groups.py \
            --config "../../$cfg" --gpu 0 --tag "$name" ) > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED $name (exit $rc), last lines of $log:" | tee -a "$SUMMARY"
        tail -15 "$log" | sed 's/^/      /' | tee -a "$SUMMARY"
        continue
    fi
    sed -n '/==== TYPE THIS BACK ====/,/==== END ====/p' "$log" | tee -a "$SUMMARY"
done

echo
echo "  all blocks: $SUMMARY"
cat "$SUMMARY"
