#!/usr/bin/env bash
# Held-out inference for every arm that produced a checkpoint. Writes one HDF5 per case
# (8,482 per arm) into output/iFEM_dense48/<ARM>/infer/. Minutes per arm.
#   bash configs/campaigns/iFEM_dense48/infer_all.sh          # every finished arm
#   bash configs/campaigns/iFEM_dense48/infer_all.sh T01 T03  # selected arms
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
cd "$AICAE_ROOT"

for arm in ${*:-$ARMS_ALL}; do
  ckpt="$OUTPUT_DIR/$arm/model.pth"
  if [ ! -f "$ckpt" ]; then
    echo "  $arm: no checkpoint -- skipped (training did not finish)"
    continue
  fi
  echo "== $arm"
  "$PYBIN" -B AI_CAE4ALL_main.py --config "$(config_path "$arm" infer)" \
    > "$OUTPUT_DIR/$arm/infer_launcher.log" 2>&1
  code=$?
  n=$(ls "$OUTPUT_DIR/$arm/infer"/*.h5 2>/dev/null | wc -l)
  echo "   exit=$code, $n prediction files"
done

echo
echo "next: \$PYBIN configs/campaigns/$CAMPAIGN/score_infer.py"
