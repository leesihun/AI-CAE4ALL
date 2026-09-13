#!/usr/bin/env bash
# Pack what is worth bringing back. Per-case prediction dumps are the bulk
# (33 arms x 8,482 files), so each arm gets its own archive.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PACKED="$OUTPUT_DIR/packed"
mkdir -p "$PACKED"

echo "== logs, checkpoints, scores, configs (small)"
cd "$AICAE_ROOT"
tar czf "$PACKED/logs_and_checkpoints.tgz" \
  $(ls output/$CAMPAIGN/*/train.log output/$CAMPAIGN/*/launcher.log \
       output/$CAMPAIGN/*/model.pth output/$CAMPAIGN/queue_gpu*.log \
       output/$CAMPAIGN/scores.json output/$CAMPAIGN/baselines.json 2>/dev/null) \
  configs/campaigns/$CAMPAIGN configs/Transolver/$CAMPAIGN \
  configs/MeshGraphNets/$CAMPAIGN configs/Neural_Operator/$CAMPAIGN
echo "   $(du -h "$PACKED/logs_and_checkpoints.tgz" | cut -f1)"

echo "== train/test visualisation dumps"
if compgen -G "output/$CAMPAIGN/*/dumps" > /dev/null || compgen -G "output/$CAMPAIGN/*/test" > /dev/null; then
  tar czf "$PACKED/visualisation.tgz" $(ls -d output/$CAMPAIGN/*/dumps output/$CAMPAIGN/*/test \
      output/$CAMPAIGN/*/train 2>/dev/null)
  echo "   $(du -h "$PACKED/visualisation.tgz" | cut -f1)"
else
  echo "   none found"
fi

echo "== per-arm held-out predictions"
for arm in $ARMS_ALL; do
  [ -d "$OUTPUT_DIR/$arm/infer" ] || continue
  tar czf "$PACKED/infer_${arm}.tgz" -C "$OUTPUT_DIR/$arm" infer
  echo "   $arm  $(du -h "$PACKED/infer_${arm}.tgz" | cut -f1)"
done
echo
echo "bring back: $PACKED"
