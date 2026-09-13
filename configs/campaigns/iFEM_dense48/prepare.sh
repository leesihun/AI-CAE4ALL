#!/usr/bin/env bash
# One-time setup on the training box. Run from the AI-CAE4ALL suite root.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

echo "== verifying the transferred dataset"
( cd "$DATASET_DIR" && sha256sum -c SHA256SUMS.txt )

echo
echo "== output tree"
for arm in $ARMS_ALL; do mkdir -p "$OUTPUT_DIR/$arm"; done
echo "  $OUTPUT_DIR/<ARM>/"
echo "  models, logs, train/test visualisation dumps and held-out predictions all land here"

echo
echo "== MeshGraphNets private training copies"
# The MGN trainer opens dataset_dir with h5py 'r+' and writes metadata/normalization_params
# into it unconditionally -- there is no opt-out key. Each MGN arm therefore gets its own
# copy, so the shared train.h5 stays byte-identical for the Transolver and DeepONet arms
# and two MGN arms on different GPUs can never contend for the same file.
n_mgn=$(echo $ARMS_MGN | wc -w)
need_mb=$(( n_mgn * $(du -m "$DATASET_DIR/train.h5" | cut -f1) ))
free_mb=$(df -Pm "$DATASET_DIR" | tail -1 | awk '{print $4}')
echo "  $n_mgn MGN arms x $(du -h "$DATASET_DIR/train.h5" | cut -f1) = ~${need_mb} MB needed, ${free_mb} MB free"
[ "$free_mb" -gt "$(( need_mb + 5000 ))" ] || die "not enough free disk for the MGN copies"
for arm in $ARMS_MGN; do
  dst="$DATASET_DIR/native_train/$arm/train.h5"
  if [ -f "$dst" ]; then
    echo "  $arm: already present"
  else
    mkdir -p "$(dirname "$dst")"
    cp "$DATASET_DIR/train.h5" "$dst"
    echo "  $arm: copied"
  fi
done

echo
echo "== disk"
df -h "$AICAE_ROOT" | tail -1
echo
echo "next: bash configs/campaigns/$CAMPAIGN/check_all.sh"
