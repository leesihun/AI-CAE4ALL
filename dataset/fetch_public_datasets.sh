#!/usr/bin/env bash
# Stage the public CAE/ML datasets described in dataset/PUBLIC_DATASETS.md.
#
# Downloads go to $RAW (default D:/CAE_datasets_raw) rather than the repo drive -- the raw
# sources total ~47 GB and C: had ~115 GB free. Override with:  RAW=/path ./fetch_public_datasets.sh
#
# Resumable: curl -C - continues a partial file instead of restarting, so re-running after an
# interruption is cheap and safe.
set -u
RAW="${RAW:-/d/CAE_datasets_raw}"
mkdir -p "$RAW"

get() {  # get <url> <dest-relative-path>
  local url="$1" dest="$RAW/$2"
  mkdir -p "$(dirname "$dest")"
  echo "[$(date +%H:%M:%S)] START $2"
  curl -L -C - --retry 5 --retry-delay 10 --retry-connrefused -o "$dest" "$url" 2>&1 | tail -2
  echo "[$(date +%H:%M:%S)] DONE  $2 -> $(du -h "$dest" 2>/dev/null | cut -f1)"
}

MGN=https://storage.googleapis.com/dm-meshgraphnets

# 1. cylinder_flow  ~15.2 GB : temporal 2D incompressible flow, 600 steps, ~1.9k nodes
# 2. deforming_plate ~10.7 GB : temporal 3D hyperelastic CONTACT, node types + world edges
for ds in cylinder_flow deforming_plate; do
  for f in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
    get "$MGN/$ds/$f" "meshgraphnets/$ds/$f"
  done
done

# 3. AirfRANS ~9.3 GB : 1000 static RANS airfoil sims, varied geometry + (Re, AoA) conditions
get "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip" "airfrans/Dataset.zip"

# 4. flag_simple ~11.4 GB : temporal cloth with DYNAMIC node types (they change per timestep,
#    which nothing else here covers). Optional -- no converter ships for it yet.
for f in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
  get "$MGN/flag_simple/$f" "meshgraphnets/flag_simple/$f"
done

echo "[$(date +%H:%M:%S)] ALL DOWNLOADS COMPLETE"
du -sh "$RAW"/* 2>/dev/null
