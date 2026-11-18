#!/usr/bin/env bash
# Is the latent in order, and does an ordered latent give a better prior? Answered from
# runs that are ALREADY TRAINED, so it needs no training and can run beside round 13:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r13/organized_135.sh [GPU]
#
# Which runs: round 8 trained the base with the per-sample KL at four weights (0.1, 1, 10,
# 100) and with the MMD at weight 10000; only their scores were read then. When none of
# round 8's checkpoints is left on this node, it takes instead the base's other two seeds
# (round 9 hybrid_base_s43, round 10 hybrid_base_s44) and round 11's local runs (no z_g,
# seeds 42/43). Those have no KL, so they cannot say whether the KL puts the latent in
# order (round 13's KL settings ask that); they still give n, s and c for the base and for
# the plainer model. SET=r8 or SET=other forces one list.
#
# For each run it runs methods/MeshGraphNets_Variational/misc/posterior_vs_prior_groups.py
# on the held-out file (forward passes only, ~10-20 GB, minutes per run, in the
# foreground, one after another on one card) and prints one line per run:
#
#   kl10 27/31/85 n120 s95 c130     floor / posterior / prior, then
#        n  a z ~ N(0,I) decoded and scored like the prior. n near the third number =
#           the decoder reads any N(0,I) draw as a valid result: the latent is in order.
#        s  the posterior's width over the spread of its means, z_l (x100)
#        c  the largest over the smallest z_l cloud size among the 8 geometries (x100).
#           100 = every geometry's latent has the same size, so a normalizer per
#           geometry would have nothing to rescale.
#
# What it settles (round 8's list): if n falls with the KL weight and the third number (the
# prior) does not, a latent in order is not what the prior lacks. If n never falls, the KL
# of round 8 did not put it in order and the question is still open (round 13's KL
# settings ask it again on the joint setting).
#
# GPU: the card to use; default the one with the most free memory. A run whose JSON is
# already there (with n in it) is not run again (REDO=1 runs it again).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES = nvidia-smi's index
METHOD=methods/MeshGraphNets_Variational
SWEEPS=configs/MeshGraphNets_Variational/hyperparameter_sweep
# label=round/job; the label is what the typed-back line starts with.
LIST_R8="base=r8/hybrid_base kl0.1=r8/hybrid_kl0.1 kl1=r8/hybrid_kl1 kl10=r8/hybrid_kl10 kl100=r8/hybrid_kl100 mmd10000=r8/hybrid_mmd10000"
LIST_OTHER="base43=r9/hybrid_base_s43 base44=r10/hybrid_base_s44 local42=r11/local_s42 local43=r11/local_s43"

"$PYTHON" -c "import cae_suite" 2>/dev/null \
    || { echo "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."; exit 2; }
MGNV_PYTHON="${MGNV_PYTHON:-}"
if [ -z "$MGNV_PYTHON" ]; then
    MGNV_PYTHON="$("$PYTHON" -c "
from pathlib import Path
from cae_suite.settings import LocalSettings
print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets_variational'))
")" || { echo "could not resolve the MGN-V interpreter (ai_cae4all.local.toml); set MGNV_PYTHON."; exit 2; }
fi
grep -q "std_normal" "$METHOD/misc/posterior_vs_prior_groups.py" \
    || { echo "posterior_vs_prior_groups.py has no std_normal: git pull first"; exit 2; }

cfg_path() { sed -n "s/^$2[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p" "$1" | head -1; }
ckpt_of() {   # round job -> checkpoint path, empty when the config is missing
    local cfg="$SWEEPS/$1/config_infer_$2.txt"
    [ -f "$cfg" ] && echo "$METHOD/$(cfg_path "$cfg" modelpath)"
}

SET="${SET:-}"
if [ -z "$SET" ]; then
    SET=other
    for e in $LIST_R8; do
        rj="${e#*=}"; ck="$(ckpt_of "${rj%%/*}" "${rj#*/}")"
        [ -n "$ck" ] && [ -f "$ck" ] && { SET=r8; break; }
    done
    [ "$SET" = "other" ] && echo "no checkpoint of round 8 is left: using the base's seeds 43/44 and round 11's local runs"
fi
case "$SET" in
    r8) LIST="$LIST_R8" ;;
    other) LIST="$LIST_OTHER" ;;
    *) echo "SET must be r8 or other"; exit 2 ;;
esac

GPU="${1:-}"
if [ -z "$GPU" ]; then
    GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
           | tr -d '\r ' | sort -t, -k2 -n -r | head -1 | cut -d, -f1)"
    GPU="${GPU:-0}"
fi
echo "card $GPU, MGN-V python $MGNV_PYTHON, runs: $SET"

ENTRIES=()   # label|json|state; state: ok, none (no checkpoint), F (the diagnosis failed)
for e in $LIST; do
    label="${e%%=*}"; rj="${e#*=}"; round="${rj%%/*}"; job="${rj#*/}"
    cfg="$SWEEPS/$round/config_infer_$job.txt"
    out="output/b300_$round/$job/diag"
    json="$out/posterior_vs_prior_groups_ho.json"
    if [ -f "$json" ] && grep -q std_normal "$json" && [ -z "${REDO:-}" ]; then
        ENTRIES+=("$label|$json|ok"); echo "$label: already there"; continue
    fi
    ck="$(ckpt_of "$round" "$job")"
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then
        ENTRIES+=("$label|$json|none"); echo "$label: no checkpoint (${ck:-$cfg})"; continue
    fi
    mkdir -p "$out"
    echo "[$(date '+%m-%d %H:%M')] $label ($round/$job) on GPU $GPU (log: $out/diag_ho.log)"
    ( cd "$METHOD" && CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$MGNV_PYTHON" -B \
        misc/posterior_vs_prior_groups.py --config "../../$cfg" --gpu 0 \
        --out "../../$out" --tag ho ) > "$out/diag_ho.log" 2>&1
    if [ $? -eq 0 ] && [ -f "$json" ]; then ENTRIES+=("$label|$json|ok")
    else ENTRIES+=("$label|$json|F"); echo "$label FAILED: $out/diag_ho.log"; fi
done

echo
echo "  (do not type) floor/posterior/prior on the 8 held-out geometries, x100; n = a decoded"
echo "  z ~ N(0,I) scored like the prior; s = posterior width / spread of its means, z_l;"
echo "  c = largest / smallest z_l cloud size among the geometries (100 = all the same)."
echo "  e360 = the base of round 13 (360 epochs, seed 42)."
echo "==== TYPE THIS BACK ===="
echo "ORG 135 $SET"
ENTRIES="$(printf '%s\n' "${ENTRIES[@]}")" "$PYTHON" -B - <<'EOF'
import json, os

def x100(v):
    return "-" if v is None else "nan" if v != v else str(int(round(v * 100)))

def line(name, path, state="ok"):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return f"{name} {state if state != 'ok' else 'none'}"
    sc = d.get("scores") or {}
    w = [((sc.get(e) or {}).get("w2") or {}).get("p50")
         for e in ("floor", "posterior_sample", "prior", "std_normal")]
    s = ((d.get("latent_geomean_over_groups") or {}).get("l") or {}).get("sigma_over_sd_means")
    # One geometry's z_l cloud size: spread of its posterior means and its sigma together.
    size = [(g["l"]["sd_post_means_rms"] ** 2 + g["l"]["posterior_sigma_rms"] ** 2) ** 0.5
            for g in (d.get("latent") or {}).values()
            if "sd_post_means_rms" in (g.get("l") or {})]
    c = max(size) / min(size) if size and min(size) > 0 else None
    return f"{name} " + "/".join(x100(v) for v in w[:3]) + f" n{x100(w[3])} s{x100(s)} c{x100(c)}"

print(line("e360", "output/b300_r9/ref_r8_hybrid_e360/diag/posterior_vs_prior_groups_ho.json"))
for entry in os.environ["ENTRIES"].splitlines():
    if entry:
        label, path, state = entry.split("|")
        print(line(label, path, state))
EOF
echo "==== END ===="
