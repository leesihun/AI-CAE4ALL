#!/usr/bin/env bash
#
# Diagnose, fix and re-measure MeshGraphNets-V's conditional prior, in one run.
#
#   cd methods/MeshGraphNets_Variational
#   nohup bash misc/fix_prior.sh 8 > ../../output/meshgraphnets-v/fix_prior.out 2>&1 &
#
# The argument is an ARM NAME in --cfg-dir; the train config is
# config_train_<arm>.txt and the eval sets are its config_infer_<arm>_<tag>.txt.
#
# WHAT IT DOES, AND WHY IN THIS ORDER
#
#   0  count realizations per part in the training file.
#      A conditional prior can only learn the within-part spread it is shown.
#      One realization per part means every condition has a single target and
#      no prior retraining can create the missing variation -- so this runs
#      first and the rest is pointless if it comes back "1".
#
#   1  BEFORE: the per-geometry posterior-vs-prior diagnostic on the current
#      checkpoint, one eval set at a time. This is what tells prior mismatch
#      apart from encode/decode loss. The existing [PriorDiag] cannot: it
#      averages Var(mu_q) over the whole validation set, so it is dominated by
#      BETWEEN-geometry variance while sd_ratio is made by the WITHIN-geometry
#      spread of one part's realizations.
#
#   2  FIX: retrain the prior alone against a frozen encoder and decoder,
#      removing all three mechanisms at once --
#        * the moving target (the encoder no longer drifts under recon loss),
#        * the ill-scaled target (latent standardization; MMD pins the
#          AGGREGATE q(z) to N(0,I), which with ~100 realizations of each of a
#          few parts leaves every per-part cloud far smaller and off-origin),
#        * the undersized trunk (rebuilt larger, encoder/decoder untouched).
#
#   3  AFTER: the same diagnostic on the retrained checkpoint, so the decision
#      is a before/after on the same geometries and the same metric.
#
# The retrained file is a drop-in: encoder/decoder weights byte-identical,
# normalization and model_config carried over. Point an inference config's
# modelpath at it and the ordinary run_sweep.sh pipeline runs unchanged.
#
# Environment:
#   CFG_DIR       config folder (default: the SAOI_sweep arm folder)
#   TAGS          eval sets (default: s26fe_main s26fe_sec sm_l345u)
#   EPOCHS        prior epochs (default 300)
#   LR            prior learning rate (default 3e-4)
#   PRIOR_HIDDEN  rebuild width, 0 = keep (default 512)
#   PRIOR_LAYERS  rebuild depth, 0 = keep (default 8)
#   N_PRIOR       prior draws per eval geometry in the diagnostic (default 500)
#   SKIP_BEFORE   1 = do not re-measure the baseline
#   GPU           card to run on (default: the config's gpu_ids -- which is the
#                 card the arm TRAINED on, so set this if that run is still live)
#   TRAIN_H5      training file for the realization count (default: the config's)
set -uo pipefail

ARM="${1:-}"
if [ -z "$ARM" ]; then
    echo "usage: bash misc/fix_prior.sh <arm name>   e.g. 8" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.." || exit 1          # methods/MeshGraphNets_Variational
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
CFG_DIR="${CFG_DIR:-../../configs/MeshGraphNets_Variational/SAOI_sweep}"
TAGS="${TAGS:-s26fe_main s26fe_sec sm_l345u}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-3e-4}"
PRIOR_HIDDEN="${PRIOR_HIDDEN:-512}"
PRIOR_LAYERS="${PRIOR_LAYERS:-8}"
N_PRIOR="${N_PRIOR:-500}"
SKIP_BEFORE="${SKIP_BEFORE:-0}"
GPU="${GPU:-}"
GPU_ARG=""
[ -n "$GPU" ] && GPU_ARG="--gpu $GPU"

TRAIN_CFG="$CFG_DIR/config_train_${ARM}.txt"
[ -f "$TRAIN_CFG" ] || { echo "no such train config: $TRAIN_CFG" >&2; exit 1; }

read_key() { sed -n "s/^$1[[:space:]]\{1,\}\([^#]*\).*/\1/p" "$2" | head -1 | sed 's/[[:space:]]*$//'; }
SRC_CKPT="$(read_key modelpath "$TRAIN_CFG")"
NEW_CKPT="${SRC_CKPT%.pth}_priorfit.pth"
TRAIN_H5="${TRAIN_H5:-$(read_key dataset_dir "$TRAIN_CFG")}"
DIAG_DIR="${DIAG_DIR:-$(dirname "$SRC_CKPT")/diag}"
mkdir -p "$DIAG_DIR"

echo "=================================================================="
echo " MeshGraphNets-V prior fix -- arm $ARM"
echo "=================================================================="
echo "  train config : $TRAIN_CFG"
echo "  checkpoint   : $SRC_CKPT"
echo "  will write   : $NEW_CKPT"
echo "  eval sets    : $TAGS"
echo "  card         : ${GPU:-from config (gpu_ids)}"
echo ""

rc=0

# ---- 0: is the spread even learnable? --------------------------------------
echo "---- 0. realizations per part -----------------------------------"
"$PYTHON" misc/count_realizations.py "$TRAIN_H5" || rc=1
echo ""
echo "  If that says ONE realization per part, stop here: the prior has never"
echo "  seen within-part variation and retraining cannot invent it."
echo ""

# ---- 1: where is the variability lost, before any change -------------------
if [ "$SKIP_BEFORE" = "1" ]; then
    echo "---- 1. BEFORE diagnostic -- skipped (SKIP_BEFORE=1) ------------"
else
    echo "---- 1. BEFORE: posterior vs prior, current checkpoint ----------"
    for tag in $TAGS; do
        cfg="$CFG_DIR/config_infer_${ARM}_${tag}.txt"
        if [ ! -f "$cfg" ]; then
            echo "  [skip] no inference config for $tag ($cfg)"
            continue
        fi
        "$PYTHON" misc/posterior_vs_prior.py --config "$cfg" \
            --n-prior "$N_PRIOR" $GPU_ARG --tag "${ARM}_${tag}_before" --out "$DIAG_DIR" || rc=1
    done
fi
echo ""

# ---- 2: fit the prior alone ------------------------------------------------
echo "---- 2. retrain the prior (encoder + decoder frozen) -------------"
"$PYTHON" misc/retrain_prior.py --config "$TRAIN_CFG" \
    --epochs "$EPOCHS" --lr "$LR" \
    --prior-hidden "$PRIOR_HIDDEN" --prior-layers "$PRIOR_LAYERS" \
    --out "$NEW_CKPT" $GPU_ARG
if [ $? -ne 0 ] || [ ! -f "$NEW_CKPT" ]; then
    echo "  prior retrain FAILED -- no AFTER measurement to make." >&2
    exit 1
fi
echo ""

# ---- 3: did it move the metric that matters? -------------------------------
echo "---- 3. AFTER: the same diagnostic on the retrained prior --------"
for tag in $TAGS; do
    cfg="$CFG_DIR/config_infer_${ARM}_${tag}.txt"
    [ -f "$cfg" ] || continue
    "$PYTHON" misc/posterior_vs_prior.py --config "$cfg" --modelpath "$NEW_CKPT" \
        --n-prior "$N_PRIOR" $GPU_ARG --tag "${ARM}_${tag}_after" --out "$DIAG_DIR" || rc=1
done

echo ""
echo "=================================================================="
echo " Read the 'prior' row's sd_ratio before vs after, per eval set."
echo " Target is 1; the 8-arm sweep never exceeded 0.526."
echo ""
echo "   moved toward posterior_mean  -> the prior WAS the bottleneck, fixed"
echo "   unchanged, posterior high    -> still the prior: read the PCA table."
echo "                                   Prior variance in the wrong directions"
echo "                                   needs a covariance fix, not more epochs"
echo "   posterior_mean also low      -> encode/decode loses it; no prior work"
echo "                                   can recover it"
echo ""
echo " JSON: $DIAG_DIR/posterior_vs_prior_${ARM}_<tag>_{before,after}.json"
echo " New checkpoint: $NEW_CKPT"
echo "=================================================================="
exit $rc
