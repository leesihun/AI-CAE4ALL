#!/usr/bin/env bash
#
# DOE-2 for MeshGraphNets-V: the beta_aux dose-response, cards 8-13.
# Train, infer, score and write the document in one command.
#
#   nohup bash configs/MeshGraphNets_Variational/SAOI_sweep3/run_sweep2.sh \
#       > output/meshgraphnets-v/saoi_sweep3/doe2.out 2>&1 &
#   tail -f output/meshgraphnets-v/saoi_sweep3/doe2.out
#
# WHY THIS EXISTS
#   DOE-1 (cards 4-7) tests the peak-to-valley term at beta_aux 0 and 100 only.
#   With realistic recon/aux magnitudes the weight that puts the term at ~30% of
#   the objective came out near 8, so 100 is roughly 13x that. Two points cannot
#   separate "the mechanism fails" from "the weight crushed the reconstruction",
#   which is the difference between abandoning an idea and re-scaling it.
#
#   Adding 3, 10 and 30 on each section makes the ladder
#
#       beta_aux   0     3    10    30   100
#                  control     ^ estimated balance point
#
#   and running it on BOTH sections means the curve's SHAPE is replicated rather
#   than fitted once. Read the shape, not the single best value: a curve that
#   rises and turns over says the weight matters; one that is flat across 1.5
#   decades says the mechanism does not.
#
# IT DOES NOT TOUCH DOE-1
#   Only the six new arms are trained. The DOE-1 arms are referenced once, at
#   the scoring step, so the document covers the whole ladder instead of six
#   rungs of ten. Arms with no dump yet are printed as such, not dropped.
#
# CARDS
#   The arms pin their own cards via gpu_ids, set when the configs were
#   generated. The default assumes a SECOND eight-card box numbered 0-7:
#   cHI-MGNflow takes 0-1 and the six MeshGraphNets-V rungs take 2-7.
#
#   The first attempt wrote them as 8-15, as though the added GPUs extended
#   one machine's numbering, and preflight refused every arm with
#   ENV-CUDA-002 before any of them ran. If the real layout differs,
#   regenerate with the cards you have -- nothing else changes:
#
#     DOE2_GPUS="<ids>" python <this folder>/gen_sweep_configs.py --pv
#
# Environment overrides (everything else is run_sweep.sh's):
#   PYTHON     interpreter (default: python)
#   STAGGER    seconds between arm launches (default: 10)
#   TRAIN      0 to skip training and go straight to inference
#   INFER      0 to skip inference
#   SCORE      0 to skip the scoring + document pass
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
STAGGER="${STAGGER:-10}"
TRAIN="${TRAIN:-1}"
INFER="${INFER:-1}"
SCORE="${SCORE:-1}"

# Printed verbatim by gen_sweep_configs.py --pv as DOSE_ARMS= and ARMS=, so
# these cannot drift from the generator without the mismatch being visible.
NEW_ARMS="2_1 2_2 2_3 2_4 2_5 2_6"
ALL_ARMS="pv_bot_a0 pv_bot_a100 pv_top_a0 pv_top_a100 $NEW_ARMS"
DOC="docs/research/SAOI_LONG_RUN_MGNV.md"

echo "=================================================================="
echo " MeshGraphNets-V DOE-2 -- beta_aux dose-response, cards 8-13"
echo "=================================================================="
echo "  new arms  : $NEW_ARMS"
echo "  ladder    : 0 / 3 / 10 / 30 / 100 on each of bot and top"
echo "  budget    : 1000 epochs x ~346 s/epoch = ~96 h per arm, all parallel"
echo ""

rc=0

# ---- train + infer the new rungs -------------------------------------------
# SCORE=0 here on purpose: a report built from this batch alone would cover six
# arms of ten and read as if the other four had failed.
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    ARMS="$NEW_ARMS" TRAIN="$TRAIN" INFER="$INFER" SCORE=0 STAGGER="$STAGGER" \
        PYTHON="$PYTHON" bash "$HERE/run_sweep.sh" || rc=1
    echo ""
    echo "DOE-2 arms finished (rc=$rc)."
else
    echo "TRAIN=0 INFER=0 -- going straight to scoring."
fi

# ---- score the whole ladder ------------------------------------------------
if [ "$SCORE" != "1" ]; then
    echo ""
    echo "SCORE=0 -- skipped. Build the document later with:"
    echo "  ARMS=\"$ALL_ARMS\" TRAIN=0 INFER=0 SCORE=1 bash $HERE/run_sweep.sh"
    exit $rc
fi

echo ""
echo "Scoring the FULL ladder so the document covers DOE-1 as well..."
echo ""
ARMS="$ALL_ARMS" TRAIN=0 INFER=0 SCORE=1 PYTHON="$PYTHON" \
    bash "$HERE/run_sweep.sh" || rc=1

echo ""
echo "Document: $DOC"
echo ""
echo "Its Verdict compares each section's control against its weighted arms on"
echo "sd_ratio, whose target is 1 and whose best over the whole 8-arm sweep was"
echo "0.526. The objective-balance table underneath shows what share of the loss"
echo "the peak-to-valley term actually carried at each weight -- which is how a"
echo "null result gets attributed to the mechanism rather than to the scaling."
exit $rc
