#!/usr/bin/env bash
#
# DOE-2 for cHI-MGNflow: capacity at the longer budget, cards 14-15.
# Train, infer, score and write the document in one command.
#
#   nohup bash configs/HI_MGNFlow/SAOI_sweepB/run_sweep2.sh \
#       > output/chi-mgnflow/saoi_sweepB/doe2.out 2>&1 &
#   tail -f output/chi-mgnflow/saoi_sweepB/doe2.out
#
# WHY CAPACITY
#   In the 8-arm sweep capacity moved W1/sd by 0.074 -- nothing -- while
#   batch_size, which at a fixed epoch count IS the optimizer step count, moved
#   it by 0.288. A dead capacity axis beside a live step axis is what a run that
#   could not train the model it had looks like: a bigger model cannot show its
#   capacity until it can be trained. Tripling the budget is exactly the
#   condition under which that ranking can invert.
#
#   k1 is the sweep's own upper level (latent_dim 192, mp_per_level 6,8,12,8,6),
#   so these arms are directly comparable to that 0.074. The budget is held at
#   the same 3000 epochs as the lr arms on purpose -- a capacity comparison at a
#   different budget answers a different question. They cost ~1.5x per epoch, so
#   expect them roughly two days after the rest.
#
# WHAT WAS CONSIDERED AND DROPPED
#   An arm at cond_var 2, to drop "Part No." on the theory that an unseen
#   categorical identifier was being extrapolated on -- which would have
#   explained a held-out bias whose sign flips between families. It is out
#   because Part No. is identically 0 in this data, and so is stress: a constant
#   channel normalizes to exactly 0 (finalize_moments floors std at 1e-8, so
#   (0-0)/1e-8 = 0, with no blow-up), and the arm would have measured nothing.
#
#   The fact still matters. flow's conditioning is THICKNESS plus the mesh and
#   nothing else -- everything the model knows about a part arrives through the
#   GNN's encoding of its geometry. That is why capacity, not conditioning, got
#   the cards.
#
# IT DOES NOT TOUCH DOE-1
#   Only the two new arms are trained. The DOE-1 arms are referenced once, at
#   the scoring step, so the document covers all six instead of two. Arms with
#   no dump yet are printed as such, not dropped.
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
#     DOE2_GPUS="<ids>" python <this folder>/gen_sweep_configs.py --long
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

# Printed verbatim by gen_sweep_configs.py --long as NEW_ARMS= and ARMS=, so
# these cannot drift from the generator without the mismatch being visible.
NEW_ARMS="2_1 2_2"
ALL_ARMS="long_bot_lr1 long_bot_lr3 long_top_lr1 long_top_lr3 $NEW_ARMS"
DOC="docs/research/SAOI_LONG_RUN_FLOW.md"

echo "=================================================================="
echo " cHI-MGNflow DOE-2 -- capacity k1 at 3000 epochs, cards 14-15"
echo "=================================================================="
echo "  new arms  : $NEW_ARMS"
echo "  capacity  : latent_dim 192, mp_per_level 6,8,12,8,6 (the sweep's k1)"
echo "  budget    : 3000 epochs, ~1.5x per epoch -- about two days behind DOE-1"
echo ""

rc=0

# ---- train + infer the new arms --------------------------------------------
# SCORE=0 here on purpose: a report built from this batch alone would cover two
# arms of six and read as if the other four had failed.
if [ "$TRAIN" = "1" ] || [ "$INFER" = "1" ]; then
    ARMS="$NEW_ARMS" TRAIN="$TRAIN" INFER="$INFER" SCORE=0 STAGGER="$STAGGER" \
        PYTHON="$PYTHON" bash "$HERE/run_sweep.sh" || rc=1
    echo ""
    echo "DOE-2 arms finished (rc=$rc)."
else
    echo "TRAIN=0 INFER=0 -- going straight to scoring."
fi

# ---- score every arm -------------------------------------------------------
if [ "$SCORE" != "1" ]; then
    echo ""
    echo "SCORE=0 -- skipped. Build the document later with:"
    echo "  ARMS=\"$ALL_ARMS\" TRAIN=0 INFER=0 SCORE=1 bash $HERE/run_sweep.sh"
    exit $rc
fi

echo ""
echo "Scoring every arm so the document covers DOE-1 as well..."
echo ""
ARMS="$ALL_ARMS" TRAIN=0 INFER=0 SCORE=1 PYTHON="$PYTHON" \
    bash "$HERE/run_sweep.sh" || rc=1

echo ""
echo "Document: $DOC"
echo ""
echo "Read the 'gain, middle half' column first: the LR anneals to 1e-8, so a"
echo "flat tail is the schedule and decides nothing. If the k1 arms gained more"
echo "there than the k0 arms did, capacity was being masked by the old budget --"
echo "which is the whole reason these two cards were spent."
exit $rc
