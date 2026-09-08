"""The peak-to-valley auxiliary loss on the decoded field.

`beta_aux` weights an MSE on `max(field) - min(field)` per graph, computed from
the DECODER OUTPUT. It replaced a head that regressed per-graph `[mean, std]`
from `z`, which could not move the measured ensemble dispersion (`sd_ratio`
0.449-0.526 on every arm of the SAOI wave-3 sweep) for three structural
reasons, and each is pinned here:

  * it must read the decoder output, not `z` -- nothing tied a z-readout's
    accuracy to the field a rollout writes;
  * it must be the extreme-value statistic that is scored, not a node-axis
    standard deviation, which is a different quantity;
  * it must group PER GRAPH, because the sensitivity it is meant to create is
    between realizations of the same part.

Also pinned: the fine-level batch index. `_forward_multiscale` keeps a
`current_batch` that the descending arm re-scatters at every level, and passing
that one produced

    RuntimeError: The expanded size of the tensor (66782) must match the
    existing size (500) at non-singleton dimension 0

on the first real run -- 500 being 5 graphs x 100 coarsest clusters. The guard
turns a size coincidence into an error instead of silently scoring coarse nodes.
"""

from pathlib import Path
import sys

import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from model.MeshGraphNets import EncoderProcessorDecoder  # noqa: E402

pv_loss = EncoderProcessorDecoder._pv_loss


class _Stub:
    """`_pv_loss` reads only these two attributes off self."""

    pv_channel = 2
    training = True


# Two graphs, 4 and 3 nodes, 3 channels. Channel 2 is the scored one.
BATCH = torch.tensor([0, 0, 0, 0, 1, 1, 1])
NUM_GRAPHS = 2
Y = torch.tensor([[9., -9.,  1.0],
                  [9., -9.,  5.0],
                  [9., -9., -2.0],
                  [9., -9.,  3.0],
                  [9., -9.,  0.5],
                  [9., -9.,  0.5],
                  [9., -9.,  4.5]])
# graph 0 channel 2: max 5.0, min -2.0 -> peak-to-valley 7.0
# graph 1 channel 2: max 4.5, min  0.5 -> peak-to-valley 4.0


def _flat(value=2.0):
    """A prediction with zero peak-to-valley on the scored channel."""
    p = Y.clone()
    p[:, 2] = value
    return p


@pytest.fixture
def stub():
    return _Stub()


def test_groups_per_graph_not_across_the_batch(stub):
    # Per graph: mean((0-7)^2, (0-4)^2) = 32.5.
    # Pooled over the batch the truth would be a single 5.0-(-2.0)=7.0 and the
    # loss 49.0, so the value distinguishes the two.
    assert pv_loss(stub, _flat(), Y, BATCH, NUM_GRAPHS) == pytest.approx(32.5)


def test_zero_when_the_peak_to_valley_matches(stub):
    assert pv_loss(stub, Y.clone(), Y, BATCH, NUM_GRAPHS) == pytest.approx(0.0)


def test_only_the_scored_channel_participates(stub):
    wrecked = Y.clone()
    wrecked[:, 0] = 1000.0
    wrecked[:, 1] = -1000.0
    assert pv_loss(stub, wrecked, Y, BATCH, NUM_GRAPHS) == pytest.approx(0.0)


def test_moving_one_extreme_node_moves_the_loss(stub):
    """The distinguishing property of max-min against the std it replaced."""
    wide = Y.clone()
    wide[1, 2] = 8.0                     # graph 0: 7.0 -> 10.0
    assert pv_loss(stub, wide, Y, BATCH, NUM_GRAPHS) == pytest.approx(4.5)


def test_z_score_normalization_only_rescales_the_loss(stub):
    """Why the term can run on `graph.y` with no denormalization.

    `graph.y` is a channel-wise z-score of the field. The mean cancels in a
    difference, so the normalized peak-to-valley is the true one over that
    channel's sigma -- and the squared loss scales by 1/sigma^2, a constant
    absorbed into `beta_aux`.
    """
    mu, sigma = 12.34, 3.0
    y_n, pred_n = Y.clone(), _flat()
    y_n[:, 2] = (Y[:, 2] - mu) / sigma
    pred_n[:, 2] = (_flat()[:, 2] - mu) / sigma
    raw = pv_loss(stub, _flat(), Y, BATCH, NUM_GRAPHS)
    assert pv_loss(stub, pred_n, y_n, BATCH, NUM_GRAPHS) == pytest.approx(raw / sigma ** 2)


def test_gradient_reaches_the_extremes_and_nothing_else(stub):
    # A non-degenerate prediction: with every node equal, max and min select the
    # SAME element and their +1/-1 gradients cancel exactly.
    pred = Y.clone()
    pred[:, 2] = torch.tensor([1.5, 3.0, 0.5, 2.0, 1.0, 1.2, 2.8])
    pred.requires_grad_(True)
    pv_loss(stub, pred * 1.0, Y, BATCH, NUM_GRAPHS).backward()

    scored, unscored = pred.grad[:, 2], pred.grad[:, 0]
    assert float(unscored.abs().sum()) == 0.0
    # Exactly each graph's argmax and argmin -- the term is the exact statistic,
    # so it drives 2 nodes per graph and no more.
    per_graph = [int((scored[BATCH == g].abs() > 0).sum()) for g in range(NUM_GRAPHS)]
    assert per_graph == [2, 2]


def test_no_op_at_inference_and_without_a_target(stub):
    stub.training = False
    assert pv_loss(stub, _flat(), Y, BATCH, NUM_GRAPHS) == 0.0
    stub.training = True
    assert pv_loss(stub, _flat(), None, BATCH, NUM_GRAPHS) == 0.0


def test_a_coarsened_batch_index_is_refused(stub):
    """`_forward_multiscale` must pass the fine-level index, not `current_batch`."""
    coarse = torch.tensor([0, 1])        # 2 rows against 7 predicted nodes
    with pytest.raises(RuntimeError, match="FINE-level"):
        pv_loss(stub, _flat(), Y, coarse, NUM_GRAPHS)
