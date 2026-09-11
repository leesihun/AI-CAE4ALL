"""`flow_head x`: the network emits the clean field, the model returns a velocity.

Pinned here because a mistake in the conversion would train a model whose
"velocity" is not one, and nothing downstream -- loss, integrator, mean readout
-- would notice.

  1. head 'v' is the untouched model: output == velocity, byte for byte;
  2. head 'x' obeys the exact path identity  v = (h - s*y_t) / (1 - s*t)  with
     t broadcast per graph, so re-applying the identity to the returned v
     recovers the network's own output h;
  3. the denominator is floored at flow_head_eps near t = 1, never at 1e-4;
  4. at t = 0 the identity is exact (denominator 1), so predict_mean's readout
     z0 + v(z0, 0) equals h + sigma_min*z0 -- the clean field the head predicted.
"""

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data, Batch

FLOW_ROOT = Path(__file__).resolve().parents[1]
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))

from model.CHiMGNFlow import CHiMGNFlow          # noqa: E402
from model.flow import SIGMA_MIN                 # noqa: E402


def _config(**over):
    cfg = {
        "input_var": 3, "output_var": 3, "cond_var": 0, "edge_var": 8,
        "positional_features": 0, "use_node_types": False, "use_world_edges": False,
        "use_multiscale": False, "latent_dim": 16, "message_passing_num": 1,
        "flow_time_freqs": 4, "use_checkpointing": False,
    }
    cfg.update(over)
    return cfg


def _batch(n_per_graph=(5, 7)):
    graphs = []
    for n in n_per_graph:
        x = torch.zeros(n, 3)                          # zeroed state block
        ei = torch.tensor([list(range(n - 1)), list(range(1, n))])
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ea = torch.randn(ei.shape[1], 8)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea))
    return Batch.from_data_list(graphs)


def _pair(head, **over):
    """Two models with identical weights, one per head."""
    torch.manual_seed(0)
    m_v = CHiMGNFlow(_config(flow_head='v', **over), 'cpu').eval()
    torch.manual_seed(0)
    m_x = CHiMGNFlow(_config(flow_head=head, **over), 'cpu').eval()
    m_x.load_state_dict(m_v.state_dict())
    return m_v, m_x


def test_head_v_is_the_untouched_model():
    m_v, m_v2 = _pair('v')
    g = _batch()
    y_t = torch.randn(g.x.shape[0], 3)
    t = torch.tensor([[0.3], [0.8]])
    with torch.no_grad():
        assert torch.equal(m_v(g, y_t, t), m_v2(g, y_t, t))


def test_head_x_obeys_the_path_identity():
    m_v, m_x = _pair('x', flow_head_eps=0.05)
    g = _batch()
    y_t = torch.randn(g.x.shape[0], 3)
    t = torch.tensor([[0.3], [0.6]])
    s = 1.0 - SIGMA_MIN
    with torch.no_grad():
        h = m_v(g, y_t, t)               # same weights: the raw network output
        v = m_x(g, y_t, t)
    denom = (1.0 - s * t)[g.batch]       # [N, 1], per graph
    # invert the identity: h = s*y_t + (1 - s*t) * v
    assert torch.allclose(s * y_t + denom * v, h, atol=1e-5)
    assert not torch.allclose(v, h)      # it really did convert


def test_denominator_is_floored_near_t_one():
    m_v, m_x = _pair('x', flow_head_eps=0.05)
    g = _batch((4,))
    y_t = torch.randn(4, 3)
    t = torch.tensor([[1.0]])            # denominator would be sigma_min = 1e-4
    with torch.no_grad():
        h = m_v(g, y_t, t)
        v = m_x(g, y_t, t)
    s = 1.0 - SIGMA_MIN
    floored = (h - s * y_t) / 0.05
    assert torch.allclose(v, floored, atol=1e-5)
    assert float(v.abs().max()) < 1e3    # not the 1e4 amplification


def test_mean_readout_recovers_the_clean_field_at_t_zero():
    m_v, m_x = _pair('x')
    g = _batch((6,))
    z0 = torch.randn(6, 3)
    t0 = torch.zeros(1, 1)
    s = 1.0 - SIGMA_MIN
    with torch.no_grad():
        h = m_v(g, z0, t0)
        v = m_x(g, z0, t0)
    # predict_mean does z0 + v(z0, 0); with the x head that is h + sigma_min*z0
    assert torch.allclose(z0 + v, h + (1 - s) * z0, atol=1e-5)
