import numpy as np
import torch

from model.gno import GNOLayer


def _reference_gno(q_pos, s_pos, s_feat, edge_index, kernel_fn, num_queries, out_dim):
    """fp64 pure-Python loop reference: mean over incoming edges, zero for
    queries with no neighbors."""
    acc = np.zeros((num_queries, out_dim), dtype=np.float64)
    cnt = np.zeros(num_queries, dtype=np.float64)
    for e in range(edge_index.shape[1]):
        q, s = edge_index[0, e], edge_index[1, e]
        msg = kernel_fn(q_pos[q], s_pos[s], s_feat[s])
        acc[q] += msg
        cnt[q] += 1
    out = np.zeros((num_queries, out_dim), dtype=np.float64)
    nonzero = cnt > 0
    out[nonzero] = acc[nonzero] / cnt[nonzero, None]
    return out


def test_matches_loop_reference():
    torch.manual_seed(0)
    layer = GNOLayer(query_dim=2, source_dim=2, source_feat_dim=3, hidden=16, out_dim=4, depth=2)
    layer.eval()

    num_q, num_s = 6, 5
    q_pos = torch.rand(num_q, 2)
    s_pos = torch.rand(num_s, 2)
    s_feat = torch.rand(num_s, 3)
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 4], [0, 1, 2, 3, 4, 0]], dtype=torch.long)

    with torch.no_grad():
        out = layer(q_pos, s_pos, s_feat, edge_index, num_queries=num_q).numpy()

    def kernel_fn(qp, sp, sf):
        msg_in = torch.tensor(np.concatenate([qp, sp, sf])[None, :], dtype=torch.float32)
        with torch.no_grad():
            return layer.kernel_mlp(msg_in)[0].numpy()

    ref = _reference_gno(q_pos.numpy(), s_pos.numpy(), s_feat.numpy(),
                         edge_index.numpy(), kernel_fn, num_q, 4)
    assert np.allclose(out, ref, atol=1e-5)


def test_empty_neighbor_query_is_zero():
    layer = GNOLayer(query_dim=2, source_dim=2, source_feat_dim=2, hidden=8, out_dim=3, depth=1)
    layer.eval()
    q_pos = torch.rand(3, 2)
    s_pos = torch.rand(2, 2)
    s_feat = torch.rand(2, 2)
    # query 1 has no edges at all
    edge_index = torch.tensor([[0, 2], [0, 1]], dtype=torch.long)
    with torch.no_grad():
        out = layer(q_pos, s_pos, s_feat, edge_index, num_queries=3)
    assert torch.all(out[1] == 0.0)


def test_fully_empty_edge_index_returns_all_zero():
    layer = GNOLayer(query_dim=2, source_dim=2, source_feat_dim=2, hidden=8, out_dim=3, depth=1)
    q_pos = torch.rand(4, 2)
    s_pos = torch.rand(3, 2)
    s_feat = torch.rand(3, 2)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = layer(q_pos, s_pos, s_feat, edge_index, num_queries=4)
    assert torch.all(out == 0.0)
    assert out.shape == (4, 3)
