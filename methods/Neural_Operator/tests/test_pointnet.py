import torch

from model.pointnet import PointNetEncoder


def test_dense_forward_shape():
    net = PointNetEncoder(in_channels=6, hidden_channels=32, out_channels=16, depth=3)
    net.eval()
    x = torch.randn(4, 50, 6)  # [B, M, C_in]
    out = net.forward_dense(x)
    assert out.shape == (4, 16)


def test_permutation_invariance_dense():
    net = PointNetEncoder(in_channels=6, hidden_channels=32, out_channels=16, depth=3)
    net.eval()
    x = torch.randn(2, 30, 6)
    perm = torch.randperm(30)
    out1 = net.forward_dense(x)
    out2 = net.forward_dense(x[:, perm, :])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_segmented_matches_dense_for_equal_sized_graphs():
    net = PointNetEncoder(in_channels=4, hidden_channels=16, out_channels=8, depth=2)
    net.eval()
    b, m, c = 3, 10, 4
    x_dense = torch.randn(b, m, c)
    out_dense = net.forward_dense(x_dense)

    x_flat = x_dense.reshape(b * m, c)
    batch = torch.arange(b).repeat_interleave(m)
    out_seg = net.forward_segmented(x_flat, batch, num_graphs=b)
    assert torch.allclose(out_dense, out_seg, atol=1e-5)


def test_segmented_permutation_invariance():
    net = PointNetEncoder(in_channels=4, hidden_channels=16, out_channels=8, depth=2)
    net.eval()
    n0, n1 = 12, 9
    x = torch.randn(n0 + n1, 4)
    batch = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])
    out1 = net.forward_segmented(x, batch, num_graphs=2)

    perm = torch.randperm(n0 + n1)
    out2 = net.forward_segmented(x[perm], batch[perm], num_graphs=2)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_rejects_non_relu_activation():
    import pytest
    with pytest.raises(ValueError):
        PointNetEncoder(4, 16, 8, activation='gelu')


def test_rejects_non_batch_norm():
    import pytest
    with pytest.raises(ValueError):
        PointNetEncoder(4, 16, 8, norm='layer')
