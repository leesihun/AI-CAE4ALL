import torch

from model.deeponet_fractional2d import FractionalLaplacianDeepONet


def test_paper_topology_and_forward_shape():
    model = FractionalLaplacianDeepONet(seed=12345)
    assert [layer.in_features for layer in model.branch_layers] == [225, 60, 60]
    assert [layer.out_features for layer in model.branch_layers] == [60, 60, 60]
    assert [layer.in_features for layer in model.trunk_layers] == [3, 60, 60]
    assert [layer.out_features for layer in model.trunk_layers] == [60, 60, 60]
    prediction = model(torch.randn(7, 225), torch.randn(7, 3))
    assert prediction.shape == (7, 1)
    assert torch.isfinite(prediction).all()


def test_seeded_truncated_xavier_is_deterministic():
    first = FractionalLaplacianDeepONet(seed=12345)
    second = FractionalLaplacianDeepONet(seed=12345)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_trunk_basis_is_bounded_but_branch_basis_is_linear():
    model = FractionalLaplacianDeepONet(seed=12345)
    with torch.no_grad():
        trunk = model.encode_trunk(torch.randn(32, 3) * 100.0)
        branch = model.encode_branch(torch.randn(32, 225) * 100.0)
    assert torch.max(torch.abs(trunk)) <= 1.0
    # This is an architecture assertion: the final branch layer is not tanh.
    expected = model.branch_layers[2](
        torch.tanh(model.branch_layers[1](
            torch.tanh(model.branch_layers[0](torch.randn(2, 225)))
        ))
    )
    assert expected.shape == (2, 60)
    assert branch.shape == (32, 60)


def test_encoded_pairing_matches_forward():
    model = FractionalLaplacianDeepONet(seed=12345).eval()
    branch = torch.randn(11, 225)
    query = torch.randn(11, 3)
    with torch.no_grad():
        direct = model(branch, query)
        encoded = model.decode_encoded(
            model.encode_branch(branch), model.encode_trunk(query)
        )
    assert torch.allclose(direct, encoded, atol=0.0, rtol=0.0)


def test_no_factory_registration_or_default_hot_path_change():
    from model.factory import MODEL_REGISTRY

    assert "deeponet_fractional_laplacian_2d" not in MODEL_REGISTRY
