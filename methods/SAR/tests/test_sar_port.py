"""The port's changes to upstream code: the dense attention path equals the
reference one, gradient checkpointing changes nothing, and the plain-PyTorch
checkpoint round-trips. (`sample_n` returning exactly n -- 5 with batch 2,
where upstream returned 6 -- is checked by test_end_to_end.)"""
import torch
from torch import nn

from sar.nn import VAE
from sar.nn.modules import transolver as tr


def _transolver():
    torch.manual_seed(0)
    return tr.Transolver(in_features=5, out_features=3, num_layers=2, d_model=16, num_heads=2,
                         num_slices=4, d_cond=6).eval()


def test_dense_attention_matches_reference():
    model = _transolver()
    batch = torch.tensor([0] * 7 + [1] * 11 + [2] * 5)
    x = torch.randn(batch.numel(), 5)
    c = torch.randn(3, 6)
    try:
        tr.set_runtime_options(dense_attention=False)
        ref = model(x, batch, c)
        tr.set_runtime_options(dense_attention=True)
        dense = model(x, batch, c)
    finally:
        tr.set_runtime_options(dense_attention=True, use_checkpointing=False)
    torch.testing.assert_close(dense, ref, rtol=1e-5, atol=1e-5)


def test_checkpointing_same_gradients():
    model = _transolver().train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    batch = torch.tensor([0] * 6 + [1] * 9)
    x = torch.randn(batch.numel(), 5)
    c = torch.randn(2, 6)
    grads = []
    try:
        for flag in (False, True):
            tr.set_runtime_options(use_checkpointing=flag)
            model.zero_grad()
            model(x, batch, c).square().sum().backward()
            grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    finally:
        tr.set_runtime_options(dense_attention=True, use_checkpointing=False)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_vae_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    vae = VAE(in_features=3, num_layers=[1], d_model=16, dim=3, latent_features=4,
              activation=nn.SELU, dropout=0.1, kl_weight=1e-6, noise_std=0.01)
    path = tmp_path / 'vae.pth'
    torch.save(vae.checkpoint_dict(stage='vae'), path)
    back = VAE.load_from_checkpoint(str(path))
    assert back.hparams == vae.hparams
    for (k, a), (_, b) in zip(vae.state_dict().items(), back.state_dict().items()):
        assert torch.equal(a, b), k
