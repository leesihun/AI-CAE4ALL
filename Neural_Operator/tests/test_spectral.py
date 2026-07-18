import numpy as np
import pytest
import torch

from model.spectral import SpectralConvNd, validate_fno_modes
from tests.reference_spectral import reference_spectral_conv_2d


def test_matches_fp64_reference_2d():
    torch.manual_seed(0)
    Cin, Cout, R1, R2 = 3, 4, 8, 10
    modes = (3, 4)
    layer = SpectralConvNd(Cin, Cout, modes)
    x = torch.randn(1, Cin, R1, R2, dtype=torch.float32)

    with torch.no_grad():
        out = layer(x)[0].numpy()  # [Cout, R1, R2]

    weight_blocks = []
    for b in range(layer.n_blocks):
        w = torch.view_as_complex(layer.weight[b].contiguous()).detach().numpy()
        weight_blocks.append(w)
    ref = reference_spectral_conv_2d(x[0].numpy(), weight_blocks, modes)

    assert np.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_shapes_2d_and_3d():
    layer2d = SpectralConvNd(2, 3, (2, 2))
    x2d = torch.randn(2, 2, 6, 6)
    out2d = layer2d(x2d)
    assert out2d.shape == (2, 3, 6, 6)
    assert layer2d.n_blocks == 2

    layer3d = SpectralConvNd(2, 3, (2, 2, 2))
    x3d = torch.randn(2, 2, 6, 6, 6)
    out3d = layer3d(x3d)
    assert out3d.shape == (2, 3, 6, 6, 6)
    assert layer3d.n_blocks == 4


def test_finite_under_bf16_autocast():
    if not torch.cuda.is_available():
        pytest.skip("bf16 autocast test requires CUDA")
    layer = SpectralConvNd(4, 4, (3, 3)).cuda()
    x = torch.randn(2, 4, 8, 8, device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = layer(x)
    assert torch.isfinite(out).all()


def test_validate_modes_rejects_nyquist_violation():
    with pytest.raises(ValueError, match="Nyquist"):
        validate_fno_modes((5, 2), (8, 8))  # 5 > 8//2=4 on non-final axis


def test_validate_modes_rejects_rfft_violation():
    with pytest.raises(ValueError, match="real-FFT"):
        validate_fno_modes((2, 6), (8, 8))  # 6 > 8//2+1=5 on final axis


def test_validate_modes_accepts_valid_config():
    validate_fno_modes((4, 5), (8, 8))  # exactly at both limits
