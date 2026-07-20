from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from general_modules.data_spec import DataSpec
from general_modules.config_validation import validate_common_config
from model.factory import _resolve_core_class
from model.gino_carcfd import (
    CarCFDGINODecoder,
    LinearKernelIntegral3d,
    NeRFSinusoidalEmbedding,
    PaperEraPositionalEmbedding,
    TuckerSpectralConv3d,
    _PaperFNOBlock3d,
    tucker_rank_from_fraction,
)

SUITE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUITE_ROOT / "dataset" / "benchmarks" / "gino_carcfd"))
from carcfd_dataset import CarCFDPaperDataset  # noqa: E402
from train_carcfd import build_paper_scheduler  # noqa: E402


def _data_spec():
    return DataSpec(
        input_var=1,
        output_var=1,
        positional_dim=0,
        node_type_dim=0,
        global_condition_dim=0,
        operator_dim=3,
        active_axes=(0, 1, 2),
        has_sdf=False,
        has_integration_weights=False,
        num_timesteps=1,
    )


def _config(**overrides):
    config = {
        "gino_variant": "paper_decoder",
        "gino_grid_resolution": [4, 4, 4],
        "gino_fno_modes": [2, 2, 2],
        "gino_fno_hidden_channels": 4,
        "gino_fno_layers": 1,
        "gino_tucker_rank": 0.7,
        "gino_channel_mlp_expansion": 1.0,
        "gino_lifting_hidden": 4,
        "gino_out_radius": 0.01,
        "gino_kernel_widths": [8],
        "gino_projection_widths": [8],
        "gino_query_chunk_size": 2,
        "gino_use_torch_cluster": False,
        "use_checkpointing": False,
    }
    config.update(overrides)
    return config


def _item(offset: float = 0.0):
    axis = torch.linspace(-1.0, 1.0, 4)
    latent_points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    pos = latent_points[[0, 5, 21, 42, 63]].clone()
    sdf = torch.randn(1, 4, 4, 4, 1) + offset
    return Data(
        x=torch.zeros(pos.shape[0], 1),
        y=torch.zeros(pos.shape[0], 1),
        pos=pos,
        latent_sdf=sdf,
    )


def test_paper_rank_fraction_matches_released_shape():
    assert tucker_rank_from_fraction((64, 64, 24, 24, 13), 0.4) == (53, 53, 20, 20, 11)


def test_factorized_tucker_contraction_matches_dense_weight():
    torch.manual_seed(2)
    layer = TuckerSpectralConv3d(2, 3, (2, 2, 2), rank_fraction=0.8)
    kept = torch.randn(1, 2, 2, 2, 2, dtype=torch.complex64)
    actual = layer.contract_kept_modes(kept)
    dense = layer.reconstructed_weight()
    expected = torch.einsum("nixyz,ioxyz->noxyz", kept, dense)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_linear_output_gno_uses_paper_era_neighbor_mean():
    layer = LinearKernelIntegral3d(
        channels=1,
        kernel_widths=(),
        radius=10.0,
        coordinate_embedding_dim=1,
        use_torch_cluster=False,
    )
    # One NeRF frequency emits sin/cos for each of three coordinates, and
    # the integral kernel receives source then query embeddings: 2 * 6.
    kernel = nn.Linear(12, 1)
    with torch.no_grad():
        kernel.weight.zero_()
        kernel.bias.fill_(2.0)
    layer.kernel = nn.Sequential(kernel)
    queries = torch.tensor([[0.0, 0.0, 0.0]])
    # The third latent source is deliberately not a neighbor: the paper-era
    # IntegralTransform divides by two neighbors, not all three grid points.
    sources = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    features = torch.tensor([[3.0], [5.0], [100.0]])
    edges = torch.tensor([[0, 0], [0, 1]])
    # (2*3 + 2*5) / number_of_neighbors with two neighbors.
    assert torch.allclose(layer(queries, sources, features, edges), torch.tensor([[8.0]]))


def test_paper_decoder_lifts_sdf_together_with_latent_grid_coordinates():
    model = CarCFDGINODecoder(_config(), _data_spec())
    assert model.include_grid_coordinates is True
    assert model.lifting[0].in_channels == 4


def test_nerf_embedding_matches_maintained_reference_order():
    embedding = NeRFSinusoidalEmbedding(in_channels=3, num_frequencies=2)
    coordinates = torch.tensor([[0.5, 0.0, -1.0]])
    actual = embedding(coordinates)
    expected = torch.tensor(
        [[
            1.0, 0.0, 0.0, -1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, -1.0, 0.0, 1.0,
        ]]
    )
    assert actual.shape == (1, 12)
    assert torch.allclose(actual, expected, atol=5e-7, rtol=0.0)


def test_paper_era_embedding_matches_public_2023_reference_formula():
    embedding = PaperEraPositionalEmbedding(in_channels=3, num_channels=4)
    coordinates = torch.tensor([[0.5, 0.0, -1.0]])
    frequencies = torch.tensor([1.0, 0.01])
    phases = coordinates.reshape(-1, 1) * frequencies
    expected = torch.cat((phases.cos(), phases.sin()), dim=1).reshape(1, 12)
    assert embedding.out_channels == 12
    assert torch.allclose(embedding(coordinates), expected, atol=1e-7, rtol=0.0)


def test_paper_decoder_applies_and_removes_one_sided_domain_padding():
    class _Capture(nn.Module):
        def __init__(self):
            super().__init__()
            self.shape = None

        def forward(self, value):
            self.shape = tuple(value.shape)
            return value

    model = CarCFDGINODecoder(
        _config(gino_domain_padding=0.25), _data_spec()
    )
    capture = _Capture()
    model.fno_blocks = nn.ModuleList([capture])
    encoded = model.encode_operator(Batch.from_data_list([_item()]))
    assert capture.shape == (1, 4, 5, 5, 5)
    assert encoded.shape == (1, 4, 4, 4, 4)


def test_exported_v2_config_reconstructs_lifting_and_strict_state_dict():
    model = CarCFDGINODecoder(
        _config(
            gino_lifting_hidden=7,
            gino_domain_padding=0.25,
            gino_pos_embedding_type="paper_2023",
            gino_coord_embed_dim=4,
        ),
        _data_spec(),
    )
    exported = model.export_model_config()
    assert exported["gino_lifting_hidden"] == 7
    assert exported["gino_domain_padding"] == 0.25
    assert exported["gino_pos_embedding_type"] == "paper_2023"
    rebuilt = CarCFDGINODecoder(exported, _data_spec())
    rebuilt.load_state_dict(model.state_dict(), strict=True)


class _Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, value):
        return value * self.value


def test_fno_block_matches_reference_skip_norm_and_final_activation_semantics():
    x = torch.ones(1, 2, 2, 2, 2)
    block = _PaperFNOBlock3d(2, (2, 2, 2), 0.8, 1.0, is_final=False)
    assert block.fno_skip.bias is None
    assert block.norm_fno.affine is False
    assert block.norm_channel_mlp.affine is False
    block.spectral = _Scale(2.0)
    block.fno_skip = _Scale(3.0)
    block.channel_mlp = _Scale(5.0)
    block.channel_mlp_skip = _Scale(7.0)
    block.norm_fno = nn.Identity()
    block.norm_channel_mlp = nn.Identity()
    block.activation = _Scale(2.0)
    # act((5 * act((2 + 3) * x)) + 7 * original_x) = 114*x.
    assert torch.equal(block(x), torch.full_like(x, 114.0))

    final = _PaperFNOBlock3d(2, (2, 2, 2), 0.8, 1.0, is_final=True)
    final.spectral = _Scale(2.0)
    final.fno_skip = _Scale(3.0)
    final.channel_mlp = _Scale(5.0)
    final.channel_mlp_skip = _Scale(7.0)
    final.norm_fno = nn.Identity()
    final.norm_channel_mlp = nn.Identity()
    final.activation = _Scale(2.0)
    # 5*x -> first activation 10*x -> MLP 50*x + skip 7*x; no second activation.
    assert torch.equal(final(x), torch.full_like(x, 57.0))


def test_centered_spectral_modes_retain_known_dc_coefficient():
    layer = TuckerSpectralConv3d(1, 1, (2, 2, 2), rank_fraction=0.8)
    layer.contract_kept_modes = lambda kept: kept
    with torch.no_grad():
        layer.bias.zero_()
    constant = torch.ones(1, 1, 4, 4, 4)
    # After fftshift, DC is at (2,2,0), inside the centered 2x2 slice.
    assert torch.allclose(layer(constant), constant, atol=1e-6, rtol=0.0)


def test_tucker_spectral_convolution_has_paper_era_physical_bias():
    layer = TuckerSpectralConv3d(2, 3, (2, 2, 2), rank_fraction=0.8)
    assert layer.bias.shape == (1, 3, 1, 1, 1)


def test_epoch_50_step_schedule_halves_learning_rate_once():
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=2.5e-4)
    scheduler = build_paper_scheduler(optimizer, {"lr_step_size": 50, "lr_gamma": 0.5})
    for _ in range(49):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.25e-4)


def test_pressure_normalization_and_inverse_include_reference_epsilon():
    dataset = CarCFDPaperDataset.__new__(CarCFDPaperDataset)
    dataset.pressure_mean = -2.0
    dataset.pressure_std = 4.0
    dataset.pressure_eps = 1.0e-7
    pressure = torch.tensor([-2.0, 2.0, 6.0], dtype=torch.float64)
    normalized = dataset.normalize_pressure(pressure)
    expected = (pressure + 2.0) / 4.0000001
    assert torch.equal(normalized, expected)
    assert torch.allclose(dataset.de_normalize_pressure(normalized), pressure, atol=0.0, rtol=0.0)


def test_paper_decoder_forward_shape_and_batch_isolation():
    torch.manual_seed(3)
    model = CarCFDGINODecoder(_config(), _data_spec()).eval()
    first = _item(0.0)
    second = _item(1.0)
    batch = Batch.from_data_list([first, second])
    assert batch.latent_sdf.shape == (2, 4, 4, 4, 1)
    with torch.no_grad():
        combined = model(batch)
        first_alone = model(Batch.from_data_list([first]))
        second_alone = model(Batch.from_data_list([second]))
    assert combined.shape == (10, 1)
    assert torch.isfinite(combined).all()
    assert torch.allclose(combined[:5], first_alone, atol=1e-5, rtol=1e-5)
    assert torch.allclose(combined[5:], second_alone, atol=1e-5, rtol=1e-5)


def test_paper_decoder_external_query_chunks_match_forward():
    model = CarCFDGINODecoder(_config(gino_query_chunk_size=0), _data_spec()).eval()
    batch = Batch.from_data_list([_item(0.0), _item(1.0)])
    with torch.no_grad():
        encoded = model.encode_operator(batch)
        full = model(batch)
        pieces = [
            model.decode_queries(encoded, batch, 0, 3),
            model.decode_queries(encoded, batch, 3, 7),
            model.decode_queries(encoded, batch, 7, 10),
        ]
    assert torch.allclose(full, torch.cat(pieces), atol=1e-5, rtol=1e-5)


def test_paper_decoder_coverage_is_output_only():
    model = CarCFDGINODecoder(_config(), _data_spec()).eval()
    report = model.coverage_preflight(Batch.from_data_list([_item()]))
    assert report["reports"][0]["output_gno"]["empty_fraction"] == 0.0
    assert "input_gno" not in report["reports"][0]


def test_paper_decoder_rejects_missing_or_mismatched_sdf():
    model = CarCFDGINODecoder(_config(), _data_spec())
    missing = SimpleNamespace()
    with pytest.raises(ValueError, match="latent_sdf"):
        model._latent_sdf(missing)
    mismatched = SimpleNamespace(latent_sdf=torch.zeros(1, 3, 3, 3, 1))
    with pytest.raises(ValueError, match="shape"):
        model._latent_sdf(mismatched)


def test_factory_dispatch_is_opt_in_and_model_split_is_rejected():
    from model.gino import MeshGINO

    assert _resolve_core_class("gino", {"gino_variant": "mesh_state"}) is MeshGINO
    assert _resolve_core_class("gino", {"gino_variant": "paper_decoder"}) is CarCFDGINODecoder
    validate_common_config(
        {"model": "gino", "mode": "train", "parallel_mode": "ddp", **_config()}
    )
    with pytest.raises(ValueError, match="paper_decoder.*model-split"):
        validate_common_config(
            {
                "model": "gino",
                "mode": "train",
                "parallel_mode": "model_split",
                **_config(),
            }
        )
