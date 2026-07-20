import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.config_validation import validate_common_config
from general_modules.data_spec import DataSpec
from model.adapters.coordinate_domain import CoordinateDomain
from model.fno import MeshFNO, validate_config
from training_profiles.setup import build_optimizer_scheduler
from training_profiles.training_loop import _paper_relative_l2_batch


def _data_spec():
    return DataSpec(
        input_var=1,
        output_var=1,
        positional_dim=0,
        node_type_dim=0,
        global_condition_dim=0,
        operator_dim=2,
        active_axes=(0, 1),
        has_sdf=False,
        has_integration_weights=False,
        num_timesteps=2,
    )


def _paper_config(**overrides):
    config = {
        'model': 'fno',
        'mode': 'train',
        'parallel_mode': 'ddp',
        'split_seed': 42,
        'input_var': 1,
        'output_var': 1,
        'positional_features': 0,
        'use_node_types': False,
        'sdf_source': 'none',
        'fno_variant': 'paper_darcy',
        'fno_grid_resolution': [85, 85],
        'fno_modes': [12, 12],
        'fno_hidden_channels': 32,
        'fno_layers': 4,
        'fno_use_channel_mlp': False,
        'fno_norm': 'none',
        'training_epochs': 500,
        'batch_size': 20,
        'learningr': 0.001,
        'weight_decay': 0.0001,
        'grad_accum_steps': 1,
        'std_noise': 0.0,
        'augment_geometry': False,
        'use_amp': False,
        'use_ema': False,
    }
    config.update(overrides)
    return config


def test_paper_variant_is_registered_and_fail_fast_validated():
    config = _paper_config()
    validate_common_config(config)
    validate_config(config, _data_spec())

    with pytest.raises(ValueError, match='fno_layers'):
        validate_config(_paper_config(fno_layers=3), _data_spec())


def test_paper_core_has_original_three_channel_shape_and_parameter_count():
    domain = CoordinateDomain(
        active_axes=(0, 1),
        grid_bound_min=torch.tensor([-1.0, -1.0]),
        grid_bound_max=torch.tensor([1.0, 1.0]),
    )
    core = MeshFNO(_paper_config(), _data_spec(), domain)

    assert core.variant == 'paper_darcy'
    assert core.in_channels == 3
    assert isinstance(core.activation, nn.ReLU)
    assert core.projection[0].in_channels == 32
    assert core.projection[0].out_channels == 128
    assert core.projection[-1].in_channels == 128
    assert core.projection[-1].out_channels == 1
    assert core._block.__func__ is MeshFNO._paper_darcy_block
    assert sum(parameter.numel() for parameter in core.parameters()) == 2_368_001

    axis = torch.linspace(-1.0, 1.0, 85)
    xx, yy = torch.meshgrid(axis, axis, indexing='ij')
    positions = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.zeros(85 * 85)], dim=1
    )
    coefficient = torch.linspace(-1.0, 1.0, 85 * 85).unsqueeze(1)
    graph = Data(x=coefficient, pos_normalized=positions)
    graph.batch = torch.zeros(85 * 85, dtype=torch.long)
    graph.ptr = torch.tensor([0, 85 * 85])
    paper_grid = core._assemble_grid(graph)
    assert paper_grid.shape == (1, 3, 85, 85)
    assert torch.allclose(paper_grid[0, 0], coefficient.reshape(85, 85), atol=1e-6)
    with torch.no_grad():
        prediction = core(graph)
    assert prediction.shape == (85 * 85, 1)
    assert torch.isfinite(prediction).all()


def test_paper_optimizer_and_scheduler_are_opt_in():
    model = nn.Linear(2, 1)
    optimizer, scheduler, warmup, step_size = build_optimizer_scheduler(
        _paper_config(), model.parameters(), total_epochs=500
    )
    assert type(optimizer) is torch.optim.Adam
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert warmup == 0
    assert step_size == 100
    assert scheduler.step_size == 100
    assert scheduler.gamma == pytest.approx(0.5)


def test_paper_loss_decodes_then_sums_per_sample_relative_l2():
    mean = torch.tensor([10.0])
    std = torch.tensor([2.0])
    target_normalized = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    prediction_normalized = target_normalized + torch.tensor([[1.0], [0.0], [0.0], [2.0]])
    ptr = torch.tensor([0, 2, 4])

    loss, detached_sum, count = _paper_relative_l2_batch(
        prediction_normalized, target_normalized, ptr, mean, std
    )
    target_physical = target_normalized * std + mean
    prediction_physical = prediction_normalized * std + mean
    expected = sum(
        torch.linalg.vector_norm(prediction_physical[start:end] - target_physical[start:end])
        / torch.linalg.vector_norm(target_physical[start:end])
        for start, end in ((0, 2), (2, 4))
    )
    assert count == 2
    assert torch.allclose(loss, expected)
    assert torch.allclose(detached_sum, expected)
