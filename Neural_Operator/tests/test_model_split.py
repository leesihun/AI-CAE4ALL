"""parallel_mode=model_split: partitioner, stage pruning/state-dict key
partition, deterministic cross-stage noise, config gating, and a real
two-process gloo pipeline whose loss and per-stage gradients must match a
single-process reference model bit-for-bit-seeded (2026-07-17).
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data

from general_modules.config_validation import validate_common_config
from general_modules.data_spec import DataSpec
from model.adapters.coordinate_domain import CoordinateDomain
from model.factory import MODEL_REGISTRY
from model.operator_wrapper import OperatorWrapper
from parallelism.partition import partition_stages
from parallelism.stages import (
    apply_noise_to_input,
    apply_noise_to_target,
    build_split_stage,
    pipeline_noise_tensor,
    run_stage_step,
)

SEED = 42


def _make_spec_domain():
    spec = DataSpec(input_var=4, output_var=4, positional_dim=4, node_type_dim=4,
                    global_condition_dim=0, operator_dim=2, active_axes=(0, 1),
                    has_sdf=False, has_integration_weights=False, num_timesteps=1)
    domain = CoordinateDomain(active_axes=(0, 1),
                              grid_bound_min=torch.zeros(2),
                              grid_bound_max=torch.ones(2),
                              out_of_bounds_policy='clamp')
    return spec, domain


def _make_graph(n=120, graph_seed=0):
    g = torch.Generator().manual_seed(graph_seed)
    pos = torch.rand(n, 3, generator=g)
    pos[:, 2] = 0.0
    x = torch.randn(n, 12, generator=g) * 0.1
    y = torch.randn(n, 4, generator=g)
    d = Data(x=x, y=y, pos_normalized=pos)
    d.batch = torch.zeros(n, dtype=torch.long)
    d.ptr = torch.tensor([0, n])
    return d


def _fno_cfg(**overrides):
    cfg = {'model': 'fno', 'output_var': 4, 'std_noise': 0.0, 'split_seed': SEED,
           'fno_grid_resolution': [8, 8], 'fno_modes': [3, 3],
           'fno_hidden_channels': 16, 'fno_layers': 2}
    cfg.update(overrides)
    return cfg


def _gino_cfg(**overrides):
    cfg = {'model': 'gino', 'output_var': 4, 'std_noise': 0.0, 'split_seed': SEED,
           'gino_grid_resolution': [6, 6], 'gino_fno_modes': [2, 3],
           'gino_fno_hidden_channels': 12, 'gino_fno_layers': 2,
           'gino_in_radius': 0.35, 'gino_out_radius': 0.35}
    cfg.update(overrides)
    return cfg


def test_partition_stages_minmax():
    # Heavy entry/exit, light middles (the GINO shape): the partitioner must
    # not lump both heavy blocks onto one stage.
    assignment = partition_stages([100.0, 1.0, 1.0, 100.0], 2)
    assert assignment == [[0, 1], [2, 3]] or assignment == [[0, 1, 2], [3]]
    stage_costs = [sum([100.0, 1.0, 1.0, 100.0][b] for b in blocks) for blocks in assignment]
    assert max(stage_costs) < 202.0  # never everything on one stage


@pytest.mark.parametrize('model_name', ['fno', 'gino'])
def test_stage_pruning_partitions_state_dict(model_name):
    """Union of the stages' state-dict keys == the full wrapper's keys, with
    no overlap: the rank-0 merge reconstructs exactly the single-GPU model."""
    spec, domain = _make_spec_domain()
    cfg = _fno_cfg() if model_name == 'fno' else _gino_cfg()

    torch.manual_seed(SEED)
    core = MODEL_REGISTRY[model_name](cfg, spec, domain)
    full_keys = set(OperatorWrapper(core, cfg).state_dict().keys())

    assignment = [[0, 1], [2, 3]]  # 2 layers -> 4 blocks
    stage_keys = []
    for stage_idx in range(2):
        stage = build_split_stage(cfg, spec, domain, stage_idx, 2, assignment)
        stage_keys.append(set(stage.state_dict().keys()))

    assert stage_keys[0] & stage_keys[1] == set()
    assert stage_keys[0] | stage_keys[1] == full_keys


def test_stage_weights_match_reference_init():
    """Seeded stage construction reproduces the reference model's weights."""
    spec, domain = _make_spec_domain()
    cfg = _fno_cfg()
    torch.manual_seed(SEED)
    ref = OperatorWrapper(MODEL_REGISTRY['fno'](cfg, spec, domain), cfg)
    ref_sd = ref.state_dict()
    stage = build_split_stage(cfg, spec, domain, 0, 2, [[0, 1], [2, 3]])
    for k, v in stage.state_dict().items():
        assert torch.equal(v, ref_sd[k]), f"stage weight {k} differs from reference"


def test_pipeline_noise_is_deterministic_across_stages():
    cfg = _fno_cfg(std_noise=0.5, noise_gamma=1, noise_std_ratio=[1.0, 1.0, 1.0, 1.0])
    n1 = pipeline_noise_tensor(cfg, 50, epoch=3, batch_idx=7, device='cpu', dtype=torch.float32)
    n2 = pipeline_noise_tensor(cfg, 50, epoch=3, batch_idx=7, device='cpu', dtype=torch.float32)
    n3 = pipeline_noise_tensor(cfg, 50, epoch=3, batch_idx=8, device='cpu', dtype=torch.float32)
    assert torch.equal(n1, n2)          # same (epoch, batch) -> identical noise
    assert not torch.equal(n1, n3)      # different batch -> different noise

    # x-perturbation on one graph copy and y-correction on an independent copy
    # must be mutually consistent (the single-wrapper contract, section 4.6).
    g_first = _make_graph(50)
    g_last = _make_graph(50)
    apply_noise_to_input(g_first, cfg, epoch=3, batch_idx=7)
    apply_noise_to_target(g_last, cfg, epoch=3, batch_idx=7)
    ref = _make_graph(50)
    added_noise = g_first.x[:, :4] - ref.x[:, :4]
    expected_y = ref.y - added_noise * torch.ones(4)
    assert torch.allclose(g_last.y, expected_y, atol=1e-6)


def test_model_split_config_gating():
    base = {'model': 'point_deeponet', 'mode': 'train', 'parallel_mode': 'model_split'}
    with pytest.raises(ValueError, match="model_split supports only"):
        validate_common_config(base)
    with pytest.raises(ValueError, match="augment_geometry"):
        validate_common_config({'model': 'fno', 'mode': 'train',
                                'parallel_mode': 'model_split', 'augment_geometry': True})
    validate_common_config({'model': 'fno', 'mode': 'train',
                            'parallel_mode': 'model_split', 'pipeline_microbatches': 4})
    validate_common_config({'model': 'gino', 'mode': 'train', 'parallel_mode': 'model_split'})


def _pipeline_worker(rank, world_size, port, model_name):
    """Two-stage CPU/gloo pipeline; each rank asserts its stage's loss/grads
    against an in-process single-model reference built with the same seed."""
    import torch.distributed as dist
    from parallelism.comm import drain_pending_sends, set_pipeline_process_groups

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    try:
        pg_data = dist.new_group(list(range(world_size)))
        pg_grad = dist.new_group(list(range(world_size)))
        set_pipeline_process_groups(pg_data, pg_grad)

        spec, domain = _make_spec_domain()
        cfg = _fno_cfg() if model_name == 'fno' else _gino_cfg()
        assignment = [[0, 1], [2, 3]]
        device = torch.device('cpu')

        stage = build_split_stage(cfg, spec, domain, rank, world_size, assignment)
        stage.train()

        graph = _make_graph(120)
        out, batch_loss_sum, batch_count = run_stage_step(
            stage, graph if (stage.is_first or stage.is_last) else None,
            cfg, device, epoch=0, batch_idx=0,
        )
        out.backward()
        drain_pending_sends()

        # Reference: the full single-process model, same seed, same graph.
        torch.manual_seed(SEED)
        ref_core = MODEL_REGISTRY[model_name](cfg, spec, domain)
        ref = OperatorWrapper(ref_core, cfg)
        ref.train()
        ref_graph = _make_graph(120)
        pred, target = ref(ref_graph, add_noise=False)
        errors = torch.nn.functional.mse_loss(pred, target, reduction='none')
        ref_loss = errors.mean(dim=-1).mean()
        ref_loss.backward()

        if stage.is_last:
            pipeline_loss = batch_loss_sum.item() / batch_count
            assert abs(pipeline_loss - ref_loss.item()) < 1e-5, (
                f"pipeline loss {pipeline_loss} != reference {ref_loss.item()}"
            )

        ref_grads = {k: p.grad for k, p in ref.named_parameters()}
        for k, p in stage.named_parameters():
            assert p.grad is not None, f"stage {rank}: no grad for {k}"
            assert torch.allclose(p.grad, ref_grads[k], atol=1e-5), (
                f"stage {rank}: grad mismatch for {k} "
                f"(max diff {(p.grad - ref_grads[k]).abs().max().item():.2e})"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('model_name', ['fno', 'gino'])
def test_two_stage_pipeline_matches_single_model(model_name):
    """End-to-end: forward+backward through a real 2-process gloo pipeline
    reproduces the single-model loss and every parameter gradient."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    mp.spawn(_pipeline_worker, args=(2, port, model_name), nprocs=2, join=True)
