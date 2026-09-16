"""Focused contracts for the upgraded conditional flow-matching prior."""

from pathlib import Path
import math
import sys

import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from model.conditional_prior import ConditionalFMPrior, build_prior_config  # noqa: E402
from training_profiles.training_loop import freeze_moments_for_residual_fit  # noqa: E402


def _config(**overrides):
    cfg = {
        'input_var': 3,
        'cond_var': 2,
        'edge_var': 8,
        'latent_dim': 16,
        'vae_latent_dim': 4,
        'positional_features': 0,
        'use_node_types': False,
        'use_multiscale': True,
        'multiscale_levels': 1,
        'num_z': 2,
        'prior_family': 'fm',
        'prior_hidden_dim': 16,
        'prior_condition_hidden_dim': 16,
        'prior_velocity_hidden_dim': 24,
        'prior_mp_layers': 1,
        'prior_fm_steps': 3,
        'prior_fm_solver': 'heun',
        'prior_fm_velocity_arch': 'residual_film',
        'prior_fm_blocks': 2,
        'prior_fm_moments': True,
        'prior_fm_moment_hidden_dim': 12,
        'prior_fm_moment_weight': 1.0,
    }
    cfg.update(overrides)
    return cfg


def test_build_prior_config_preserves_fm_v2_architecture_keys():
    cfg = build_prior_config(_config())
    assert cfg['prior_condition_hidden_dim'] == 16
    assert cfg['prior_velocity_hidden_dim'] == 24
    assert cfg['prior_fm_velocity_arch'] == 'residual_film'
    assert cfg['prior_fm_blocks'] == 2
    assert cfg['prior_fm_moments'] is True
    assert cfg['prior_fm_moment_hidden_dim'] == 12


def test_moment_head_starts_as_identity_and_maps_residuals_explicitly():
    prior = ConditionalFMPrior(_config())
    cond = torch.randn(3, prior.hidden_dim)
    mean, scale = prior.moment_parameters(cond)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(scale, torch.ones_like(scale))

    wanted_mean = torch.linspace(-1.0, 1.0, prior.flat_dim)
    wanted_scale = torch.linspace(0.5, 1.5, prior.flat_dim)
    with torch.no_grad():
        prior.moment_head[4].bias[:prior.flat_dim].copy_(wanted_mean)
        prior.moment_head[4].bias[prior.flat_dim:].copy_(wanted_scale.log())
    residual = torch.full((3, prior.flat_dim), 2.0)
    mapped = prior._flow_to_standardized_z(residual, cond)
    expected = wanted_mean + 2.0 * wanted_scale
    assert torch.allclose(mapped, expected.expand_as(mapped), atol=1e-6)


def test_residual_fm_and_moment_likelihood_train_their_own_parameters():
    torch.manual_seed(7)
    prior = ConditionalFMPrior(_config())
    cond = torch.randn(6, prior.hidden_dim)
    target_mu = torch.randn(6, prior.num_z, prior.z_dim) + 1.5
    target_logvar = torch.full_like(target_mu, math.log(0.4 ** 2))
    target = target_mu + torch.exp(0.5 * target_logvar) * torch.randn_like(target_mu)
    loss = prior.fm_loss(cond, target, target_mu=target_mu, target_logvar=target_logvar)
    loss.backward()
    assert torch.isfinite(loss)
    assert prior.moment_head[4].weight.grad is not None
    assert prior.moment_head[4].weight.grad.abs().sum() > 0
    assert prior.velocity_net.out.weight.grad is not None
    assert prior.velocity_net.out.weight.grad.abs().sum() > 0


def test_flow_loss_cannot_move_moment_coordinate_system():
    torch.manual_seed(9)
    prior = ConditionalFMPrior(_config(prior_fm_moment_weight=0.0))
    cond = torch.randn(5, prior.hidden_dim)
    target = torch.randn(5, prior.num_z, prior.z_dim)
    prior.fm_loss(cond, target).backward()
    grads = [p.grad for p in prior.moment_head.parameters()]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)
    assert prior.velocity_net.out.weight.grad.abs().sum() > 0


def test_legacy_defaults_keep_the_original_velocity_state_dict_layout():
    cfg = _config(
        prior_fm_velocity_arch='mlp',
        prior_fm_moments=False,
        prior_velocity_hidden_dim=16,
    )
    prior = ConditionalFMPrior(cfg)
    keys = set(prior.state_dict())
    assert 'velocity_net.0.weight' in keys
    assert 'velocity_net.4.weight' in keys
    assert not any(key.startswith('moment_head.') for key in keys)
    assert prior.z_shift.shape == (prior.flat_dim,)
    assert prior.z_scale.shape == (prior.flat_dim,)


def test_residual_stage_freezes_condition_and_moments():
    class Holder(torch.nn.Module):
        def __init__(self, prior):
            super().__init__()
            self.prior = prior

    prior = ConditionalFMPrior(_config())
    holder = Holder(prior)
    cfg = {'learningr': 1e-3, 'weight_decay': 0.0, 'warmup_epochs': 1}
    optimizer, _ = freeze_moments_for_residual_fit(holder, cfg, remaining_epochs=5)
    trainable = {id(p) for group in optimizer.param_groups for p in group['params']}
    velocity = {id(p) for p in prior.velocity_net.parameters()}
    assert trainable == velocity
    assert all(p.requires_grad for p in prior.velocity_net.parameters())
    assert all(not p.requires_grad for p in prior.moment_head.parameters())
    assert all(not p.requires_grad for p in prior.node_encoder.parameters())
