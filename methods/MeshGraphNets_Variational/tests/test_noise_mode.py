"""stochastic_mode noise: z ~ N(0, I) in training and inference, K-draw CRPS loss.

What must hold:

  1. the default is the old VAE (same modules); noise builds no posterior
     encoder and no prior, and the key persists in the saved model_config;
  2. z is drawn fresh every forward in the latent_type layout, in training
     and in eval, so two forwards differ;
  3. the training loss is the fair CRPS of noise_draws forwards, its gradient
     reaches the simulator, and the reg/aux/prior terms are zero;
  4. validation scores N(0, I) draws with the CRPS (best_by crps works);
  5. a checkpoint round-trips; noise_draws < 2 and posterior-only schedules
     are refused;
  6. noise_loss energy scores each graph's whole field as one vector: it is
     the CRPS for a one-node graph, separates joint structure the per-node
     CRPS cannot, stays finite for identical draws, and noise_loss crps
     (the default) is the old loss.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from model.latent import Latent                                     # noqa: E402
from model.MeshGraphNets import MeshGraphNets                       # noqa: E402
from training_profiles.setup import build_model_config              # noqa: E402
from training_profiles.training_loop import (                       # noqa: E402
    _build_loss_weights, _crps_from_samples, _energy_score_from_samples,
    _ObjectiveTerms, _compose_loss,
    evaluate_vae_learned_prior_epoch, train_epoch,
)

from test_coarse_prior_variants import _build, _jitter              # noqa: E402
from test_latent_type import _write_grid                            # noqa: E402

LATENTS = ["global", "hybrid", "local"]


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("noise_mode") / "grid.h5"
    _write_grid(path)
    return path


def _noise(dataset_path, latent_type="hybrid", **kw):
    return _build(dataset_path, latent_type, stochastic_mode="noise", noise_draws=3, **kw)


def _terms(cfg, model):
    return _ObjectiveTerms(cfg, model, False, torch.float32,
                           _build_loss_weights(cfg, torch.device("cpu")))


# -- build --------------------------------------------------------------------

def test_default_is_vae(dataset_path):
    cfg, model, _ = _build(dataset_path)
    assert not model.noise_mode and model.model.vae_encoder is not None
    assert build_model_config(cfg)["stochastic_mode"] == "vae"


@pytest.mark.parametrize("latent_type", LATENTS)
def test_noise_builds_no_encoder_and_no_prior(dataset_path, latent_type):
    cfg, model, _ = _noise(dataset_path, latent_type)
    assert model.noise_mode and model.model.noise_mode
    assert model.model.vae_encoder is None
    assert getattr(model, "prior", None) is None
    assert not any(k.startswith(("prior.", "model.vae_encoder.")) for k in model.state_dict())
    saved = build_model_config(cfg)
    assert saved["stochastic_mode"] == "noise"
    with pytest.raises(RuntimeError, match="no posterior encoder"):
        model.model.posterior(None)


def test_bad_mode_is_rejected(dataset_path):
    with pytest.raises(ValueError, match="stochastic_mode must be one of"):
        _build(dataset_path, stochastic_mode="gan")


# -- sampling -----------------------------------------------------------------

@pytest.mark.parametrize("latent_type", LATENTS)
@pytest.mark.parametrize("training", [True, False])
def test_each_forward_draws_a_new_z(dataset_path, latent_type, training):
    _, model, loader = _noise(dataset_path, latent_type)
    _jitter(model)   # AdaLN-Zero gates start at zero: z would have no effect
    model.train(training)
    graph = next(iter(loader))
    with torch.no_grad():
        a, *_ = model(graph.clone(), add_noise=False)
        b, *_ = model(graph.clone(), add_noise=False)
    assert torch.isfinite(a).all() and not torch.allclose(a, b)
    z = model.model.standard_normal_z(graph.clone())
    if latent_type == "global":
        assert torch.is_tensor(z)
    else:
        assert isinstance(z, Latent) and (z.g is None) == (latent_type == "local")


# -- objective ------------------------------------------------------------------

@pytest.mark.parametrize("latent_type", LATENTS)
def test_loss_is_the_crps_of_k_draws_and_reaches_the_simulator(dataset_path, latent_type):
    cfg, model, loader = _noise(dataset_path, latent_type)
    _jitter(model)
    terms = _terms(cfg, model)
    assert terms.noise_mode and terms.noise_draws == 3
    graph = next(iter(loader))
    calls = []
    real = model.forward

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(kwargs.get("add_noise"))
        return out

    model.forward = spy
    try:
        out = _compose_loss(model, graph.clone(), terms)
    finally:
        del model.forward
    pred, loss, crps_sum, *rest = out
    assert len(calls) == 3 and calls[0] is None and calls[1:] == [False, False]
    assert torch.isfinite(loss) and loss > 0
    assert all(float(t) == 0.0 for t in rest)
    loss.backward()
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads and any(g.abs().max() > 0 for g in grads)


def test_crps_rewards_spread_that_matches(dataset_path):
    """The spread term is what a deterministic loss lacks: a sharp but wrong
    ensemble scores worse than one that brackets the target."""
    y = torch.zeros(4, 1)
    sharp = torch.full((4, 4, 1), 1.0)
    spread = torch.tensor([-1.5, -0.5, 0.5, 1.5]).view(4, 1, 1).expand(4, 4, 1)
    assert _crps_from_samples(spread, y).mean() < _crps_from_samples(sharp, y).mean()


@pytest.mark.parametrize("latent_type", ["global", "hybrid"])
def test_checkpointing_gives_the_same_loss_and_gradients(dataset_path, latent_type):
    """Several jobs per card turn use_checkpointing on: z is drawn outside the
    checkpointed blocks, so recomputing a block must see the same draw."""
    results = []
    for ckpt in (False, True):
        cfg, model, loader = _noise(dataset_path, latent_type, use_checkpointing=ckpt)
        _jitter(model)
        model.train()
        graph = next(iter(loader))
        torch.manual_seed(7)
        _, loss, *_ = _compose_loss(model, graph.clone(), _terms(cfg, model))
        loss.backward()
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (l0, g0), (l1, g1) = results
    assert torch.allclose(l0, l1, rtol=1e-5, atol=1e-6) and len(g0) == len(g1)
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(g0, g1))


def test_too_few_draws_is_refused(dataset_path):
    cfg, model, _ = _build(dataset_path, stochastic_mode="noise", noise_draws=1)
    with pytest.raises(ValueError, match="noise_draws must be >= 2"):
        _terms(cfg, model)


# -- the whole recipe -------------------------------------------------------------

@pytest.mark.parametrize("latent_type", ["global", "hybrid"])
def test_epoch_validation_and_checkpoint_round_trip(dataset_path, latent_type):
    cfg, model, loader = _noise(dataset_path, latent_type, vae_valid_prior_samples=3)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = train_epoch(model, loader, optimizer, device, cfg, 0)
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))

    val = evaluate_vae_learned_prior_epoch(model, loader, device, cfg, 0)
    assert val is not None and np.isfinite(val["mean"]) and np.isfinite(val["crps"])

    torch.manual_seed(1)
    twin = MeshGraphNets(dict(cfg, **build_model_config(cfg)), "cpu")
    assert twin.noise_mode
    twin.load_state_dict(model.state_dict())
    graph = next(iter(loader))
    model.eval(), twin.eval()
    z = model.model.standard_normal_z(graph.clone())
    with torch.no_grad():
        a, *_ = model(graph.clone(), fixed_z=z)
        b, *_ = twin(graph.clone(), fixed_z=z)
    assert torch.equal(a, b)


# -- noise_loss energy ------------------------------------------------------------

def test_energy_score_is_the_crps_on_a_one_node_graph():
    """One node, one feature: the norm is |.|, so the two fair estimators agree."""
    torch.manual_seed(0)
    samples, y = torch.randn(5, 1, 1), torch.randn(1, 1)
    energy = _energy_score_from_samples(samples, y, torch.zeros(1, dtype=torch.long), 1)
    assert torch.allclose(energy, _crps_from_samples(samples, y).view(1), atol=1e-5)


def test_energy_score_sees_joint_structure_the_crps_cannot():
    """Truth: both nodes move together (+a, +a) or (-a, -a). A generator with the
    right marginals but independent nodes has the same per-node CRPS and a
    worse energy score."""
    gen = torch.Generator().manual_seed(0)
    S, T = 8, 4000
    batch = torch.zeros(2, dtype=torch.long)
    sign = lambda *shape: torch.randint(0, 2, shape, generator=gen).float() * 2 - 1   # noqa: E731
    scores = {}
    for name in ("joint", "independent"):
        crps = energy = 0.0
        for _ in range(T):
            y = sign(1).expand(2).reshape(2, 1)
            s = sign(S, 1, 1).expand(S, 2, 1) if name == "joint" else sign(S, 2, 1)
            crps += float(_crps_from_samples(s, y).mean())
            energy += float(_energy_score_from_samples(s, y, batch, 1))
        scores[name] = (crps / T, energy / T)
    assert abs(scores["joint"][0] - scores["independent"][0]) < 0.03
    assert scores["independent"][1] > scores["joint"][1] + 0.05


def test_energy_score_is_per_graph_and_size_free():
    """Two graphs in a batch score separately, and a graph repeated node for
    node scores the same (RMS, not sum, over nodes)."""
    torch.manual_seed(1)
    s, y = torch.randn(4, 3, 2), torch.randn(3, 2)
    one = _energy_score_from_samples(s, y, torch.zeros(3, dtype=torch.long), 1)
    twice = _energy_score_from_samples(torch.cat([s, s], 1), torch.cat([y, y]),
                                       torch.zeros(6, dtype=torch.long), 1)
    both = _energy_score_from_samples(torch.cat([s, 2 * s], 1), torch.cat([y, y]),
                                      torch.tensor([0, 0, 0, 1, 1, 1]), 2)
    assert torch.allclose(one, twice, atol=1e-6)
    assert torch.allclose(both[0], one[0], atol=1e-6) and not torch.allclose(both[1], one[0])


def test_energy_score_has_a_finite_gradient_for_identical_draws():
    s = torch.ones(3, 4, 2, requires_grad=True)
    _energy_score_from_samples(s, torch.ones(4, 2), torch.zeros(4, dtype=torch.long), 1).sum().backward()
    assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize("latent_type", ["global", "hybrid"])
def test_energy_loss_trains_and_crps_stays_the_default(dataset_path, latent_type):
    cfg, model, loader = _noise(dataset_path, latent_type, noise_loss="energy")
    _jitter(model)
    terms = _terms(cfg, model)
    assert terms.noise_loss == "energy" and build_model_config(cfg)["noise_loss"] == "energy"
    graph = next(iter(loader))
    pred, loss, score_sum, *rest = _compose_loss(model, graph.clone(), terms)
    assert torch.isfinite(loss) and all(float(t) == 0.0 for t in rest)
    assert torch.allclose(score_sum / pred.shape[0], loss.detach(), rtol=1e-5)
    loss.backward()
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads and any(g.abs().max() > 0 for g in grads)

    # the same draws under noise_loss crps (default) give the old loss
    cfg2, model2, _ = _noise(dataset_path, latent_type)
    model2.load_state_dict(model.state_dict())
    terms2 = _terms(cfg2, model2)
    assert terms2.noise_loss == "crps"
    torch.manual_seed(3)
    _, crps_loss, *_ = _compose_loss(model2, graph.clone(), terms2)
    torch.manual_seed(3)
    _, energy_loss, *_ = _compose_loss(model, graph.clone(), terms)
    assert not torch.allclose(crps_loss, energy_loss)

    metrics = train_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=1e-3),
                          torch.device("cpu"), cfg, 0)
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
    val = evaluate_vae_learned_prior_epoch(model, loader, torch.device("cpu"), cfg, 0)
    assert val is not None and np.isfinite(val["crps"])


def test_bad_noise_loss_is_refused(dataset_path):
    cfg, model, _ = _noise(dataset_path, noise_loss="mse")
    with pytest.raises(ValueError, match="noise_loss must be one of"):
        _terms(cfg, model)
