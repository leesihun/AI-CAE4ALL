"""prior_trunk: where the coarse prior's per-coarse-node condition comes from.

  fine        (default) the prior's own blocks on the fine mesh + cluster mean
  multiscale  the prior's own descending arm
  decoder     the simulator's descending arm at z = 0, read without gradient

What must hold:

  1. the default is the old prior: same modules, same parameter names;
  2. every trunk trains and samples the declared shapes, with either velocity
     network and either latent placement;
  3. `multiscale` sees past its own cluster where `fine` cannot;
  4. `decoder` owns no trunk and none of the simulator's weights, its loss
     gives the simulator no gradient through the condition, and what it reads
     depends on neither the target nor the latent;
  5. an EMA copy's prior reads the EMA copy's simulator, not the live one;
  6. the full recipe (joint epoch, learned-prior validation, prior-only tail)
     runs, and a checkpoint round-trips.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from model.conditional_prior import PRIOR_TRUNKS                    # noqa: E402
from model.latent import Latent                                     # noqa: E402
from model.MeshGraphNets import MeshGraphNets                       # noqa: E402
from training_profiles.setup import build_model_config              # noqa: E402
from training_profiles.training_loop import (                       # noqa: E402
    build_ema_model, evaluate_vae_learned_prior_epoch, freeze_for_prior_fit,
    train_epoch, train_prior_epoch,
)

from test_coarse_prior_variants import _build, _jitter              # noqa: E402
from test_latent_type import LATENT_CH, _write_grid                 # noqa: E402

TRUNK_MODULES = ("node_encoder", "edge_encoder", "mp_layers")


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("prior_trunk") / "grid.h5"
    _write_grid(path)
    return path


def _prefixes(module):
    return {k.split(".")[0] for k in module.state_dict()}


# -- defaults ---------------------------------------------------------------

def test_default_trunk_is_fine_with_the_old_modules(dataset_path):
    cfg, model, _ = _build(dataset_path)
    _, explicit, _ = _build(dataset_path, prior_trunk="fine")
    assert PRIOR_TRUNKS[0] == "fine" and model.prior.trunk == "fine"
    names = _prefixes(model.prior)
    assert set(TRUNK_MODULES) <= names
    assert not names & {"ms_edge_encoders", "ms_layers", "geom_encoder"}
    # same names and, built from the same seed, the same initial weights
    a, b = model.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert build_model_config(cfg)["prior_trunk"] == "fine"


def test_bad_trunk_is_rejected(dataset_path):
    with pytest.raises(ValueError, match="prior_trunk must be one of"):
        _build(dataset_path, prior_trunk="coarse")


def test_global_latent_ignores_the_key(dataset_path):
    _, model, _ = _build(dataset_path, latent_type="global", prior_trunk="decoder")
    assert not hasattr(model.prior, "trunk")


# -- every trunk trains and samples -------------------------------------------

@pytest.mark.parametrize("trunk", ["multiscale", "decoder"])
@pytest.mark.parametrize("arch", ["gnn", "transformer"])
@pytest.mark.parametrize("latent_type", ["local", "hybrid"])
def test_trunk_trains_and_samples_the_declared_shapes(dataset_path, trunk, arch, latent_type):
    cfg, model, loader = _build(dataset_path, latent_type, prior_trunk=trunk,
                                prior_arch=arch, prior_tf_heads=2, prior_fm_draws=2)
    prior = model.prior
    assert prior.trunk == trunk and build_model_config(cfg)["prior_trunk"] == trunk
    _jitter(prior)   # the transformer's zero-initialized heads pass no gradient back
    graph = next(iter(loader))
    with torch.no_grad():
        z, mu, logvar = model.model.posterior(graph.clone())
    cond = prior.condition(graph.clone())
    assert cond["x"].shape == (z.l.shape[0], prior.hidden_dim)
    if "pooled" in cond:
        assert cond["pooled"].shape == (3, prior.hidden_dim)
    loss = prior.fm_loss(cond, z, target_mu=mu, target_logvar=logvar)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    own = "ms_layers" if trunk == "multiscale" else "geom_encoder"
    grads = [p.grad for n, p in prior.named_parameters() if n.startswith(own)]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)

    draws = prior.sample_n(graph.clone(), 2)
    assert draws.l.shape == (z.l.shape[0], 2, LATENT_CH)
    assert (draws.g is None) == (latent_type == "local")


# -- multiscale ---------------------------------------------------------------

def test_multiscale_builds_one_stage_per_level_beside_the_fine_blocks(dataset_path):
    _, model, _ = _build(dataset_path, prior_trunk="multiscale")
    prior = model.prior
    assert set(TRUNK_MODULES) <= _prefixes(prior)
    assert len(prior.ms_edge_encoders) == len(prior.ms_layers) == prior.levels
    assert all(len(stage) == prior.num_mp_layers for stage in prior.ms_layers)


@pytest.mark.parametrize("trunk,reaches", [("fine", False), ("multiscale", True)])
def test_only_multiscale_carries_a_far_node_into_a_coarse_node(dataset_path, trunk, reaches):
    """Change the input of the fine nodes of one coarse cluster. With one
    fine block, `fine` moves only the clusters within one fine hop; the
    coarse-level blocks of `multiscale` carry it further."""
    _, model, loader = _build(dataset_path, prior_trunk=trunk, prior_mp_layers=1)
    prior = model.prior.eval()
    _jitter(prior)
    graph = next(iter(loader))
    with torch.no_grad():
        base = prior.condition(graph.clone())
        from model.latent import coarsest_view
        cv = coarsest_view(graph, prior.levels)
        first = torch.nonzero(cv["batch"] == 0).flatten()
        src = int(first[0])
        moved_nodes = cv["parent"] == src
        # clusters within one fine hop of the changed nodes
        ei = graph.edge_index
        touched = torch.zeros(cv["n_c"], dtype=torch.bool)
        touched[src] = True
        hit = moved_nodes[ei[0]] | moved_nodes[ei[1]]
        touched[cv["parent"][ei[0][hit]]] = True
        touched[cv["parent"][ei[1][hit]]] = True
        changed = graph.clone()
        changed.x = changed.x.clone()
        changed.x[moved_nodes] += 1.0
        out = prior.condition(changed)
    same_graph = cv["batch"] == 0
    far = same_graph & ~touched
    assert far.any(), "the toy grid has no cluster beyond one fine hop"
    delta = (out["x"] - base["x"]).abs().amax(dim=1)
    assert delta[src] > 0
    assert bool(delta[far].max() > 1e-6) is reaches
    # never across graphs
    assert delta[~same_graph].max() == 0


# -- decoder -----------------------------------------------------------------

def test_decoder_trunk_owns_no_trunk_and_no_simulator_weight(dataset_path):
    _, model, _ = _build(dataset_path, prior_trunk="decoder")
    prior = model.prior
    names = _prefixes(prior)
    assert "geom_encoder" in names and not names & set(TRUNK_MODULES)
    sim_ids = {id(p) for p in model.model.parameters()}
    assert not sim_ids & {id(p) for p in prior.parameters()}
    assert not any("_geometry_fn" in k or k.startswith("model.") for k in prior.state_dict())
    # the same optimizer split the trainer makes still covers every parameter once
    prior_ids = {id(p) for p in prior.parameters()}
    assert len(prior_ids) + len(sim_ids) == len(list(model.parameters()))


def test_decoder_trunk_needs_a_bound_simulator(dataset_path):
    from model.conditional_prior import build_conditional_prior
    cfg, _, loader = _build(dataset_path, prior_trunk="decoder")
    lone = build_conditional_prior(cfg)
    with pytest.raises(RuntimeError, match="no simulator is bound"):
        lone.condition(next(iter(loader)))


@pytest.mark.parametrize("latent_type", ["local", "hybrid"])
@pytest.mark.parametrize("inject", ["all", "coarsest"])
def test_geometry_features_ignore_the_target_and_take_no_gradient(dataset_path, latent_type, inject):
    _, model, loader = _build(dataset_path, latent_type, prior_trunk="decoder",
                              local_inject=inject)
    _jitter(model)
    graph = next(iter(loader))
    feats = model.model.geometry_features(graph.clone())
    assert feats.shape[1] == model.model.latent_dim and not feats.requires_grad
    assert torch.isfinite(feats).all() and feats.abs().max() > 0

    other = graph.clone()
    other.y = other.y + 3.0
    assert torch.equal(model.model.geometry_features(other), feats)
    # train / eval do not change it: nothing on the path is stochastic
    model.eval()
    assert torch.equal(model.model.geometry_features(graph.clone()), feats)
    model.train()

    # it is the descending arm of a real decode at z = 0
    sim = model.model
    seen = {}
    real = sim._run_processor_blocks

    def spy(blocks, level_graph, fusers, z_per_node):
        if blocks is sim.coarsest_blocks:
            seen["x"] = level_graph.x.detach().clone()
        return real(blocks, level_graph, fusers, z_per_node)

    sim._run_processor_blocks = spy
    try:
        with torch.no_grad():
            z0 = Latent(None if latent_type == "local" else
                        torch.zeros(3, sim.num_z, sim.vae_latent_dim),
                        torch.zeros(feats.shape[0], LATENT_CH))
            sim(graph.clone(), use_posterior=False, fixed_z=z0)
    finally:
        del sim._run_processor_blocks
    assert torch.allclose(seen["x"], feats, atol=1e-6)

    # the prior's loss reaches the prior, not the simulator
    with torch.no_grad():
        z, mu, logvar = sim.posterior(graph.clone())
    model.zero_grad()
    model.prior.fm_loss(model.prior.condition(graph.clone()), z,
                        target_mu=mu, target_logvar=logvar).backward()
    assert all(p.grad is None for p in sim.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0
               for p in model.prior.geom_encoder.parameters())


def test_ema_copy_reads_its_own_simulator(dataset_path):
    cfg, model, loader = _build(dataset_path, prior_trunk="decoder")
    ema = build_ema_model(model, cfg)
    copy = ema.module if hasattr(ema, "module") else ema
    assert copy.prior._geometry_fn.__self__ is copy.model
    assert copy.prior._geometry_fn.__self__ is not model.model
    # moving the live simulator leaves the copy's condition where it was
    graph = next(iter(loader))
    with torch.no_grad():
        before = copy.prior.condition(graph.clone())["x"].clone()
        live_before = model.prior.condition(graph.clone())["x"].clone()
    _jitter(model.model)
    with torch.no_grad():
        assert torch.equal(copy.prior.condition(graph.clone())["x"], before)
        assert not torch.equal(model.prior.condition(graph.clone())["x"], live_before)


# -- the whole recipe ----------------------------------------------------------

@pytest.mark.parametrize("trunk,arch", [("decoder", "gnn"), ("multiscale", "gnn"),
                                        ("decoder", "transformer")])
def test_full_recipe_runs_and_a_checkpoint_round_trips(dataset_path, trunk, arch):
    extra = dict(prior_arch="transformer", prior_tf_heads=2, prior_fm_draws=2,
                 local_inject="coarsest", lambda_mmd_g=0.0) if arch == "transformer" else {}
    cfg, model, loader = _build(dataset_path, prior_trunk=trunk,
                                prior_grad_to_encoder=1.0, **extra)
    ema = build_ema_model(model, cfg)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in model.prior.parameters()]
    joint = train_epoch(model, loader, optimizer, device, cfg, 0, ema_model=ema)
    assert all(np.isfinite(v) for v in joint.values() if isinstance(v, (int, float)))
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.prior.parameters()))

    val = evaluate_vae_learned_prior_epoch(model, loader, device, cfg, 0)
    assert np.isfinite(val["mean"])

    # a fresh model built from the saved config loads the weights and agrees
    torch.manual_seed(1)
    twin = MeshGraphNets(dict(cfg, **build_model_config(cfg)), "cpu")
    twin.load_state_dict(model.state_dict())
    graph = next(iter(loader))
    model.eval(), twin.eval()
    with torch.no_grad():
        assert torch.equal(twin.prior.condition(graph.clone())["x"],
                           model.prior.condition(graph.clone())["x"])
    model.train()

    optimizer, _ = freeze_for_prior_fit(model, ema, loader, device, cfg, remaining_epochs=1)
    tail = train_prior_epoch(model, loader, optimizer, device, cfg, 1, ema_model=ema)
    assert np.isfinite(tail["prior_loss_mean"]) and tail["prior_loss_mean"] > 0
    assert isinstance(model.prior.sample(next(iter(loader)).clone()), Latent)
