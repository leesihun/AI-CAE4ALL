"""The coarse-prior and latent-placement switches added for the hybrid prior.

  local_inject   all | coarsest   -- which V-cycle levels read z_l
  lambda_mmd_g   >= 0             -- weight of the z_g MMD (default lambda_mmd)
  prior_arch     gnn | transformer
  prior_param    velocity | x     -- what the prior's heads predict
  prior_fm_draws K >= 1           -- (z0, t) draws per posterior sample
  prior_cond_noise s >= 0         -- relative noise on the prior's context (training)
  prior_tf_dropout p in [0, 1)    -- the transformer prior's dropout (training)

What must hold:

  1. the defaults are the old model: same prior modules, same z_l broadcast,
     same summed MMD;
  2. each switch does exactly what it names (zeroed z_l columns, a scaled or
     absent z_g MMD, x -> v conversion, a permutation-equivariant transformer
     that sees the whole coarse graph);
  3. the full recipe (joint epoch, learned-prior validation, prior-only tail)
     runs with every switch on at once.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.mesh_dataset import MeshGraphDataset          # noqa: E402
from model.conditional_prior import CoarseFMPrior                   # noqa: E402
from model.latent import Latent, coarsest_parent_maps               # noqa: E402
from model.MeshGraphNets import MeshGraphNets                       # noqa: E402
from training_profiles.setup import build_model_config              # noqa: E402
from training_profiles.training_loop import (                       # noqa: E402
    build_ema_model, evaluate_vae_learned_prior_epoch, freeze_for_prior_fit,
    train_epoch, train_prior_epoch,
)
from torch_geometric.loader import DataLoader                       # noqa: E402

from test_latent_type import LATENT_CH, VAE_DIM, _config, _write_grid  # noqa: E402


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("coarse_prior_variants") / "grid.h5"
    _write_grid(path)
    return path


def _build(dataset_path, latent_type="hybrid", **overrides):
    cfg = _config(dataset_path, latent_type, "mmd")
    cfg.update(overrides)
    dataset = MeshGraphDataset(str(dataset_path), cfg)
    train, _, _ = dataset.split(0.5, 0.25, 0.25, seed=0)
    train.prepare_preprocessing()
    cfg["num_timesteps"] = train.num_timesteps
    loader = DataLoader(train, batch_size=3, shuffle=False)
    torch.manual_seed(0)
    model = MeshGraphNets(cfg, "cpu")
    return cfg, model, loader


def _jitter(module, scale=0.2, seed=1):
    """Random weights everywhere, so zero-initialized heads and gates do not
    make a property hold trivially."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))


# -- defaults ---------------------------------------------------------------

def test_defaults_build_the_gnn_prior_with_its_old_modules(dataset_path):
    cfg, model, _ = _build(dataset_path)
    prior = model.prior
    assert isinstance(prior, CoarseFMPrior)
    assert (prior.arch, prior.param, prior.fm_draws) == ("gnn", "velocity", 1)
    keys = set(prior.state_dict())
    for name in ("coarse_edge_encoder", "in_proj", "blocks", "time_mod", "out_head",
                 "g_in", "g_update", "g_out", "g_head", "pool"):
        assert any(k.startswith(name + ".") for k in keys), name
    assert not any(k.startswith(("tf_blocks", "l_in", "cond_mlp")) for k in keys)
    assert model.model.local_inject == "all" and model.model.mmd_g_scale == 1.0

    saved = build_model_config(cfg)
    assert (saved["local_inject"], saved["prior_arch"], saved["prior_param"]) == (
        "all", "gnn", "velocity")


# -- local_inject -----------------------------------------------------------

@pytest.mark.parametrize("inject", ["all", "coarsest"])
def test_local_inject_zeroes_z_l_on_finer_levels_only(dataset_path, inject):
    _, model, loader = _build(dataset_path, local_inject=inject)
    sim = model.model
    graph = next(iter(loader))
    with torch.no_grad():
        z, _, _ = sim.posterior(graph.clone())
    level_data = {i: {"ftc": graph[f"fine_to_coarse_{i}"]} for i in range(2)}
    maps = coarsest_parent_maps(level_data)
    batch0 = graph.batch
    fine = sim._level_z(z, 0, batch0, maps)
    assert fine.shape == (graph.num_nodes, LATENT_CH + VAE_DIM)
    # z_g columns are the same either way
    assert torch.equal(fine[:, LATENT_CH:], z.g[batch0, 0])
    if inject == "all":
        assert torch.equal(fine[:, :LATENT_CH], z.l[maps[0]])
    else:
        assert torch.count_nonzero(fine[:, :LATENT_CH]) == 0
    # the coarsest level always reads z_l directly
    coarse_batch = torch.zeros(z.l.shape[0], dtype=torch.long)
    coarsest = sim._level_z(z, 2, coarse_batch, maps)
    assert torch.equal(coarsest[:, :LATENT_CH], z.l)


def test_local_inject_coarsest_changes_the_prediction_not_the_parameters(dataset_path):
    _, base, loader = _build(dataset_path)
    _, coarse_only, _ = _build(dataset_path, local_inject="coarsest")
    assert base.state_dict().keys() == coarse_only.state_dict().keys()
    coarse_only.load_state_dict(base.state_dict())
    _jitter(base)
    coarse_only.load_state_dict(base.state_dict())
    graph = next(iter(loader))
    base.eval(); coarse_only.eval()
    with torch.no_grad():
        z = base.prior.sample(graph.clone())
        a, *_ = base(graph.clone(), fixed_z=z)
        b, *_ = coarse_only(graph.clone(), fixed_z=z)
    assert torch.isfinite(b).all() and not torch.allclose(a, b)


# -- lambda_mmd_g -----------------------------------------------------------

def test_lambda_mmd_g_scales_or_drops_the_z_g_term(dataset_path, monkeypatch):
    # The real estimator draws its N(0, I) reference sample at random; a
    # deterministic stand-in isolates how _mmd composes the two parts.
    from model.vae import GNNVariationalEncoder
    monkeypatch.setattr(GNNVariationalEncoder, "mmd_loss",
                        staticmethod(lambda z, bandwidth="fixed": z.float().pow(2).mean()))
    _, model, loader = _build(dataset_path)          # lambda_mmd 0.2 in _config
    sim = model.model
    graph = next(iter(loader))
    with torch.no_grad():
        z, _, _ = sim.posterior(graph.clone())
    full = sim._mmd(z)
    g_only = sim._mmd(z.g)
    l_only = full - g_only
    for lam_g, expect in ((0.0, l_only), (0.6, l_only + 3.0 * g_only), (0.2, full)):
        _, other, _ = _build(dataset_path, lambda_mmd_g=lam_g)
        assert other.model.mmd_g_scale == pytest.approx(lam_g / 0.2)
        assert torch.allclose(other.model._mmd(z), expect, rtol=1e-5, atol=1e-7)


def test_lambda_mmd_g_rejects_bad_values(dataset_path):
    with pytest.raises(ValueError, match="lambda_mmd_g must be >= 0"):
        _build(dataset_path, lambda_mmd_g=-1.0)
    with pytest.raises(ValueError, match="needs lambda_mmd > 0"):
        _build(dataset_path, lambda_mmd=0.0, lambda_mmd_g=1.0)


def test_lambda_mmd_g_is_hybrid_only(dataset_path):
    # with global, z_g is the whole latent and lambda_mmd alone weighs it
    _, model, _ = _build(dataset_path, latent_type="global", lambda_mmd_g=0.0)
    assert model.model.mmd_g_scale == 1.0


# -- prior_arch transformer / prior_param x / prior_fm_draws ------------------

@pytest.mark.parametrize("latent_type", ["local", "hybrid"])
@pytest.mark.parametrize("param", ["velocity", "x"])
def test_transformer_prior_trains_and_samples_the_declared_shapes(dataset_path, latent_type, param):
    _, model, loader = _build(dataset_path, latent_type, prior_arch="transformer",
                              prior_param=param, prior_fm_draws=3, prior_tf_heads=2)
    prior = model.prior
    assert prior.arch == "transformer" and not hasattr(prior, "coarse_edge_encoder")
    graph = next(iter(loader))
    with torch.no_grad():
        z, mu, logvar = model.model.posterior(graph.clone())
    cond = prior.condition(graph.clone())
    assert cond["pe"].shape == (z.l.shape[0], 27)
    loss = prior.fm_loss(cond, z, target_mu=mu, target_logvar=logvar)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    grads = [p.grad for n, p in prior.named_parameters() if n.startswith("l_in")]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    draws = prior.sample_n(graph.clone(), 2)
    assert draws.l.shape == (z.l.shape[0], 2, LATENT_CH)
    if latent_type == "hybrid":
        assert draws.g.shape == (3, 2) + tuple(z.g.shape[1:])
    else:
        assert draws.g is None


def test_transformer_velocity_is_permutation_equivariant(dataset_path):
    _, model, loader = _build(dataset_path, prior_arch="transformer", prior_tf_heads=2)
    prior = model.prior.eval()
    _jitter(prior)
    graph = next(iter(loader))
    with torch.no_grad():
        ctx = prior._float_ctx(prior.condition(graph.clone()))
        n_c = ctx["x"].shape[0]
        # shuffle the coarse nodes inside each graph (rows stay graph-sorted)
        perm = torch.cat([idx[torch.randperm(idx.numel())]
                          for idx in (torch.nonzero(ctx["batch"] == b).flatten()
                                      for b in range(ctx["B"]))])
        zl = torch.randn(n_c, LATENT_CH)
        zg = torch.randn(ctx["B"], prior.flat_dim)
        t = torch.rand(ctx["B"], 1)
        v_l, v_g = prior.velocity(zl, zg, t, ctx)
        ctx_p = dict(ctx, x=ctx["x"][perm], pe=ctx["pe"][perm])
        v_lp, v_gp = prior.velocity(zl[perm], zg, t, ctx_p)
    assert v_l.abs().max() > 0
    assert torch.allclose(v_lp, v_l[perm], atol=1e-5)
    assert torch.allclose(v_gp, v_g, atol=1e-5)


def test_transformer_node_sees_the_whole_coarse_graph(dataset_path):
    """One node's z_l moves the velocity of every node of its graph -- and of
    no node of another graph. A 1-block GNN reaches only its neighbours."""
    _, model, loader = _build(dataset_path, prior_arch="transformer",
                              prior_fm_blocks=1, prior_tf_heads=2)
    prior = model.prior.eval()
    _jitter(prior)
    graph = next(iter(loader))
    with torch.no_grad():
        ctx = prior._float_ctx(prior.condition(graph.clone()))
        zl = torch.randn(ctx["x"].shape[0], LATENT_CH)
        zg = torch.randn(ctx["B"], prior.flat_dim)
        t = torch.full((ctx["B"], 1), 0.5)
        v0, _ = prior.velocity(zl, zg, t, ctx)
        zl2 = zl.clone()
        zl2[0] += 3.0                              # node 0 belongs to graph 0
        v1, _ = prior.velocity(zl2, zg, t, ctx)
    moved = (v1 - v0).abs().amax(dim=1) > 1e-6
    in_graph0 = ctx["batch"] == 0
    assert moved[in_graph0].all()
    assert not moved[~in_graph0].any()


def test_x_param_converts_heads_to_the_path_velocity(dataset_path):
    _, model, loader = _build(dataset_path, prior_arch="transformer",
                              prior_param="x", prior_tf_heads=2)
    prior = model.prior.eval()
    _jitter(prior)
    graph = next(iter(loader))
    with torch.no_grad():
        ctx = prior._float_ctx(prior.condition(graph.clone()))
        zl = torch.randn(ctx["x"].shape[0], LATENT_CH)
        zg = torch.randn(ctx["B"], prior.flat_dim)
        t = torch.tensor([[0.1], [0.5], [0.999]])[:ctx["B"]]
        x_l, x_g = prior._transformer_net(zl, zg, t, ctx)
        v_l, v_g = prior.velocity(zl, zg, t, ctx)
    s = 1.0 - prior.sigma_min
    den = (1.0 - s * t).clamp(min=0.05)
    assert torch.allclose(v_l, (x_l - s * zl) / den[ctx["batch"]], atol=1e-6)
    assert torch.allclose(v_g, (x_g - s * zg) / den, atol=1e-6)


def test_fm_draws_replicate_the_context_draw_major(dataset_path):
    _, model, loader = _build(dataset_path, prior_arch="transformer", prior_tf_heads=2)
    prior = model.prior
    graph = next(iter(loader))
    ctx = prior.condition(graph.clone())
    rep = prior._replicate(ctx, 3)
    n_c, B = ctx["x"].shape[0], ctx["B"]
    assert rep["B"] == 3 * B and rep["x"].shape[0] == 3 * n_c
    assert torch.equal(rep["batch"][n_c:2 * n_c], ctx["batch"] + B)
    assert torch.equal(rep["pe"][2 * n_c:], ctx["pe"])
    assert torch.equal(rep["pooled"][B:2 * B], ctx["pooled"])
    assert torch.all(rep["batch"][1:] >= rep["batch"][:-1])     # still graph-sorted


# -- transformer dropout / prior_cond_noise ---------------------------------

def _velocity_pairs(prior, loader):
    """Two eval and two train velocities of the same input."""
    graph = next(iter(loader))
    with torch.no_grad():
        ctx = prior._float_ctx(prior.condition(graph.clone()))
        zl = torch.randn(ctx["x"].shape[0], LATENT_CH)
        zg = torch.randn(ctx["B"], prior.flat_dim)
        t = torch.full((ctx["B"], 1), 0.5)
        prior.eval()
        e1, _ = prior.velocity(zl, zg, t, ctx)
        e2, _ = prior.velocity(zl, zg, t, ctx)
        prior.train()
        t1, _ = prior.velocity(zl, zg, t, ctx)
        t2, _ = prior.velocity(zl, zg, t, ctx)
    return e1, e2, t1, t2


def test_transformer_dropout_is_training_only_and_keeps_the_state_dict(dataset_path):
    _, model, loader = _build(dataset_path, prior_arch="transformer", prior_tf_heads=2,
                              prior_tf_dropout=0.1)
    prior = model.prior
    assert all(b.dropout == 0.1 for b in prior.tf_blocks)
    _jitter(prior)
    e1, e2, t1, t2 = _velocity_pairs(prior, loader)
    assert torch.equal(e1, e2)
    assert not torch.allclose(t1, t2)
    # functional dropout: the blocks' parameters are the same with or without it
    block = {k.split(".", 2)[2] for k in prior.state_dict() if k.startswith("tf_blocks.0.")}
    assert block == {f"{m}.{w}" for m in ("qkv", "proj", "mlp.0", "mlp.2", "ada.1")
                     for w in ("weight", "bias")}


def test_transformer_dropout_defaults_to_off(dataset_path):
    _, model, loader = _build(dataset_path, prior_arch="transformer", prior_tf_heads=2)
    prior = model.prior
    assert all(b.dropout == 0.0 for b in prior.tf_blocks)
    _jitter(prior)
    _, _, t1, t2 = _velocity_pairs(prior, loader)
    assert torch.equal(t1, t2)


@pytest.mark.parametrize("arch", ["gnn", "transformer"])
def test_cond_noise_perturbs_the_training_context_only(dataset_path, arch, monkeypatch):
    sigma = 0.3
    _, model, loader = _build(dataset_path, prior_arch=arch, prior_tf_heads=2,
                              prior_fm_draws=2, prior_cond_noise=sigma)
    prior = model.prior
    assert prior.cond_noise == sigma
    graph = next(iter(loader))
    with torch.no_grad():
        z, _, _ = model.model.posterior(graph.clone())
        cond = prior.condition(graph.clone())
    clean = prior._replicate(prior._float_ctx(cond), 2)
    seen = []
    real_velocity = prior.velocity
    def spy(zl_t, zg_t, t, ctx):
        seen.append({k: ctx[k] for k in ("x", "pooled", "batch") if k in ctx})
        return real_velocity(zl_t, zg_t, t, ctx)
    monkeypatch.setattr(prior, "velocity", spy)

    prior.train()
    assert torch.isfinite(prior.fm_loss(cond, z))
    for key in ("x", "pooled"):
        noise = seen[-1][key] - clean[key]
        rel = noise.pow(2).mean(-1).sqrt() / clean[key].pow(2).mean(-1).sqrt()
        assert abs(rel.mean().item() - sigma) < 0.35 * sigma
        n_c = cond[key].shape[0]
        assert not torch.allclose(noise[:n_c], noise[n_c:])   # own noise per draw
    assert torch.equal(seen[-1]["batch"], clean["batch"])

    prior.eval()
    prior.fm_loss(cond, z)
    for key in ("x", "pooled"):
        assert torch.equal(seen[-1][key], clean[key])


@pytest.mark.parametrize("key,value,match", [
    ("prior_arch", "mlp", "prior_arch must be one of"),
    ("prior_param", "eps", "prior_param must be one of"),
    ("prior_fm_draws", 0, "prior_fm_draws must be >= 1"),
    ("prior_cond_noise", -0.1, "prior_cond_noise must be >= 0"),
    ("prior_tf_dropout", -0.1, "prior_tf_dropout must be in"),
    ("prior_tf_dropout", 1.0, "prior_tf_dropout must be in"),
    ("prior_tf_heads", 5, "prior_tf_heads"),
    ("local_inject", "fine", "local_inject must be one of"),
])
def test_bad_values_are_rejected(dataset_path, key, value, match):
    overrides = {key: value}
    if key == "prior_tf_heads":
        overrides["prior_arch"] = "transformer"
    with pytest.raises(ValueError, match=match):
        _build(dataset_path, **overrides)


# -- everything on at once --------------------------------------------------

def test_full_recipe_runs_with_every_switch_on(dataset_path):
    cfg, model, loader = _build(
        dataset_path, local_inject="coarsest", lambda_mmd_g=0.0,
        prior_arch="transformer", prior_param="x", prior_fm_draws=2, prior_tf_heads=2,
        prior_cond_noise=0.1, prior_tf_dropout=0.1)
    ema = build_ema_model(model, cfg)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in model.parameters()]
    joint = train_epoch(model, loader, optimizer, device, cfg, 0, ema_model=ema)
    assert all(np.isfinite(v) for v in joint.values() if isinstance(v, (int, float)))
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))

    val = evaluate_vae_learned_prior_epoch(model, loader, device, cfg, 0)
    assert np.isfinite(val["mean"])

    optimizer, _ = freeze_for_prior_fit(model, ema, loader, device, cfg, remaining_epochs=1)
    prior_before = [p.detach().clone() for p in model.prior.parameters()]
    tail = train_prior_epoch(model, loader, optimizer, device, cfg, 1, ema_model=ema)
    assert np.isfinite(tail["prior_loss_mean"]) and tail["prior_loss_mean"] > 0
    assert any(not torch.equal(b, p.detach())
               for b, p in zip(prior_before, model.prior.parameters()))
    assert isinstance(model.prior.sample(next(iter(loader)).clone()), Latent)
