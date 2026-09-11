"""The prior-only tail: freeze_for_prior_fit and train_prior_epoch.

Three things must hold, each a way the recipe could be silently wrong:

  1. after the freeze, NO simulator parameter changes across a prior epoch
     and the prior's DO -- otherwise "frozen target" is a label, not a fact;
  2. the fitted latent standardization is copied into the EMA shadow. rollout
     loads EMA weights, and AveragedModel does not track buffers: a prior
     trained on whitened targets sampled through identity buffers would be
     wrong in a way nothing downstream would flag;
  3. the returned optimizer holds ONLY prior parameters, on a fresh schedule.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import h5py

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.mesh_dataset import MeshGraphDataset          # noqa: E402
from model.MeshGraphNets import MeshGraphNets                       # noqa: E402
from training_profiles.training_loop import (                       # noqa: E402
    build_ema_model, freeze_for_prior_fit, train_prior_epoch,
)
from torch_geometric.loader import DataLoader                       # noqa: E402

NUM_NODES = 12


def _write_dataset(path, num_samples=6):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        for sid in range(num_samples):
            ref = rng.random((NUM_NODES, 3)).astype(np.float32) * 10.0
            nodal = np.zeros((7, 1, NUM_NODES), dtype=np.float32)
            nodal[:3, 0, :] = ref.T
            nodal[3:7, 0, :] = rng.random((4, NUM_NODES)).astype(np.float32) * 0.1
            g = handle.create_group(f"data/{sid}")
            g.create_dataset("nodal_data", data=nodal)
            edges = np.array([list(range(NUM_NODES - 1)), list(range(1, NUM_NODES))],
                             dtype=np.int64)
            g.create_dataset("mesh_edge", data=edges)


def _config(dataset_path):
    return {
        "dataset_dir": str(dataset_path), "input_var": 4, "output_var": 4,
        "edge_var": 8, "positional_features": 0, "use_node_types": False,
        "use_world_edges": False, "use_multiscale": False, "augment_geometry": False,
        "std_noise": 0.0, "latent_dim": 16, "message_passing_num": 1,
        "use_vae": True, "vae_latent_dim": 4, "vae_mp_layers": 1,
        "recon_loss": "mse", "alpha_recon": 1.0, "lambda_mmd": 0.2, "beta_aux": 0.0,
        "prior_type": "gnn_e2e", "use_conditional_prior": True, "prior_family": "fm",
        "prior_hidden_dim": 16, "prior_mp_layers": 1, "prior_fm_steps": 4,
        "prior_nll_weight": 1.0, "prior_grad_to_encoder": 0.0,
        "use_ema": True, "ema_decay": 0.5, "use_amp": False, "use_compile": False,
        "learningr": 1e-3, "training_epochs": 4, "warmup_epochs": 1, "batch_size": 3,
        "num_timesteps": 1,
    }


@pytest.fixture
def setup(tmp_path):
    ds_path = tmp_path / "tiny.h5"
    _write_dataset(ds_path)
    cfg = _config(ds_path)
    dataset = MeshGraphDataset(str(ds_path), cfg)
    train, _, _ = dataset.split(0.5, 0.25, 0.25, seed=0)
    train.prepare_preprocessing()
    cfg["num_timesteps"] = train.num_timesteps
    loader = DataLoader(train, batch_size=3, shuffle=False)
    torch.manual_seed(0)
    model = MeshGraphNets(cfg, "cpu")
    ema = build_ema_model(model, cfg)
    return cfg, model, ema, loader


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _changed(before, params):
    return [not torch.equal(b, p.detach()) for b, p in zip(before, params)]


def test_freeze_isolates_the_prior(setup):
    cfg, model, ema, loader = setup
    sim_params = list(model.model.parameters())
    prior_params = list(model.prior.parameters())

    optimizer, scheduler = freeze_for_prior_fit(model, ema, loader, torch.device("cpu"),
                                                cfg, remaining_epochs=2)

    # 3. the optimizer holds only the prior
    opt_ids = {id(p) for grp in optimizer.param_groups for p in grp["params"]}
    assert opt_ids == {id(p) for p in prior_params}
    assert all(not p.requires_grad for p in sim_params)
    assert all(p.requires_grad for p in prior_params)

    # 1. a prior epoch moves the prior and nothing else
    sim_before = _snapshot(sim_params)
    prior_before = _snapshot(prior_params)
    metrics = train_prior_epoch(model, loader, optimizer, torch.device("cpu"), cfg, 0,
                                ema_model=ema)
    assert not any(_changed(sim_before, sim_params)), "a simulator parameter moved"
    assert any(_changed(prior_before, prior_params)), "no prior parameter moved"
    assert metrics["prior_loss_mean"] > 0 and metrics["mean"] == 0.0


def test_standardization_is_fitted_and_mirrored_into_ema(setup):
    cfg, model, ema, loader = setup
    assert torch.equal(model.prior.z_scale, torch.ones_like(model.prior.z_scale))

    freeze_for_prior_fit(model, ema, loader, torch.device("cpu"), cfg, remaining_epochs=1)

    # fitted: no longer identity, finite, positive
    assert not torch.equal(model.prior.z_scale, torch.ones_like(model.prior.z_scale))
    assert torch.isfinite(model.prior.z_shift).all()
    assert (model.prior.z_scale > 0).all()
    # 2. mirrored into the EMA shadow, which rollout actually loads
    ema_prior = ema.module.prior
    assert torch.equal(ema_prior.z_shift, model.prior.z_shift)
    assert torch.equal(ema_prior.z_scale, model.prior.z_scale)


def test_fitted_standardization_matches_the_posterior(setup):
    """shift = E[mu], scale = sqrt(Var(mu) + E[sigma^2]) over the loader."""
    cfg, model, ema, loader = setup
    inner = model.model
    mus, lvs = [], []
    with torch.no_grad():
        for g in loader:
            batch = g.batch
            _, mu, lv = inner.vae_encoder(g.y, g.edge_index, g.edge_attr, batch,
                                          x=(g.x if inner.vae_graph_aware else None))
            mus.append(mu.float())
            lvs.append(lv.float())
    mu = torch.cat(mus).reshape(-1, model.prior.flat_dim).double()
    var = torch.cat(lvs).reshape(-1, model.prior.flat_dim).double().exp()
    want_shift = mu.mean(0).float()
    want_scale = (mu.var(0, unbiased=False) + var.mean(0)).sqrt().float()

    freeze_for_prior_fit(model, ema, loader, torch.device("cpu"), cfg, remaining_epochs=1)
    assert torch.allclose(model.prior.z_shift, want_shift, atol=1e-5)
    assert torch.allclose(model.prior.z_scale, want_scale, atol=1e-5)
