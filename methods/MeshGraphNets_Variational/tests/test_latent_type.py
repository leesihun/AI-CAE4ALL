"""latent_type (global / local / hybrid) x latent_reg (mmd / kl).

What must hold for every one of the six combinations:

  1. the latent has the placement it claims: a plain [B, num_z, D] tensor for
     global, Latent(None, z_l [n_coarse, latent_ch]) for local and
     Latent(z_g, z_l) for hybrid -- and the prior samples the same structure;
  2. the active regularizer is the one named: `latent_reg kl` puts the
     per-sample KL in the objective, not MMD, and its weight is kl_weight;
  3. the whole training recipe runs on it: a joint epoch, the learned-prior
     validation pass, and the prior-only tail (freeze + prior epoch), with
     finite numbers -- the Latent pair must survive every torch.cat / reparam /
     standardization site, not only the forward pass.
"""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.mesh_dataset import MeshGraphDataset          # noqa: E402
from model.latent import Latent                                     # noqa: E402
from model.MeshGraphNets import MeshGraphNets                       # noqa: E402
from training_profiles.training_loop import (                       # noqa: E402
    build_ema_model, evaluate_vae_learned_prior_epoch, freeze_for_prior_fit,
    latent_reg_weight, train_epoch, train_prior_epoch,
)
from torch_geometric.loader import DataLoader                       # noqa: E402

NX, NY = 12, 10
LATENT_CH = 3
VAE_DIM = 4
COMBOS = [(lt, reg) for lt in ("global", "local", "hybrid") for reg in ("mmd", "kl")]


def _write_grid(path, num_samples=6):
    """A triangulated 12x10 grid -- multiscale needs a mesh that coarsens."""
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(NX, dtype=np.float32), np.arange(NY, dtype=np.float32))
    pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(NX * NY, np.float32)], 1)
    edges = []
    for j in range(NY):
        for i in range(NX):
            k = j * NX + i
            if i + 1 < NX:
                edges.append((k, k + 1))
            if j + 1 < NY:
                edges.append((k, k + NX))
            if i + 1 < NX and j + 1 < NY:
                edges.append((k, k + NX + 1))
    edge_index = np.array(edges, dtype=np.int64).T
    with h5py.File(path, "w") as handle:
        for sid in range(num_samples):
            nodal = np.zeros((6, 1, pos.shape[0]), np.float32)
            nodal[:3, 0] = pos.T
            nodal[3:6, 0] = rng.normal(size=(3, pos.shape[0])).astype(np.float32) * 0.1
            g = handle.create_group(f"data/{sid}")
            g.create_dataset("nodal_data", data=nodal)
            g.create_dataset("mesh_edge", data=edge_index)


def _config(dataset_path, latent_type, latent_reg):
    return {
        "dataset_dir": str(dataset_path), "input_var": 3, "output_var": 3, "edge_var": 8,
        "positional_features": 0, "use_node_types": False, "use_world_edges": False,
        "use_multiscale": True, "multiscale_levels": 2, "mp_per_level": [1, 1, 1, 1, 1],
        "coarsening_type": "bfs", "augment_geometry": False, "std_noise": 0.0,
        "latent_dim": 16, "message_passing_num": 1,
        "use_vae": True, "vae_latent_dim": VAE_DIM, "vae_mp_layers": 1,
        "z_conditioning": "adaln",
        "latent_type": latent_type, "latent_reg": latent_reg, "latent_ch": LATENT_CH,
        "vae_local_mp_layers": 1, "kl_weight": 1e-3,
        "recon_loss": "mse", "alpha_recon": 1.0, "lambda_mmd": 0.2, "beta_aux": 0.0,
        "prior_type": "gnn_e2e", "use_conditional_prior": True, "prior_family": "fm",
        "prior_hidden_dim": 16, "prior_mp_layers": 1, "prior_fm_steps": 3,
        "prior_nll_weight": 1.0, "prior_grad_to_encoder": 0.0,
        "use_ema": True, "ema_decay": 0.5, "use_amp": False, "use_compile": False,
        "learningr": 1e-3, "training_epochs": 3, "warmup_epochs": 1, "batch_size": 3,
        "num_timesteps": 1, "num_vae_samples": 2,
    }


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("latent_type") / "grid.h5"
    _write_grid(path)
    return path


def _build(dataset_path, latent_type, latent_reg):
    cfg = _config(dataset_path, latent_type, latent_reg)
    dataset = MeshGraphDataset(str(dataset_path), cfg)
    train, _, _ = dataset.split(0.5, 0.25, 0.25, seed=0)
    train.prepare_preprocessing()
    cfg["num_timesteps"] = train.num_timesteps
    loader = DataLoader(train, batch_size=3, shuffle=False)
    torch.manual_seed(0)
    model = MeshGraphNets(cfg, "cpu")
    return cfg, model, loader


def _finite(metrics):
    return all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))


@pytest.mark.parametrize("latent_type", ["global", "local", "hybrid"])
def test_latent_and_prior_sample_have_the_declared_placement(dataset_path, latent_type):
    cfg, model, loader = _build(dataset_path, latent_type, "mmd")
    graph = next(iter(loader))
    model.eval()
    with torch.no_grad():
        z, _, _ = model.model.posterior(graph.clone())
        prior_z = model.prior.sample(graph.clone())
        pred, *_ = model(graph.clone(), fixed_z=prior_z)
    assert torch.isfinite(pred).all()

    if latent_type == "global":
        assert torch.is_tensor(z) and z.dim() == 3 and z.shape[0] == 3 and z.shape[-1] == VAE_DIM
        assert torch.is_tensor(prior_z) and prior_z.shape == z.shape
        return

    assert isinstance(z, Latent) and isinstance(prior_z, Latent)
    z_g, z_l = z
    assert z_l.dim() == 2 and z_l.shape[1] == LATENT_CH
    # one row per coarsest-level node across the batch: fewer than the fine nodes
    assert 3 <= z_l.shape[0] < graph.num_nodes
    assert prior_z.l.shape == z_l.shape
    if latent_type == "local":
        assert z_g is None and prior_z.g is None
    else:
        assert z_g is not None and z_g.shape[0] == 3 and z_g.shape[-1] == VAE_DIM
        assert prior_z.g.shape == z_g.shape


@pytest.mark.parametrize("latent_type", ["global", "local", "hybrid"])
def test_kl_and_mmd_are_both_reported_and_kl_is_per_sample(dataset_path, latent_type):
    _, model, loader = _build(dataset_path, latent_type, "kl")
    graph = next(iter(loader))
    model.train()
    _, _, losses, _, _ = model(graph.clone(), compute_prior_path=False)
    assert "kl" in losses and "mmd" in losses
    assert torch.isfinite(losses["kl"]) and losses["kl"] >= 0

    # per-sample KL(q || N(0,I)) is zero exactly when q is N(0,I): it does not
    # depend on the other samples in the batch (MMD does).
    _, mu, logvar = model.model.posterior(graph.clone())
    parts = [(mu, logvar)] if torch.is_tensor(mu) else [
        (m, lv) for m, lv in zip(mu, logvar) if m is not None]
    # a per-coordinate mean within each part, summed over the parts (hybrid: g + l)
    kl_manual = sum((0.5 * (m.pow(2) + lv.exp() - 1.0 - lv)).mean() for m, lv in parts)
    # posterior() resamples nothing deterministic away: mu/logvar are the same
    # functions of the graph, so the two KLs agree in train mode too.
    assert torch.allclose(losses["kl"], kl_manual, rtol=1e-4, atol=1e-6)


def test_latent_reg_weight_routes_to_the_named_key():
    assert latent_reg_weight({"latent_reg": "kl"}) == ("kl", 1e-3)
    assert latent_reg_weight({"latent_reg": "kl", "kl_weight": "1e-2"}) == ("kl", 1e-2)
    assert latent_reg_weight({"lambda_mmd": 3}) == ("mmd", 3.0)
    assert latent_reg_weight({"latent_reg": "mmd", "kl_weight": 5.0, "lambda_mmd": 2}) == ("mmd", 2.0)


@pytest.mark.parametrize("latent_type,latent_reg", COMBOS)
def test_full_recipe_runs(dataset_path, latent_type, latent_reg):
    cfg, model, loader = _build(dataset_path, latent_type, latent_reg)
    ema = build_ema_model(model, cfg)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # joint epoch: simulator + VAE + prior move together
    before = [p.detach().clone() for p in model.parameters()]
    joint = train_epoch(model, loader, optimizer, device, cfg, 0, ema_model=ema)
    assert _finite(joint) and "reg_mean" in joint and "prior_loss_mean" in joint
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))

    # learned-prior validation: sample z from p(z|graph) the way inference does
    val = evaluate_vae_learned_prior_epoch(model, loader, device, cfg, 0)
    assert np.isfinite(val["mean"])

    # prior-only tail: the simulator stays put, the prior moves
    optimizer, _ = freeze_for_prior_fit(model, ema, loader, device, cfg, remaining_epochs=1)
    sim_params = list(model.model.parameters())
    prior_params = list(model.prior.parameters())
    sim_before = [p.detach().clone() for p in sim_params]
    prior_before = [p.detach().clone() for p in prior_params]
    tail = train_prior_epoch(model, loader, optimizer, device, cfg, 1, ema_model=ema)
    assert np.isfinite(tail["prior_loss_mean"]) and tail["prior_loss_mean"] > 0
    assert all(torch.equal(b, p.detach()) for b, p in zip(sim_before, sim_params))
    assert any(not torch.equal(b, p.detach()) for b, p in zip(prior_before, prior_params))
    # the standardization the freeze fitted is finite and positive, per part
    scales = [model.prior.zl_scale] if latent_type != "global" else []
    if latent_type != "local":
        scales.append(model.prior.z_scale)
    assert all(torch.isfinite(s).all() and (s > 0).all() for s in scales)
