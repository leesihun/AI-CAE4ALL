"""AR-OT / AR-RT time integration for the variational simulator.

Beyond the shared contract (AR-OT is the untouched default; AR-RT unrolls the
whole trajectory with gradients through all of it), two variational-specific
decisions are pinned here: the latent is resampled at every unrolled step, and
the loss composition is the one-step objective evaluated per step and averaged.
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

from general_modules.mesh_dataset import MeshGraphDataset  # noqa: E402
from general_modules.time_integration import (  # noqa: E402
    resolve_rollout_window,
    resolve_time_integration,
)
from training_profiles.ar_rollout import RolloutContext, rollout_loss  # noqa: E402


NUM_NODES = 12
NUM_TIMESTEPS = 5
INPUT_VAR = 3
OUTPUT_VAR = 3


def _write_dataset(path, num_samples=4, num_timesteps=None):
    """A tiny grid-graph trajectory in the repository's HDF5 layout."""
    num_timesteps = num_timesteps or NUM_TIMESTEPS
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        for sample_id in range(num_samples):
            ref_pos = rng.random((NUM_NODES, 3)).astype(np.float32) * 10.0
            nodal = np.zeros((7, num_timesteps, NUM_NODES), dtype=np.float32)
            nodal[:3, :, :] = ref_pos.T[:, None, :]
            for t in range(num_timesteps):
                nodal[3:7, t, :] = (
                    rng.random((4, NUM_NODES)).astype(np.float32) * 0.1 + 0.05 * t
                )
            group = handle.create_group(f"data/{sample_id}")
            group.create_dataset("nodal_data", data=nodal)
            edges = np.array(
                [[i for i in range(NUM_NODES - 1)], [i + 1 for i in range(NUM_NODES - 1)]],
                dtype=np.int64,
            )
            group.create_dataset("mesh_edge", data=edges)


def _base_config(dataset_path, **overrides):
    config = {
        "dataset_dir": str(dataset_path),
        "input_var": INPUT_VAR,
        "output_var": OUTPUT_VAR,
        "edge_var": 8,
        "positional_features": 0,
        "use_node_types": False,
        "use_world_edges": False,
        "use_multiscale": False,
        "augment_geometry": False,
        "std_noise": 0.0,
        "latent_dim": 16,
        "message_passing_num": 1,
        # VAE path on, conditional prior off (the prior has its own tests).
        "use_vae": True,
        "vae_latent_dim": 8,
        "vae_mp_layers": 1,
        "recon_loss": "mse",
        "alpha_recon": 1.0,
        "lambda_mmd": 0.2,
        "beta_aux": 1.0,
        "prior_type": "",
    }
    config.update(overrides)
    return config


def _prepared_dataset(dataset_path, config):
    dataset = MeshGraphDataset(str(dataset_path), config)
    train, _, _ = dataset.split(0.5, 0.25, 0.25, seed=0)
    config["num_timesteps"] = train.num_timesteps
    return train


def _norm_stats(dataset):
    from training_profiles.setup import build_normalization_dict

    stats = build_normalization_dict(dataset)
    stats["world_edge_radius"] = dataset.world_edge_radius
    return stats


def _graph_on_cpu(dataset):
    graph = dataset[0]
    graph.batch = torch.zeros(NUM_NODES, dtype=torch.long)
    graph.ptr = torch.tensor([0, NUM_NODES], dtype=torch.long)
    return graph


def test_ar_ot_is_the_default(tmp_path):
    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path)
    assert resolve_time_integration(config) == "ar_ot"
    assert resolve_rollout_window(config, NUM_TIMESTEPS) == 1

    dataset = _prepared_dataset(path, config)
    assert len(dataset) == len(dataset.sample_ids) * (NUM_TIMESTEPS - 1)
    assert getattr(dataset[0], "y_seq", None) is None


def test_ar_rt_full_trajectory_window(tmp_path):
    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path, time_integration="AR-RT")
    dataset = _prepared_dataset(path, config)

    assert dataset.rollout_window == NUM_TIMESTEPS - 1
    assert len(dataset) == len(dataset.sample_ids)

    item = dataset[0]
    assert item.y_seq.shape == (NUM_NODES, NUM_TIMESTEPS - 1, OUTPUT_VAR)
    assert item.state0.shape == (NUM_NODES, INPUT_VAR)


def _rollout_once(tmp_path, **config_overrides):
    """Run one AR-RT unroll, returning (model, loss, recon_count, latents)."""
    from model.MeshGraphNets import MeshGraphNets

    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path, time_integration="ar_rt", **config_overrides)
    dataset = _prepared_dataset(path, config)
    config["_norm_stats"] = _norm_stats(dataset)

    torch.manual_seed(0)
    model = MeshGraphNets(config, "cpu")
    model.train()

    latents = []

    def compose(graph):
        predicted, target, vae_losses, aux, _ = model(graph, compute_prior_path=False)
        # mu is the posterior mean; a fresh reparameterized draw per step is
        # what "latent resampled every step" means in practice.
        if vae_losses.get("mu") is not None:
            latents.append(vae_losses["mu"].detach().clone())
        errors = torch.nn.functional.mse_loss(predicted, target, reduction="none")
        per_node = errors.mean(dim=-1)
        loss = per_node.mean() + 0.2 * vae_losses["mmd"]
        aux_tensor = torch.as_tensor(aux, dtype=torch.float32)
        return predicted, loss, per_node.sum(), vae_losses["mmd"], aux_tensor

    ctx = RolloutContext(config, torch.device("cpu"))
    loss, _, _, recon_count, extras = rollout_loss(
        model, _graph_on_cpu(dataset), ctx, compose, training=True
    )
    return model, loss, recon_count, latents, extras


def test_rollout_backpropagates_through_the_whole_unroll(tmp_path):
    model, loss, recon_count, _, extras = _rollout_once(tmp_path)
    loss.backward()

    # Every step of the trajectory is scored, not just the last.
    assert recon_count == (NUM_TIMESTEPS - 1) * NUM_NODES
    # The composition's extra terms survive as trajectory averages.
    assert len(extras) == 2 and all(torch.isfinite(term) for term in extras)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rollout produced no gradients"
    assert any(torch.any(g != 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_posterior_is_re_encoded_at_every_step(tmp_path):
    """The latent is resampled per step, not drawn once for the trajectory.

    The posterior conditions on `graph.y`, which the rollout rewrites to the
    step's correction, so a per-step re-encode must produce a different mu at
    each step. Identical latents would mean the rollout reused one draw.
    """
    _, _, _, latents, _ = _rollout_once(tmp_path)

    assert len(latents) == NUM_TIMESTEPS - 1
    first = latents[0]
    assert any(not torch.allclose(first, other) for other in latents[1:])


def test_vae_terms_are_averaged_over_the_trajectory(tmp_path):
    """Composition is unchanged: each step's objective, averaged over steps."""
    from model.MeshGraphNets import MeshGraphNets

    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path, time_integration="ar_rt")
    dataset = _prepared_dataset(path, config)
    config["_norm_stats"] = _norm_stats(dataset)

    torch.manual_seed(0)
    model = MeshGraphNets(config, "cpu")
    model.eval()

    per_step = []

    def compose(graph):
        predicted, target, vae_losses, aux, _ = model(graph, compute_prior_path=False)
        errors = torch.nn.functional.mse_loss(predicted, target, reduction="none")
        per_node = errors.mean(dim=-1)
        loss = per_node.mean() + 0.2 * vae_losses["mmd"]
        per_step.append(loss.detach().clone())
        return (predicted, loss, per_node.sum(), vae_losses["mmd"],
                torch.as_tensor(aux, dtype=torch.float32))

    ctx = RolloutContext(config, torch.device("cpu"))
    with torch.no_grad():
        total, _, _, _, _ = rollout_loss(
            model, _graph_on_cpu(dataset), ctx, compose, training=False
        )

    torch.testing.assert_close(total, torch.stack(per_step).mean(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("scheme", ["ar_ot", "ar_rt"])
def test_train_epoch_runs_under_both_schemes(tmp_path, scheme):
    """Smoke test for the loop itself, not just the rollout helper.

    train_epoch's body is shared between the two schemes, so this is what
    catches wiring mistakes in the objective bundle, the logging accumulators
    and the optimizer step.
    """
    from torch_geometric.loader import DataLoader

    from model.MeshGraphNets import MeshGraphNets
    from training_profiles.training_loop import train_epoch

    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path, time_integration=scheme, use_amp=False)
    dataset = _prepared_dataset(path, config)
    config["_norm_stats"] = _norm_stats(dataset)

    torch.manual_seed(0)
    model = MeshGraphNets(config, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = train_epoch(model, loader, optimizer, torch.device("cpu"), config, epoch=0)

    assert np.isfinite(metrics["mean"])
    assert metrics["count"] > 0
    assert "mmd_mean" in metrics  # the VAE terms still reach the epoch summary
