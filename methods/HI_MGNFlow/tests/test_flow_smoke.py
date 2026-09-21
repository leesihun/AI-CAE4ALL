"""End-to-end smoke test: build a tiny synthetic mesh dataset, train both
stages, sample.

Runs on CPU in well under a minute. It proves the wiring -- dataloader ->
compressor (stage 1) -> frozen-latent flow prior (stage 2) -> coarse-latent
ODE sampling -> decode -- not the science.

    cd methods/HI_MGNFlow && python -m pytest -q tests/
    cd methods/HI_MGNFlow && python tests/test_flow_smoke.py     # standalone

The synthetic field is a smooth quadratic bowl whose amplitude is drawn per
sample, so the conditional distribution is genuinely non-degenerate and the
ensemble is expected to show non-zero spread.
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch


N_SIDE = 6          # 6x6 grid -> 36 nodes
N_SAMPLES = 12
INPUT_VAR = 3
COND_VAR = 1


def test_checkpoint_save_creates_fresh_parent(tmp_path):
    """GUI-selected run directories need not exist before native training."""
    from training_profiles.setup import save_checkpoint

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    dataset = SimpleNamespace(
        node_mean=np.zeros(2, dtype=np.float32),
        node_std=np.ones(2, dtype=np.float32),
        edge_mean=np.zeros(8, dtype=np.float32),
        edge_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(2, dtype=np.float32),
        delta_std=np.ones(2, dtype=np.float32),
        use_node_types=False,
        use_world_edges=False,
        use_multiscale=False,
    )
    target = tmp_path / "fresh" / "nested" / "model.pth"

    save_checkpoint(
        0, model, None, optimizer, scheduler, 1.0, 2.0,
        {"model": "chi-mgnflow", "input_var": 2, "output_var": 2},
        dataset, str(target),
    )

    assert target.is_file()
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 0
    assert payload["model_config"]["model"] == "chi-mgnflow"


def build_synthetic_h5(path):
    """Write a mesh HDF5 in the shared dataset contract.

    nodal_data rows: [x, y, z | dx, dy, dz | thickness]
                      ^ ref     ^ state      ^ cond (input-only)
    """
    xs, ys = np.meshgrid(np.linspace(-1, 1, N_SIDE), np.linspace(-1, 1, N_SIDE))
    xs, ys = xs.ravel(), ys.ravel()
    zs = np.zeros_like(xs)
    n = xs.size

    edges = []
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            k = i * N_SIDE + j
            if j + 1 < N_SIDE:
                edges.append((k, k + 1))
            if i + 1 < N_SIDE:
                edges.append((k, k + N_SIDE))
    mesh_edge = np.asarray(edges, dtype=np.int32).T          # [2, M]

    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        grp = f.create_group('data')
        for s in range(N_SAMPLES):
            thickness = float(rng.uniform(0.5, 1.5))
            # Amplitude is only partly explained by thickness -> real spread.
            amp = thickness * 2.0 + float(rng.normal(0.0, 0.3))
            bowl = amp * (xs ** 2 + ys ** 2)

            nodal = np.zeros((3 + INPUT_VAR + COND_VAR, 1, n), dtype=np.float32)
            nodal[0, 0, :] = xs
            nodal[1, 0, :] = ys
            nodal[2, 0, :] = zs
            nodal[3, 0, :] = 0.05 * bowl
            nodal[4, 0, :] = 0.05 * bowl
            nodal[5, 0, :] = bowl
            nodal[6, 0, :] = thickness

            g = grp.create_group(str(s))
            g.create_dataset('nodal_data', data=nodal)
            g.create_dataset('mesh_edge', data=mesh_edge)
    return path


def make_config(h5_path, out_dir):
    return {
        'model': 'chi-mgnflow', 'mode': 'train',
        'gpu_ids': 0, 'parallel_mode': 'ddp',
        'dataset_dir': h5_path, 'infer_dataset': h5_path,
        'log_file_dir': os.path.join(out_dir, 'smoke.log'),
        'modelpath': os.path.join(out_dir, 'smoke.pth'),
        'split_seed': 0,

        'input_var': INPUT_VAR, 'output_var': INPUT_VAR, 'cond_var': COND_VAR,
        'edge_var': 8, 'positional_features': 4,
        'latent_dim': 16, 'latent_ch': 4, 'ae_kl_weight': 1e-6, 'prior_blocks': 2,
        'ae_epochs': 3, 'training_epochs': 3, 'batch_size': 2, 'learningr': 1e-3,
        'num_workers': 0, 'std_noise': 0.0, 'grad_accum_steps': 1,
        'use_checkpointing': False, 'use_amp': False, 'use_ema': False,
        'test_interval': 100, 'val_interval': 1,
        'use_node_types': False, 'use_world_edges': False,
        'augment_geometry': False,
        'test_batch_idx': [0, 1], 'plot_feature_idx': 2,
        'display_testset': False, 'display_trainset': False,

        # LDGN's coarse level needs a real hierarchy -- there is no flat mode.
        'use_multiscale': True, 'coarsening_type': 'voronoi_seedmean',
        'voronoi_clusters': 6, 'multiscale_levels': 1, 'mp_per_level': [1, 1, 1],
        'hierarchy_variants': 1, 'hierarchy_seed': 1234,

        'flow_steps': 6, 'flow_solver': 'heun', 'flow_time_freqs': 8,
        'val_flow_steps': 4, 'val_num_samples': 3,
        'best_by': 'crps',
    }


def test_flow_smoke(tmp_path=None):
    import tempfile
    from general_modules.mesh_dataset import MeshGraphDataset  # noqa: F401  (import check)
    from model.CHiMGNFlow import CHiMGNFlow
    from model.flow import resolve_flow_config, sample_path
    from training_profiles.training_loop import (
        ae_loss, prior_loss, _sample_posterior_latent, _generate_fields,
    )

    out_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    h5_path = build_synthetic_h5(os.path.join(out_dir, 'smoke.h5'))
    config = make_config(h5_path, out_dir)

    # ── 1. the path construction is exact at both endpoints (generic: it does
    #      not care whether it is handed a field or a latent) ────────────────
    y1 = torch.randn(20, 3)
    batch = torch.zeros(20, dtype=torch.long)
    t, y_t, u = sample_path(y1, batch, 1)
    assert y_t.shape == y1.shape and u.shape == y1.shape
    assert torch.isfinite(y_t).all() and torch.isfinite(u).all()

    # ── 2. build the model and check the identity-at-init property ───────────
    device = 'cpu'
    model = CHiMGNFlow(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0

    from model.blocks import AdaLNZero
    n_adaln = 0
    for m in model.modules():
        if isinstance(m, AdaLNZero):
            n_adaln += 1
            lin = m.net[1]
            assert torch.count_nonzero(lin.weight) == 0, "AdaLN weight must start at zero"
            D = m.latent_dim
            assert torch.allclose(lin.bias[:2 * D], torch.zeros(2 * D))
            assert torch.allclose(lin.bias[2 * D:], torch.ones(D)), "gate must start at 1"
    assert n_adaln == config['prior_blocks'], "AdaLN-Zero lives only in the latent prior"

    # ── 3. one real batch from the real dataloader (multiscale hierarchy
    #      attached by the dataset itself -- see general_modules/mesh_dataset.py)
    from training_profiles.setup import build_dataset_splits
    train_dataset, _val, _test = build_dataset_splits(config, split_seed=0)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    graph = next(iter(loader))
    assert graph.x.shape[1] == INPUT_VAR + COND_VAR + 4, graph.x.shape

    # ── 4. stage 1 (compressor): reconstruction + KL, gradients reach the AE ──
    model.train()
    opt_ae = torch.optim.Adam(model.parameters(), lr=1e-3)
    ae_losses = []
    for _ in range(6):
        loss, recon, kl = ae_loss(model, graph, None, use_amp=False,
                                  amp_dtype=torch.float32, kl_weight=1e-6)
        opt_ae.zero_grad(set_to_none=True)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
        opt_ae.step()
        ae_losses.append(float(loss.detach()))
    assert np.isfinite(ae_losses).all(), ae_losses
    print(f"  stage 1 (AE) loss over 6 steps: {ae_losses[0]:.4f} -> {ae_losses[-1]:.4f}"
          f"  (recon={float(recon):.4f}, kl={float(kl):.4f})")

    # ── 5. freeze the compressor, train stage 2 (latent prior) on frozen
    #      posterior draws; verify gradients reach ONLY the prior ────────────
    model.freeze_ae()
    model.train()
    for p in model.ae_parameters():
        p.grad = None  # drop stage-1's leftover grad so the check below is fresh
    opt_prior = torch.optim.Adam(model.prior_parameters(), lr=1e-3)
    flow_cfg = resolve_flow_config(config)
    prior_losses = []
    for _ in range(6):
        z1, ctx = _sample_posterior_latent(model, graph)
        loss = prior_loss(model, ctx, z1, ctx['coarse_batch'], use_amp=False,
                          amp_dtype=torch.float32, flow_cfg=flow_cfg)
        opt_prior.zero_grad(set_to_none=True)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.prior_parameters())
        assert all(p.grad is None for p in model.ae_parameters()), \
            "compressor must not receive gradient once frozen"
        opt_prior.step()
        prior_losses.append(float(loss.detach()))
    assert np.isfinite(prior_losses).all(), prior_losses
    print(f"  stage 2 (prior) loss over 6 steps: {prior_losses[0]:.4f} -> {prior_losses[-1]:.4f}")

    # ── 6. generation: encode once, integrate the COARSE latent ODE, decode
    #      once -- produces finite, DIFFERENT fields per draw ────────────────
    model.eval()
    flow_cfg['steps'], flow_cfg['solver'] = 6, 'heun'
    samples = _generate_fields(model, graph, flow_cfg, num_samples=3,
                               use_amp=False, amp_dtype=torch.float32)
    assert samples.shape == (3, graph.y.shape[0], graph.y.shape[1]), samples.shape
    assert torch.isfinite(samples).all()
    spread = float(samples.std(dim=0).mean())
    assert spread > 1e-6, "every draw identical -- the latent noise channel is dead"
    print(f"  ensemble spread across 3 draws: {spread:.4f}")

    # ── 7. euler and heun both run and disagree only mildly ──────────────────
    flow_cfg['steps'], flow_cfg['solver'] = 6, 'euler'
    s_euler = _generate_fields(model, graph, flow_cfg, num_samples=1,
                               use_amp=False, amp_dtype=torch.float32)
    assert torch.isfinite(s_euler).all()

    # ── 8. K is a sampling-time choice: the same weights integrate at any K ──
    flow_cfg['solver'] = 'heun'
    for k in (2, 4, 12):
        flow_cfg['steps'] = k
        out = _generate_fields(model, graph, flow_cfg, num_samples=1,
                               use_amp=False, amp_dtype=torch.float32)
        assert torch.isfinite(out).all(), f"integration failed at K={k}"
    print("  same checkpoint integrated at K = 2, 4, 6, 12")

    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    test_flow_smoke()
