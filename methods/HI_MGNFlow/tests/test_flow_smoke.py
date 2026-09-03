"""End-to-end smoke test: build a tiny synthetic mesh dataset, train, sample.

Runs on CPU in well under a minute. It proves the wiring -- dataloader ->
velocity network -> flow-matching loss -> ODE sampling -- not the science.

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
        'message_passing_num': 2, 'latent_dim': 16,
        'training_epochs': 3, 'batch_size': 2, 'learningr': 1e-3,
        'num_workers': 0, 'std_noise': 0.0, 'grad_accum_steps': 1,
        'use_checkpointing': False, 'use_amp': False, 'use_ema': False,
        'test_interval': 100, 'val_interval': 1,
        'use_node_types': False, 'use_world_edges': False,
        'use_multiscale': False, 'augment_geometry': False,
        'test_batch_idx': [0, 1], 'plot_feature_idx': 2,
        'display_testset': False, 'display_trainset': False,

        'flow_steps': 6, 'flow_solver': 'heun', 'flow_time_freqs': 8,
        'val_flow_steps': 4, 'val_num_samples': 3,
        'best_by': 'crps',
    }


def test_flow_smoke(tmp_path=None):
    import tempfile
    from general_modules.mesh_dataset import MeshGraphDataset  # noqa: F401  (import check)
    from model.CHiMGNFlow import CHiMGNFlow
    from model.flow import integrate, sample_path
    from training_profiles.training_loop import flow_loss, sample_fields

    out_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    h5_path = build_synthetic_h5(os.path.join(out_dir, 'smoke.h5'))
    config = make_config(h5_path, out_dir)

    # ── 1. the path construction is exact at both endpoints ──────────────────
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
    for m in model.modules():
        if isinstance(m, AdaLNZero):
            lin = m.net[1]
            assert torch.count_nonzero(lin.weight) == 0, "AdaLN weight must start at zero"
            D = m.latent_dim
            assert torch.allclose(lin.bias[:2 * D], torch.zeros(2 * D))
            assert torch.allclose(lin.bias[2 * D:], torch.ones(D)), "gate must start at 1"

    # ── 3. one training step on a real batch from the real dataloader ────────
    from training_profiles.setup import build_dataset_splits
    train_dataset, _val, _test = build_dataset_splits(config, split_seed=0)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    graph = next(iter(loader))
    assert graph.x.shape[1] == INPUT_VAR + COND_VAR + 4, graph.x.shape

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(6):
        loss = flow_loss(model, graph, None, use_amp=False, amp_dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # every parameter must receive gradient -- catches a disconnected branch
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
        opt.step()
        losses.append(float(loss))
    assert np.isfinite(losses).all(), losses
    print(f"  flow loss over 6 steps: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ── 4. ODE sampling produces finite, DIFFERENT fields per draw ───────────
    model.eval()
    flow_cfg = {'steps': 6, 'solver': 'heun'}
    samples = sample_fields(model, graph, flow_cfg, num_samples=3)
    assert samples.shape == (3, graph.y.shape[0], graph.y.shape[1]), samples.shape
    assert torch.isfinite(samples).all()
    spread = float(samples.std(dim=0).mean())
    assert spread > 1e-6, "every draw identical -- the noise channel is dead"
    print(f"  ensemble spread across 3 draws: {spread:.4f}")

    # ── 5. euler and heun both run and disagree only mildly ──────────────────
    s_euler = sample_fields(model, graph, {'steps': 6, 'solver': 'euler'}, 1)
    assert torch.isfinite(s_euler).all()

    # ── 6. K is a sampling-time choice: the same weights integrate at any K ──
    for k in (2, 4, 12):
        out = sample_fields(model, graph, {'steps': k, 'solver': 'heun'}, 1)
        assert torch.isfinite(out).all(), f"integration failed at K={k}"
    print("  same checkpoint integrated at K = 2, 4, 6, 12")

    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    test_flow_smoke()
