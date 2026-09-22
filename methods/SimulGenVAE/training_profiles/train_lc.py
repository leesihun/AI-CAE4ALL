"""Stage 2: latent-conditioner (LC) training.

Loads the frozen hierarchical VAE, encodes every FOM sample to its **dual** latent
target -- a main latent ``[Dm]`` and a stack of hierarchical latents ``[L, Dh]``
(``L = len(num_filter_enc) - 1``) -- scales each, and trains the conditioner to
regress both from the physical parameter / image conditions. Loss follows the
SimulGenVAE convention ``10 * MSE(main) + MSE(hier)``.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.field_viz import plot_latent_parity
from general_modules.fom_dataset import (
    apply_minmax,
    build_dataset_splits,
    fit_minmax,
    read_conditions,
)
from model.common import add_sn
from training_profiles.setup import (
    append_log,
    artifact_dir,
    build_lc,
    build_optimizer_scheduler,
    build_vae,
    init_log_file,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


@torch.no_grad()
def _encode_latents(vae, data_nct, device, batch_size):
    """Encode the full FOM array [N, C, T] -> (main [N, Dm], hier [N, L, Dh])."""
    vae.eval()
    mains, hiers = [], []
    n = data_nct.shape[0]
    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(data_nct[start:start + batch_size]).to(device)
        mu, _log_var, xs = vae.encoder(chunk)         # xs: list[L] of [B, Dh]
        mains.append(mu.detach().cpu().numpy())
        hiers.append(torch.stack(xs, dim=1).detach().cpu().numpy())  # [B, L, Dh]
    return np.concatenate(mains, 0), np.concatenate(hiers, 0)


@torch.no_grad()
def run_latent_parity_test(lc, val_loader, config, device, epoch, out_dir):
    """Periodic parity picture for the conditioner (rank 0 only).

    The conditioner regresses a frozen latent target, so predicted-vs-true over
    every latent coordinate is the honest view: a model that has learned only
    the latent mean collapses onto a horizontal band instead of the diagonal.
    """
    if val_loader is None:
        return None
    was_training = lc.training
    lc.eval()

    true_main, pred_main, true_hier, pred_hier = [], [], [], []
    for x, y_main, y_hier in val_loader:
        out_main, out_hier = lc(x.to(device, non_blocking=True))
        true_main.append(y_main.numpy())
        pred_main.append(out_main.float().cpu().numpy())
        true_hier.append(y_hier.numpy())
        pred_hier.append(out_hier.float().cpu().numpy())

    if was_training:
        lc.train()
    if not true_main:
        return None

    written = plot_latent_parity(
        np.concatenate(true_main, 0), np.concatenate(pred_main, 0),
        np.concatenate(true_hier, 0), np.concatenate(pred_hier, 0),
        os.path.join(out_dir, f'lc_parity_epoch{epoch:05d}.png'),
        epoch=epoch, dpi=int(config.get('plot_dpi', 140)),
    )
    if written:
        print(f'  [viz] {written}')
    return written


def lc_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))
    rank0 = D.is_main_process()
    world_size = D.get_world_size()

    # The frozen VAE owns both its field scaler and its holdout partition.
    vae_path = config.get('vae_modelpath', '../../output/simulgenvae/simulgenvae_vae.pth')
    ckpt = load_checkpoint(vae_path, device)
    if ckpt.get('stage') != 'vae':
        raise ValueError(f"vae_modelpath is a {ckpt.get('stage')!r} checkpoint, expected 'vae'.")
    if 'split_manifest' not in ckpt or 'normalization' not in ckpt:
        raise ValueError('VAE checkpoint lacks train-only split/scaler provenance; retrain the VAE.')
    train_dataset, val_dataset, test_dataset = build_dataset_splits(
        config, split_seed, normalization=ckpt['normalization'],
        split_manifest=ckpt['split_manifest'])
    data_nct = train_dataset.data
    num_channels, num_time = train_dataset.num_channels, train_dataset.num_time
    train_indices = train_dataset.indices
    vae, num_filter_enc = build_vae(config, num_channels, num_time,
                                    int(config.get('batch_size', 16)))
    vae.apply(add_sn)   # match the spectral-norm parametrization saved by the VAE stage
    vae.load_state_dict(ckpt['model_state'])
    vae = vae.to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    num_levels = len(num_filter_enc) - 1

    if rank0:
        print('\nEncoding FOM samples to hierarchical latent targets...')
    main, hier = _encode_latents(vae, data_nct, device, int(config.get('batch_size', 16)))

    conditions, input_shape = read_conditions(config)
    if conditions.shape[0] != main.shape[0]:
        raise ValueError(
            f"Condition count ({conditions.shape[0]}) != sample count ({main.shape[0]}). "
            "Condition rows/images must be ordered to match sorted sample IDs.")

    # Scale main + hier latent targets and (csv) inputs; keep scalers for the checkpoint.
    main_norm = fit_minmax(main[train_indices])
    hier_norm = fit_minmax(hier[train_indices])
    main_s = apply_minmax(main, main_norm)
    hier_s = apply_minmax(hier, hier_norm)
    if str(config.get('lc_data_type', 'csv')).lower() in ('csv', 'hdf5'):
        input_norm = fit_minmax(conditions[train_indices])
        conditions = apply_minmax(conditions, input_norm)
    else:
        input_norm = None

    dataset = TensorDataset(
        torch.from_numpy(np.float32(conditions)),
        torch.from_numpy(np.float32(main_s)),
        torch.from_numpy(np.float32(hier_s)))
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_dataset.indices)

    batch_size = int(config.get('batch_size', 16))
    if world_size > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=D.get_rank(), shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if rank0 else None

    lc = build_lc(config, input_shape, num_levels).to(device)
    train_lc_model, _ = D.wrap_model(lc, config, device)

    total_epochs = int(config.get('training_epochs', 500))
    optimizer, scheduler = build_optimizer_scheduler(config, train_lc_model.parameters(), total_epochs)
    mse = nn.MSELoss()

    modelpath = config.get('lc_modelpath', '../../output/simulgenvae/simulgenvae_lc.pth')
    log_file = init_log_file(config, config_filename) if rank0 else None
    test_interval = max(1, int(config.get('test_interval', 100)))
    display_testset = bool(config.get('display_testset', True))
    viz_dir = artifact_dir(config, modelpath) if rank0 else None
    if rank0:
        print('\n' + '=' * 60)
        print(f'Starting latent-conditioner training ({num_levels} hierarchical levels)...')
        print('=' * 60 + '\n')
    start_time = time.time()

    def payload(epoch):
        return {
            'stage': 'lc',
            'epoch': epoch,
            'model_state': D.full_state_dict(train_lc_model, False),
            'config': config,
            'normalization': {'main': main_norm, 'hier': hier_norm, 'input': input_norm},
            'num_levels': num_levels,
            'input_shape': input_shape,
            'split_manifest': train_dataset.split_manifest,
        }

    for epoch in range(total_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_lc_model.train()
        loss_sum, batches = 0.0, 0
        for x, y_main, y_hier in train_loader:
            x = x.to(device, non_blocking=True)
            y_main = y_main.to(device, non_blocking=True)
            y_hier = y_hier.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred_main, pred_hier = train_lc_model(x)
            loss = 10.0 * mse(pred_main, y_main) + mse(pred_hier, y_hier)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_lc_model.parameters(), max_norm=10.0)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        scheduler.step()
        train_loss = D.reduce_epoch_mean(loss_sum, batches, device)

        last_epoch = epoch == total_epochs - 1
        if rank0 and (epoch % 100 == 0 or last_epoch):
            val_loss = _validate(D.unwrap_model(train_lc_model), val_loader, device, mse)
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}/{total_epochs} LC train {train_loss:.4e} val {val_loss:.4e} LR {lr:.2e}')
            append_log(log_file, f'Elapsed {time.time()-start_time:.2f}s Epoch {epoch} '
                                 f'LC train {train_loss:.4e} val {val_loss:.4e} LR {lr:.4e}')

        if rank0 and display_testset and (epoch % test_interval == 0 or last_epoch):
            run_latent_parity_test(D.unwrap_model(train_lc_model), val_loader, config,
                                   device, epoch, viz_dir)

    if rank0:
        save_checkpoint(modelpath, payload(total_epochs - 1))
        print(f'\nLatent conditioner saved to {modelpath}')
    D.barrier()


@torch.no_grad()
def _validate(lc, val_loader, device, mse):
    if val_loader is None:
        return float('nan')
    lc.eval()
    total, batches = 0.0, 0
    for x, y_main, y_hier in val_loader:
        x, y_main, y_hier = x.to(device), y_main.to(device), y_hier.to(device)
        pred_main, pred_hier = lc(x)
        total += float((10.0 * mse(pred_main, y_main) + mse(pred_hier, y_hier)).item())
        batches += 1
    lc.train()
    return total / max(batches, 1)
