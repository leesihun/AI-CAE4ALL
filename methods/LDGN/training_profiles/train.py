"""The two LDGN training stages on the shared mesh HDF5.

Stage 1 (`train_vae`): the multi-scale variational graph autoencoder (VGAE)
whose latent space the diffusion model runs in (upstream
examples/Wing/train_vgae.py).
Stage 2 (`train_ldgn`): the latent diffusion graph net over the frozen VGAE
(upstream examples/Wing/train_ldgn.py).

Both stages optimise exactly as upstream (Adam, gradient clipping,
ReduceLROnPlateau on the epoch-mean train loss, stop once the LR falls below
`*_lr_threshold`) and keep the LAST weights, as upstream's checkpoint did.
What this file adds around `Model.fit` is the suite's plumbing: the grouped
split, logging, an optional validation loss (logged only, it selects nothing),
the periodic pictures, resume, and a checkpoint that carries the train-split
normalization so the sampler can undo it.
"""
import os
import time

import h5py
import numpy as np
import torch
from torch import nn

from dgn4cfd.loader import DataLoader
from dgn4cfd.nn import VGAE, LatentDiffusionGraphNet, TrainingSettings
from dgn4cfd.nn.diffusion import DiffusionProcess
from dgn4cfd.nn.model import decode_arch, encode_arch
from general_modules import mesh_dataset as md
from general_modules.resume_state import TrainingResume

from .common import (append_log, as_float, as_list, check_normalization_matches,
                     checkpoint_config, init_stage_log, load_checkpoint,
                     resolve_device, save_checkpoint, training_generator)


VAL_SEED = 12345
EDGE_FEATURES = 3     # edge_attr = scaled pos[dst] - pos[src]


# ---------------------------------------------------------------------------
# Architecture from the config
# ---------------------------------------------------------------------------

def vae_arch(config, stats):
    """Upstream Wing VGAE arch; `cond_node_features` follows the data."""
    depths = as_list(config.get('vae_depths', [2, 2, 1]))
    if len(depths) < 2:
        raise ValueError(f'vae_depths needs at least 2 entries (one pooling level); got {depths}')
    return {
        'in_node_features': int(stats['output_var']),
        'cond_node_features': md.cond_features(stats),
        'cond_edge_features': EDGE_FEATURES,
        'latent_node_features': int(config.get('vae_latent_features', 1)),
        'depths': depths,
        'fnns_depth': int(config.get('vae_fnns_depth', 2)),
        'fnns_width': int(config.get('vae_fnns_width', 126)),
        'activation': nn.SELU,
        'aggr': str(config.get('vae_aggr', 'sum')),
        'dropout': float(config.get('vae_dropout', 0.1)),
        'norm_latents': bool(config.get('vae_norm_latents', True)),
        'dim': 3,
        'scalar_rel_pos': True,
    }


def ldgn_arch(config, vae_arch_):
    """Upstream Wing LDGN arch; its inputs are the VGAE's latent widths."""
    depths = as_list(config.get('ldgn_depths', [1, 2, 2, 2]))
    return {
        'in_node_features': int(vae_arch_['latent_node_features']),
        'cond_node_features': int(vae_arch_['fnns_width']),
        'cond_edge_features': int(vae_arch_['fnns_width']),
        'depths': depths,
        'fnns_depth': int(config.get('ldgn_fnns_depth', 2)),
        'fnns_width': int(config.get('ldgn_fnns_width', 128)),
        'aggr': str(config.get('ldgn_aggr', 'sum')),
        'dropout': float(config.get('ldgn_dropout', 0.1)),
        'dim': 3,
        'scalar_rel_pos': True,
    }


def num_scales_for(vae_depths, ldgn_depths):
    """Levels the LDGN reads: the VGAE's, then one per LDGN depth past the
    VGAE's coarsest (upstream Wing: 3 + 4 - 1 = 6)."""
    return len(vae_depths) + len(ldgn_depths) - 1


def diffusion_process(config):
    return DiffusionProcess(
        num_steps=int(config.get('ldgn_diffusion_steps', 1000)),
        schedule_type=str(config.get('ldgn_schedule', 'linear')),
    )


# ---------------------------------------------------------------------------
# Data shared by both stages
# ---------------------------------------------------------------------------

def _split_groups(config, log=print):
    train_ids, val_ids, test_ids = md.split_ids(config)
    attr = md.group_attr(config)
    h5_path = config['dataset_dir']
    train = md.group_ids(h5_path, train_ids, attr)
    val = md.group_ids(h5_path, val_ids, attr) if val_ids else {}
    log(f'Split (split_seed {config.get("split_seed", 42)}, grouped by {attr!r}): '
        f'{len(train)} train / {len(val)} val / '
        f'{len(md.group_ids(h5_path, test_ids, attr)) if test_ids else 0} test systems; '
        f'{len(train_ids)} / {len(val_ids)} / {len(test_ids)} draws')
    return train, val


def _loader(dataset, batch_size, num_workers=0, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


# ---------------------------------------------------------------------------
# Epoch callback: log line, validation loss, periodic pictures
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validation_losses(model, dataset, batch_size, device):
    """Mean of the stage's own training losses over a fixed set of val draws,
    under a fixed RNG (the VGAE samples its latent, the LDGN a diffusion step
    and noise)."""
    if dataset is None or len(dataset) == 0:
        return {}
    devices = [device.index] if device.type == 'cuda' and device.index is not None else []
    was_training = model.training
    model.eval()
    sums, count = {}, 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VAL_SEED)
        for graph in _loader(dataset, batch_size):
            graph = graph.to(device)
            model.training_step(graph)
            for name, value in model.losses.items():
                v = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                sums[name] = sums.get(name, 0.0) + v
            count += 1
    model.train(was_training)
    return {k: v / max(count, 1) for k, v in sums.items()}


def _due(epoch, interval, max_epochs):
    interval = int(interval or 0)
    if interval <= 0:
        return False
    return (epoch + 1) % interval == 0 or epoch == max_epochs - 1


class _EpochReporter:
    def __init__(self, config, log_path, val_dataset, batch_size, device, max_epochs,
                 elapsed_before=0.0, picture=None, extra_names=()):
        self.config = config
        self.log_path = log_path
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.max_epochs = max_epochs
        self.elapsed_before = float(elapsed_before)
        self.start = time.time()
        self.picture = picture
        self.extra_names = extra_names
        self.val_interval = int(config.get('val_interval', 10) or 0)
        self.test_interval = int(config.get('test_interval', 0) or 0)

    def elapsed(self):
        return self.elapsed_before + (time.time() - self.start)

    def extra_state(self):
        return {'elapsed': self.elapsed()}

    def __call__(self, epoch, stats, model):
        body = f"Train {stats.get('train_loss', float('nan')):.6e}"
        val = {}
        if self.val_dataset is not None and _due(epoch, self.val_interval, self.max_epochs):
            val = _validation_losses(model, self.val_dataset, self.batch_size, self.device)
            if 'train_loss' in val:
                body += f" Valid {val['train_loss']:.6e}"
        body += f" LR {stats['lr']:.4e}"
        # Every loss component, train and (on validation epochs) val, for the loss plots
        extras = [f'{short}={stats[name]:.4e}' for name, short in self.extra_names if name in stats]
        extras += [f'val_{short}={val[name]:.4e}' for name, short in self.extra_names if name in val]
        if extras:
            body += ' | ' + ' '.join(extras)
        # stdout leads with `Epoch k/max`, the line the Studio's metrics reader
        # parses (studio_backend/training_metrics.py); the log file keeps the
        # MGN-V `Elapsed: ...s Epoch k` line that score_rank.py reads.
        print(f"Epoch {epoch}/{self.max_epochs} {body}", flush=True)
        append_log(self.log_path, f"Elapsed: {self.elapsed():.2f}s Epoch {epoch} {body}")
        display = bool(self.config.get('display_testset', True)) or bool(self.config.get('display_trainset', False))
        if self.picture is not None and display and _due(epoch, self.test_interval, self.max_epochs):
            try:
                self.picture(model, epoch)
            except Exception as exc:  # a picture never stops a training
                print(f'  (periodic visualization skipped: {exc})')


# ---------------------------------------------------------------------------
# Periodic pictures
# ---------------------------------------------------------------------------

def _plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print('  (matplotlib not installed: periodic visualization skipped)')
        return None


def _vae_parity_picture(config, datasets, stats, device):
    """Reconstruction vs truth, one panel per output channel (physical units)."""
    @torch.no_grad()
    def draw(model, epoch):
        plt = _plt()
        if plt is None:
            return
        out_dir = config.get('log_dir') or '.'
        for prefix, dataset in datasets:
            if dataset is None or len(dataset) == 0:
                continue
            graph = _loader(dataset, 1).__iter__().__next__().to(device)
            graph.field = graph.target
            was_training = model.training
            model.eval()
            pred = model(graph)[0]
            model.train(was_training)
            p = md.denormalize_target(pred.cpu().numpy(), stats)
            t = md.denormalize_target(graph.target.cpu().numpy(), stats)
            keep = np.random.default_rng(0).permutation(len(t))[:20000]
            n = p.shape[1]
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            for c in range(n):
                ax = axes[0, c]
                ax.scatter(t[keep, c], p[keep, c], s=1, alpha=0.3)
                lo, hi = float(min(t[:, c].min(), p[:, c].min())), float(max(t[:, c].max(), p[:, c].max()))
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
                ax.set_xlabel(f'truth ch{c}')
                ax.set_ylabel(f'VGAE reconstruction ch{c}')
            sid = int(graph.sample_id[0])
            fig.suptitle(f'VGAE epoch {epoch}: sample {sid} ({prefix or "val"})')
            fig.tight_layout()
            sub = os.path.join(out_dir, prefix) if prefix else out_dir
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, f'vae_parity_ep{epoch}.png')
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f'  periodic picture: {path}')
    return draw


def _ldgn_sample_picture(config, systems, stats, device):
    """LDGN samples of one system vs every FEA draw of it: histogram of the
    spread statistic used by the inference score (default max |u|)."""
    from inference_profiles.sampler import sample_system, spread_of
    from inference_profiles.spread import _spread_index, _spread_stat_expr

    @torch.no_grad()
    def draw(model, epoch):
        plt = _plt()
        if plt is None:
            return
        out_dir = config.get('log_dir') or '.'
        idx, stat = _spread_index(config)
        label = str(config.get('spread_label', 'u'))
        n = int(config.get('viz_num_samples', 16))
        for prefix, (h5_path, key, sids, static) in systems:
            gen = sample_system(model, static, h5_path, sids[0], stats, config, n, device,
                                seed=int(config.get('sample_seed', 0)))
            gen_vals = [spread_of(g, idx, stat) for g in gen]
            truth = []
            with h5py.File(h5_path, 'r') as f:
                out = int(stats['output_var'])
                for sid in sids:
                    field = f[f'data/{sid}/nodal_data'][3:3 + out, -1, :].T
                    truth.append(spread_of(field, idx, stat))
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.linspace(min(min(truth), min(gen_vals)), max(max(truth), max(gen_vals)) + 1e-12, 25)
            ax.hist(truth, bins=bins, alpha=0.55, density=True, label=f'FEA draws (n={len(truth)})')
            ax.hist(gen_vals, bins=bins, alpha=0.55, density=True, label=f'LDGN samples (n={len(gen_vals)})')
            ax.set_xlabel(_spread_stat_expr(label, stat))
            ax.set_ylabel('density')
            ax.legend()
            ax.set_title(f'LDGN epoch {epoch}: system {key} ({prefix or "val"})')
            fig.tight_layout()
            sub = os.path.join(out_dir, prefix) if prefix else out_dir
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, f'ldgn_samples_ep{epoch}.png')
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f'  periodic picture: {path}')
        model.train()
    return draw


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _settings(config, prefix, generator):
    lr = as_float(config.get(f'{prefix}_learningr'))
    return TrainingSettings(
        model_name=prefix,
        lr=lr,
        max_epochs=int(config.get(f'{prefix}_training_epochs')),
        batch_size=int(config.get(f'{prefix}_batch_size')),
        num_workers=int(config.get('num_workers', 0)),
        gradient_clip_val=as_float(config.get('grad_clip', 1.0)) or None,
        accumulate_grad_batches=int(config.get(f'{prefix}_grad_accum', 1)),
        lr_scheduler_factor=as_float(config.get(f'{prefix}_lr_factor', 0.1)),
        lr_scheduler_patience=int(config.get(f'{prefix}_lr_patience', 250)),
        lr_threshold=as_float(config.get(f'{prefix}_lr_threshold', 0.0)),
        kl_reg=as_float(config.get('vae_kl_reg', 1e-6)),
        lambda_vlb=as_float(config.get('ldgn_lambda_vlb', 1e-3)),
        step_sampler=str(config.get('ldgn_step_sampler', 'importance')),
        generator=generator,
    )


def _datasets(config, train_groups, val_groups, statics, val_statics, stats):
    train_set = md.SystemDrawDataset(config['dataset_dir'], train_groups, statics, stats, config,
                                     mode='random', draws_per_epoch=int(config.get('draws_per_epoch', 1)))
    val_set = (md.SystemDrawDataset(config['dataset_dir'], val_groups, val_statics, stats, config,
                                    mode='fixed', draws_per_system=int(config.get('val_draws_per_geometry', 1)))
               if val_groups else None)
    return train_set, val_set


# ---------------------------------------------------------------------------
# Stage 1: VGAE
# ---------------------------------------------------------------------------

def train_vae(config, config_filename):
    # `[Stage k/2] Training <name>` is the header the Studio splits a two-stage log at.
    print('\n[Stage 1/2] Training VAE (LDGN stage 1: variational graph autoencoder)', flush=True)
    device = resolve_device(config)
    generator = training_generator(config)
    modelpath = config['vae_modelpath']

    train_groups, val_groups = _split_groups(config)
    stats = md.compute_normalization(config, train_groups)
    arch = vae_arch(config, stats)
    num_scales = len(arch['depths'])
    statics = md.build_statics(config, train_groups, stats, num_scales, what='train systems')
    val_statics = md.build_statics(config, val_groups, stats, num_scales, what='val systems') if val_groups else {}

    model = VGAE(arch=arch, device=device)
    print(f'VGAE: {model.num_params:,} parameters, depths {arch["depths"]}, '
          f'{arch["latent_node_features"]} latent feature(s) per level-{num_scales} node')

    train_set, val_set = _datasets(config, train_groups, val_groups, statics, val_statics, stats)
    ts = _settings(config, 'vae', generator)
    resume = TrainingResume(config, modelpath)
    state = resume.load()
    log_path = init_stage_log(config, config_filename, 'vae',
                              resumed_at=None if state is None else state['epoch'] + 1)
    shown = []
    if bool(config.get('display_testset', True)) and val_set is not None:
        shown.append(('', val_set))
    if bool(config.get('display_trainset', False)):
        shown.append(('train', md.SystemDrawDataset(config['dataset_dir'], train_groups, statics, stats,
                                                    config, mode='fixed', draws_per_system=1)))
    reporter = _EpochReporter(
        config, log_path, val_set, ts.batch_size, device, ts.max_epochs,
        elapsed_before=0.0 if state is None else state.get('elapsed', 0.0),
        picture=_vae_parity_picture(config, shown, stats, device),
        extra_names=(('reconstruction_loss', 'recon'), ('kl_divergence', 'kl')))
    final = model.fit(ts, train_set, epoch_callback=reporter, resume=resume, start_state=state,
                      extra_state=reporter.extra_state)

    save_checkpoint(modelpath, model.checkpoint_dict(
        stage='vae', normalization=stats, config=checkpoint_config(config), num_scales=num_scales,
        epoch=int(final.get('epoch', -1)), stopped_early=bool(final.get('stopped_early', False)),
        elapsed=reporter.elapsed()))
    resume.finish()
    append_log(log_path, f'==== VAE finished: {modelpath} ====')
    return modelpath


# ---------------------------------------------------------------------------
# Stage 2: LDGN
# ---------------------------------------------------------------------------

def train_ldgn(config, config_filename):
    print('\n[Stage 2/2] Training LDGN (LDGN stage 2: latent diffusion graph net)', flush=True)
    device = resolve_device(config)
    generator = training_generator(config)
    vae_path = config['vae_modelpath']
    modelpath = config['modelpath']

    vae_ckpt = load_checkpoint(vae_path)
    if vae_ckpt.get('stage') != 'vae':
        raise ValueError(f'{vae_path} is not an LDGN-route VGAE checkpoint (stage {vae_ckpt.get("stage")!r})')
    stats = vae_ckpt['normalization']
    check_normalization_matches(stats, config, vae_path)

    vae_arch_ = decode_arch(vae_ckpt['arch'])
    arch = ldgn_arch(config, vae_arch_)
    num_scales = num_scales_for(vae_arch_['depths'], arch['depths'])

    train_groups, val_groups = _split_groups(config)
    statics = md.build_statics(config, train_groups, stats, num_scales, what='train systems')
    val_statics = md.build_statics(config, val_groups, stats, num_scales, what='val systems') if val_groups else {}

    model = LatentDiffusionGraphNet(
        autoencoder_checkpoint=vae_ckpt,
        arch=arch,
        diffusion_process=diffusion_process(config),
        learnable_variance=bool(config.get('ldgn_learnable_variance', True)),
        device=device,
    )
    print(f'LDGN: {model.num_learnable_params:,} trainable parameters ({model.num_params:,} with the '
          f'frozen VGAE), depths {arch["depths"]} from level {model.scale_0 + 1} of {num_scales}, '
          f'{model.diffusion_process.num_steps} diffusion steps')

    train_set, val_set = _datasets(config, train_groups, val_groups, statics, val_statics, stats)
    ts = _settings(config, 'ldgn', generator)
    resume = TrainingResume(config, modelpath, upstream=(vae_path,))
    state = resume.load()
    log_path = init_stage_log(config, config_filename, 'ldgn',
                              resumed_at=None if state is None else state['epoch'] + 1)
    systems = []
    if bool(config.get('display_testset', True)) and val_groups:
        key = next(iter(val_groups))
        systems.append(('', (config['dataset_dir'], key, val_groups[key], val_statics[key])))
    if bool(config.get('display_trainset', False)):
        key = next(iter(train_groups))
        systems.append(('train', (config['dataset_dir'], key, train_groups[key], statics[key])))
    reporter = _EpochReporter(
        config, log_path, val_set, ts.batch_size, device, ts.max_epochs,
        elapsed_before=0.0 if state is None else state.get('elapsed', 0.0),
        picture=_ldgn_sample_picture(config, systems, stats, device),
        extra_names=(('mse_loss', 'mse'), ('vlb_loss', 'vlb')))
    final = model.fit(ts, train_set, epoch_callback=reporter, resume=resume, start_state=state,
                      extra_state=reporter.extra_state)

    save_checkpoint(modelpath, model.checkpoint_dict(
        stage='ldgn', normalization=stats, config=checkpoint_config(config),
        autoencoder_arch=encode_arch(vae_arch_), num_scales=num_scales, vae_modelpath=str(vae_path),
        epoch=int(final.get('epoch', -1)), stopped_early=bool(final.get('stopped_early', False)),
        elapsed=reporter.elapsed()))
    resume.finish()
    append_log(log_path, f'==== LDGN finished: {modelpath} ====')
    return modelpath
