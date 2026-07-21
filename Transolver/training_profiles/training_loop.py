"""
Train/validate/test loops (mirrors MeshGraphNets' training_profiles/training_loop.py),
with topology-independent visualization boundary (section 12 Phase 4): this repo's
graphs carry no edge_attr, so periodic evaluation reports numeric metrics
(denormalized per-feature RMSE/MAE, per-sample relative L2) and optionally
dumps raw prediction HDF5 files. Full 3D mesh/triangle visualization (pyvista)
is deliberately out of scope for this pass -- it is MGN-specific plumbing
(edges_to_triangles, mesh_utils_fast) that this architecture does not need for
its numeric evaluation contract, and section 12 only requires visualization to
be decoupled from edge attributes, not reimplemented.
"""

import os
import time

import h5py
import numpy as np
import torch
import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from general_modules.time_integration import resolve_rollout_window
from training_profiles.ar_rollout import (
    RolloutContext,
    ar_rt_enabled,
    describe_ar_rt,
    rollout_loss,
)


def build_ema_model(model, config):
    """Create an EMA shadow model if use_ema is enabled."""
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=decay))
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _build_loss_weights(config, device):
    """Build per-feature loss weights normalized to sum to 1."""
    loss_weights = config.get('feature_loss_weights', None)
    if loss_weights is not None:
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights]
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)
        loss_weights = loss_weights / loss_weights.sum()
    return loss_weights


def _per_node_loss(errors, loss_weights):
    """Reduce feature errors to one scalar per node."""
    if loss_weights is not None:
        return torch.sum(errors * loss_weights, dim=-1)
    return torch.mean(errors, dim=-1)


def _loss_from_errors(errors, loss_weights):
    """Return mean loss used for backprop plus exact aggregation stats.

    The batch sum is returned as a detached 0-dim GPU tensor, not a Python
    float: .item() here would force a CPU<->GPU sync on every batch.
    """
    per_node = _per_node_loss(errors, loss_weights)
    loss_sum = per_node.sum()
    loss_count = per_node.numel()
    return loss_sum / loss_count, loss_sum.detach(), loss_count


def _build_rollout_context(config, device):
    """Return an AR-RT context, or None when the run is AR-OT."""
    return RolloutContext(config, device) if ar_rt_enabled(config) else None


def _forward_and_loss(model, graph, ctx, loss_weights, training):
    """One optimization target for a batch, under whichever scheme is active.

    AR-OT evaluates the model once on a ground-truth input pair; AR-RT unrolls
    it over the trajectory. Both return `(loss, loss_sum, loss_count)`.
    """
    def loss_fn(prediction, target):
        errors = torch.nn.functional.mse_loss(prediction, target, reduction='none')
        return _loss_from_errors(errors, loss_weights)

    if ctx is not None:
        return rollout_loss(model, graph, ctx, loss_fn, training=training)

    predicted, target = model(graph) if training else model(graph, add_noise=False)
    return loss_fn(predicted, target)


def _move_graph_to_device(graph, device, config):
    non_blocking = bool(config.get('_pin_memory', False)) and getattr(device, 'type', None) == 'cuda'
    return graph.to(device, non_blocking=non_blocking)


def _accum_window_size(batch_idx, total_batches, actual_accum):
    window_start = (batch_idx // actual_accum) * actual_accum
    window_end = min(window_start + actual_accum, total_batches)
    return window_end - window_start


def log_training_config(config):
    """Log loss weights and architecture switches to stdout."""
    loss_weights_cfg = config.get('feature_loss_weights', None)
    if loss_weights_cfg is not None:
        if not isinstance(loss_weights_cfg, list):
            loss_weights_cfg = [loss_weights_cfg]
        w = torch.tensor(loss_weights_cfg, dtype=torch.float32)
        w_normalized = (w / w.sum()).tolist()
        print(f"Per-feature loss weights (raw):         {loss_weights_cfg}")
        print(f"Per-feature loss weights (normalized):  {[f'{v:.4f}' for v in w_normalized]}")
    else:
        print("Per-feature loss weights: equal (default)")
    print(f"attention_kernel: {config.get('attention_kernel', 'naive')}, "
          f"chunk_size: {config.get('chunk_size', 0)}")
    if ar_rt_enabled(config):
        print(describe_ar_rt(resolve_rollout_window(config, int(config.get('num_timesteps', 1)))))
    else:
        print("Time integration: AR-OT (one-step / teacher-forced training)")


def train_epoch(model, dataloader, optimizer, device, config, epoch, ema_model=None):
    """max_train_batches (config, default 0 = unlimited) caps how many
    batches run per epoch. This does not change what full-dataset training
    computes -- it exists so a smoke/verification run on a dataset with very
    many items per epoch (e.g. a large temporal dataset) can exercise real
    forward/backward passes on real data without waiting for a full epoch."""
    model.train()
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss_count = 0

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    grad_accum_steps = config.get('grad_accum_steps', 1)
    max_train_batches = int(config.get('max_train_batches', 0))
    total_batches = len(dataloader)
    if max_train_batches > 0:
        total_batches = min(total_batches, max_train_batches)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps
    max_grad_norm = float(config.get('max_grad_norm', 3.0))

    rollout_ctx = _build_rollout_context(config, device)

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm.tqdm(dataloader, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        if batch_idx >= total_batches:
            break
        graph = _move_graph_to_device(graph, device, config)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            loss, batch_loss_sum, batch_loss_count = _forward_and_loss(
                model, graph, rollout_ctx, loss_weights, training=True,
            )
            scaled_loss = loss / _accum_window_size(batch_idx, total_batches, actual_accum)

        scaled_loss.backward()

        total_loss_sum += batch_loss_sum.double()
        total_loss_count += batch_loss_count

        is_last_batch = batch_idx == total_batches - 1
        if (batch_idx + 1) % actual_accum == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            loss_val = batch_loss_sum.item() / batch_loss_count
            pbar.set_postfix({'loss': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'})

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def _evaluate_epoch(model, dataloader, device, config, *, progress_name='Validation'):
    """max_val_batches (config, default 0 = unlimited): see train_epoch's
    max_train_batches docstring -- same rationale, applied to validation."""
    model.eval()

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    max_val_batches = int(config.get('max_val_batches', 0))
    # Validation follows the training scheme: under AR-RT the reported loss is
    # the rollout loss, so best-checkpoint selection optimizes the quantity the
    # run is actually training for.
    rollout_ctx = _build_rollout_context(config, device)

    with torch.no_grad():
        total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        total_loss_count = 0

        total_batches = len(dataloader)
        if max_val_batches > 0:
            total_batches = min(total_batches, max_val_batches)

        pbar = tqdm.tqdm(dataloader, desc=progress_name, total=total_batches)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= total_batches:
                break
            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                _, batch_loss_sum, batch_loss_count = _forward_and_loss(
                    model, graph, rollout_ctx, loss_weights, training=False,
                )

            if batch_idx % 10 == 0:
                mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                loss_val = batch_loss_sum.item() / batch_loss_count
                pbar.set_postfix({'loss': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss_sum += batch_loss_sum.double()
            total_loss_count += batch_loss_count

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def validate_epoch(model, dataloader, device, config, epoch=0):
    return _evaluate_epoch(model, dataloader, device, config, progress_name='Validation')


def _scalar_attr(graph, name):
    value = getattr(graph, name, None)
    if value is None:
        return None
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, '__getitem__') and len(value) > 0:
        return int(value[0])
    return int(value)


def run_periodic_test(model, test_loader, device, config, epoch, train_dataset):
    """Test-set evaluation plus optional train-set reconstruction check.
    Shared by the single-GPU and DDP launchers at test_interval cadence."""
    start = time.time()
    test_loss = test_model(model, test_loader, device, config, epoch, train_dataset)
    print(f"  Test loss: {test_loss:.2e} ({time.time() - start:.1f}s)")

    if config.get('display_trainset', True):
        viz_indices = config.get('test_batch_idx', [0, 1, 2, 3])
        viz_indices = [i for i in viz_indices if i < len(train_dataset)]
        if viz_indices:
            viz_loader = DataLoader(
                Subset(train_dataset, viz_indices),
                batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
            )
            viz_loss = test_model(model, viz_loader, device, config, epoch,
                                  train_dataset, output_prefix='train')
            print(f"  Train reconstruction loss: {viz_loss:.2e}")
    return test_loss


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    """Numeric evaluation: normalized MSE (for the loss curve) plus
    denormalized per-feature RMSE/MAE and per-sample relative L2 (section 9's
    fair-comparison metrics). Optionally dumps raw prediction HDF5 files
    (numeric arrays only -- no mesh/triangle visualization, section 12)."""
    model.eval()

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    total_test = len(dataloader)
    max_test_batches = int(config.get('test_max_batches', 200))
    effective_total = min(max_test_batches, total_test)

    delta_mean = delta_std = None
    if dataset is not None:
        delta_mean = dataset.delta_mean
        delta_std = dataset.delta_std

    output_var = config.get('output_var')
    rmse_sum = np.zeros(output_var, dtype=np.float64)
    mae_sum = np.zeros(output_var, dtype=np.float64)
    l2re_sum = 0.0
    n_eval_samples = 0

    write_hdf5 = config.get('write_test_predictions', False)
    gpu_ids = str(config.get('gpu_ids'))

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0

        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break

            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                predicted, target = model(graph)
                errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
                loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            pbar.set_postfix({'loss': f'{loss.item():.2e}'})
            total_loss_sum += batch_loss_sum.item()
            total_loss_count += batch_loss_count

            predicted_np = predicted.float().cpu().numpy()
            target_np = target.float().cpu().numpy()

            if delta_mean is not None and delta_std is not None:
                predicted_denorm = predicted_np * delta_std + delta_mean
                target_denorm = target_np * delta_std + delta_mean

                diff = predicted_denorm - target_denorm
                rmse_sum += np.mean(diff ** 2, axis=0)
                mae_sum += np.mean(np.abs(diff), axis=0)
                num = np.linalg.norm(diff)
                den = np.linalg.norm(target_denorm)
                l2re_sum += (num / den) if den > 0 else 0.0
                n_eval_samples += 1

                if write_hdf5 and batch_idx in config.get('test_batch_idx', [0, 1, 2, 3]):
                    sample_id = _scalar_attr(graph, 'sample_id')
                    time_idx = _scalar_attr(graph, 'time_idx')
                    filename = (f'sample{sample_id}_t{time_idx}' if time_idx is not None
                               else f'sample{sample_id}' if sample_id is not None
                               else f'batch{batch_idx}')
                    output_path = f'outputs/{output_prefix}/{gpu_ids}/{epoch}/{filename}.h5'
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('predicted_denorm', data=predicted_denorm)
                        f.create_dataset('target_denorm', data=target_denorm)
                        f.create_dataset('pos', data=graph.pos.cpu().numpy())

        if n_eval_samples > 0:
            rmse = np.sqrt(rmse_sum / n_eval_samples)
            mae = mae_sum / n_eval_samples
            l2re = l2re_sum / n_eval_samples
            print(f"  Denormalized per-feature RMSE: {rmse}")
            print(f"  Denormalized per-feature MAE:  {mae}")
            print(f"  Mean per-sample relative L2:   {l2re:.4f}")

    return total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
