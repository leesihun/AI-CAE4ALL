"""Train/validate/test epoch loops (IMPLEMENTATION_PLAN.md section 12), ported
from MeshGraphNets' training_profiles/training_loop.py (commit c559458) with
the exact loss/accumulation/AMP/EMA/logging behavior preserved. The model call
signature (`model(graph, add_noise=None) -> (prediction, target_or_none)`) is
identical to MGN's, so this file never branches on which of the four
architectures is selected (section 6.2/A.7).

One addition beyond the MGN original: `update_ema` also copies BatchNorm
running-mean/var buffers from the live model into the EMA shadow after every
parameter update. `torch.optim.swa_utils.AveragedModel` only averages
*parameters*; MGN never needed this because it has no BatchNorm anywhere, but
Point-DeepONet's PointNet branch does (section 5.2 item 6 / 12.4).
"""

import os
import time

import tqdm
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from general_modules.mesh_utils_fast import (
    edges_to_triangles_gpu,
    edges_to_triangles_optimized,
    render_plot_data,
    save_inference_results_fast,
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


def update_ema(ema_model, model) -> None:
    """Update EMA parameters, then copy (not average) BatchNorm buffers."""
    ema_model.update_parameters(model)
    for b_ema, b_model in zip(ema_model.module.buffers(), model.buffers()):
        b_ema.data.copy_(b_model.data)


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
    float: .item() here would force a CPU<->GPU sync on every batch and
    serialize the CUDA pipeline. Callers accumulate on-device and convert
    once per epoch (or every N batches for progress display).
    """
    per_node = _per_node_loss(errors, loss_weights)
    loss_sum = per_node.sum()
    loss_count = per_node.numel()
    return loss_sum / loss_count, loss_sum.detach(), loss_count


def _move_graph_to_device(graph, device, config):
    non_blocking = bool(config.get('_pin_memory', False)) and getattr(device, 'type', None) == 'cuda'
    return graph.to(device, non_blocking=non_blocking)


def _accum_window_size(batch_idx, total_batches, actual_accum):
    """Return the number of batches in the current accumulation window."""
    window_start = (batch_idx // actual_accum) * actual_accum
    window_end = min(window_start + actual_accum, total_batches)
    return window_end - window_start


def _is_paper_darcy(config) -> bool:
    return (str(config.get('model', '')).lower() == 'fno'
            and str(config.get('fno_variant', 'mesh')).lower() == 'paper_darcy')


def _paper_relative_l2_batch(prediction, target, ptr, target_mean, target_std):
    """Decoded per-sample relative L2 used by the paper-era Darcy recipe.

    The returned optimization loss is a SUM over samples, matching the
    released ``LpLoss(size_average=False)`` behavior. Epoch statistics retain
    an explicit sample count so reporting is the corresponding mean.
    """
    prediction = prediction * target_std + target_mean
    target = target * target_std + target_mean
    values = []
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        difference_norm = torch.linalg.vector_norm(prediction[start:end] - target[start:end])
        target_norm = torch.linalg.vector_norm(target[start:end])
        values.append(difference_norm / target_norm)
    per_sample = torch.stack(values)
    loss_sum = per_sample.sum()
    return loss_sum, loss_sum.detach(), per_sample.numel()


def _paper_target_stats(config, device):
    try:
        mean = config['_paper_target_mean']
        std = config['_paper_target_std']
    except KeyError as exc:
        raise RuntimeError(
            "paper_darcy loss requires train-derived _paper_target_mean/std"
        ) from exc
    return (
        torch.as_tensor(mean, dtype=torch.float32, device=device),
        torch.as_tensor(std, dtype=torch.float32, device=device),
    )


def _train_epoch_paper_darcy(model, dataloader, optimizer, device, config, epoch):
    """Isolated paper loss path; the normal MSE hot loop remains unchanged."""
    model.train()
    getattr(model, 'module', model).set_epoch(epoch)
    target_mean, target_std = _paper_target_stats(config, device)
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss_count = 0

    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch_idx, graph in enumerate(pbar):
        graph = _move_graph_to_device(graph, device, config)
        optimizer.zero_grad(set_to_none=True)
        prediction, target = model(graph)
        loss, batch_loss_sum, batch_loss_count = _paper_relative_l2_batch(
            prediction, target, graph.ptr, target_mean, target_std
        )
        loss.backward()
        # The released Darcy recipe does not apply gradient clipping.
        optimizer.step()

        total_loss_sum += batch_loss_sum.double()
        total_loss_count += batch_loss_count
        if batch_idx % 10 == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'relative_l2': f'{batch_loss_sum.item() / batch_loss_count:.2e}',
                'mem': f'{mem_gb:.1f}GB',
            })

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def _evaluate_epoch_paper_darcy(model, dataloader, device, config, progress_name):
    model.eval()
    target_mean, target_std = _paper_target_stats(config, device)
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss_count = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for batch_idx, graph in enumerate(pbar):
            graph = _move_graph_to_device(graph, device, config)
            prediction, target = model(graph, add_noise=False)
            _, batch_loss_sum, batch_loss_count = _paper_relative_l2_batch(
                prediction, target, graph.ptr, target_mean, target_std
            )
            total_loss_sum += batch_loss_sum.double()
            total_loss_count += batch_loss_count
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'relative_l2': f'{batch_loss_sum.item() / batch_loss_count:.2e}'
                })
    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


# Batches skipped before profiling starts (allocator/cudnn warmup): 2 wait + 2 warmup.
_PROFILE_SKIP_BATCHES = 4


def _start_profiler(config, epoch):
    """Start a torch profiler for the first `profile_batches` batches of epoch 0.

    Set `profile_batches N` in the config to enable. Returns None when disabled.
    """
    profile_batches = int(config.get('profile_batches', 0))
    if profile_batches <= 0 or epoch != 0:
        return None

    from torch.profiler import ProfilerActivity, profile, schedule
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    profiler = profile(
        activities=activities,
        schedule=schedule(wait=2, warmup=2, active=profile_batches, repeat=1),
    )
    profiler.start()
    tqdm.tqdm.write(
        f"Profiling batches {_PROFILE_SKIP_BATCHES}..{_PROFILE_SKIP_BATCHES + profile_batches - 1} "
        f"of epoch 0 (profile_batches={profile_batches})"
    )
    return profiler


def _finish_profiler(profiler, config):
    """Stop the profiler, print a kernel-time summary, and export a chrome trace."""
    profiler.stop()
    sort_key = 'self_cuda_time_total' if torch.cuda.is_available() else 'self_cpu_time_total'
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=30))
    trace_path = os.path.join(config.get('log_dir', '.'), 'train_profile_trace.json')
    try:
        profiler.export_chrome_trace(trace_path)
        print(f"Profiler trace written to {trace_path} (open in chrome://tracing or https://ui.perfetto.dev)")
    except Exception as e:
        print(f"Warning: could not export profiler trace: {e}")


def log_training_config(config):
    """Log loss weights and the resolved model/adapter switches to stdout."""
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
    print(f"Model: {config.get('model')}")
    if _is_paper_darcy(config):
        print("Loss: decoded mean per-sample relative L2 (paper_darcy opt-in)")


def train_epoch(model, dataloader, optimizer, device, config, epoch, ema_model=None):
    if _is_paper_darcy(config):
        return _train_epoch_paper_darcy(
            model, dataloader, optimizer, device, config, epoch
        )
    model.train()
    # DistributedDataParallel does not delegate custom methods to the wrapped
    # module (torch.compile's OptimizedModule does); unwrap so the epoch
    # reaches OperatorWrapper.set_epoch in the DDP path too.
    getattr(model, 'module', model).set_epoch(epoch)
    # On-device accumulator: adding batch sums tensor-to-tensor keeps the loop
    # free of CPU<->GPU syncs; converted to a Python float once at epoch end.
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss_count = 0

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    optimizer.zero_grad(set_to_none=True)

    profiler = _start_profiler(config, epoch)
    profile_end_batch = _PROFILE_SKIP_BATCHES + int(config.get('profile_batches', 0))

    pbar = tqdm.tqdm(dataloader, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        graph = _move_graph_to_device(graph, device, config)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            predicted_acc, target_acc = model(graph)
            errors = torch.nn.functional.mse_loss(predicted_acc, target_acc, reduction='none')
            loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            scaled_loss = loss / _accum_window_size(batch_idx, total_batches, actual_accum)

        scaled_loss.backward()

        total_loss_sum += batch_loss_sum.double()
        total_loss_count += batch_loss_count

        is_last_batch = batch_idx == total_batches - 1
        if (batch_idx + 1) % actual_accum == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get('max_grad_norm', 3.0)))
            optimizer.step()
            if ema_model is not None:
                update_ema(ema_model, model)
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            # The only deliberate sync in the loop: one .item() per 10 batches
            # to keep the progress bar live without stalling the pipeline.
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            loss_val = batch_loss_sum.item() / batch_loss_count
            pbar.set_postfix({'loss': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'})

        if profiler is not None:
            profiler.step()
            if batch_idx + 1 >= profile_end_batch:
                _finish_profiler(profiler, config)
                profiler = None

    if profiler is not None:  # dataloader shorter than the profiling window
        _finish_profiler(profiler, config)
        profiler = None

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def _evaluate_epoch(model, dataloader, device, config, *, progress_name='Validation'):
    if _is_paper_darcy(config):
        return _evaluate_epoch_paper_darcy(
            model, dataloader, device, config, progress_name
        )
    model.eval()

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        total_loss_count = 0

        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for batch_idx, graph in enumerate(pbar):
            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target = model(graph, add_noise=False)
                errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
            _, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

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
    """Extract a scalar graph attribute that may be a tensor, list, or scalar."""
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


def _as_list(value):
    """Normalize a config value that may be a bare scalar (single-element
    lists collapse to a plain int/float during parsing, section 11.1) into
    a list."""
    if value is None:
        return value
    if isinstance(value, list):
        return value
    return [value]


def run_periodic_test(model, test_loader, device, config, epoch, train_dataset):
    """Test-set evaluation plus optional train-set reconstruction visualization.

    Shared by the single-GPU and DDP launchers at `test_interval` cadence.
    Returns the test loss.
    """
    start = time.time()
    test_loss = test_model(model, test_loader, device, config, epoch, train_dataset)
    print(f"  Test loss: {test_loss:.2e} ({time.time() - start:.1f}s)")

    if config.get('display_trainset', True):
        viz_indices = _as_list(config.get('test_batch_idx', [0, 1, 2, 3, 4, 5, 6, 7]))
        viz_indices = [i for i in viz_indices if i < len(train_dataset)]
        if viz_indices:
            viz_loader = DataLoader(
                Subset(train_dataset, viz_indices),
                batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
            )
            viz_config = dict(config)
            viz_config['test_batch_idx'] = list(range(len(viz_indices)))
            viz_loss = test_model(model, viz_loader, device, viz_config, epoch,
                                  train_dataset, output_prefix='train')
            print(f"  Train reconstruction loss: {viz_loss:.2e}")
    return test_loss


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    model.eval()

    loss_weights = _build_loss_weights(config, device)
    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else (device != 'cpu')
    mesh_device = device if use_gpu else 'cpu'
    faces_cache = {}

    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    total_test = len(dataloader)
    max_test_batches = int(config.get('test_max_batches', 200))
    effective_total = min(max_test_batches, total_test)
    if effective_total < total_test:
        print(f"  Test: evaluating {effective_total}/{total_test} samples (set test_max_batches in config to change)")

    delta_mean = None
    delta_std = None
    if dataset is not None:
        delta_mean = dataset.delta_mean
        delta_std = dataset.delta_std
        if delta_mean is not None and delta_std is not None:
            print(f"Using denormalization: delta_mean={delta_mean}, delta_std={delta_std}")

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break

            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target = model(graph)
                errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
                loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            # Test runs rarely and already syncs for visualization output;
            # a per-batch .item() here is harmless.
            total_loss_sum += batch_loss_sum.item()
            total_loss_count += batch_loss_count

            if batch_idx in _as_list(config.get('test_batch_idx', [0, 1, 2, 3])):
                gpu_ids = str(config.get('gpu_ids'))
                sample_id = _scalar_attr(graph, 'sample_id')
                time_idx = _scalar_attr(graph, 'time_idx')

                if sample_id is not None and time_idx is not None:
                    filename = f'sample{sample_id}_t{time_idx}'
                elif sample_id is not None:
                    filename = f'sample{sample_id}'
                else:
                    filename = f'batch{batch_idx}'

                # config['log_dir'] is set by init_log_file() from log_file_dir,
                # the same directory the checkpoint/log already live in -- keep
                # periodic viz output there instead of a bare cwd-relative path.
                viz_base = config.get('log_dir', 'outputs')
                output_path = os.path.join(viz_base, output_prefix, gpu_ids, str(epoch), f'{filename}.h5')
                predicted_np = predicted.float().cpu().numpy() if hasattr(predicted, 'cpu') else predicted
                target_np = target.float().cpu().numpy() if hasattr(target, 'cpu') else target

                if delta_mean is not None and delta_std is not None:
                    predicted_denorm = predicted_np * delta_std + delta_mean
                    target_denorm = target_np * delta_std + delta_mean
                else:
                    predicted_denorm = predicted_np
                    target_denorm = target_np

                cached_faces = faces_cache.get(sample_id)
                if cached_faces is None and sample_id is not None:
                    if use_gpu and torch.cuda.is_available():
                        edge_index_gpu = graph.edge_index.to(mesh_device)
                        cached_faces = edges_to_triangles_gpu(edge_index_gpu, device=mesh_device)
                    else:
                        ei_np = (
                            graph.edge_index.cpu().numpy()
                            if hasattr(graph.edge_index, 'cpu')
                            else np.array(graph.edge_index)
                        )
                        cached_faces = edges_to_triangles_optimized(ei_np)
                    faces_cache[sample_id] = cached_faces

                plot_data = save_inference_results_fast(
                    output_path, graph,
                    predicted_norm=predicted_np, target_norm=target_np,
                    predicted_denorm=predicted_denorm, target_denorm=target_denorm,
                    skip_visualization=not config.get('display_testset', True),
                    device=mesh_device,
                    feature_idx=config.get('plot_feature_idx', -1),
                    precomputed_faces=cached_faces,
                )
                if plot_data:
                    plot_data_queue.append(plot_data)

        if plot_data_queue:
            print(f"\nRendering {len(plot_data_queue)} visualizations...")
            failed = 0
            for pd in plot_data_queue:
                if not render_plot_data(pd):
                    failed += 1
            if failed:
                print(f"Visualization done with {failed}/{len(plot_data_queue)} failures.")
            else:
                print("All visualizations complete!")

    return total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
