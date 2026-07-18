"""Node-sharded training launcher (IMPLEMENTATION_PLAN.md section 6.6 / Phase 7).

`parallel_mode node_shard` pools VRAM across ranks by splitting ONE mesh's nodes
across GPUs instead of giving each rank a whole mesh (that is `parallel_mode ddp`).
Weights are replicated; each rank holds ~N/world_size nodes' worth of activations;
the slice aggregates (num/den) are all-reduced inside every attention forward
(model/physics_attention.py), so the physics tokens are the exact whole-mesh
tokens and the result is single-process-identical (proven bit-for-bit by
tests/test_node_shard.py). Communication per layer is M*(H*D + H) values,
independent of N.

Gradient contract (the subtle part, pinned by the exactness test)
-----------------------------------------------------------------
The forward num/den reduce is autograd-aware, so after each rank backprops its
own shard loss `L_r = local_sum_r / global_count`, the token-pathway parameter
gradients already carry the global upstream factor G = dL/d(num) on every rank
(G is identical across ranks because the reduce is symmetric). The remaining
per-node "local pathway" gradients are still rank-local. A single SUM all-reduce
of every parameter's .grad after backward therefore yields exactly

    sum_r [ G * d(num_acc_r)/dtheta  +  dL_r/dtheta|_local ]  =  dL/dtheta,

with no double counting. This is why we do NOT wrap in DDP (which would MEAN-
reduce) and instead SUM-reduce grads manually here.

v1 scope: batch of one sharded mesh per optimizer micro-step (ptr-batching kept
orthogonal to sharding per the plan); full-mesh training (no node subsampling,
so physics tokens are unperturbed); normalized-MSE validation. Sharded test-set
denormalized metrics / HDF5 dumps (which need predictions gathered onto one rank)
are a documented follow-up, not part of v1.
"""

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.data import Data

from general_modules.load_config import load_config  # noqa: F401  (parity import)
from model.Transolver import Transolver
from training_profiles.setup import (
    build_dataset_splits,
    build_optimizer_scheduler,
    init_log_file,
    log_model_summary,
    save_checkpoint,
)
from training_profiles.training_loop import (
    _build_loss_weights,
    _per_node_loss,
    build_ema_model,
    log_training_config,
)


def _setup_process_group(rank, world_size, port):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(port)
    # NCCL only when it is actually built (Linux). On a Windows CUDA box NCCL is
    # unavailable, so fall back to gloo instead of crashing; the fast path is the
    # Linux/NCCL rig this is meant for.
    backend = 'nccl' if (torch.cuda.is_available() and dist.is_nccl_available()) else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _broadcast_module(model, src=0):
    """Replicate rank `src`'s parameters and buffers onto every rank, so all
    shards start from identical weights (node_shard does not use DDP, which would
    otherwise do this broadcast at construction)."""
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
        for b in model.buffers():
            dist.broadcast(b.data, src=src)


def _all_reduce_grads_sum(model, group):
    """SUM-reduce every parameter gradient across the shard group (see module
    docstring for why SUM, not mean)."""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)


def _epoch_order(n, epoch, seed, shuffle=True):
    """Sample visitation order for one epoch -- identical on every rank so all
    ranks shard the same mesh at the same step."""
    if not shuffle:
        return list(range(n))
    rng = np.random.default_rng(int(seed) * 100003 + int(epoch))
    return rng.permutation(n).tolist()


def _step_aug_seed(seed, epoch, step):
    """Deterministic per-step augmentation seed, identical across ranks, so
    every rank's np.random-driven geometric augmentation produces the SAME
    rotated mesh before it is sliced into node shards."""
    return (int(seed) * 1_000_003 + int(epoch) * 10_007 + int(step)) % (2 ** 31)


def _slice_graph_for_rank(graph, rank, world_size):
    """Take this rank's disjoint node shard of a full graph (strided, for load
    balance and geometric representativeness). Any partition covering all nodes
    is numerically valid because num/den is a global sum; strided is just even.

    Returns (sharded_graph, N_global). The sharded graph carries only what the
    forward consumes (x, pos_normalized, y) plus pos/ids for eval/logging -- no
    edge_index, since positional features are already baked into x.
    """
    N = graph.x.shape[0]
    if N < world_size:
        raise ValueError(
            f"node_shard needs at least world_size ({world_size}) nodes per mesh; "
            f"got a {N}-node graph. Use fewer ranks or a larger mesh."
        )
    idx = torch.arange(rank, N, world_size, dtype=torch.long)
    y = getattr(graph, 'y', None)
    pos = getattr(graph, 'pos', None)
    sharded = Data(
        x=graph.x[idx],
        y=(y[idx] if y is not None else None),
        pos=(pos[idx] if pos is not None else None),
        pos_normalized=graph.pos_normalized[idx],
        sample_id=getattr(graph, 'sample_id', None),
        time_idx=getattr(graph, 'time_idx', None),
    )
    return sharded, N


def _load_sharded(dataset, sample_idx, rank, world_size, seed, epoch, step, augment):
    """Deterministically build the full (optionally augmented) graph identically
    on every rank, then return this rank's node shard."""
    if augment:
        np.random.seed(_step_aug_seed(seed, epoch, step))
    full = dataset[sample_idx]
    sharded, _ = _slice_graph_for_rank(full, rank, world_size)
    return sharded


def _global_scalar(value, device, group):
    """SUM a python scalar across the shard group; returns a python number."""
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t.item()


def _shard_train_epoch(model, dataset, optimizer, device, config, epoch, *,
                       rank, world_size, group, ema_model, loss_weights):
    model.train()
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    augment = bool(config.get('augment_geometry', False))

    n = len(dataset)
    order = _epoch_order(n, epoch, config.get('split_seed', 42),
                         shuffle=True)
    max_train_batches = int(config.get('max_train_batches', 0))
    total_batches = min(n, max_train_batches) if max_train_batches > 0 else n
    grad_accum_steps = int(config.get('grad_accum_steps', 1))
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps
    max_grad_norm = float(config.get('max_grad_norm', 3.0))

    total_loss_sum = 0.0
    total_loss_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_batches):
        graph = _load_sharded(dataset, order[step], rank, world_size,
                              config.get('split_seed', 42), epoch, step, augment)
        graph = graph.to(device)

        window = min(actual_accum, total_batches - (step // actual_accum) * actual_accum)
        with torch.amp.autocast('cuda', dtype=amp_dtype,
                                enabled=use_amp and device.type == 'cuda'):
            predicted, target = model(graph)
            errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
            per_node = _per_node_loss(errors, loss_weights)
            local_sum = per_node.sum()
            local_count = per_node.numel()

        # Normalize by the GLOBAL node count so the summed-across-ranks gradient
        # is the gradient of the true whole-mesh mean loss.
        global_count = _global_scalar(local_count, device, group)
        scaled_loss = local_sum / global_count / window
        scaled_loss.backward()

        total_loss_sum += float(local_sum.detach().item())
        total_loss_count += local_count

        is_last = step == total_batches - 1
        if (step + 1) % actual_accum == 0 or is_last:
            _all_reduce_grads_sum(model, group)   # SUM: see module docstring
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

    # Reduce the epoch's running loss to the global mean for reporting.
    g_sum = _global_scalar(total_loss_sum, device, group)
    g_count = _global_scalar(total_loss_count, device, group)
    mean = g_sum / g_count if g_count > 0 else float('nan')
    return {'mean': mean, 'sum': g_sum, 'count': int(g_count)}


def _shard_validate(model, dataset, device, config, *, rank, world_size, group, loss_weights):
    model.eval()
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    n = len(dataset)
    order = _epoch_order(n, 0, config.get('split_seed', 42), shuffle=False)
    max_val_batches = int(config.get('max_val_batches', 0))
    total_batches = min(n, max_val_batches) if max_val_batches > 0 else n

    local_sum = 0.0
    local_count = 0
    with torch.no_grad():
        for step in range(total_batches):
            graph = _load_sharded(dataset, order[step], rank, world_size,
                                  config.get('split_seed', 42), 0, step, augment=False)
            graph = graph.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                    enabled=use_amp and device.type == 'cuda'):
                predicted, target = model(graph, add_noise=False)
                errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
            per_node = _per_node_loss(errors, loss_weights)
            local_sum += float(per_node.sum().item())
            local_count += per_node.numel()

    g_sum = _global_scalar(local_sum, device, group)
    g_count = _global_scalar(local_count, device, group)
    mean = g_sum / g_count if g_count > 0 else float('nan')
    return {'mean': mean, 'sum': g_sum, 'count': int(g_count)}


def shard_worker(rank, world_size, config, gpu_ids, config_filename='config.txt'):
    """Entry point for torch.multiprocessing.spawn under parallel_mode node_shard."""
    port = config.get('_ddp_port', '29500')
    _setup_process_group(rank, world_size, port)

    if torch.cuda.is_available():
        gpu_id = gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    is_main = rank == 0
    print(f'[rank {rank}] node_shard on device {device}')

    if config.get('attention_kernel') != 'slice_space':
        raise ValueError("node_shard requires attention_kernel 'slice_space'.")

    # ---- Dataset (all ranks build identically from the same split_seed) ----
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, _ = build_dataset_splits(config, split_seed)
    if is_main and config.get('write_preprocessing', False):
        train_dataset.write_preprocessing_to_hdf5(split_seed)
    dist.barrier()

    # ---- Model: build, replicate rank 0's weights, enable sharding ----
    model = Transolver(config, str(device)).to(device)
    _broadcast_module(model, src=0)

    # Build EMA BEFORE attaching the shard group: AveragedModel deep-copies the
    # model, and a ProcessGroup handle is unpicklable. The EMA shadow needs the
    # group too (it runs sharded validation forwards), so set it afterward.
    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    shard_group = dist.new_group(ranks=list(range(world_size)))
    model.set_shard_group(shard_group)
    if ema_model is not None:
        ema_model.module.set_shard_group(shard_group)

    if is_main:
        log_model_summary(model, config, ema_model)
        log_training_config(config)

    total_epochs = config.get('training_epochs')
    optimizer, scheduler, _, _ = build_optimizer_scheduler(config, model.parameters(), total_epochs)
    loss_weights = _build_loss_weights(config, device)

    log_file = init_log_file(config, config_filename) if is_main else None
    modelname = config.get('modelpath')
    val_interval = int(config.get('val_interval', 1))
    train_loss = valid_loss = float('nan')
    start_time = time.time()

    try:
        for epoch in range(total_epochs):
            train_metrics = _shard_train_epoch(
                model, train_dataset, optimizer, device, config, epoch,
                rank=rank, world_size=world_size, group=shard_group,
                ema_model=ema_model, loss_weights=loss_weights,
            )
            train_loss = train_metrics['mean']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                valid_metrics = _shard_validate(
                    eval_model, val_dataset, device, config,
                    rank=rank, world_size=world_size, group=shard_group,
                    loss_weights=loss_weights,
                )
                valid_loss = valid_metrics['mean']

            if is_main:
                msg = f"Epoch {epoch}/{total_epochs} TrainOpt: {train_loss:.2e} LR: {current_lr:.2e}"
                if do_val:
                    msg += f" Valid: {valid_loss:.2e}"
                print(msg)
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"Elapsed: {time.time() - start_time:.2f}s Epoch {epoch} "
                                f"TrainOpt {train_loss:.4e} Valid {valid_loss:.4e} "
                                f"LR: {current_lr:.4e}\n")
            dist.barrier()

        if is_main:
            # Weights are replicated and in-sync across ranks; rank 0's are
            # authoritative. Disable sharding on the saved model config path is
            # unnecessary: the checkpoint is architecturally identical to a
            # single-GPU model and loads for single-GPU inference unchanged.
            save_checkpoint(
                epoch, model, ema_model, optimizer, scheduler,
                train_loss, valid_loss, config, train_dataset, modelname,
            )
            print(f"\nNode-sharded training finished. Model saved at epoch {epoch}.")
    except KeyboardInterrupt:
        if is_main:
            print("\nTraining interrupted by user. No checkpoint saved.")

    dist.destroy_process_group()
