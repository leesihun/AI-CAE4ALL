"""Launcher and training loop for deterministic parallel_mode=model_split,
ported from MeshGraphNets' parallelism/launcher.py. One process per GPU, one
pipeline stage per process, 1F1B micro-batch schedule, per-stage optimizers
with a global gradient-norm clip, and a rank-0 merged checkpoint that is
byte-compatible with the single-GPU checkpoint contract (section 13) -- so
inference_profiles/rollout.py loads it with zero model-split awareness.

Scope matches the MGN original: the split worker trains and checkpoints; it
runs no validation/test epochs (evaluate the merged checkpoint with a normal
single-GPU inference run).
"""

from __future__ import annotations

import datetime
import itertools
import os
import socket
import time
import traceback
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException
from torch_geometric.loader import DataLoader

from parallelism.checkpoint_io import merge_stage_state_dicts_to_rank0
from parallelism.comm import drain_pending_sends, set_pipeline_process_groups
from parallelism.partition import partition_stages, partition_summary
from parallelism.stages import (
    build_probe_core,
    build_split_stage_from_dataset,
    run_stage_step,
)
from training_profiles.setup import (
    SCHEMA_VERSION,
    build_dataset_splits,
    build_dependency_versions,
    build_normalization_dict,
    build_optimizer_scheduler,
    cleanup_dataloaders,
    init_log_file,
    _lightweight_file_fingerprint,
)
from training_profiles.training_loop import _build_loss_weights, update_ema


def launch_model_split(config: dict, config_filename: str = 'config.txt') -> None:
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if len(gpu_ids) < 2:
        print("[model_split] only one GPU; falling back to single-GPU training.")
        from training_profiles.single_training import single_worker
        single_worker(config, config_filename)
        return

    num_stages = len(gpu_ids)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        config['_ddp_port'] = str(s.getsockname()[1])

    if max(1, int(config.get('pipeline_microbatches', 2 * num_stages))) == 1:
        print("[model_split] note: pipeline_microbatches=1 disables 1F1B overlap; "
              "stages run sequentially.")

    print(f"[model_split] spawning {num_stages} processes on GPUs {gpu_ids} "
          f"(port {config['_ddp_port']})...")
    try:
        mp.spawn(
            _split_worker,
            args=(num_stages, config, gpu_ids, config_filename),
            nprocs=num_stages,
            join=True,
        )
        print("[model_split] training completed.")
    except (KeyboardInterrupt, ProcessExitedException):
        print("\n[model_split] training interrupted.")
    except Exception as e:
        print(f"\n[model_split] training failed: {e}")
        traceback.print_exc()


def _split_worker(rank: int, num_stages: int, config: dict, gpu_ids: list,
                  config_filename: str) -> None:
    try:
        _split_worker_inner(rank, num_stages, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _split_worker_inner(rank: int, num_stages: int, config: dict, gpu_ids: list,
                        config_filename: str) -> None:
    gpu_id = gpu_ids[rank]
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config['_ddp_port']
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=num_stages,
        timeout=datetime.timedelta(minutes=60),
    )

    # Separate communicators for downstream activations and upstream gradients:
    # 1F1B interleaves the two directions, which deadlocks on a single
    # communicator (kernel-order serialization on NCCL, blocking sends on gloo).
    pg_data = dist.new_group(list(range(num_stages)))
    pg_grad = dist.new_group(list(range(num_stages)))
    set_pipeline_process_groups(pg_data, pg_grad)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    if rank == 0:
        print(f"[model_split rank=0] using device {device} "
              f"(stages on GPUs {gpu_ids})")

    split_seed = int(config.get('split_seed', 42))
    train_dataset, _, _ = build_dataset_splits(config, split_seed)
    dist.barrier(device_ids=[gpu_id]) if torch.cuda.is_available() else dist.barrier()

    # Every rank iterates the SAME loader order (shuffle=False): the first
    # stage consumes graph.x, the last stage consumes graph.pos/y for the loss,
    # and both must be looking at the same sample at the same micro-batch slot.
    pin_memory = torch.cuda.is_available()
    num_workers = int(config.get('num_workers', 0))
    mp_context = 'spawn' if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['batch_size']),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=int(config.get('prefetch_factor', 4)) if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )
    config['_pin_memory'] = pin_memory

    assignment = _profile_and_partition(rank, config, train_dataset, train_loader,
                                        device, num_stages)
    if rank == 0:
        print(f"[model_split rank=0] stage assignment: {assignment}")

    stage, data_spec, coordinate_domain = build_split_stage_from_dataset(
        config, train_dataset, rank, num_stages, assignment,
    )
    stage = stage.to(device)
    if rank == 0:
        total_params = sum(p.numel() for p in stage.parameters())
        print(f"[model_split rank=0] stage 0 built: blocks={stage.my_blocks}, "
              f"params={total_params:,}")

    if config.get('use_compile', False):
        try:
            stage = torch.compile(stage, dynamic=True)
            if rank == 0:
                print("[model_split rank=0] torch.compile applied.")
        except Exception as e:
            if rank == 0:
                print(f"[model_split rank=0] torch.compile failed ({e}); running eager.")

    ema_model = None
    if config.get('use_ema', False):
        from training_profiles.training_loop import build_ema_model
        ema_model = build_ema_model(stage, config)
        ema_model = ema_model.to(device)
        if rank == 0:
            print(f"[model_split rank=0] EMA enabled (decay={config.get('ema_decay', 0.999)}).")

    total_epochs = int(config['training_epochs'])
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, [p for p in stage.parameters() if p.requires_grad], total_epochs,
    )
    if rank == 0:
        print(f"[model_split rank=0] optimizer ready; warmup={warmup_epochs}, "
              f"cosine_T0={cosine_T0}")

    microbatches = max(1, int(config.get('pipeline_microbatches', 2 * num_stages)))
    if rank == 0:
        eff_batch = int(config['batch_size']) * microbatches
        print(f"[model_split rank=0] 1F1B pipeline: microbatches={microbatches} "
              f"(one optimizer step per {microbatches} batches, effective batch={eff_batch}; "
              f"in-flight activations per stage <= {num_stages})")

    log_file = init_log_file(config, config_filename) if rank == 0 else None
    checkpoint_interval = int(config.get('checkpoint_interval', 0))
    modelpath = config.get('modelpath')

    start_time = time.time()
    train_loss = float('nan')
    for epoch in range(total_epochs):
        train_loss = _train_one_epoch(
            stage=stage, loader=train_loader, optimizer=optimizer, device=device,
            config=config, ema_model=ema_model, microbatches=microbatches, epoch=epoch,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"[model_split] epoch {epoch}/{total_epochs} "
                  f"train_loss={train_loss:.2e} lr={current_lr:.2e} "
                  f"elapsed={time.time() - start_time:.1f}s")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Elapsed: {time.time() - start_time:.2f}s Epoch {epoch} "
                            f"TrainOpt {train_loss:.4e} LR: {current_lr:.4e}\n")

        # Collective merge: every rank must enter _save_merged_checkpoint.
        if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            _save_merged_checkpoint(
                stage=stage, ema_model=ema_model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, train_loss=train_loss,
                config=config, train_dataset=train_dataset, data_spec=data_spec,
                coordinate_domain=coordinate_domain, modelpath=modelpath,
                config_filename=config_filename, rank=rank,
            )
            if rank == 0:
                print(f"  Periodic checkpoint saved at epoch {epoch}")

    _save_merged_checkpoint(
        stage=stage, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler,
        epoch=epoch, train_loss=train_loss, config=config,
        train_dataset=train_dataset, data_spec=data_spec,
        coordinate_domain=coordinate_domain, modelpath=modelpath,
        config_filename=config_filename, rank=rank,
    )
    if rank == 0:
        print(f"[model_split] training finished. Merged model saved at epoch {epoch} "
              f"with train_loss={train_loss:.2e}")
    cleanup_dataloaders(train_loader)


def _profile_and_partition(rank, config, train_dataset, train_loader, device,
                           num_stages):
    """Rank 0 measures per-block activation costs on a CPU probe batch and
    runs the DP partitioner; the assignment is broadcast to all ranks."""
    if rank == 0:
        from general_modules.data_spec import build_data_spec_from_dataset
        from model.adapters.coordinate_domain import CoordinateDomain

        data_spec = build_data_spec_from_dataset(train_dataset, config)
        domain = CoordinateDomain.from_dataset(
            train_dataset,
            out_of_bounds_policy=str(config.get('out_of_bounds_policy', 'error')).lower(),
        )
        probe_core = build_probe_core(config, data_spec, domain)
        try:
            probe = next(iter(train_loader))  # CPU tensors; never leaves host
            block_costs = probe_core.pipeline_block_costs(probe)
            del probe
            assignment = partition_stages(block_costs, num_stages)
            print("[model_split rank=0] " + partition_summary(block_costs, assignment))
        except Exception as e:
            n_blocks = probe_core.pipeline_num_blocks()
            print(f"[model_split rank=0] WARNING: cost probe failed ({e}); "
                  "falling back to equal split.")
            assignment = partition_stages([1.0] * n_blocks, num_stages)
        del probe_core
        payload = [assignment]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _clip_grads_global(params, max_norm: float, device) -> None:
    """Clip by the global grad norm across all stages (matches the single-GPU
    path; per-stage clip_grad_norm_ would be a stricter, different operation)."""
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        total_sq = torch.stack(
            [torch.linalg.vector_norm(g, 2) for g in grads]
        ).square().sum().to(device=device, dtype=torch.float32)
    else:
        total_sq = torch.zeros((), device=device, dtype=torch.float32)
    dist.all_reduce(total_sq)
    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    if grads:
        torch._foreach_mul_(grads, clip_coef.to(grads[0].device))


def _train_one_epoch(*, stage, loader, optimizer, device, config, ema_model,
                     microbatches: int, epoch: int) -> float:
    """1F1B pipeline schedule over groups of `microbatches` loader batches.

    Each group is one optimizer step (gradient accumulation). Stage i runs
    (num_stages - 1 - i) warmup forwards, then alternates one-forward/
    one-backward, then drains -- so at most (num_stages - i) micro-batch
    activation sets are resident per stage. All stages issue forwards and
    backwards in the same FIFO order, keeping the P2P send/recv sequence
    matched pairwise between neighbors.
    """
    stage.train()
    raw_stage = getattr(stage, '_orig_mod', stage)
    num_stages = raw_stage.num_stages
    is_last = raw_stage.is_last
    is_first = raw_stage.is_first
    rank_warmup = max(num_stages - 1 - raw_stage.stage_idx, 0)

    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    max_grad_norm = float(config.get('max_grad_norm', 3.0))
    loss_weights = _build_loss_weights(config, device)

    params = [p for p in stage.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_count = 0
    in_flight = deque()
    batch_counter = itertools.count()

    def _forward(graph, loss_scale):
        b_idx = next(batch_counter)
        if is_first or is_last:
            graph = graph.to(device, non_blocking=config.get('_pin_memory', False))
        else:
            graph = None  # middle stages never touch the graph
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            return run_stage_step(stage, graph, config, device, epoch, b_idx,
                                  loss_weights=loss_weights, loss_scale=loss_scale)

    def _backward_oldest():
        nonlocal loss_count
        sync_loss, batch_loss_sum, batch_count = in_flight.popleft()
        sync_loss.backward()
        if is_last and batch_loss_sum is not None:
            loss_sum.add_(batch_loss_sum.double())
            loss_count += batch_count

    data_iter = iter(loader)
    while True:
        group = deque(itertools.islice(data_iter, microbatches))
        if not group:
            break
        loss_scale = 1.0 / len(group)
        optimizer.zero_grad(set_to_none=True)

        for _ in range(min(rank_warmup, len(group))):
            in_flight.append(_forward(group.popleft(), loss_scale))
        while group:
            in_flight.append(_forward(group.popleft(), loss_scale))
            _backward_oldest()
        while in_flight:
            _backward_oldest()

        drain_pending_sends()
        _clip_grads_global(params, max_grad_norm, device)
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, stage)

    loss_tensor = torch.zeros(2, device=device, dtype=torch.float64)
    if is_last:
        loss_tensor[0] = loss_sum
        loss_tensor[1] = float(loss_count)
    dist.broadcast(loss_tensor, src=num_stages - 1)
    if loss_tensor[1].item() > 0:
        return float(loss_tensor[0].item() / loss_tensor[1].item())
    return 0.0


def _save_merged_checkpoint(*, stage, ema_model, optimizer, scheduler, epoch,
                            train_loss, config, train_dataset, data_spec,
                            coordinate_domain, modelpath, config_filename,
                            rank) -> None:
    """Merge per-stage state dicts to rank 0 and write the section 13
    checkpoint contract. Collective: every rank must call this."""
    raw_stage = getattr(stage, '_orig_mod', stage)
    merged = merge_stage_state_dicts_to_rank0(raw_stage.state_dict())
    ema_merged = None
    if ema_model is not None:
        ema_merged = merge_stage_state_dicts_to_rank0(ema_model.state_dict())

    if rank != 0:
        return

    save_dict = {
        'schema_version': SCHEMA_VERSION,
        'selected_model': raw_stage.model_name,
        'epoch': epoch,
        'model_state_dict': merged,
        'optimizer_state_dict': optimizer.state_dict(),  # rank-0 stage's share
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'valid_loss': train_loss,
        'model_config': raw_stage.model_config_export,
        'adapter_config': coordinate_domain.to_dict(),
        'data_config': data_spec.to_dict(),
        'normalization': build_normalization_dict(train_dataset),
        'dependency_versions': build_dependency_versions(),
        'rng_states': {
            'torch': torch.get_rng_state(),
            'numpy': __import__('numpy').random.get_state(),
        },
        'source_reference': {
            'config_file': os.path.abspath(config_filename) if config_filename else None,
            'dataset': _lightweight_file_fingerprint(config.get('dataset_dir')),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'split_stage_assignment': None,  # filled below for provenance
    }
    if ema_merged:
        save_dict['ema_state_dict'] = ema_merged
    save_dict['split_stage_assignment'] = {
        'num_stages': raw_stage.num_stages,
        'stage_blocks': raw_stage.my_blocks,
    }
    model_dir = os.path.dirname(modelpath)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(save_dict, modelpath)
