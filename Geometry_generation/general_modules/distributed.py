"""
Distributed training helpers for SDFFlow (DDP + FSDP/model-split).

Design (matches the MeshGraphNets convention in this repo):
  * The suite still launches one `python SDFFlow_main.py --config ...` process.
    When `parallel_mode` is `ddp`/`fsdp` and >1 GPU is requested, that process
    self-spawns one worker per GPU with `torch.multiprocessing.spawn` and a
    self-picked free TCP port. No `torchrun` is required.
  * Each rank initializes its own NCCL (CUDA) / gloo (CPU) process group, pins
    its GPU, and runs the normal stage dispatch. Rank 0 owns validation,
    logging, periodic tests, and checkpoint writes; all ranks share gradients.

`parallel_mode`:
  * `single` (default) — unchanged single-process/single-GPU behavior.
  * `ddp`    — Distributed Data Parallel. Replicates the model per GPU, shards
               the batch, all-reduces gradients. Use when the model fits on one
               GPU (on a 288 GB B300 that is almost everything).
  * `fsdp`   — Fully Sharded Data Parallel: SDFFlow's "model split". Sharded
               params/grads/optimizer across GPUs for models too large for one
               GPU (multi-billion-parameter DiTs). Requires CUDA.

Everything degrades to a no-op when a process group is not initialized, so the
same worker code runs single-process and distributed.
"""

import contextlib
import datetime
import os
import socket

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Mode / device resolution
# ---------------------------------------------------------------------------

def resolve_gpu_ids(config):
    """Return the configured GPU IDs as a list of ints."""
    gpu_ids = config.get('gpu_ids', 0)
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    return [int(g) for g in gpu_ids]


def parallel_mode(config):
    mode = str(config.get('parallel_mode', 'single')).lower()
    if mode not in ('single', 'ddp', 'fsdp'):
        raise ValueError(f"parallel_mode must be single, ddp, or fsdp; got '{mode}'")
    return mode


def should_distribute(config):
    """True when the run should spawn multiple ranks."""
    if parallel_mode(config) == 'single':
        return False
    if not torch.cuda.is_available():
        # gloo multi-process on CPU is only used by the test harness, which sets
        # the env flag below explicitly. Production distributed runs need CUDA.
        return os.environ.get('SDFFLOW_FORCE_CPU_DIST') == '1' and len(resolve_gpu_ids(config)) > 1
    return len(resolve_gpu_ids(config)) > 1 and torch.cuda.device_count() > 1


# ---------------------------------------------------------------------------
# Rank helpers (safe to call without an initialized process group)
# ---------------------------------------------------------------------------

def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def barrier(device_ids=None):
    if is_dist():
        if device_ids is not None and dist.get_backend() == 'nccl':
            dist.barrier(device_ids=device_ids)
        else:
            dist.barrier()


def all_reduce_sum(tensor):
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_scalar(value, device, src=0):
    """Broadcast a python float from `src` to every rank; returns the float."""
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if is_dist():
        dist.broadcast(t, src=src)
    return float(t.item())


def reduce_epoch_mean(loss_sum, count, device):
    """Sum (loss_sum, count) across ranks and return the global mean."""
    t = torch.tensor([float(loss_sum), float(count)], device=device, dtype=torch.float64)
    all_reduce_sum(t)
    return float(t[0].item() / t[1].item()) if t[1].item() > 0 else 0.0


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def init_process_group(rank, world_size, gpu_id, port):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = port
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=60))
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


def cleanup():
    if is_dist():
        dist.destroy_process_group()


def spawn_workers(entry_fn, config, config_filename):
    """Spawn one process per GPU. `entry_fn(rank, world_size, gpu_ids, config,
    config_filename)` runs inside each process."""
    import torch.multiprocessing as mp

    gpu_ids = resolve_gpu_ids(config)
    world_size = len(gpu_ids)
    config = dict(config)
    config['_ddp_port'] = pick_free_port()
    print(f'[distributed] parallel_mode={parallel_mode(config)} spawning '
          f'{world_size} ranks on GPUs {gpu_ids} (port {config["_ddp_port"]})')
    mp.spawn(_worker_entry, args=(entry_fn, world_size, gpu_ids, config, config_filename),
             nprocs=world_size, join=True)


def _worker_entry(rank, entry_fn, world_size, gpu_ids, config, config_filename):
    import traceback
    gpu_id = gpu_ids[rank]
    try:
        init_process_group(rank, world_size, gpu_id, config['_ddp_port'])
        entry_fn(rank, world_size, gpu_ids, config, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        cleanup()


def worker_device(rank, gpu_ids):
    if torch.cuda.is_available():
        gpu_id = gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Model wrapping / unwrapping
# ---------------------------------------------------------------------------

def wrap_model(model, config, device):
    """Wrap `model` for the active parallel_mode. Returns (wrapped, is_fsdp).

    Single-process (no initialized group) returns the model untouched.
    """
    if not is_dist():
        return model, False

    mode = parallel_mode(config)
    if mode == 'fsdp':
        return _wrap_fsdp(model, config, device), True

    from torch.nn.parallel import DistributedDataParallel as DDP
    if torch.cuda.is_available():
        gpu_id = device.index
        wrapped = DDP(model, device_ids=[gpu_id], broadcast_buffers=True,
                      find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        wrapped = DDP(model, broadcast_buffers=True, find_unused_parameters=False)
    return wrapped, False


def _wrap_fsdp(model, config, device):
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError('parallel_mode=fsdp requires CUDA (NCCL).')

    min_params = int(config.get('fsdp_min_params', 1_000_000))
    auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=min_params)

    mp_policy = None
    if bool(config.get('use_amp', True)) and torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.float32,
                                   buffer_dtype=torch.bfloat16)

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device.index,
        use_orig_params=True,
    )


def unwrap_model(model):
    """Strip DDP / compiled wrappers to reach the raw module (FSDP stays wrapped;
    use full_state_dict for its parameters)."""
    model = getattr(model, '_orig_mod', model)  # torch.compile
    model = getattr(model, 'module', model)      # DDP / AveragedModel
    return model


def full_state_dict(model, is_fsdp):
    """Return a full (unsharded) state dict on rank 0 (empty elsewhere for FSDP).

    For DDP / single, returns the raw module's state dict on every rank.
    """
    if is_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return unwrap_model(model).state_dict()


def main_process_only():
    """Context manager: body runs on rank 0, others wait at the exit barrier."""
    return _MainOnly()


class _MainOnly(contextlib.AbstractContextManager):
    def __enter__(self):
        return is_main_process()

    def __exit__(self, *exc):
        barrier()
        return False
