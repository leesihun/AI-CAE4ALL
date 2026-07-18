# Transolver
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import socket

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from general_modules.load_config import load_config
from inference_profiles.rollout import run_inference
from training_profiles.distributed_training import train_worker
from training_profiles.sharded_training import shard_worker
from training_profiles.single_training import single_worker


def main():
    parser = argparse.ArgumentParser(description='Transolver deterministic runtime')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to config file (default: config.txt)')
    args = parser.parse_args()

    print('\n' * 3)
    print("""
    Transolver
    Deterministic simulator runtime
    """)

    config = load_config(args.config)
    run_mode = config.get('mode')
    if run_mode not in ('train', 'inference'):
        raise ValueError(f"Unsupported mode '{run_mode}'. This checkout supports only 'train' and 'inference'.")

    model = config.get('model')

    print('\n' * 2)
    print(f'           Config file   : {args.config}')
    print(f'           Selected Model: {model}')
    print(f'           Running in    : {run_mode} mode')
    print('\n' * 2)

    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    world_size = len(gpu_ids)
    use_distributed = world_size > 1

    print("GPU Configuration:")
    print(f"  gpu_ids: {gpu_ids}")
    print(f"  world_size (auto-calculated): {world_size}")
    print(f"  use_distributed (auto-calculated): {use_distributed}")
    print('\n' * 2)
    print(f"Current absolute path: {os.path.abspath('.')}")

    # 'ddp'        -> data parallelism (one full mesh per rank; ~world_size throughput)
    # 'node_shard' -> Phase 7 VRAM pooling (one mesh's nodes split across ranks)
    # (config['parallel_mode'] is already normalized: 'model_split' -> 'node_shard')
    parallel_mode = str(config.get('parallel_mode', 'ddp')).lower().strip()

    if run_mode == 'inference':
        run_inference(config, args.config)
    elif use_distributed is False:
        single_worker(config, args.config)
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            config['_ddp_port'] = str(s.getsockname()[1])
        worker = shard_worker if parallel_mode == 'node_shard' else train_worker
        launch_label = 'node-sharded' if parallel_mode == 'node_shard' else 'data-parallel (DDP)'
        print(f"Starting {launch_label} training with {world_size} processes on "
              f"GPUs {gpu_ids} (port {config['_ddp_port']})...")
        try:
            mp.spawn(
                worker,
                args=(world_size, config, gpu_ids, args.config),
                nprocs=world_size,
                join=True
            )
            print(f"{launch_label} training completed.")
        except (KeyboardInterrupt, ProcessExitedException):
            print("\nTraining interrupted by user. All worker processes terminated.")
        except Exception as e:
            print(f"\nDistributed training failed: {e}")


if __name__ == "__main__":
    main()
