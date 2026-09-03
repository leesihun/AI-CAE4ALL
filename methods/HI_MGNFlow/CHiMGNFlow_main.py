# cHI-MGNflow -- conditional flow matching on mesh fields
import os
import sys
# Must be set before h5py is imported (transitively via data_loader/mesh_dataset)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# Expandable CUDA memory segments: prevents OOM-with-free-memory on variable-size graphs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# The banner uses box-drawing glyphs. When the launcher pipes stdout, Python
# picks the ANSI codepage (cp949 on a Korean Windows) and the very first
# print raises UnicodeEncodeError before any training starts. Force UTF-8
# on the captured streams; errors='replace' keeps a exotic console alive.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import torch
# TF32: ~3x faster fp32 matmuls on Ampere/Hopper with negligible precision loss.
# Affects the ODE integrator, which runs in fp32 outside autocast.
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import socket
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException
from general_modules.config_validation import validate_runtime_config
from general_modules.load_config import load_config
from training_profiles.distributed_training import train_worker
from training_profiles.single_training import single_worker
from inference_profiles.rollout import run_rollout

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='cHI-MGNflow Training')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to config file (default: config.txt)')
    args = parser.parse_args()

    print('\n'*3)

    # Display ASCII art banner
    print("""
    ██████╗ █████╗ ███████╗    ███╗   ███╗██╗         ███████╗██╗   ██╗██╗████████╗███████╗
   ██╔════╝██╔══██╗██╔════╝    ████╗ ████║██║         ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝
   ██║     ███████║█████╗      ██╔████╔██║██║         ███████╗██║   ██║██║   ██║   █████╗
   ██║     ██╔══██║██╔══╝      ██║╚██╔╝██║██║         ╚════██║██║   ██║██║   ██║   ██╔══╝
   ╚██████╗██║  ██║███████╗    ██║ ╚═╝ ██║███████╗    ███████║╚██████╔╝██║   ██║   ███████╗
    ╚═════╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝
    """)
    print(" " * 64 + "Version 1.0.0, 2026-01-06")
    print(" " * 50 + "Developed by SiHun Lee, Ph. D., MX, SEC")
    print()

    # Load configuration files
    config = load_config(args.config)

    run_mode = config.get('mode')
    model = config.get('model')

    print('\n'*2)
    print(f'           Config file   : {args.config}')
    print(f'           Selected Model: {model}, Based on Nvidia physicsNeMo implementation')
    print(f'           Running in    : {run_mode} mode')
    print('\n'*2)
    
    # Auto-configure distributed training from gpu_ids
    gpu_ids = config.get('gpu_ids')  # Default to GPU 0 if not specified

    # Ensure gpu_ids is a list
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    # Auto-calculate world_size and use_distributed
    world_size = len(gpu_ids)
    use_distributed = world_size > 1

    # cHI-MGNflow has a DDP trainer; a one-GPU list naturally selects its
    # single-device worker. Validate before importing or starting a worker so a
    # stale model-split config fails with an architecture-specific explanation.
    parallel_mode = validate_runtime_config(config)

    print(f"GPU Configuration:")
    print(f"  gpu_ids: {gpu_ids}")
    print(f"  world_size (auto-calculated): {world_size}")
    print(f"  use_distributed (auto-calculated): {use_distributed}")
    print(f"  parallel_mode: {parallel_mode}")
    print('\n'*2)

    # Display the current absolute path
    print(f"Current absolute path: {os.path.abspath('.')}")

    if run_mode == 'inference':
        # Inference mode: autoregressive rollout
        run_rollout(config, args.config)

    elif use_distributed==False:
        single_worker(config, args.config)

    else:
        # Find a free port once, before spawning, so workers never collide
        # with zombie processes from prior runs.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            config['_ddp_port'] = str(s.getsockname()[1])
        print(f"Starting distributed training with {world_size} processes on GPUs {gpu_ids} (port {config['_ddp_port']})...")
        try:
            mp.spawn(
                train_worker,
                args=(world_size, config, gpu_ids, args.config),
                nprocs=world_size,
                join=True
            )
            print("Distributed training completed.")
        except (KeyboardInterrupt, ProcessExitedException):
            print("\nTraining interrupted by user. All worker processes terminated.")
        except Exception as e:
            print(f"\nDistributed training failed: {e}")


if __name__ == "__main__":
    main()
