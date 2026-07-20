#!/usr/bin/env python3
"""Phase 2 performance gate (IMPLEMENTATION_PLAN.md section 12): benchmark the
naive and slice_space Physics-Attention kernels on the largest sample of a
real dataset, forward+backward wall time and peak CUDA memory.

Usage:
    python misc/bench_attention_kernels.py --dataset dataset/ex1.h5
    python misc/bench_attention_kernels.py --dataset dataset/ex1.h5 --chunk-size 8000

Expectation from section 6.3: naive is faster (fewer attention FLOPs at this
profile), slice_space uses less peak memory. `naive` stays the shipped
default at ex1 scale; if slice_space is NOT lighter, that contradicts the
theory and should be investigated before relying on the memory ladder.
"""
import argparse
import os
import sys
import time

import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.physics_attention import PhysicsAttentionIrregular


def largest_sample_node_count(dataset_path: str) -> int:
    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted(int(k) for k in f['data'].keys())
        max_n = 0
        for sid in sample_ids:
            n = f[f'data/{sid}/nodal_data'].shape[2]
            max_n = max(max_n, n)
    return max_n


def bench_kernel(module, x, ptr, kernel, chunk_size, use_checkpointing, n_iters=3):
    device = x.device
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(n_iters):
        x_iter = x.clone().requires_grad_(True)
        start = time.time()
        out = module(x_iter, ptr, attention_kernel=kernel, chunk_size=chunk_size,
                     use_checkpointing=use_checkpointing)
        out.pow(2).sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else float('nan')
    return min(times), peak_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--slice-num', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='0 = single tile (whole mesh) for the slice_space kernel')
    parser.add_argument('--use-checkpointing', action='store_true')
    parser.add_argument('--n', type=int, default=None,
                        help='Override node count instead of reading the dataset\'s largest sample')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = args.n if args.n is not None else largest_sample_node_count(args.dataset)
    print(f"Benchmarking on N={n:,} nodes, device={device}")
    print(f"Profile: latent_dim={args.latent_dim}, num_heads={args.num_heads}, "
          f"slice_num={args.slice_num}, chunk_size={args.chunk_size}, "
          f"use_checkpointing={args.use_checkpointing}")

    torch.manual_seed(0)
    dim_head = args.latent_dim // args.num_heads
    module = PhysicsAttentionIrregular(
        dim=args.latent_dim, heads=args.num_heads, dim_head=dim_head, slice_num=args.slice_num,
    ).to(device)
    module.train()

    x = torch.randn(n, args.latent_dim, device=device)
    ptr = torch.tensor([0, n], device=device)

    naive_time, naive_mem = bench_kernel(module, x, ptr, 'naive', 0, args.use_checkpointing)
    slice_time, slice_mem = bench_kernel(
        module, x, ptr, 'slice_space', args.chunk_size, args.use_checkpointing)

    print(f"\n{'kernel':<14}{'time (s)':>12}{'peak mem (MB)':>16}")
    print(f"{'naive':<14}{naive_time:>12.4f}{naive_mem:>16.1f}")
    print(f"{'slice_space':<14}{slice_time:>12.4f}{slice_mem:>16.1f}")

    if device.type == 'cuda':
        if slice_mem < naive_mem:
            print(f"\nslice_space uses {100 * (1 - slice_mem / naive_mem):.1f}% less peak memory "
                  f"(expected: 6.3's FLOPs-for-memory trade).")
        else:
            print(f"\nWARNING: slice_space did NOT use less memory than naive here "
                  f"-- contradicts section 6.3's theory; investigate before relying on the "
                  f"6.6 scaling ladder.")


if __name__ == '__main__':
    main()
