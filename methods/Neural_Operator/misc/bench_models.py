#!/usr/bin/env python3
"""Per-stage timing benchmark (IMPLEMENTATION_PLAN.md section 18). Profiles
common data loading plus forward/backward/optimizer time for the selected
model on a real config, with CUDA warmup and explicit synchronization
(never inferring memory from nvidia-smi alone).

Usage:
    python misc/bench_models.py --config ex1/config_train_deeponet.txt --num-batches 10
"""

import argparse
import os
import sys
import time

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--warmup-batches', type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    gpu_ids = config.get('gpu_ids')
    if isinstance(gpu_ids, list):
        gpu_ids = gpu_ids[0]
    device = torch.device(f'cuda:{gpu_ids}') if torch.cuda.is_available() and gpu_ids >= 0 else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    dataset = MeshGraphDataset(config['dataset_dir'], dict(config))
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=int(config.get('split_seed', 42)))
    model, data_spec, coordinate_domain = build_model(config, train)
    model = model.to(device)

    loader = DataLoader(train, batch_size=config.get('batch_size', 1), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == 'cuda'))

    data_times, fwd_times, bwd_times, opt_times = [], [], [], []
    model.train()

    it = iter(loader)
    total = args.warmup_batches + args.num_batches
    for i in range(total):
        t0 = time.time()
        try:
            graph = next(it)
        except StopIteration:
            it = iter(loader)
            graph = next(it)
        graph = graph.to(device)
        _sync(device)
        t1 = time.time()

        optimizer.zero_grad(set_to_none=True)
        pred, target = model(graph, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        _sync(device)
        t2 = time.time()

        loss.backward()
        _sync(device)
        t3 = time.time()

        optimizer.step()
        _sync(device)
        t4 = time.time()

        if i >= args.warmup_batches:
            data_times.append(t1 - t0)
            fwd_times.append(t2 - t1)
            bwd_times.append(t3 - t2)
            opt_times.append(t4 - t3)

    def summarize(name, times):
        if not times:
            return
        print(f"  {name:10s}: mean={sum(times) / len(times) * 1000:.1f}ms  "
              f"min={min(times) * 1000:.1f}ms  max={max(times) * 1000:.1f}ms")

    print(f"=== Benchmark: model='{config.get('model')}' on {config['dataset_dir']} "
          f"({args.num_batches} batches after {args.warmup_batches} warmup) ===")
    summarize('data', data_times)
    summarize('forward', fwd_times)
    summarize('backward', bwd_times)
    summarize('optimizer', opt_times)

    if device.type == 'cuda':
        print(f"  peak allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"  peak reserved:  {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
