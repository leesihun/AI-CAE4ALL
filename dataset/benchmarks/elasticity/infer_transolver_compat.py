#!/usr/bin/env python3
"""Benchmark-only Transolver static inference compatibility wrapper.

The production rollout currently dereferences ``graph.part_ids`` even when
``use_node_types=False``.  This wrapper intentionally leaves that runtime
untouched and differs only by using ``getattr(..., None)`` while writing the
same native output format.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
TRANSOLVER_ROOT = SUITE_ROOT / "transolver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    sys.path.insert(0, str(TRANSOLVER_ROOT))
    os.chdir(TRANSOLVER_ROOT)

    from general_modules.load_config import load_config
    from general_modules.mesh_dataset import MeshGraphDataset, denormalize_delta
    from inference_profiles.rollout import (
        _build_model_and_load_weights,
        _load_checkpoint_and_overlay,
        _resolve_device,
        _write_sample_hdf5,
    )

    config = load_config(str(config_path))
    if config.get("model") != "transolver" or config.get("mode") != "inference":
        raise ValueError("This wrapper accepts only a Transolver inference config")
    if config.get("infer_mode", "direct") != "direct":
        raise ValueError("The Elasticity compatibility wrapper supports direct inference only")

    device = _resolve_device(config)
    checkpoint = _load_checkpoint_and_overlay(config, device)
    model = _build_model_and_load_weights(config, checkpoint, device)
    dataset = MeshGraphDataset(config["infer_dataset"], config)
    if dataset.num_timesteps != 1:
        raise ValueError("The Elasticity benchmark requires static T=1 samples")
    dataset.inherit_preprocessing_from_dict(checkpoint["normalization"])

    output_dir = config.get("inference_output_dir", "outputs/rollout")
    modelpath = config["modelpath"]
    output_dim = config["output_var"]
    model.eval()
    print("Benchmark compatibility: absent part_ids will be written as None.")
    with torch.no_grad():
        for idx, sample_id in enumerate(dataset.sample_ids):
            start = time.time()
            graph = dataset[idx].to(device)
            predicted_norm, _ = model(graph, add_noise=False)
            predicted = denormalize_delta(
                predicted_norm.cpu().numpy(), dataset.delta_mean, dataset.delta_std
            )
            elapsed = time.time() - start
            part_ids_tensor = getattr(graph, "part_ids", None)
            part_ids = part_ids_tensor.cpu().numpy() if part_ids_tensor is not None else None
            _write_sample_hdf5(
                output_dir=output_dir,
                sample_id=sample_id,
                steps=0,
                ref_pos=graph.pos.cpu().numpy(),
                output_states=predicted[None, :, :],
                output_dim=output_dim,
                part_ids=part_ids,
                checkpoint_path=modelpath,
                config_filename=str(config_path),
                elapsed_s=elapsed,
            )
    print(f"Inference complete. Processed {len(dataset.sample_ids)} sample(s).")


if __name__ == "__main__":
    main()
