"""CLI wiring for infer_mode decoupled (section 11): builds the checkpoint-led
model exactly like inference_profiles.rollout, then calls
Transolver.forward_decoupled instead of the ordinary forward. Static (T=1)
datasets only for now -- temporal decoupled rollout is not implemented in this
pass (direct temporal rollout is; see rollout.py).
"""

import time

import numpy as np
import torch

from general_modules.mesh_dataset import MeshGraphDataset, denormalize_delta
from inference_profiles.rollout import (
    _build_model_and_load_weights, _load_checkpoint_and_overlay, _resolve_device,
    _write_sample_hdf5,
)


def run_decoupled_inference(config, config_filename='config.txt'):
    print("\n" + "=" * 60)
    print("DECOUPLED TWO-STAGE INFERENCE")
    print("=" * 60)

    device = _resolve_device(config)
    checkpoint = _load_checkpoint_and_overlay(config, device)
    model = _build_model_and_load_weights(config, checkpoint, device)

    infer_dataset_path = config.get('infer_dataset')
    dataset = MeshGraphDataset(infer_dataset_path, config)
    dataset.inherit_preprocessing_from_dict(checkpoint['normalization'])

    if dataset.num_timesteps > 1:
        raise NotImplementedError(
            "infer_mode decoupled currently supports static (T=1) datasets only; "
            "use infer_mode direct for temporal rollout."
        )

    infer_chunk_size = int(config.get('infer_chunk_size', 0))
    output_dim = config['output_var']
    output_dir = config.get('inference_output_dir', 'outputs/rollout')
    modelpath = config.get('modelpath')

    with torch.no_grad():
        for idx, sample_id in enumerate(dataset.sample_ids):
            start = time.time()
            graph = dataset[idx].to(device)
            pred_norm, _ = model.forward_decoupled(graph, infer_chunk_size=infer_chunk_size)
            pred_denorm = denormalize_delta(pred_norm.cpu().numpy(), dataset.delta_mean, dataset.delta_std)
            elapsed = time.time() - start

            ref_pos = graph.pos.cpu().numpy()
            part_ids = graph.part_ids.cpu().numpy() if graph.part_ids is not None else None
            output_states = pred_denorm[np.newaxis, :, :]

            _write_sample_hdf5(
                output_dir, sample_id, steps=0, ref_pos=ref_pos, output_states=output_states,
                output_dim=output_dim, part_ids=part_ids, checkpoint_path=modelpath,
                config_filename=config_filename, elapsed_s=elapsed,
            )

    print(f"\nDecoupled inference complete. Processed {len(dataset.sample_ids)} sample(s).")
