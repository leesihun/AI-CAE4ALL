"""Pipeline stage construction and per-stage forward step for
parallel_mode=model_split, adapted from MeshGraphNets' parallelism/model_split.py
to the grid-latent operator cores (fno, gino).

Design differences from the MGN original, both deliberate:

  * A stage is a *pruned full core*: every rank seeds torch with `split_seed`,
    builds the complete core (identical weights everywhere), then drops the
    submodules outside its assigned block range. Surviving parameters keep
    the exact state-dict keys of the single-GPU OperatorWrapper, so the
    rank-0 merged checkpoint loads through the normal
    `build_model_from_checkpoint` path with strict=True.
  * The MGN split launcher silently dropped the noise target correction when
    first and last stage differ (each stage holds its own graph copy). Here
    both boundary stages regenerate the identical noise tensor from a
    deterministic per-(epoch, batch) seed, so the section 4.6 contract holds
    exactly as in single-GPU training.

DeepONet/Point-DeepONet have no sequential latent stack to cut and are
rejected at config validation, never here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from general_modules.config_validation import validate_model_config
from general_modules.data_spec import build_data_spec_from_dataset
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.point_sampling import stable_hash
from parallelism.comm import recv_upstream, send_downstream

SPLIT_CAPABLE_MODELS = ("fno", "gino")


def pipeline_noise_tensor(config, num_nodes: int, epoch: int, batch_idx: int,
                          device, dtype) -> Optional[torch.Tensor]:
    """The noise tensor for one batch, identical on every stage that asks.

    Seeded from (split_seed, epoch, batch_idx) via the same stable_hash used
    for Point-DeepONet sensor sampling, generated on CPU so the value cannot
    depend on which GPU a stage runs on."""
    std_noise = float(config.get('std_noise', 0.0))
    if std_noise <= 0:
        return None
    seed = stable_hash(int(config.get('split_seed', 42)), epoch, batch_idx) % (2 ** 31 - 1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    output_var = int(config['output_var'])
    noise = torch.randn(num_nodes, output_var, generator=gen) * std_noise
    return noise.to(device=device, dtype=dtype)


def apply_noise_to_input(graph, config, epoch: int, batch_idx: int) -> None:
    """First-stage half of the section 4.6 noise contract (x perturbation)."""
    noise = pipeline_noise_tensor(config, graph.x.shape[0], epoch, batch_idx,
                                  graph.x.device, graph.x.dtype)
    if noise is None:
        return
    noise_padded = torch.zeros_like(graph.x)
    noise_padded[:, :noise.shape[1]] = noise
    graph.x = graph.x + noise_padded


def apply_noise_to_target(graph, config, epoch: int, batch_idx: int) -> None:
    """Last-stage half of the noise contract (target correction), using the
    identical regenerated noise tensor."""
    noise = pipeline_noise_tensor(config, graph.x.shape[0], epoch, batch_idx,
                                  graph.y.device, graph.y.dtype)
    if noise is None:
        return
    ratio = config.get('noise_std_ratio', None)
    if ratio is None:
        return
    ratio_t = torch.tensor(ratio, device=graph.y.device, dtype=graph.y.dtype)
    graph.y = graph.y - float(config.get('noise_gamma', 1)) * noise * ratio_t


class OperatorSplitStage(nn.Module):
    """One pipeline stage: a pruned core plus its block range.

    Holds the core under the attribute name `core` so `state_dict()` keys are
    exactly the OperatorWrapper's ('core.*')."""

    def __init__(self, config, core, stage_idx: int, num_stages: int,
                 assignment: Sequence[Sequence[int]], model_config_export: dict):
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)
        self.is_first = self.stage_idx == 0
        self.is_last = self.stage_idx == self.num_stages - 1
        self.my_blocks = sorted(int(b) for b in assignment[stage_idx])
        self.model_name = core.model_name
        self.model_config_export = dict(model_config_export)

        num_blocks = core.pipeline_num_blocks()
        flat = sorted(b for blocks in assignment for b in blocks)
        if flat != list(range(num_blocks)):
            raise ValueError(
                f"stage assignment {assignment} does not cover blocks "
                f"0..{num_blocks - 1} exactly once."
            )
        if self.is_first and 0 not in self.my_blocks:
            raise ValueError("first stage must own block 0 (entry).")
        if self.is_last and (num_blocks - 1) not in self.my_blocks:
            raise ValueError("last stage must own the exit block.")

        core.prune_to_pipeline_blocks(self.my_blocks)
        self.core = core

    @property
    def latent_blocks(self) -> List[int]:
        last = self.core.pipeline_num_blocks() - 1
        return [b for b in self.my_blocks if 0 < b < last]

    def run_latent_blocks(self, h: torch.Tensor) -> torch.Tensor:
        for b in self.latent_blocks:
            h = self.core.pipeline_block(h, b)
        return h


def build_split_stage(config, data_spec, coordinate_domain, stage_idx: int,
                      num_stages: int, assignment: Sequence[Sequence[int]],
                      seed: Optional[int] = None):
    """Construct one stage. `torch.manual_seed(seed)` before core construction
    makes every rank build bitwise-identical weights (and makes a single-GPU
    rebuild with the same seed reproduce the initialization)."""
    from model.factory import MODEL_REGISTRY  # local import: avoids a cycle

    model_name = str(config.get('model', '')).lower()
    if model_name not in SPLIT_CAPABLE_MODELS:
        raise ValueError(
            f"parallel_mode=model_split supports only {SPLIT_CAPABLE_MODELS} "
            f"(sequential latent stack); got model='{model_name}'."
        )
    validate_model_config(config, data_spec)

    if seed is None:
        seed = int(config.get('split_seed', 42))
    torch.manual_seed(seed)
    core = MODEL_REGISTRY[model_name](config, data_spec, coordinate_domain)
    model_config_export = core.export_model_config()
    return OperatorSplitStage(config, core, stage_idx, num_stages, assignment,
                              model_config_export)


def build_split_stage_from_dataset(config, train_dataset, stage_idx: int,
                                   num_stages: int, assignment: Sequence[Sequence[int]]):
    """Dataset-driven variant used by the launcher; mirrors factory.build_model's
    spec/domain derivation so the checkpointed metadata is identical."""
    data_spec = build_data_spec_from_dataset(train_dataset, config)
    coordinate_domain = CoordinateDomain.from_dataset(
        train_dataset,
        out_of_bounds_policy=str(config.get('out_of_bounds_policy', 'error')).lower(),
    )
    stage = build_split_stage(config, data_spec, coordinate_domain,
                              stage_idx, num_stages, assignment)
    return stage, data_spec, coordinate_domain


def build_probe_core(config, data_spec, coordinate_domain):
    """Throwaway full core for the rank-0 cost probe (CPU, discarded after
    `pipeline_block_costs`)."""
    from model.factory import MODEL_REGISTRY

    model_name = str(config.get('model', '')).lower()
    torch.manual_seed(int(config.get('split_seed', 42)))
    return MODEL_REGISTRY[model_name](config, data_spec, coordinate_domain)


def run_stage_step(stage: OperatorSplitStage, graph, config, device,
                   epoch: int, batch_idx: int, loss_weights=None,
                   loss_scale: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
    """Execute one micro-batch forward on one pipeline stage.

    Returns (tensor to backward through, detached unscaled batch loss sum or
    None, node-loss count or None). Only the last stage reports a loss; other
    stages return a send sentinel whose backward blocks on the downstream
    gradient. Loss semantics match training_profiles/training_loop.py exactly
    (per-node feature-weighted MSE)."""
    from training_profiles.training_loop import _loss_from_errors

    if stage.is_first:
        if stage.training:
            apply_noise_to_input(graph, config, epoch, batch_idx)
        h = stage.core.pipeline_entry(graph)
        h = stage.run_latent_blocks(h)
        if not stage.is_last:
            sentinel = send_downstream(h, stage.stage_idx + 1)
            return sentinel.sum(), None, None
    else:
        h = recv_upstream(stage.stage_idx - 1, device)
        h = stage.run_latent_blocks(h)
        if not stage.is_last:
            sentinel = send_downstream(h, stage.stage_idx + 1)
            return sentinel.sum(), None, None

    # Last stage: decode and compute the shared loss.
    if stage.training:
        apply_noise_to_target(graph, config, epoch, batch_idx)
    predicted = stage.core.pipeline_exit(h, graph)
    errors = torch.nn.functional.mse_loss(predicted, graph.y, reduction='none')
    loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
    return loss * loss_scale, batch_loss_sum, batch_loss_count
