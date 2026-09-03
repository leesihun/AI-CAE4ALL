"""State-dict merge for pipeline-split operator models.

Ported from MeshGraphNets' parallelism/checkpoint_io.py. Every stage's state
dict uses the exact key names of the full single-GPU OperatorWrapper (stages
are pruned full cores, see parallelism/stages.py), so the merged dict loads
into `model.factory.build_model_from_checkpoint` with strict=True and
inference needs no model-split awareness at all.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist


def merge_stage_state_dicts_to_rank0(
    stage_sd: Dict[str, torch.Tensor],
    group=None,
) -> Dict[str, torch.Tensor]:
    """Gather per-stage state dicts to rank 0 and merge into a single dict.

    Collective: every rank must call this. Returns the merged dict on rank 0
    and an empty dict elsewhere.
    """
    if not dist.is_available() or not dist.is_initialized():
        return dict(stage_sd)

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    cpu_sd = {k: v.detach().cpu() for k, v in stage_sd.items()}
    gathered: List[Dict[str, torch.Tensor]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, cpu_sd, group=group)

    if rank != 0:
        return {}

    merged: Dict[str, torch.Tensor] = {}
    for sd in gathered:
        for k, v in sd.items():
            if k in merged and not torch.equal(merged[k], v):
                print(f"  [merge_state_dict] WARNING: conflicting duplicate key '{k}' "
                      "across stages; last writer wins")
            merged[k] = v
    return merged
