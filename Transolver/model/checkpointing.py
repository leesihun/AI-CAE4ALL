"""Block-level activation checkpointing (mirrors MeshGraphNets'
model/checkpointing.py). This is coarse-grained -- it recomputes an entire
TransolverBlock (attention + FFN) during backward -- and is independent of the
slice_space kernel's own finer-grained per-tile checkpointing internal to
model/physics_attention.py (section 6.3/6.6). Both may be enabled together;
they simply nest.

Set `use_checkpointing True` in config to enable.
"""

from torch.utils.checkpoint import checkpoint


def run_checkpointed(fn, *args, enabled: bool = True):
    """Run fn(*args), optionally under non-reentrant gradient checkpointing."""
    if enabled:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)
