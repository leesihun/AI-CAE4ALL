"""CPU-only device resolution for the inference bundle.

The bundle intentionally never ships CUDA wheels or torch_cluster (see
INFERENCE_BUNDLE_PLAN.md section 1, item 6). Every driver calls
`resolve_device()` instead of touching `torch.cuda` directly.
"""

import torch


def resolve_device(requested: str = "cpu") -> torch.device:
    requested = (requested or "cpu").lower()
    if requested not in ("cpu", "auto"):
        print(f"[device] '{requested}' requested but this bundle is CPU-only; using cpu.")
    return torch.device("cpu")
