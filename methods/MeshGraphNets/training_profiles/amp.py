"""AMP dtype selection.

bfloat16 needs native tensor-core support (compute capability >= 8.0, i.e.
Ampere and later). On older cards cuBLAS silently falls back for bf16 GEMMs:
measured on an RTX 2080 SUPER (sm_75), a 4-block GnBlock fwd+bwd takes 746 ms
in bf16 vs 135 ms in fp16 and 199 ms in plain fp32.

`torch.cuda.is_bf16_supported()` is NOT a usable gate — it returns True on
sm_75 because it counts emulation. Check the capability directly.
"""

import torch


def bf16_is_native(device=None) -> bool:
    """True when the device has native bfloat16 tensor cores (sm_80+)."""
    if not torch.cuda.is_available():
        return False
    index = None
    if device is not None and getattr(device, 'type', None) == 'cuda':
        index = device.index
    return torch.cuda.get_device_capability(index)[0] >= 8


def resolve_amp_dtype(device=None):
    """Return the autocast dtype to use: bfloat16 where native, else float16.

    float16 needs loss scaling; see `build_grad_scaler`.
    """
    return torch.bfloat16 if bf16_is_native(device) else torch.float16


def build_grad_scaler(amp_dtype, enabled: bool):
    """Return a GradScaler that is active only for enabled float16 autocast.

    bfloat16 has fp32's exponent range and needs no loss scaling, so the
    scaler is constructed disabled and every call becomes a passthrough.
    """
    return torch.amp.GradScaler(
        'cuda', enabled=bool(enabled) and amp_dtype is torch.float16
    )


def describe_amp(amp_dtype) -> str:
    """Short label for logging, e.g. 'float16 (bfloat16 not native on sm_75)'."""
    if amp_dtype is torch.bfloat16:
        return 'bfloat16'
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'float16 (bfloat16 not native on sm_{major}{minor})'
    return 'float16'
