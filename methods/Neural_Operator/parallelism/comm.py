"""Point-to-point pipeline communication for model-split training.

Ported from MeshGraphNets' parallelism/model_split.py bundle protocol,
simplified for this repository: the boundary between operator pipeline stages
is always exactly ONE tensor (the latent grid [B, hidden, *resolution]), so
the multi-tensor bundle packing is unnecessary. What is kept unchanged:

  * Two separate process groups (downstream activations vs upstream
    gradients). NCCL runs each communicator on its own stream, so a queued
    data-send can never serialize ahead of a grad op and deadlock 1F1B.
  * Tracked async sends, swept opportunistically and fully drained before
    every optimizer step.
  * A shape/dtype header per send: the last micro-batch group of an epoch may
    have a smaller batch dimension, so the receiver cannot assume a shape.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist

_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float64: 1, torch.bfloat16: 2, torch.float16: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

_MAX_DIMS = 6
_HEADER_LEN = 2 + _MAX_DIMS  # [ndim, dtype_code, s0..s5]

_PG_DATA = None
_PG_GRAD = None

# In-flight async sends: (work, buffer) pairs kept alive until the transfer
# completes. Swept opportunistically; fully drained before each optimizer step.
_PENDING_SENDS: List[tuple] = []


def set_pipeline_process_groups(pg_data, pg_grad) -> None:
    global _PG_DATA, _PG_GRAD
    _PG_DATA = pg_data
    _PG_GRAD = pg_grad


def _isend_tracked(t: torch.Tensor, dst: int, group) -> None:
    work = dist.isend(t, dst=dst, group=group)
    _PENDING_SENDS.append((work, t))
    if len(_PENDING_SENDS) > 16:
        _PENDING_SENDS[:] = [(w, b) for w, b in _PENDING_SENDS if not w.is_completed()]


def drain_pending_sends() -> None:
    """Block until every outstanding isend finished; call before optimizer.step()."""
    for work, _ in _PENDING_SENDS:
        work.wait()
    _PENDING_SENDS.clear()


def _send_tensor(t: torch.Tensor, dst: int, group) -> None:
    if t.dim() > _MAX_DIMS:
        raise ValueError(f"pipeline tensor has {t.dim()} dims, max {_MAX_DIMS}")
    meta = [t.dim(), _DTYPE_TO_CODE[t.dtype]] + list(t.shape) + [0] * (_MAX_DIMS - t.dim())
    header = torch.tensor(meta, dtype=torch.long, device=t.device)
    _isend_tracked(header, dst, group)
    _isend_tracked(t.contiguous(), dst, group)


def _recv_tensor(src: int, device: torch.device, group) -> torch.Tensor:
    header = torch.empty(_HEADER_LEN, dtype=torch.long, device=device)
    dist.recv(header, src=src, group=group)
    meta = header.tolist()
    ndim, dtype_code = meta[0], meta[1]
    shape = meta[2:2 + ndim]
    buf = torch.empty(shape, dtype=_CODE_TO_DTYPE[dtype_code], device=device)
    dist.recv(buf, src=src, group=group)
    return buf


class GridSend(torch.autograd.Function):
    """Send the latent grid downstream; receive its gradient in backward."""

    @staticmethod
    def forward(ctx, dst: int, h: torch.Tensor) -> torch.Tensor:
        ctx.dst = dst
        ctx.device = h.device
        ctx.shape = tuple(h.shape)
        ctx.dtype = h.dtype
        _send_tensor(h, dst, _PG_DATA)
        return torch.zeros((), device=h.device, dtype=h.dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, _grad_sentinel):
        buf = torch.empty(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        dist.recv(buf, src=ctx.dst, group=_PG_GRAD)
        return None, buf


class GridRecv(torch.autograd.Function):
    """Receive the latent grid from upstream; send its gradient back in backward."""

    @staticmethod
    def forward(ctx, src: int, device: torch.device, anchor: torch.Tensor) -> torch.Tensor:
        ctx.src = src
        ctx.device = device
        h = _recv_tensor(src, device, _PG_DATA)
        ctx.shape = tuple(h.shape)
        ctx.dtype = h.dtype
        return h

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # The upstream stage blocks on this gradient unconditionally, so send
        # zeros rather than nothing if autograd delivered no grad.
        if grad is None:
            grad = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        if grad.dtype != ctx.dtype:
            grad = grad.to(ctx.dtype)
        _isend_tracked(grad.contiguous(), ctx.src, _PG_GRAD)
        return None, None, None


def send_downstream(h: torch.Tensor, dst: int) -> torch.Tensor:
    """Returns a 0-dim sentinel; `.backward()` on it blocks until the
    downstream stage returns the boundary gradient."""
    return GridSend.apply(dst, h)


def recv_upstream(src: int, device: torch.device) -> torch.Tensor:
    anchor = torch.zeros((), device=device, requires_grad=True)
    return GridRecv.apply(src, device, anchor)
