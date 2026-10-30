"""`use_compile`: each plain GnBlock as one torch.compile'd tensor function, eager when that fails.

A GnBlock is ~20 small kernels (three split Linears, two gathers, SiLUs, a
LayerNorm, a scatter, two more split Linears, two more Linears, another
LayerNorm, two residual adds). On ex3 they are memory-bound, and the step spends most of its
time on them rather than in the matmuls (2026-10-27 profile: GEMM 22% of
kernel time). Compiled as one function Inductor fuses the elementwise work into
the gathers and the scatter: a level-0 block 46 -> 30 ms, the whole training
step 1.36x on a 3090, outputs equal to eager up to bf16 rounding.

The weights go in as tensors, so every block of every width shares one compiled
graph (`dynamic=True`: node and edge counts vary per batch). Whole-model
`torch.compile` is not used: the Data objects, checkpointing and the prior's
control flow are what made it unreliable.

Failure is never fatal. `enable()` first runs every block width the model has
through the compiled function, forward AND backward (Inductor compiles the
backward lazily, on the first backward, where a failure could not be caught),
under the run's own autocast setting, and compares it with eager. Any exception
or a mismatch turns compilation off for the run and training continues eager;
so does any later exception raised inside the compiled forward (out-of-memory and
checkpoint recomputation's own early stop excepted, which are re-raised). Each
case prints one line saying why.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

_compiled = None
_disabled_reason = None


def _passthrough(exc) -> bool:
    """Exceptions that are not a compile failure and must reach the caller:
    out-of-memory, and the control flow of torch.utils.checkpoint (non-reentrant
    recomputation stops early by raising through the forward)."""
    return (isinstance(exc, torch.cuda.OutOfMemoryError)
            or type(exc).__module__ == 'torch.utils.checkpoint')


def _layer_norm(h, w, b, eps: float, amp_ln: bool):
    out = F.layer_norm(h, (h.shape[-1],), w, b, eps)
    # Under autocast layer_norm returns fp32; AMP keeps the MLP's dtype, as
    # eager AmpLayerNorm does in its own kernel (model/mlp.py).
    return out.to(h.dtype) if amp_ln else out


def gn_tensors(x, ea, s, r, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb,
               nW0, nb0, nW1, nb1, nW2, nb2, nlnw, nlnb, eps: float, amp_ln: bool):
    """GnBlock.forward_tensors without world edges: EdgeBlock and NodeBlock
    (both with the split first Linear), sum aggregation, both residual adds."""
    D = x.shape[-1]
    h = (F.linear(x, eW0[:, :D])[s] + F.linear(x, eW0[:, D:2 * D])[r]
         + F.linear(ea, eW0[:, 2 * D:], eb0))
    h = F.silu(F.linear(F.silu(h), eW1, eb1))
    e_out = _layer_norm(F.linear(h, eW2, eb2), elnw, elnb, eps, amp_ln)
    agg = scatter(e_out, r, dim=0, dim_size=x.shape[0], reduce='sum')
    n = F.silu(F.linear(x, nW0[:, :D], nb0) + F.linear(agg, nW0[:, D:]))
    n = F.silu(F.linear(n, nW1, nb1))
    n = _layer_norm(F.linear(n, nW2, nb2), nlnw, nlnb, eps, amp_ln)
    return x + n, ea + e_out


def _mlp_params(seq):
    l0, l1, l2, ln = seq[0], seq[2], seq[4], seq[5]
    return (l0.weight, l0.bias, l1.weight, l1.bias, l2.weight, l2.bias, ln.weight, ln.bias)


def _plain_mlp(seq):
    return (isinstance(seq, nn.Sequential) and len(seq) == 6
            and all(isinstance(seq[i], nn.Linear) for i in (0, 2, 4))
            and all(isinstance(seq[i], nn.SiLU) for i in (1, 3))
            and isinstance(seq[5], nn.LayerNorm) and seq[5].weight is not None)


def eligible(block) -> bool:
    """A GnBlock the tensor function reproduces: no world edges, plain MLPs."""
    from model.blocks import NodeBlock
    return (not block.use_world_edges and type(block.nb_module) is NodeBlock
            and _plain_mlp(block.eb_module.net) and _plain_mlp(block.nb_module.net))


def _args(block, x, ea, ei):
    ln = block.nb_module.net[5]
    return (x, ea, ei[0], ei[1], *_mlp_params(block.eb_module.net),
            *_mlp_params(block.nb_module.net), float(ln.eps), bool(getattr(ln, 'in_amp_dtype', False)))


def _disable(reason):
    global _disabled_reason
    if _disabled_reason is None:
        _disabled_reason = reason
        print(f'use_compile: compiled GnBlocks OFF, training continues eager ({reason})', flush=True)


def run(block, x, ea, ei):
    """The compiled block's (x, edge_attr), or None when the caller should run eager."""
    if _compiled is None or _disabled_reason is not None:
        return None
    try:
        return _compiled(*_args(block, x, ea, ei))
    except Exception as exc:   # compile or kernel failure: eager from here on
        if _passthrough(exc):
            raise
        _disable(f'{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ""}')
        return None


def _self_test(width, device, amp_dtype, use_amp):
    """Compiled vs eager on one small random graph, forward and backward."""
    from model.encoder_decoder import GnBlock
    torch.manual_seed(0)   # the caller restores the RNG
    block = GnBlock(width).to(device)
    n, e = 1000, 4000
    x0 = torch.randn(n, width, device=device)
    ea0 = torch.randn(e, width, device=device)
    ei = torch.randint(0, n, (2, e), device=device)
    outs = []
    for fn in (None, _compiled):
        x, ea = x0.clone().requires_grad_(), ea0.clone().requires_grad_()
        with torch.enable_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            if fn is None:
                ox, oe, _ = block.forward_tensors(x, ea, ei)
            else:
                ox, oe = fn(*_args(block, x, ea, ei))
            loss = ox.float().square().mean() + oe.float().square().mean()
        grads = torch.autograd.grad(loss, [x, ea, *block.parameters()])
        outs.append((ox.detach().float(), oe.detach().float(), [g.float() for g in grads]))
    (ax, ae, ag), (bx, be, bg) = outs
    tol = 0.1 if use_amp else 1e-3
    worst = max([(ax - bx).abs().max().item(), (ae - be).abs().max().item()]
                + [((g1 - g2).abs().max() / (g1.abs().max() + 1e-12)).item() for g1, g2 in zip(ag, bg)])
    finite = all(torch.isfinite(t).all().item() for t in (bx, be, *bg))
    if not finite or worst > tol:
        raise RuntimeError(f'compiled block differs from eager at width {width} '
                           f'(max diff {worst:.3g}, finite {finite})')


def enable(models, device, amp_dtype, use_amp):
    """Mark every eligible GnBlock of `models` (the model, its EMA copy) compiled,
    after a self-test of each block width.

    Returns the number of blocks marked (0 when compilation is off)."""
    global _compiled
    from model.encoder_decoder import GnBlock
    models = [m for m in models if m is not None]
    blocks = [b for m in models for b in m.modules() if isinstance(b, GnBlock) and eligible(b)]
    if not blocks:
        return 0
    if _disabled_reason is not None:
        return 0
    device = torch.device(device)
    if device.type != 'cuda':
        _disable('not a CUDA device')
        return 0
    # The self-test GnBlocks must use the model's LayerNorm setting.
    from model.mlp import AmpLayerNorm
    amp_ln = any(getattr(n, 'in_amp_dtype', False) for m in models for n in m.modules()
                 if isinstance(n, AmpLayerNorm))
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
    saved = AmpLayerNorm.in_amp_dtype
    try:
        _compiled = torch.compile(gn_tensors, dynamic=True)
        AmpLayerNorm.in_amp_dtype = amp_ln
        for width in sorted({b.nb_module.net[5].normalized_shape[0] for b in blocks}):
            _self_test(width, device, amp_dtype, use_amp)
    except Exception as exc:
        if _passthrough(exc):
            raise
        _disable(f'self-test: {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ""}')
        return 0
    finally:
        AmpLayerNorm.in_amp_dtype = saved
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], device)
    for b in blocks:
        b.compiled = True
    print(f'use_compile: {len(blocks)} GnBlocks compiled (self-test passed; eager if a later call fails)',
          flush=True)
    return len(blocks)
