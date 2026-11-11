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

Two more paths ride on the same switch (2026-11-10, ex3 r9 hybrid bs4 on a
3090: 0.759 -> 0.543 s/batch together with the host-sync removals elsewhere,
the 40-step loss trajectory inside the run-to-run spread):

  fused     under bf16 autocast on sm_80+ with Triton installed, a block of
            width 64/128 runs the fused edge kernel of model/fused_gn.py
            instead of the compiled function (a level-0 block 130 -> 72 ms).
            MGNV_FUSED=0 in the environment turns it off.
  extras    the plain encoder MLPs (checkpointing.run_mlp) and each
            UnpoolBlock with its skip merge, compiled the same way.

Failure is never fatal. `enable()` first runs every block width the model has
through the compiled function, forward AND backward (Inductor compiles the
backward lazily, on the first backward, where a failure could not be caught),
under the run's own autocast setting, and compares it with eager. Any exception
or a mismatch turns compilation off for the run and training continues eager;
so does any later exception raised inside the compiled forward (out-of-memory and
checkpoint recomputation's own early stop excepted, which are re-raised). The
fused and extras paths get their own self-test and fall back on their own: a
failure there turns off only that path. Each case prints one line saying why.
"""

import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

_compiled = None
_disabled_reason = None
# Off-reasons of the two optional paths (None while on); set by enable().
_fused_off = 'not enabled'
_extras_off = 'not enabled'
_fused_node = None
_mlp_c = None
_unpool_c = None
# CSR views of each edge_index, cached by tensor identity: the encoder's blocks
# share one edge_index per level, so it is built once per level and step.
_csr_cache = {}


def _passthrough(exc) -> bool:
    """Exceptions that are not a compile failure and must reach the caller:
    out-of-memory, and the control flow of torch.utils.checkpoint (non-reentrant
    recomputation stops early by raising through the forward)."""
    return (isinstance(exc, torch.cuda.OutOfMemoryError)
            or type(exc).__module__ == 'torch.utils.checkpoint')


def _why(exc):
    return f'{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ""}'


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
    # fp32 scatter: Inductor's bf16 scatter_add falls back to aten on sm_86 (-3% step there)
    agg = scatter(e_out.float(), r, dim=0, dim_size=x.shape[0], reduce='sum')
    n = F.silu(F.linear(x, nW0[:, :D], nb0) + F.linear(agg, nW0[:, D:]))
    n = F.silu(F.linear(n, nW1, nb1))
    n = _layer_norm(F.linear(n, nW2, nb2), nlnw, nlnb, eps, amp_ln)
    return x + n, ea + e_out


def mlp_tensors(x, W0, b0, W1, b1, W2, b2, lnw, lnb, eps: float, amp_ln: bool):
    """A build_mlp Sequential (Linear, SiLU, Linear, SiLU, Linear, LayerNorm)."""
    h = F.silu(F.linear(x, W0, b0))
    h = F.silu(F.linear(h, W1, b1))
    return _layer_norm(F.linear(h, W2, b2), lnw, lnb, eps, amp_ln)


def unpool_tensors(h_c, h_s, src, dst, rel_pos, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb,
                   nW0, nb0, nW1, nb1, nW2, nb2, nlnw, nlnb, sW, sb, eps: float, amp_ln: bool):
    """UnpoolBlock.forward, then the skip merge of MeshGraphNets._unpool_merge."""
    D = h_c.shape[-1]
    h = (F.linear(h_c, eW0[:, :D], eb0)[src] + F.linear(h_s, eW0[:, D:2 * D])[dst]
         + F.linear(rel_pos, eW0[:, 2 * D:]))
    h = F.silu(F.linear(F.silu(h), eW1, eb1))
    msg = _layer_norm(F.linear(h, eW2, eb2), elnw, elnb, eps, amp_ln)
    agg = scatter(msg, dst, dim=0, dim_size=h_s.shape[0], reduce='sum')
    n = F.silu(F.linear(h_s, nW0[:, :D], nb0) + F.linear(agg, nW0[:, D:]))
    n = F.silu(F.linear(n, nW1, nb1))
    h_up = _layer_norm(F.linear(n, nW2, nb2), nlnw, nlnb, eps, amp_ln)
    return F.linear(h_s, sW[:, :D], sb) + F.linear(h_up, sW[:, D:])


def _mlp_params(seq):
    l0, l1, l2, ln = seq[0], seq[2], seq[4], seq[5]
    return (l0.weight, l0.bias, l1.weight, l1.bias, l2.weight, l2.bias, ln.weight, ln.bias)


def _ln_args(seq):
    ln = seq[5]
    return float(ln.eps), bool(getattr(ln, 'in_amp_dtype', False))


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
    return (x, ea, ei[0], ei[1], *_mlp_params(block.eb_module.net),
            *_mlp_params(block.nb_module.net), *_ln_args(block.nb_module.net))


def _disable(reason):
    global _disabled_reason
    if _disabled_reason is None:
        _disabled_reason = reason
        print(f'use_compile: compiled GnBlocks OFF, training continues eager ({reason})', flush=True)


def _disable_fused(reason):
    global _fused_off
    if _fused_off is None:
        print(f'use_compile: fused Triton GnBlocks OFF, compiled GnBlocks continue ({reason})', flush=True)
    _fused_off = reason


def _disable_extras(reason):
    global _extras_off
    if _extras_off is None:
        print(f'use_compile: compiled MLPs/UnpoolBlocks OFF, they continue eager ({reason})', flush=True)
    _extras_off = reason


def _fused_unavailable(device):
    """Why the fused path cannot run on `device`, or None when it can."""
    if os.environ.get('MGNV_FUSED', '') == '0':
        return 'MGNV_FUSED=0'
    if torch.cuda.get_device_capability(device) < (8, 0):
        return 'needs sm_80+ for bf16 tensor-core dots'
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return f'triton not importable ({type(exc).__name__})'
    return None


# ----------------------------------------------------------------------------- run

def _csr(ei, n):
    from model import fused_gn
    hit = _csr_cache.get(id(ei))
    if hit is not None and hit[0]() is ei and hit[1]['n'] == n:
        return hit[1]
    for k in [k for k, v in _csr_cache.items() if v[0]() is None]:
        del _csr_cache[k]
    csr = fused_gn.csr_unsorted(ei, n)
    _csr_cache[id(ei)] = (weakref.ref(ei), csr)
    return csr


def _fused_ok(x, ei):
    return (_fused_off is None and x.is_cuda and x.shape[-1] in (64, 128) and ei.shape[1] > 0
            and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16)


def run_fused(block, x, ea, ei):
    """The fused block's (x, edge_attr). Saves the edge pre-activations for
    backward only when a backward can follow."""
    global _fused_node
    from model import fused_gn
    if _fused_node is None:
        _fused_node = torch.compile(fused_gn.node_tensors, dynamic=True)
    a = _args(block, x, ea, ei)
    return fused_gn.fused_gn(x, ea, _csr(ei, x.shape[0]), *a[4:], node_fn=_fused_node,
                             save=torch.is_grad_enabled())


def run(block, x, ea, ei):
    """The compiled block's (x, edge_attr), or None when the caller should run eager."""
    if _compiled is None or _disabled_reason is not None:
        return None
    if _fused_ok(x, ei):
        try:
            return run_fused(block, x, ea, ei)
        except Exception as exc:   # kernel failure: the compiled function from here on
            if _passthrough(exc):
                raise
            _disable_fused(_why(exc))
    try:
        return _compiled(*_args(block, x, ea, ei))
    except Exception as exc:   # compile or kernel failure: eager from here on
        if _passthrough(exc):
            raise
        _disable(_why(exc))
        return None


def _extras_ready(t):
    return _extras_off is None and _compiled is not None and _disabled_reason is None and t.is_cuda


def run_mlp(seq, x):
    """The compiled build_mlp Sequential's output, or None for eager."""
    global _mlp_c
    if not (_extras_ready(x) and _plain_mlp(seq)):
        return None
    if _mlp_c is None:
        _mlp_c = torch.compile(mlp_tensors, dynamic=True)
    try:
        return _mlp_c(x, *_mlp_params(seq), *_ln_args(seq))
    except Exception as exc:
        if _passthrough(exc):
            raise
        _disable_extras(f'mlp {_why(exc)}')
        return None


def run_unpool(block, skip_proj, h_c, h_s, up_ei, rel_pos):
    """The compiled UnpoolBlock + skip merge, or None for eager."""
    global _unpool_c
    if not (_extras_ready(h_c) and _plain_mlp(block.edge_mlp) and _plain_mlp(block.node_mlp)):
        return None
    if _unpool_c is None:
        _unpool_c = torch.compile(unpool_tensors, dynamic=True)
    try:
        return _unpool_c(h_c, h_s, up_ei[0], up_ei[1], rel_pos,
                         *_mlp_params(block.edge_mlp), *_mlp_params(block.node_mlp),
                         skip_proj.weight, skip_proj.bias, *_ln_args(block.node_mlp))
    except Exception as exc:
        if _passthrough(exc):
            raise
        _disable_extras(f'unpool {_why(exc)}')
        return None


# ----------------------------------------------------------------------------- self-tests

def _compare(name, eager_fn, test_fn, inputs, params, use_amp, device, amp_dtype):
    """Forward and backward of test_fn against eager_fn on the same inputs."""
    outs = []
    for fn in (eager_fn, test_fn):
        xs = [t.clone().requires_grad_() for t in inputs]
        with torch.enable_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            ys = fn(*xs)
            loss = sum(y.float().square().mean() for y in ys)
        grads = torch.autograd.grad(loss, [*xs, *params])
        outs.append(([y.detach().float() for y in ys], [g.float() for g in grads]))
    (ay, ag), (by, bg) = outs
    tol = 0.1 if use_amp else 1e-3
    worst = max([(a - b).abs().max().item() for a, b in zip(ay, by)]
                + [((g1 - g2).abs().max() / (g1.abs().max() + 1e-12)).item() for g1, g2 in zip(ag, bg)])
    finite = all(torch.isfinite(t).all().item() for t in (*by, *bg))
    if not finite or worst > tol:
        raise RuntimeError(f'{name} differs from eager (max diff {worst:.3g}, finite {finite})')


def _graph(n, e, width, device):
    return (torch.randn(n, width, device=device), torch.randn(e, width, device=device),
            torch.randint(0, n, (2, e), device=device))


def _self_test(width, device, amp_dtype, use_amp, fused=False):
    """Compiled (or fused) vs eager GnBlock on one small random graph."""
    from model.encoder_decoder import GnBlock
    torch.manual_seed(0)   # the caller restores the RNG
    block = GnBlock(width).to(device)
    x0, ea0, ei = _graph(1000, 4000, width, device)

    def eager(x, ea):
        return block.forward_tensors(x, ea, ei)[:2]

    if fused:
        def test(x, ea):
            return run_fused(block, x, ea, ei)
    else:
        def test(x, ea):
            return _compiled(*_args(block, x, ea, ei))
    _compare(f'{"fused" if fused else "compiled"} block at width {width}', eager, test,
             [x0, ea0], list(block.parameters()), use_amp, device, amp_dtype)


def _self_test_extras(width, device, amp_dtype, use_amp):
    """Compiled vs eager encoder MLP and UnpoolBlock + skip merge."""
    from model.blocks import UnpoolBlock, split_linear
    from model.mlp import build_mlp
    global _mlp_c, _unpool_c
    torch.manual_seed(0)
    mlp = build_mlp(8, width, width).to(device)
    _mlp_c = _mlp_c or torch.compile(mlp_tensors, dynamic=True)
    _compare(f'compiled MLP at width {width}', lambda t: (mlp(t),),
             lambda t: (_mlp_c(t, *_mlp_params(mlp), *_ln_args(mlp)),),
             [torch.randn(1000, 8, device=device)], list(mlp.parameters()), use_amp, device, amp_dtype)

    unpool = UnpoolBlock(width, build_mlp).to(device)
    skip = nn.Linear(2 * width, width).to(device)
    h_c, h_s = torch.randn(300, width, device=device), torch.randn(1000, width, device=device)
    up_ei = torch.stack([torch.randint(0, 300, (3000,), device=device),
                         torch.randint(0, 1000, (3000,), device=device)])
    rel_pos = torch.randn(3000, 3, device=device)
    _unpool_c = _unpool_c or torch.compile(unpool_tensors, dynamic=True)

    def eager(hc, hs):
        h_up = unpool(h_coarse=hc, h_fine_skip=hs, unpool_edge_index=up_ei, rel_pos=rel_pos)
        return (split_linear(skip, [(hs, None), (h_up, None)]),)

    def test(hc, hs):
        return (_unpool_c(hc, hs, up_ei[0], up_ei[1], rel_pos, *_mlp_params(unpool.edge_mlp),
                          *_mlp_params(unpool.node_mlp), skip.weight, skip.bias,
                          *_ln_args(unpool.node_mlp)),)
    _compare(f'compiled UnpoolBlock at width {width}', eager, test, [h_c, h_s],
             [*unpool.parameters(), *skip.parameters()], use_amp, device, amp_dtype)


def enable(models, device, amp_dtype, use_amp):
    """Mark every eligible GnBlock of `models` (the model, its EMA copy) compiled,
    after a self-test of each block width; then self-test the fused and extras
    paths, each of which only turns itself off on failure.

    Returns the number of blocks marked (0 when compilation is off)."""
    global _compiled, _fused_off, _extras_off
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
    widths = sorted({b.nb_module.net[5].normalized_shape[0] for b in blocks})
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
    saved = AmpLayerNorm.in_amp_dtype
    try:
        AmpLayerNorm.in_amp_dtype = amp_ln
        try:
            _compiled = torch.compile(gn_tensors, dynamic=True)
            for width in widths:
                _self_test(width, device, amp_dtype, use_amp)
        except Exception as exc:
            if _passthrough(exc):
                raise
            _disable(f'self-test: {_why(exc)}')
            return 0

        _fused_off = None
        reason = _fused_unavailable(device) if (use_amp and amp_dtype == torch.bfloat16) else 'needs bf16 autocast'
        if reason is not None:
            _fused_off = reason
            print(f'use_compile: fused Triton GnBlocks not used ({reason})', flush=True)
        else:
            try:
                for width in [w for w in widths if w in (64, 128)]:
                    _self_test(width, device, amp_dtype, use_amp, fused=True)
            except Exception as exc:
                if _passthrough(exc):
                    raise
                _disable_fused(f'self-test: {_why(exc)}')

        _extras_off = None
        try:
            for width in widths:
                _self_test_extras(width, device, amp_dtype, use_amp)
        except Exception as exc:
            if _passthrough(exc):
                raise
            _disable_extras(f'self-test: {_why(exc)}')
    finally:
        AmpLayerNorm.in_amp_dtype = saved
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], device)
    for b in blocks:
        b.compiled = True
    print(f'use_compile: {len(blocks)} GnBlocks compiled (self-test passed; eager if a later call fails)'
          f'; fused Triton {"ON" if _fused_off is None else "off"}'
          f', compiled MLPs/UnpoolBlocks {"ON" if _extras_off is None else "off"}', flush=True)
    return len(blocks)
