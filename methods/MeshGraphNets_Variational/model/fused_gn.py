"""Fused Triton GnBlock for bf16 autocast on CUDA (sm_80+); used by compiled_block.

The same math as compiled_block.gn_tensors (no world edges, plain MLPs):

  forward   one edge kernel (sender/receiver gathers + the three edge-MLP GEMMs
            + LayerNorm + the edge residual; nothing intermediate leaves
            registers but the edge output and, with save=True, the three
            pre-activations) and one CSR segment sum for the aggregation. The
            node MLP stays torch (torch.compile'd by the caller).
  backward  one edge kernel that backprops LayerNorm and the three GEMMs, then
            cuBLAS for the weight gradients and CSR segment sums (no atomics)
            for the node gradients.

Measured on ex3 (RTX 3090, r9 hybrid, bs4, checkpointing): a level-0 block
130 -> 72 ms, the training step -14%, the 40-step loss trajectory inside the
run-to-run spread of the compiled path. Edges stay in their given order; the
CSR views go through argsort permutations, built without a host sync.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- CSR
def csr_unsorted(edge_index, n):
    """CSR views of edges in their given order (no reordering of edge tensors):
    in-edges through perm_r, out-edges through perm_s. No host sync: degrees by
    an integer scatter_add (bincount reads min/max back), and the segment-sum
    kernel bounds its loop per block from rowptr instead of a host max degree."""
    s, r = edge_index[0].contiguous(), edge_index[1].contiguous()
    perm_r = torch.argsort(r)
    perm_s = torch.argsort(s)
    ones = torch.ones_like(r)
    deg_r = torch.zeros(n, dtype=torch.long, device=r.device).scatter_add_(0, r, ones)
    deg_s = torch.zeros(n, dtype=torch.long, device=r.device).scatter_add_(0, s, ones)
    rowptr_r = torch.zeros(n + 1, dtype=torch.long, device=r.device)
    rowptr_r[1:] = torch.cumsum(deg_r, 0)
    rowptr_s = torch.zeros(n + 1, dtype=torch.long, device=r.device)
    rowptr_s[1:] = torch.cumsum(deg_s, 0)
    return dict(s=s, r=r, rowptr_r=rowptr_r, rowptr_s=rowptr_s, perm_r=perm_r, perm_s=perm_s, n=n)


def _layer_norm(h, w, b, eps, amp_ln):
    out = F.layer_norm(h, (h.shape[-1],), w, b, eps)
    return out.to(h.dtype) if amp_ln else out


# ----------------------------------------------------------------------------- Triton
def _cfgs():
    return [triton.Config({'BE': be}, num_warps=w, num_stages=1)
            for be, w in ((32, 4), (32, 8), (64, 4), (64, 8), (128, 8), (128, 16))]


@triton.jit
def _silu(h):
    return h * tl.sigmoid(h)


@triton.jit
def _dsilu(h):
    sg = tl.sigmoid(h)
    return sg * (1.0 + h * (1.0 - sg))


@triton.jit
def _edge_recompute(PSR, s_ptr, r_ptr, EA, W0eT, b0, W1T, b1, W2T, b2, offs_e, me, stride_psr,
                    D: tl.constexpr):
    offs_d = tl.arange(0, D)
    m2 = me[:, None]
    s = tl.load(s_ptr + offs_e, mask=me, other=0)
    r = tl.load(r_ptr + offs_e, mask=me, other=0)
    hs = tl.load(PSR + s[:, None] * stride_psr + offs_d[None, :], mask=m2, other=0.).to(tl.float32)
    hr = tl.load(PSR + r[:, None] * stride_psr + D + offs_d[None, :], mask=m2, other=0.).to(tl.float32)
    ea = tl.load(EA + offs_e[:, None] * D + offs_d[None, :], mask=m2, other=0.)
    wdd = offs_d[:, None] * D + offs_d[None, :]
    h0 = tl.dot(ea.to(tl.bfloat16), tl.load(W0eT + wdd)) + hs + hr + tl.load(b0 + offs_d)[None, :]
    a0 = _silu(h0).to(tl.bfloat16)
    h1 = tl.dot(a0, tl.load(W1T + wdd)) + tl.load(b1 + offs_d)[None, :]
    a1 = _silu(h1).to(tl.bfloat16)
    h2 = tl.dot(a1, tl.load(W2T + wdd)) + tl.load(b2 + offs_d)[None, :]
    return r, ea, h0, a0, h1, a1, h2


@triton.autotune(configs=_cfgs(), key=['D'])
@triton.jit
def _edge_fwd(PSR, s_ptr, r_ptr, EA, W0eT, b0, W1T, b1, W2T, b2, lnw, lnb, EOUT, EANEW, H0, H1, H2,
              E, eps, stride_psr, D: tl.constexpr, SAVE: tl.constexpr, BE: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = pid * BE + tl.arange(0, BE)
    me = offs_e < E
    offs_d = tl.arange(0, D)
    r, ea, h0, a0, h1, a1, h2 = _edge_recompute(PSR, s_ptr, r_ptr, EA, W0eT, b0, W1T, b1, W2T, b2,
                                                offs_e, me, stride_psr, D)
    mean = tl.sum(h2, 1) / D
    xc = h2 - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / D + eps)
    y = xc * rstd[:, None] * tl.load(lnw + offs_d)[None, :] + tl.load(lnb + offs_d)[None, :]
    ptr = offs_e[:, None] * D + offs_d[None, :]
    m2 = me[:, None]
    tl.store(EOUT + ptr, y.to(tl.bfloat16), mask=m2)
    tl.store(EANEW + ptr, (ea.to(tl.float32) + y).to(EANEW.dtype.element_ty), mask=m2)
    if SAVE:
        tl.store(H0 + ptr, h0.to(tl.bfloat16), mask=m2)
        tl.store(H1 + ptr, h1.to(tl.bfloat16), mask=m2)
        tl.store(H2 + ptr, h2.to(tl.bfloat16), mask=m2)


@triton.autotune(configs=_cfgs(), key=['D'], reset_to_zero=[])
@triton.jit
def _edge_bwd(PSR, s_ptr, r_ptr, EA, W0eT, b0, W1T, b1, W2T, b2, lnw, W0e, W1, W2,
              GEANEW, GAGG, GEA, GH0, GH1, GH2, A0, A1, PART, H0, H1, H2,
              E, eps, stride_psr, D: tl.constexpr, SAVE: tl.constexpr, BE: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = pid * BE + tl.arange(0, BE)
    me = offs_e < E
    m2 = me[:, None]
    offs_d = tl.arange(0, D)
    if SAVE:
        hp = offs_e[:, None] * D + offs_d[None, :]
        r = tl.load(r_ptr + offs_e, mask=me, other=0)
        h0 = tl.load(H0 + hp, mask=m2, other=0.).to(tl.float32)
        h1 = tl.load(H1 + hp, mask=m2, other=0.).to(tl.float32)
        h2 = tl.load(H2 + hp, mask=m2, other=0.).to(tl.float32)
        a0 = _silu(h0).to(tl.bfloat16)
        a1 = _silu(h1).to(tl.bfloat16)
    else:
        r, ea, h0, a0, h1, a1, h2 = _edge_recompute(PSR, s_ptr, r_ptr, EA, W0eT, b0, W1T, b1, W2T, b2,
                                                    offs_e, me, stride_psr, D)
    mean = tl.sum(h2, 1) / D
    xc = h2 - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / D + eps)
    xhat = xc * rstd[:, None]
    ptr = offs_e[:, None] * D + offs_d[None, :]
    gea_new = tl.load(GEANEW + ptr, mask=m2, other=0.).to(tl.float32)
    gy = gea_new + tl.load(GAGG + r[:, None] * D + offs_d[None, :], mask=m2, other=0.).to(tl.float32)
    dlnw = tl.sum(gy * xhat, 0)
    dlnb = tl.sum(gy, 0)
    gx = gy * tl.load(lnw + offs_d)[None, :]
    gh2 = rstd[:, None] * (gx - (tl.sum(gx, 1) / D)[:, None] - xhat * (tl.sum(gx * xhat, 1) / D)[:, None])
    wdd = offs_d[:, None] * D + offs_d[None, :]
    gh2b = gh2.to(tl.bfloat16)
    tl.store(GH2 + ptr, gh2b, mask=m2)
    tl.store(A1 + ptr, a1, mask=m2)
    gh1 = tl.dot(gh2b, tl.load(W2 + wdd)) * _dsilu(h1)
    gh1b = gh1.to(tl.bfloat16)
    tl.store(GH1 + ptr, gh1b, mask=m2)
    tl.store(A0 + ptr, a0, mask=m2)
    gh0 = tl.dot(gh1b, tl.load(W1 + wdd)) * _dsilu(h0)
    gh0b = gh0.to(tl.bfloat16)
    tl.store(GH0 + ptr, gh0b, mask=m2)
    gea = gea_new + tl.dot(gh0b, tl.load(W0e + wdd))
    tl.store(GEA + ptr, gea.to(GEA.dtype.element_ty), mask=m2)
    pp = PART + pid * 5 * D + offs_d
    tl.store(pp, tl.sum(gh0, 0))
    tl.store(pp + D, tl.sum(gh1, 0))
    tl.store(pp + 2 * D, tl.sum(gh2, 0))
    tl.store(pp + 3 * D, dlnw)
    tl.store(pp + 4 * D, dlnb)


@triton.jit
def _segsum_k(SRC, PERM, ROWPTR, OUT, N, out_stride, out_off,
              HAS_PERM: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mn = offs_n < N
    offs_d = tl.arange(0, D)
    start = tl.load(ROWPTR + offs_n, mask=mn, other=0)
    end = tl.load(ROWPTR + offs_n + 1, mask=mn, other=0)
    acc = tl.zeros([BN, D], dtype=tl.float32)
    kmax = tl.max(end - start, axis=0).to(tl.int32)     # this block's max degree
    for k in range(0, kmax):
        idx = start + k
        m = mn & (idx < end)
        if HAS_PERM:
            idx = tl.load(PERM + idx, mask=m, other=0)
        acc += tl.load(SRC + idx[:, None] * D + offs_d[None, :], mask=m[:, None], other=0.).to(tl.float32)
    tl.store(OUT + offs_n[:, None] * out_stride + out_off + offs_d[None, :],
             acc.to(OUT.dtype.element_ty), mask=mn[:, None])


def segsum(src, rowptr, n, perm=None, out=None, out_off=0):
    """out[v] = sum of src rows rowptr[v]:rowptr[v+1] (through perm when given)."""
    D = src.shape[1]
    if out is None:
        out = torch.empty(n, D, device=src.device, dtype=src.dtype)
    BN = 32
    _segsum_k[(triton.cdiv(n, BN),)](src, perm if perm is not None else rowptr, rowptr, out, n,
                                     out.stride(0), out_off, HAS_PERM=perm is not None, D=D, BN=BN,
                                     num_warps=4)
    return out


class _FusedEdge(torch.autograd.Function):
    """(x, ea, edge-MLP params) -> (ea + e_out, agg = segment sum of e_out by receiver)."""

    @staticmethod
    def forward(ctx, x, ea, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb, csr, eps, save):
        D = x.shape[1]
        bf = torch.bfloat16
        xb = x.to(bf)
        W0sr = eW0[:, :2 * D].to(bf)                              # [D, 2D]: rows out, cols (s|r) in
        psr = F.linear(xb, torch.cat([W0sr[:, :D], W0sr[:, D:]], 0))  # [N, 2D] = (x Ws^T | x Wr^T)
        W0e = eW0[:, 2 * D:].to(bf).contiguous()
        W1, W2 = eW1.to(bf).contiguous(), eW2.to(bf).contiguous()
        f32 = lambda t: t.float().contiguous()
        ea_c = ea.contiguous()
        E = ea_c.shape[0]
        e_out = torch.empty(E, D, device=x.device, dtype=bf)
        ea_new = torch.empty_like(ea_c)
        grid = lambda m: (triton.cdiv(E, m['BE']),)
        # save: keep the three [E, D] pre-activations for backward (faster);
        # otherwise backward recomputes them from the inputs (less memory).
        hs = [torch.empty(E, D, device=x.device, dtype=bf) for _ in range(3)] if save else [e_out] * 3
        _edge_fwd[grid](psr, csr['s'], csr['r'], ea_c, W0e.t().contiguous(), f32(eb0),
                        W1.t().contiguous(), f32(eb1), W2.t().contiguous(), f32(eb2),
                        f32(elnw), f32(elnb), e_out, ea_new, *hs, E, eps, psr.stride(0), D=D, SAVE=save)
        agg = segsum(e_out, csr['rowptr_r'], x.shape[0], perm=csr['perm_r'])
        ctx.save_for_backward(x, ea_c, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb, *(hs if save else ()))
        ctx.csr, ctx.eps, ctx.save = csr, eps, save
        return ea_new, agg

    @staticmethod
    def backward(ctx, g_ea_new, g_agg):
        x, ea, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb, *hs = ctx.saved_tensors
        csr, eps = ctx.csr, ctx.eps
        D, N, E = x.shape[1], x.shape[0], ea.shape[0]
        bf = torch.bfloat16
        xb = x.to(bf)
        Wsr = torch.cat([eW0[:, :D], eW0[:, D:2 * D]], 0).to(bf)   # [2D, D]
        psr = F.linear(xb, Wsr)
        W0e = eW0[:, 2 * D:].to(bf).contiguous()
        W1, W2 = eW1.to(bf).contiguous(), eW2.to(bf).contiguous()
        f32 = lambda t: t.float().contiguous()
        if g_agg is None:
            g_agg = torch.zeros(N, D, device=x.device, dtype=bf)
        g_ea_new = g_ea_new.contiguous()
        g_agg = g_agg.contiguous()
        g_ea = torch.empty_like(ea)
        gh0, gh1, gh2, a0, a1 = (torch.empty(E, D, device=x.device, dtype=bf) for _ in range(5))
        best = {}

        def grid(m):
            best['BE'] = m['BE']
            return (triton.cdiv(E, m['BE']),)
        # partials sized for the smallest tile; autotune may pick a larger one
        part = torch.zeros(triton.cdiv(E, 32), 5 * D, device=x.device, dtype=torch.float32)
        _edge_bwd[grid](psr, csr['s'], csr['r'], ea, W0e.t().contiguous(), f32(eb0),
                        W1.t().contiguous(), f32(eb1), W2.t().contiguous(), f32(eb2), f32(elnw),
                        W0e, W1, W2, g_ea_new, g_agg, g_ea, gh0, gh1, gh2, a0, a1, part,
                        *(hs or [gh0] * 3), E, eps, psr.stride(0), D=D, SAVE=ctx.save)
        p = part[:triton.cdiv(E, best['BE'])].sum(0)
        db0, db1, db2, dlnw, dlnb = p.split(D)
        dW2 = (gh2.t() @ a1).float()
        dW1 = (gh1.t() @ a0).float()
        dW0e = (gh0.t() @ ea.to(bf)).float()
        g_psr = torch.empty(N, 2 * D, device=x.device, dtype=bf)
        segsum(gh0, csr['rowptr_s'], N, perm=csr['perm_s'], out=g_psr, out_off=0)
        segsum(gh0, csr['rowptr_r'], N, perm=csr['perm_r'], out=g_psr, out_off=D)
        dWsr = (g_psr.t() @ xb).float()                            # [2D, D]
        g_x = (g_psr @ Wsr).to(x.dtype)
        dW0 = torch.cat([dWsr[:D], dWsr[D:], dW0e], 1)
        return (g_x, g_ea, dW0, db0, dW1, db1, dW2, db2, dlnw, dlnb, None, None, None)


def fused_gn(x, ea, csr, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb,
             nW0, nb0, nW1, nb1, nW2, nb2, nlnw, nlnb, eps: float, amp_ln: bool, node_fn=None,
             save=True):
    """One GnBlock: (x + node update, ea + edge update). `csr` = csr_unsorted(edge_index, N)."""
    ea_new, agg = _FusedEdge.apply(x, ea, eW0, eb0, eW1, eb1, eW2, eb2, elnw, elnb, csr, eps, save)
    return (node_fn or node_tensors)(x, agg, nW0, nb0, nW1, nb1, nW2, nb2, nlnw, nlnb, eps, amp_ln), ea_new


def node_tensors(x, agg, nW0, nb0, nW1, nb1, nW2, nb2, nlnw, nlnb, eps: float, amp_ln: bool):
    D = x.shape[-1]
    n = F.silu(F.linear(x, nW0[:, :D], nb0) + F.linear(agg, nW0[:, D:]))
    n = F.silu(F.linear(n, nW1, nb1))
    n = _layer_norm(F.linear(n, nW2, nb2), nlnw, nlnb, eps, amp_ln)
    return x + n
