#!/usr/bin/env python3
"""Execute the OFFICIAL Transolver v1 and Transolver-3 model code directly
against this repo's implementation, with weights mapped between the two
layouts, and assert numerical agreement in fp64.

This is stronger than the unit tests in tests/ (which compare against an
independent transcription of the official math): here the upstream authors'
own Python files are imported and run, so even a shared misreading of their
code would be caught.

Usage:
    python misc/verify_against_official.py --t3-root <dir> [--v1-file <path>]

  --t3-root  directory containing a `models/` package with
             Transolver_chunk_opt_matrix_mul.py and
             Transolver_chunk_opt_matrix_mul_amortize.py
             (i.e. a clone of github.com/thuml/Transolver-3)
  --v1-file  path to the official v1 Physics_Attention.py
             (PDE-Solving-StandardBenchmark/model/Physics_Attention.py from
             github.com/thuml/Transolver); optional.

Expected results (all fp64, CPU):
  - every comparison against the chunk/num-den paths: < 1e-12 relative error;
  - Transolver-3's own non-chunk `Physics_Attention.forward` differs from the
    v1-exact result by ~eps/norm (documented convention difference, plan 6.3):
    asserted to land in (1e-10, 1e-5) to prove the two conventions are
    distinguishable at this scale.
"""
import argparse
import importlib.util
import os
import sys
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.physics_attention import PhysicsAttentionIrregular  # noqa: E402
from model.Transolver import Transolver  # noqa: E402

# --- test dimensions (small, fp64, CPU) ---
N, C, H, D, M, L, OUT = 257, 32, 4, 8, 8, 3, 4
CHUNK = 64  # -> tiles [64, 64, 64, 64, 1]; the size-1 tail is a stress case
FUN_DIM, SPACE_DIM = 12, 3  # their preprocess in-dim = 15 = my node_input(12) + pos(3)

FAILURES = []


def check(name, err, bound=1e-12, lower=None):
    ok = err < bound and (lower is None or err > lower)
    band = f"< {bound:g}" if lower is None else f"in ({lower:g}, {bound:g})"
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: rel err = {err:.3e} (expected {band})")
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _install_einops_stub():
    """T3's model file imports rearrange without using it; official v1 uses
    exactly 'b h n d -> b n (h d)'. Provide precisely that much."""
    if 'einops' in sys.modules:
        return
    try:
        import einops  # noqa: F401
        return
    except ImportError:
        pass

    stub = types.ModuleType('einops')

    def rearrange(t, pattern, **kw):
        if pattern.replace(' ', '') == 'bhnd->bn(hd)':
            b, h, n, d = t.shape
            return t.permute(0, 2, 1, 3).reshape(b, n, h * d)
        raise NotImplementedError(f"einops stub: unsupported pattern {pattern!r}")

    def repeat(*a, **k):
        raise NotImplementedError("einops stub: repeat not supported")

    stub.rearrange = rearrange
    stub.repeat = repeat
    sys.modules['einops'] = stub


def _import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# weight mapping: my v1 layout -> their fused layout
# ---------------------------------------------------------------------------

def map_attention_to_t3(mine, theirs):
    """theirs.in_project = Linear(dim, 2*H*D): first half rows = fx (value),
    second half = x (assignment) -- verified from their _get_fused_weight_slice
    ([:H*D] used as w_fx, [H*D:] fused with the slice projector)."""
    with torch.no_grad():
        theirs.in_project.weight.copy_(torch.cat(
            [mine.in_project_fx.weight, mine.in_project_x.weight], dim=0))
        theirs.in_project.bias.copy_(torch.cat(
            [mine.in_project_fx.bias, mine.in_project_x.bias], dim=0))
        theirs.in_project_slice.weight.copy_(mine.in_project_slice.weight)
        theirs.in_project_slice.bias.copy_(mine.in_project_slice.bias)
        theirs.to_q.weight.copy_(mine.to_q.weight)
        theirs.to_k.weight.copy_(mine.to_k.weight)
        theirs.to_v.weight.copy_(mine.to_v.weight)
        theirs.to_out_linear.weight.copy_(mine.to_out.weight)
        theirs.to_out_linear.bias.copy_(mine.to_out.bias)
        theirs.temperature.copy_(mine.temperature)


def map_attention_to_v1(mine, theirs):
    """Official v1 layout: separate in_project_x / in_project_fx, to_out is
    Sequential(Linear, Dropout)."""
    with torch.no_grad():
        theirs.in_project_x.weight.copy_(mine.in_project_x.weight)
        theirs.in_project_x.bias.copy_(mine.in_project_x.bias)
        theirs.in_project_fx.weight.copy_(mine.in_project_fx.weight)
        theirs.in_project_fx.bias.copy_(mine.in_project_fx.bias)
        theirs.in_project_slice.weight.copy_(mine.in_project_slice.weight)
        theirs.in_project_slice.bias.copy_(mine.in_project_slice.bias)
        theirs.to_q.weight.copy_(mine.to_q.weight)
        theirs.to_k.weight.copy_(mine.to_k.weight)
        theirs.to_v.weight.copy_(mine.to_v.weight)
        theirs.to_out[0].weight.copy_(mine.to_out.weight)
        theirs.to_out[0].bias.copy_(mine.to_out.bias)
        theirs.temperature.copy_(mine.temperature)


def map_block_to_t3(my_block, their_block):
    with torch.no_grad():
        their_block.ln_1.weight.copy_(my_block.ln_1.weight)
        their_block.ln_1.bias.copy_(my_block.ln_1.bias)
        their_block.ln_2.weight.copy_(my_block.ln_2.weight)
        their_block.ln_2.bias.copy_(my_block.ln_2.bias)
        their_block.mlp.linear_pre[0].weight.copy_(my_block.ffn.linear_pre.weight)
        their_block.mlp.linear_pre[0].bias.copy_(my_block.ffn.linear_pre.bias)
        their_block.mlp.linear_post.weight.copy_(my_block.ffn.linear_post.weight)
        their_block.mlp.linear_post.bias.copy_(my_block.ffn.linear_post.bias)
        if my_block.last_layer:
            their_block.ln_3.weight.copy_(my_block.ln_3.weight)
            their_block.ln_3.bias.copy_(my_block.ln_3.bias)
            their_block.mlp2.weight.copy_(my_block.head.weight)
            their_block.mlp2.bias.copy_(my_block.head.bias)
    map_attention_to_t3(my_block.attn, their_block.Attn)


def map_model_to_t3(my_model, their_model):
    with torch.no_grad():
        their_model.preprocess.linear_pre[0].weight.copy_(my_model.preprocess[0].weight)
        their_model.preprocess.linear_pre[0].bias.copy_(my_model.preprocess[0].bias)
        their_model.preprocess.linear_post.weight.copy_(my_model.preprocess[2].weight)
        their_model.preprocess.linear_post.bias.copy_(my_model.preprocess[2].bias)
        their_model.placeholder.copy_(my_model.placeholder)
    for mb, tb in zip(my_model.blocks, their_model.blocks):
        map_block_to_t3(mb, tb)


def _my_attention_module(seed=0):
    torch.manual_seed(seed)
    m = PhysicsAttentionIrregular(dim=C, heads=H, dim_head=D, slice_num=M, dropout=0.0)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'temperature' in name:
                continue
            scale = 0.3 if 'in_project_slice' in name else (
                0.2 if name.split('.')[0] in ('to_q', 'to_k', 'to_v') else 0.1)
            p.copy_(torch.randn_like(p) * scale)
    return m.double().eval()


def _my_full_model(seed=1):
    torch.manual_seed(seed)
    cfg = dict(
        model='transolver', input_var=FUN_DIM, output_var=OUT, positional_features=0,
        use_node_types=False, latent_dim=C, num_layers=L, num_heads=H, slice_num=M,
        attention_kernel='slice_space', chunk_size=CHUNK, mlp_ratio=1, dropout=0.0,
        temperature_init=0.5, temperature_min=0.1, temperature_max=5.0,
        num_timesteps=1, use_checkpointing=False, std_noise=0.0,
    )
    return Transolver(cfg).double().eval()


def verify_attention_level(t3_models):
    print("\n=== L1: attention level (their Physics_Attention_Irregular_Mesh) ===")
    mine = _my_attention_module()
    theirs = t3_models.Physics_Attention_Irregular_Mesh(
        dim=C, heads=H, dim_head=D, slice_num=M, dropout=0.0).double().eval()
    map_attention_to_t3(mine, theirs)

    torch.manual_seed(2)
    x = torch.randn(N, C, dtype=torch.float64)
    xb = x[None]  # their batch-first [1, N, C]

    # slice weights
    fused_w, fused_b = mine._fused_slice_weights()
    W_mine = mine._slice_weights(x, fused_w, fused_b)
    W_theirs = theirs.chunk_weights(xb).squeeze(0)
    check('chunk_weights == _slice_weights', rel(W_mine, W_theirs))

    # chunk_stats (num/den)
    num_m, den_m, _ = mine._chunk_stats(x, fused_w, fused_b)
    num_t, den_t = theirs.chunk_stats(xb)
    check('chunk_stats num', rel(num_m, num_t.squeeze(0)))
    check('chunk_stats den', rel(den_m, den_t.squeeze(0)))

    # full chunked attention: compose their pieces the way forward_chunks does
    tokens_t = num_t / (den_t[..., None] + 1e-5)
    out_tok_t = theirs.slice_attend(tokens_t)
    out_t = theirs.chunk_deslice_to_out(xb, out_tok_t).squeeze(0)

    ptr = torch.tensor([0, N])
    out_slice = mine(x, ptr, attention_kernel='slice_space', chunk_size=0)
    out_naive = mine(x, ptr, attention_kernel='naive')
    out_tiled = mine(x, ptr, attention_kernel='slice_space', chunk_size=CHUNK)
    check('their composed chunk path == my slice_space', rel(out_slice, out_t))
    check('their composed chunk path == my naive', rel(out_naive, out_t))
    check('their composed chunk path == my slice_space tiled', rel(out_tiled, out_t))

    # their NON-chunk forward: the normalize-first bias convention. Must be
    # measurably different (else this harness could not catch the bug class)
    # but small (~eps/norm).
    out_fwd_t = theirs.forward(xb).squeeze(0)
    check('their non-chunk forward vs chunk path (known eps-convention gap)',
          rel(out_fwd_t, out_t), bound=1e-5, lower=1e-10)


def verify_block_level(t3_models):
    print("\n=== L2: block level (their Transolver_block.forward_chunks) ===")
    my_model = _my_full_model()

    for idx, last in ((0, False), (L - 1, True)):
        their_block = t3_models.Transolver_block(
            num_heads=H, hidden_dim=C, dropout=0.0, act='gelu', mlp_ratio=1,
            last_layer=last, out_dim=OUT, slice_num=M).double().eval()
        my_block = my_model.blocks[idx]
        map_block_to_t3(my_block, their_block)

        torch.manual_seed(3 + idx)
        fx = torch.randn(N, C, dtype=torch.float64)
        chunks = [c[None] for c in torch.split(fx, CHUNK)]

        with torch.no_grad():
            out_theirs = torch.cat(
                [c.squeeze(0) for c in their_block.forward_chunks(chunks, use_checkpoint=False)], dim=0)
            out_mine = my_block(fx, torch.tensor([0, N]),
                                attention_kernel='slice_space', chunk_size=CHUNK)
        check(f'block[{idx}] (last_layer={last}) forward_chunks == my block', rel(out_mine, out_theirs))


def verify_model_level(t3_models):
    print("\n=== L3: full model (their Model, input_list chunked path) ===")
    my_model = _my_full_model()
    their_model = t3_models.Model(
        space_dim=SPACE_DIM, n_layers=L, n_hidden=C, dropout=0.0, n_head=H,
        act='gelu', mlp_ratio=1, fun_dim=FUN_DIM, out_dim=OUT, slice_num=M,
        unified_pos=False).double().eval()
    map_model_to_t3(my_model, their_model)

    torch.manual_seed(7)
    x_feat = torch.randn(N, FUN_DIM, dtype=torch.float64)
    pos = torch.randn(N, SPACE_DIM, dtype=torch.float64)
    inp = torch.cat([pos, x_feat], dim=-1)  # my wrapper order: pos first
    chunks = [c[None] for c in torch.split(inp, CHUNK)]

    from torch_geometric.data import Data
    graph = Data(x=x_feat, pos_normalized=pos)

    with torch.no_grad():
        out_theirs = torch.cat(
            [c.squeeze(0) for c in their_model(chunks, use_checkpoint=False, input_list=True)], dim=0)
        out_mine_chunked, _ = my_model(graph, add_noise=False)
        my_model.chunk_size = 0
        out_mine_single, _ = my_model(graph, add_noise=False)
        my_model.attention_kernel = 'naive'
        out_mine_naive, _ = my_model(graph, add_noise=False)
        my_model.attention_kernel = 'slice_space'
        my_model.chunk_size = CHUNK

    check('their Model == my Transolver (slice_space, tiled)', rel(out_mine_chunked, out_theirs))
    check('their Model == my Transolver (slice_space, 1 tile)', rel(out_mine_single, out_theirs))
    check('their Model == my Transolver (naive)', rel(out_mine_naive, out_theirs))

    # --- gradients: backward through both models, compare input and
    # parameter gradients across the two weight layouts ---
    inp_theirs = inp.clone().requires_grad_(True)
    chunks_g = [c[None] for c in torch.split(inp_theirs, CHUNK)]
    out_t = torch.cat(
        [c.squeeze(0) for c in their_model(chunks_g, use_checkpoint=False, input_list=True)], dim=0)
    out_t.pow(2).sum().backward()

    x_mine = x_feat.clone().requires_grad_(True)
    pos_mine = pos.clone().requires_grad_(True)
    from torch_geometric.data import Data as _Data
    graph_g = _Data(x=x_mine, pos_normalized=pos_mine)
    out_m, _ = my_model(graph_g, add_noise=False)
    out_m.pow(2).sum().backward()

    grad_mine_inp = torch.cat([pos_mine.grad, x_mine.grad], dim=-1)
    check('input gradients match', rel(grad_mine_inp, inp_theirs.grad))

    b_mine, b_theirs = my_model.blocks[0], their_model.blocks[0]
    grad_inproj_mine = torch.cat(
        [b_mine.attn.in_project_fx.weight.grad, b_mine.attn.in_project_x.weight.grad], dim=0)
    check('block0 in_project weight grads match', rel(grad_inproj_mine, b_theirs.Attn.in_project.weight.grad))
    check('block0 slice projector weight grads match',
          rel(b_mine.attn.in_project_slice.weight.grad, b_theirs.Attn.in_project_slice.weight.grad))
    check('block0 to_out weight grads match',
          rel(b_mine.attn.to_out.weight.grad, b_theirs.Attn.to_out_linear.weight.grad))
    check('block0 temperature grads match',
          rel(b_mine.attn.temperature.grad, b_theirs.Attn.temperature.grad))
    check('placeholder grads match', rel(my_model.placeholder.grad, their_model.placeholder.grad))
    check('preprocess weight grads match',
          rel(my_model.preprocess[0].weight.grad, their_model.preprocess.linear_pre[0].weight.grad))
    my_model.zero_grad(set_to_none=True)
    their_model.zero_grad(set_to_none=True)

    return my_model, their_model, graph, chunks, inp


def verify_decoupled(t3_models, t3_amortize, my_model, their_model, graph, chunks, inp):
    print("\n=== L4: decoupled two-stage inference (their amortize module) ===")
    kwargs = dict(space_dim=SPACE_DIM, n_layers=L, n_hidden=C, dropout=0.0, n_head=H,
                  act='gelu', mlp_ratio=1, fun_dim=FUN_DIM, out_dim=OUT, slice_num=M,
                  unified_pos=False)
    caching = t3_amortize.PhysicalStateCachingModel(**kwargs).double().eval()
    decoding = t3_amortize.FullMeshDecodingModel(**kwargs).double().eval()
    caching.load_state_dict(their_model.state_dict())
    decoding.load_state_dict(their_model.state_dict())

    with torch.no_grad():
        # their Stage 1, exactly as test_decoupled_inference does it:
        # layer-by-layer, re-running the prefix under the growing cache
        state_cache = []
        for layer in range(L):
            num_total = den_total = None
            for ck in chunks:
                _, num, den = caching([ck], state_cache, layer, use_checkpoint=False)
                num_total = num if num_total is None else num_total + num
                den_total = den if den_total is None else den_total + den
            state_cache.append(num_total / (den_total + 1e-5)[..., None])

        # their Stage 2 on the full mesh in one chunk
        out_theirs = decoding([inp[None]], state_cache, use_checkpoint=False)[0].squeeze(0)

        out_mine_dec, _ = my_model.forward_decoupled(graph, infer_chunk_size=CHUNK)
        out_mine_direct, _ = my_model(graph, add_noise=False)

    check('their decoupled == my forward_decoupled', rel(out_mine_dec, out_theirs))
    check('their decoupled == my direct forward', rel(out_mine_direct, out_theirs))

    # cross-check the caches themselves, layer by layer
    fx_cache, ptr_cache = my_model._embed(graph)
    for layer, block in enumerate(my_model.blocks):
        toks = block.compute_tokens(fx_cache, ptr_cache, CHUNK)[0]
        check(f'layer {layer} physics-token cache', rel(toks, state_cache[layer].squeeze(0)))
        fx_cache = block.forward_with_tokens(fx_cache, ptr_cache, [toks], CHUNK)


def verify_v1(v1_file):
    print("\n=== L0: official v1 attention (thuml/Transolver Physics_Attention.py) ===")
    v1 = _import_from_path('official_v1_physics_attention', v1_file)
    mine = _my_attention_module()
    theirs = v1.Physics_Attention_Irregular_Mesh(
        dim=C, heads=H, dim_head=D, dropout=0.0, slice_num=M).double().eval()
    map_attention_to_v1(mine, theirs)

    torch.manual_seed(11)
    x = torch.randn(N, C, dtype=torch.float64)
    with torch.no_grad():
        out_theirs = theirs(x[None]).squeeze(0)
        ptr = torch.tensor([0, N])
        out_naive = mine(x, ptr, attention_kernel='naive')
        out_slice = mine(x, ptr, attention_kernel='slice_space', chunk_size=CHUNK)
    check('official v1 forward == my naive', rel(out_naive, out_theirs))
    check('official v1 forward == my slice_space tiled', rel(out_slice, out_theirs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t3-root', required=True,
                        help='directory containing the Transolver-3 models/ package')
    parser.add_argument('--v1-file', default=None,
                        help='path to official v1 Physics_Attention.py (optional)')
    args = parser.parse_args()

    _install_einops_stub()
    sys.path.insert(0, os.path.abspath(args.t3_root))
    import models.Transolver_chunk_opt_matrix_mul as t3_models
    import models.Transolver_chunk_opt_matrix_mul_amortize as t3_amortize

    if args.v1_file:
        verify_v1(args.v1_file)
    verify_attention_level(t3_models)
    verify_block_level(t3_models)
    ctx = verify_model_level(t3_models)
    verify_decoupled(t3_models, t3_amortize, *ctx)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == '__main__':
    main()
