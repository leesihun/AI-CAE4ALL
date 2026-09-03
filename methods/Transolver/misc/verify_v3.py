#!/usr/bin/env python3
"""Verify the Transolver-3 feature set this checkout ships: the
aggregate-then-project kernel, attention tiling, and amortized training.

Unlike misc/verify_against_official.py this needs no upstream clone -- it
checks the repo against itself, in fp64 on CPU, so it can run anywhere:

  L1  kernel equivalence   naive == slice_space untiled == slice_space tiled,
                           at whole-model level. This is what makes flipping
                           the default kernel a memory/speed change and not a
                           numerical one.
  L2  amortized identity   forward_amortized with node budgets above the mesh
                           size reproduces the ordinary forward exactly.
  L3  gradient coverage    with real (smaller) budgets, EVERY parameter still
                           receives a finite gradient. This is the guard on
                           amortization's one real failure mode: run the token
                           cache stream under no_grad and in_project_fx --
                           which appears nowhere else -- silently stops
                           training.
  L4  token consistency    tokens estimated from a k-node subsample converge
                           to the whole-mesh tokens as k grows, which is the
                           claim amortized training rests on.
  L5  peak memory (CUDA)   tiling + per-tile recompute actually lowers peak
                           memory, and tiling without recompute does not.

Usage:
    python misc/verify_v3.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Transolver import Transolver  # noqa: E402
from model.amortized import sample_stream_index  # noqa: E402
from model.physics_attention import PhysicsAttentionIrregular, make_tile_ranges  # noqa: E402

N, C, H, M, L, IN, OUT = 1301, 32, 4, 8, 3, 5, 3
CHUNK = 128  # -> tiles of 128 with a ragged 21-node tail

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


class Graph:
    """Minimal stand-in for the PyG Data object the model consumes."""

    def __init__(self, n, num_graphs=1):
        total = n * num_graphs
        self.x = torch.randn(total, IN, dtype=torch.float64)
        self.y = torch.randn(total, OUT, dtype=torch.float64)
        self.pos_normalized = torch.randn(total, 3, dtype=torch.float64)
        self.ptr = torch.arange(num_graphs + 1, dtype=torch.long) * n


def build_model(seed=0, **overrides):
    torch.manual_seed(seed)
    cfg = dict(
        model='transolver', input_var=IN, output_var=OUT, positional_features=0,
        use_node_types=False, latent_dim=C, num_layers=L, num_heads=H, slice_num=M,
        attention_kernel='slice_space', chunk_size=CHUNK, mlp_ratio=2, dropout=0.0,
        temperature_init=0.5, temperature_min=0.1, temperature_max=5.0,
        num_timesteps=1, use_checkpointing=False, std_noise=0.0,
    )
    cfg.update(overrides)
    return Transolver(cfg).double()


# ---------------------------------------------------------------------------


def verify_kernels():
    print("\n=== L1: kernel equivalence (naive / untiled / tiled) ===")
    model = build_model().eval()
    graph = Graph(N)

    with torch.no_grad():
        model.attention_kernel, model.chunk_size = 'naive', 0
        out_naive, _ = model(graph, add_noise=False)
        model.attention_kernel, model.chunk_size = 'slice_space', 0
        out_flat, _ = model(graph, add_noise=False)
        model.chunk_size = CHUNK
        out_tiled, _ = model(graph, add_noise=False)

    # slice_space applies the bias convention proven exact against v1 (plan
    # 6.3), so all three agree to fp64 round-off, not merely to eps/norm.
    check('slice_space untiled == naive', rel(out_flat, out_naive) < 1e-12,
          f'rel err {rel(out_flat, out_naive):.2e}')
    check('slice_space tiled == untiled', rel(out_tiled, out_flat) < 1e-12,
          f'rel err {rel(out_tiled, out_flat):.2e}')


def verify_amortized_identity():
    print("\n=== L2: amortized forward with above-mesh budgets == ordinary forward ===")
    graph = Graph(N, num_graphs=2)

    plain = build_model().eval()
    with torch.no_grad():
        out_plain, y_plain = plain(graph, add_noise=False)

    # Budgets larger than every graph: sample_stream_index reports "nothing
    # dropped" and both streams are the full mesh.
    amort = build_model(amortized_training=True,
                        amortized_cache_nodes=N * 10,
                        amortized_query_nodes=N * 10).train()
    with torch.no_grad():
        out_amort, y_amort = amort.forward_amortized(graph, add_noise=False)

    check('prediction matches', rel(out_amort, out_plain) < 1e-12,
          f'rel err {rel(out_amort, out_plain):.2e}')
    check('target passed through unsliced', torch.equal(y_amort, y_plain))
    check('eval() disables amortization', not amort.eval()._amortized_active())
    check('train() re-enables it', amort.train()._amortized_active())

    # infer_mode decoupled runs the same two-stage split with cache == query.
    # It shares compute_layer_tokens / decode_with_tokens with amortization, so
    # this pins both callers to the ordinary forward at once.
    with torch.no_grad():
        out_dec, _ = plain.forward_decoupled(graph, infer_chunk_size=CHUNK)
    check('forward_decoupled matches the ordinary forward',
          rel(out_dec, out_plain) < 1e-12, f'rel err {rel(out_dec, out_plain):.2e}')


def verify_gradient_coverage():
    print("\n=== L3: every parameter still trains under real node budgets ===")
    graph = Graph(N, num_graphs=2)
    cache_budget, query_budget = 400, 90

    model = build_model(amortized_training=True,
                        amortized_cache_nodes=cache_budget,
                        amortized_query_nodes=query_budget).train()
    torch.manual_seed(7)
    predicted, target = model(graph, add_noise=False)

    expected = query_budget * 2
    check('prediction is restricted to the query stream',
          predicted.shape == (expected, OUT), f'got {tuple(predicted.shape)}')
    check('target is sliced to the same rows',
          target.shape == (expected, OUT), f'got {tuple(target.shape)}')

    torch.nn.functional.mse_loss(predicted, target).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    check('no parameter is left without a gradient', not missing,
          f'{len(missing)} missing: {missing[:4]}')
    check('every gradient is finite', not nonfinite, f'{len(nonfinite)} non-finite')

    # in_project_fx exists only inside the aggregate pass, so it is the one
    # parameter a no_grad token cache would silently freeze. Assert it moved.
    fx_grads = [p.grad.abs().sum().item() for n, p in model.named_parameters()
                if 'in_project_fx' in n]
    check('in_project_fx receives gradient from the cache stream',
          bool(fx_grads) and all(g > 0 for g in fx_grads), f'{len(fx_grads)} tensors')


def verify_token_consistency():
    print("\n=== L4: subsampled tokens converge to whole-mesh tokens ===")
    torch.manual_seed(3)
    attn = PhysicsAttentionIrregular(dim=C, heads=H, dim_head=C // H, slice_num=M).double().eval()
    # Structured, non-i.i.d. features: a uniform subsample has to survive real
    # spatial variation, not just noise that averages out trivially.
    coords = torch.linspace(-1, 1, 20000, dtype=torch.float64)[:, None]
    x = torch.cat([torch.sin(3 * coords * (i + 1)) for i in range(C)], dim=1)
    ptr = torch.tensor([0, x.shape[0]])

    with torch.no_grad():
        full = attn.compute_layer_tokens(x, make_tile_ranges(x.shape[0], 0))
        errors = []
        for k in (500, 2000, 8000):
            torch.manual_seed(11)
            index, _ = sample_stream_index(ptr, k)
            sub = attn.compute_layer_tokens(x[index], make_tile_ranges(k, 0))
            errors.append(rel(sub, full))
            print(f"       k={k:>5}: rel err vs whole mesh = {errors[-1]:.3e}")

    # The claim is not an absolute accuracy bound -- that depends entirely on
    # how rough the field is, and this one is deliberately rough (C distinct
    # sine frequencies over the domain). The claim is that this is a proper
    # Monte Carlo estimator, i.e. error ~ 1/sqrt(k). Each step here raises k
    # by 4x, so a consistent estimator must halve the error each time; a
    # BIASED one would flatten out at some nonzero floor instead.
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    print(f"       error ratio per 4x more nodes: "
          f"{', '.join(f'{r:.2f}' for r in ratios)}  (1/sqrt(k) predicts 2.00)")
    check('converges at the 1/sqrt(k) Monte Carlo rate',
          all(1.7 < r < 2.4 for r in ratios), f'ratios {[round(r, 2) for r in ratios]}')


def verify_peak_memory():
    print("\n=== L5: what each memory knob actually does (one attention layer) ===")
    if not torch.cuda.is_available():
        print("  [SKIP] no CUDA device")
        return

    device = torch.device('cuda')
    torch.manual_seed(0)
    big_n, big_c, big_m = 60000, 128, 128
    attn = PhysicsAttentionIrregular(
        dim=big_c, heads=8, dim_head=big_c // 8, slice_num=big_m).to(device).train()
    ptr = torch.tensor([0, big_n], device=device)

    def measure(kernel, chunk, recompute):
        """(retained, peak) in MB. `retained` is what is still held after the
        forward -- i.e. the activations backward will need, which is the term
        that accumulates once per layer. `peak` adds the largest transient."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        x = torch.randn(big_n, big_c, device=device, requires_grad=True)
        out = attn(x, ptr, attention_kernel=kernel, chunk_size=chunk,
                   use_checkpointing=recompute)
        torch.cuda.synchronize()
        retained = (torch.cuda.memory_allocated(device) - base) / 1e6
        out.pow(2).sum().backward()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated(device) - base) / 1e6
        del x, out
        attn.zero_grad(set_to_none=True)
        return retained, peak

    naive = measure('naive', 0, False)
    flat = measure('slice_space', 0, False)
    tiled = measure('slice_space', 8192, False)
    tiled_ck = measure('slice_space', 8192, True)
    print(f"       {'':<38}{'retained':>10}{'peak':>10}")
    for label, (r, p) in [('naive', naive), ('slice_space untiled', flat),
                          ('slice_space tiled 8192', tiled),
                          ('slice_space tiled + recompute', tiled_ck)]:
        print(f"       {label:<38}{r:>9.1f}M{p:>9.1f}M")

    # THE property that makes tiling a memory technique: a tiled forward
    # streams, dropping each tile's [H, tile, M] once it is folded into the
    # aggregates and rebuilding it in backward. So the N-scaled attention term
    # leaves retained memory entirely. If per-tile recompute ever gets
    # re-coupled to use_checkpointing (it was, and that made tiling a no-op at
    # use_checkpointing False), this is the check that fails.
    check('tiling streams: retained attention memory collapses',
          tiled[0] < 0.15 * flat[0], f'{flat[0]:.0f}M -> {tiled[0]:.0f}M retained')
    check('tiling lowers peak vs untiled', tiled[1] < flat[1],
          f'{100 * (1 - tiled[1] / flat[1]):.0f}% lower peak')
    check('tiling beats naive, no block checkpointing needed', tiled[1] < naive[1],
          f'{100 * (1 - tiled[1] / naive[1]):.0f}% lower peak')
    # Block-level checkpointing is orthogonal and composes on top.
    check('block checkpointing composes with tiling', tiled_ck[1] <= tiled[1] * 1.05,
          f'{tiled[1]:.0f}M -> {tiled_ck[1]:.0f}M peak')
    # Regression guard on the UNTILED path reusing pass 1's assignment matrix
    # in pass 2. Recomputing it instead allocates a second [H, N, M] -- the
    # largest tensor in the layer -- pushing this ratio from ~1.6 to ~2.4.
    check('untiled pass 2 reuses pass 1 slice weights',
          flat[0] < 2.0 * naive[0], f'{flat[0] / naive[0]:.2f}x naive retained')


def main():
    torch.manual_seed(0)
    verify_kernels()
    verify_amortized_identity()
    verify_gradient_coverage()
    verify_token_consistency()
    verify_peak_memory()
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == '__main__':
    main()
