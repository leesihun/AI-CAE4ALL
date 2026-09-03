"""Node subsampling for amortized (two-stream) training.

Amortized training is the training-time use of the same two-stage split that
`infer_mode decoupled` uses at inference: build each layer's physics tokens
from one node stream (the CACHE stream), then decode a second, much smaller
node stream (the QUERY stream) against those tokens and put the loss there.

Why that is worth doing
-----------------------
Both streams cost O(L * nodes * latent_dim) in retained activations. The
ordinary forward has exactly one stream and it is the full mesh, so training
memory is pinned to N. Splitting lets the two roles carry different node
budgets, and they want very different ones:

  * the CACHE stream only has to estimate `slice_num` global aggregates, so it
    tolerates a coarse subsample of the mesh;
  * the QUERY stream only has to carry enough nodes for a low-variance loss
    estimate, which is far fewer than N on a 200k-node mesh.

Cost becomes O(L * (cache_nodes + query_nodes) * latent_dim). On ex2's ~200k
node graphs, cache 32768 + query 8192 is a ~4.9x reduction.

Why the subsampled tokens are still the right tokens
----------------------------------------------------
A layer's token is `num / (den + EPS)` where `num` and `den` are both plain
SUMS over nodes (see `PhysicsAttentionIrregular._chunk_stats`). Under uniform
sampling of k of N nodes, both sums scale by the same k/N in expectation, and
the ratio cancels it -- so the subsampled token is a consistent estimator of
the whole-mesh token with no rescaling needed. (Only the fixed EPS in the
denominator does not scale, and it is 1e-5 against a denominator that sums to
k.) This is a ratio estimator: consistent, with sampling noise that acts as
stochastic regularization during training, exactly like neighbor sampling in
GraphSAGE. It is NOT exact, which is why it applies to training only --
`_amortized_active()` requires `self.training`, so validation, test and
inference always run the full mesh.

Every parameter still gets a gradient: the cache stream is run WITH autograd
(not under no_grad), so `in_project_fx`, which appears only in the aggregate
pass, trains normally. A no_grad cache would silently freeze it.
"""

import torch


def sample_stream_index(ptr: torch.Tensor, budget: int):
    """Per-graph uniform node subsample without replacement.

    Args:
        ptr: [B + 1] packed-graph boundaries, as carried by PyG batches.
        budget: max nodes to keep per graph. `<= 0` means "keep everything".

    Returns:
        (index, stream_ptr). `index` is a [sum_k] gather index into the packed
        node rows, sorted within each graph so the contiguous tiles the
        slice_space kernel cuts stay spatially coherent; `stream_ptr` is the
        matching [B + 1] boundary vector.

        `index` is None when nothing was actually dropped (budget <= 0, or
        every graph is smaller than the budget). Callers use that to skip the
        gather entirely and reuse the packed tensors as-is, so a budget larger
        than the mesh degenerates to the ordinary forward rather than paying
        for a full-size copy.
    """
    if budget <= 0:
        return None, ptr

    bounds = ptr.tolist()
    device = ptr.device
    pieces = []
    sizes = []
    subsampled = False

    for i in range(len(bounds) - 1):
        start, end = int(bounds[i]), int(bounds[i + 1])
        n = end - start
        if budget >= n:
            pieces.append(torch.arange(start, end, device=device))
            sizes.append(n)
        else:
            keep = torch.randperm(n, device=device)[:budget]
            pieces.append(torch.sort(keep).values + start)
            sizes.append(budget)
            subsampled = True

    if not subsampled:
        return None, ptr

    index = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
    stream_ptr = torch.zeros(len(sizes) + 1, dtype=torch.long, device=device)
    stream_ptr[1:] = torch.tensor(sizes, dtype=torch.long, device=device).cumsum(0)
    return index, stream_ptr


def describe_amortized(cache_nodes: int, query_nodes: int) -> str:
    def budget(n):
        return 'full mesh' if n <= 0 else f'{n:,} nodes/graph'
    return (f"Amortized training: ENABLED (token cache stream = {budget(cache_nodes)}, "
            f"decoded query stream = {budget(query_nodes)}; loss is computed on the "
            f"query stream only, eval/inference always run the full mesh)")
