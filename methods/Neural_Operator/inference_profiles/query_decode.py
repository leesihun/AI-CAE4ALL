"""Exact chunked query decoding (IMPLEMENTATION_PLAN.md section 14.3). Given
an already-encoded operator representation, decodes output nodes in original
order in chunks of `chunk_size` -- a memory control that must reproduce the
same values as a direct, unchunked decode (tests/test_query_chunking.py).
"""

import torch


def decode_in_chunks(model, encoded, graph, chunk_size: int) -> torch.Tensor:
    """Decode all N query nodes of `graph` via `model.decode_queries`,
    optionally split into chunks of `chunk_size` nodes (0 or >= N: no
    chunking). Concatenates results in original node order.
    """
    n = graph.x.shape[0]
    if chunk_size <= 0 or chunk_size >= n:
        return model.decode_queries(encoded, graph, 0, n)

    outputs = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        outputs.append(model.decode_queries(encoded, graph, start, end))
    return torch.cat(outputs, dim=0)
