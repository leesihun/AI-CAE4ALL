"""Public model wrapper: noise contract + batch synthesis + dispatch
(IMPLEMENTATION_PLAN.md sections 4.6, 6.2, Appendix A.1).

This is the only class training/inference code touches. It never branches on
`model_name` -- all architecture-specific behavior lives in the wrapped core.
"""

import torch
import torch.nn as nn


def ensure_batch_ptr(graph):
    """Synthesize `batch`/`ptr` for a bare (unbatched) PyG Data object.

    `torch_geometric.loader.DataLoader` always produces a `Batch` with these
    attributes, even at batch_size=1. Only the rollout path (which builds a
    raw `Data` object by hand, section 14.2) needs this synthesis.
    """
    n = graph.x.shape[0]
    if getattr(graph, 'batch', None) is None:
        graph.batch = torch.zeros(n, dtype=torch.long, device=graph.x.device)
    if getattr(graph, 'ptr', None) is None:
        graph.ptr = torch.tensor([0, n], dtype=torch.long, device=graph.x.device)
    return graph


class OperatorWrapper(nn.Module):
    model_name: str

    def __init__(self, core, config):
        super().__init__()
        self.core = core
        self.model_name = core.model_name
        self.output_var = int(config['output_var'])
        self.std_noise = float(config.get('std_noise', 0.0))
        self.noise_gamma = float(config.get('noise_gamma', 1))
        noise_std_ratio = config.get('noise_std_ratio', None)
        self._noise_std_ratio = (
            torch.tensor(noise_std_ratio, dtype=torch.float32)
            if noise_std_ratio is not None else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.core.set_epoch(epoch)

    def supports_query_chunking(self) -> bool:
        return self.core.supports_query_chunking()

    def _apply_noise(self, graph):
        """Exact port of MeshGraphNets' noise contract (section 4.6), minus
        edge noise (this repository has no edge_attr)."""
        noise = torch.randn(
            graph.x.shape[0], self.output_var,
            device=graph.x.device, dtype=graph.x.dtype,
        ) * self.std_noise
        noise_padded = torch.zeros_like(graph.x)
        noise_padded[:, :self.output_var] = noise
        graph.x = graph.x + noise_padded

        target = getattr(graph, 'y', None)
        if self._noise_std_ratio is not None and target is not None:
            ratio = self._noise_std_ratio.to(device=graph.x.device, dtype=graph.x.dtype)
            graph.y = target - self.noise_gamma * noise * ratio
        return graph

    def forward(self, graph, add_noise=None):
        if add_noise is None:
            add_noise = self.training
        graph = ensure_batch_ptr(graph)
        if add_noise and self.std_noise > 0:
            graph = self._apply_noise(graph)

        prediction = self.core(graph)
        return prediction, getattr(graph, 'y', None)

    def encode_operator(self, graph):
        graph = ensure_batch_ptr(graph)
        return self.core.encode_operator(graph)

    def decode_queries(self, encoded, graph, start: int, end: int):
        return self.core.decode_queries(encoded, graph, start, end)

    def export_model_config(self) -> dict:
        return self.core.export_model_config()
