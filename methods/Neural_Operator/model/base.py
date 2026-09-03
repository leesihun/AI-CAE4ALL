"""Shared operator-model protocol (IMPLEMENTATION_PLAN.md section 6.2).

Every architecture's *core* module (the clean forward pass, no noise) follows
this interface. `model/operator_wrapper.py` wraps a core with the noise
contract and batch/ptr synthesis so training/inference code never branches on
which architecture is selected.
"""

import torch.nn as nn


class OperatorCore(nn.Module):
    """Base class for the four architecture cores (DeepONet, PointDeepONet,
    MeshFNO, MeshGINO). Subclasses must set `model_name` and implement
    `forward(graph) -> prediction [sum_N, output_var]` plus `export_model_config`.
    """

    model_name: str = "operator_core"

    def forward(self, graph):
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Deterministic sampling context (Point-DeepONet only); no-op otherwise."""
        return None

    def supports_query_chunking(self) -> bool:
        return False

    def encode_operator(self, graph):
        raise NotImplementedError(
            f"{self.model_name} does not implement encode_operator/decode_queries."
        )

    def decode_queries(self, encoded, graph, start: int, end: int):
        raise NotImplementedError(
            f"{self.model_name} does not implement encode_operator/decode_queries."
        )

    def export_model_config(self) -> dict:
        raise NotImplementedError
