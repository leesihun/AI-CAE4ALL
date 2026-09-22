"""Actual v3 execution paths, beyond config-name checks."""
import torch
from torch_geometric.data import Data
from model.Transolver import Transolver


def model_and_graph(**overrides):
    torch.manual_seed(42)
    cfg = dict(input_var=3, output_var=3, cond_var=2, positional_features=0,
               latent_dim=32, num_layers=2, num_heads=4, slice_num=8, mlp_ratio=2,
               attention_kernel='slice_space', chunk_size=7, use_checkpointing=True,
               dropout=0.0, std_noise=0.0)
    cfg.update(overrides)
    graph = Data(x=torch.randn(23, 5), pos_normalized=torch.randn(23, 3), y=torch.randn(23, 3))
    return Transolver(cfg), graph


def test_tiled_and_decoupled_inference_agree():
    model, graph = model_and_graph()
    model.eval()
    with torch.no_grad():
        full, _ = model(graph)
        decoupled, _ = model.forward_decoupled(graph, infer_chunk_size=5)
    torch.testing.assert_close(full, decoupled, rtol=2e-5, atol=2e-6)


def test_amortized_training_has_aligned_targets_and_gradients():
    model, graph = model_and_graph(amortized_training=True, amortized_cache_nodes=11,
                                  amortized_query_nodes=9)
    model.train()
    prediction, target = model(graph)
    assert prediction.shape == target.shape == (9, 3)
    ((prediction-target)**2).mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    model.eval()
    with torch.no_grad():
        assert model(graph)[0].shape == (23, 3)
