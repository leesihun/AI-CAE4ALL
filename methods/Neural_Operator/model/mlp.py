import torch.nn.init as init
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def build_mlp(in_size, hidden_size, out_size, layer_norm=True):
    """Two-hidden-layer SiLU MLP following the MeshGraphNets architecture.

    LayerNorm is appended on the output when layer_norm=True (all non-decoder uses).
    """
    layers = [
        nn.Linear(in_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, out_size),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))
    return nn.Sequential(*layers)


_ACTIVATIONS = {
    'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'tanh': nn.Tanh,
}


def build_deep_mlp(in_size, hidden_size, out_size, depth, activation='silu',
                    layer_norm=False):
    """Configurable-depth MLP used by DeepONet/Point-DeepONet branch/trunk/
    refiner heads (IMPLEMENTATION_PLAN.md section 11.2's *_depth/*_activation
    keys). `depth` hidden Linear+activation blocks followed by one final
    linear projection (no activation on the output, matching DeepONet's
    linear modal heads, section 8.2).
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'; expected one of {list(_ACTIVATIONS)}.")
    act_cls = _ACTIVATIONS[activation]
    layers = []
    prev = in_size
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden_size), act_cls()]
        prev = hidden_size
    layers.append(nn.Linear(prev, out_size))
    if layer_norm:
        layers.append(nn.LayerNorm(out_size))
    return nn.Sequential(*layers)
