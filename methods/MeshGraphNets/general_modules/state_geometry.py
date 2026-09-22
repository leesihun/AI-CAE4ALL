"""Explicit state-to-geometry mapping, shared by data, statistics and rollouts.

State columns are physical fields, NOT necessarily displacement. The legacy
default is the first min(3, input_var) columns; new configs must choose explicitly.
A -1 index means zero displacement for that spatial axis.
"""
import numpy as np
import torch


def displacement_indices(config, input_dim):
    mode = str(config.get('geometry_state_mode', 'displacement')).lower()
    if mode not in ('fixed', 'displacement'):
        raise ValueError("geometry_state_mode must be 'fixed' or 'displacement'")
    if mode == 'fixed':
        return (-1, -1, -1)
    indices = config.get('displacement_state_indices')
    if indices is None:
        return tuple(i if i < input_dim else -1 for i in range(3))
    if not isinstance(indices, (list, tuple)):
        indices = [indices]
    if len(indices) != 3 or any(int(i) != i or int(i) < -1 or int(i) >= input_dim for i in indices):
        raise ValueError("displacement_state_indices needs three state-column indices (-1 pads zero)")
    return tuple(int(i) for i in indices)


def displacement_from_state(state, config=None, input_dim=None):
    width = state.shape[-1] if input_dim is None else int(input_dim)
    if width > state.shape[-1]:
        raise ValueError("input_dim exceeds the supplied state width")
    indices = displacement_indices(config or {}, width)
    zero = (torch.zeros_like(state[..., 0]) if torch.is_tensor(state)
            else np.zeros_like(state[..., 0]))
    columns = [zero if i == -1 else state[..., i] for i in indices]
    return (torch.stack(columns, dim=-1) if torch.is_tensor(state)
            else np.stack(columns, axis=-1))


def deformed_positions(reference_pos, state, config=None, input_dim=None):
    return reference_pos + displacement_from_state(state, config, input_dim)
