"""Autoregressive rollout training (AR-RT) for Transolver.

Implements the scheme from NVIDIA/GM's crash-dynamics study
(arXiv:2510.15201) as its reference implementation does
(`physicsnemo/examples/structural_mechanics/crash/rollout.py`), which trains
exactly this scheme on Transolver for a Body-in-White crash:

  * unroll the full trajectory, `num_time_steps - 1` steps;
  * feed each prediction straight into the next step with no detach, so
    gradients flow through the entire unroll;
  * gradient-checkpoint every step while training, which is what makes that
    affordable;
  * inject no noise and use no rollout-length curriculum.

Two deliberate differences from the reference:

1.  Integration order. The reference predicts acceleration and integrates
    twice; this repository predicts a first-order state delta, which also
    covers non-kinematic outputs such as stress. AR-RT is orthogonal to that
    choice, so the first-order integrator is kept.

2.  Loss space. Each step's target is the correction back onto the ground
    truth from wherever the rollout currently is, `s_gt[k+1] - s_pred[k]`,
    expressed in AR-OT's normalized-delta space. Because
    `s_pred[k+1] = s_pred[k] + denorm(pred)`, the residual is exactly
    `(s_pred[k+1] - s_gt[k+1]) / delta_std` — a per-channel-scaled state loss
    over the trajectory, on a scale that keeps AR-OT learning rates
    transferable.

Unlike MeshGraphNets, no geometry has to be rebuilt between steps: Transolver
consumes reference positions (`pos_normalized`) and node features only, and
reference geometry is trajectory-invariant. Only the physical channels of
`graph.x` are re-normalized per step.
"""

import torch
from torch.utils.checkpoint import checkpoint

from general_modules.time_integration import AR_RT, resolve_time_integration


def ar_rt_enabled(config) -> bool:
    return resolve_time_integration(config) == AR_RT


def describe_ar_rt(window: int) -> str:
    return (f"Time integration: AR-RT ({window}-step rollout, BPTT through the "
            f"full unroll, per-step gradient checkpointing)")


class RolloutContext:
    """Device-resident constants for an unroll: normalization stats.

    Built once per epoch — every tensor here is a constant of the run, so
    rebuilding them per batch would only add H2D copies to the hot loop.
    """

    def __init__(self, config, device, dtype=torch.float32):
        stats = config.get('_norm_stats', None)
        if stats is None:
            raise RuntimeError(
                "AR-RT needs dataset normalization stats in config['_norm_stats']; "
                "they are injected by training_profiles.setup.build_dataset_splits."
            )

        def to_tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)

        self.input_var = int(config['input_var'])
        self.output_var = int(config['output_var'])
        # node_mean/std cover [physical | positional] features; only the
        # physical head is re-normalized per step (positional features and the
        # node-type one-hot are geometry-static).
        self.node_mean = to_tensor(stats['node_mean'])[:self.input_var]
        self.node_std = to_tensor(stats['node_std'])[:self.input_var]
        self.delta_mean = to_tensor(stats['delta_mean'])
        self.delta_std = to_tensor(stats['delta_std'])


def _apply_state(graph, state, ctx, static_node_features):
    """Write the node features implied by `state` onto `graph`, on-device.

    This is the training-time twin of the per-step input construction in
    `inference_profiles/rollout.py`; keeping the two in step is what makes
    AR-RT train the model under the conditions it is deployed in.
    """
    normalized = (state[:, :ctx.input_var] - ctx.node_mean) / ctx.node_std
    graph.x = torch.cat([normalized, static_node_features], dim=1)
    return graph


def rollout_loss(model, graph, ctx, loss_fn, training=True):
    """Unroll the model over the trajectory and average the per-step losses.

    `loss_fn(prediction, target) -> (loss, loss_sum, loss_count)` is supplied
    by the training loop so feature weighting and the sync-free accumulation
    contract stay in one place. Returns the same triple as the one-step path.
    """
    if getattr(graph, 'y_seq', None) is None:
        raise RuntimeError(
            "AR-RT expects graph.y_seq from the dataset; the dataloader was "
            "built with time_integration ar_ot. Rebuild it with ar_rt."
        )

    steps = int(graph.y_seq.shape[1])
    state = graph.state0
    static_node_features = graph.x[:, ctx.input_var:]
    output_var = ctx.output_var

    total_loss = None
    total_sum = None
    total_count = 0

    def run_step(current_state):
        _apply_state(graph, current_state, ctx, static_node_features)
        prediction, _ = model(graph, add_noise=False)
        return prediction

    for step in range(steps):
        # Checkpoint while training (as the reference does): only the per-step
        # state tensors stay live, and each step's activations are recomputed
        # during backward.
        if training:
            prediction = checkpoint(run_step, state, use_reentrant=False)
        else:
            prediction = run_step(state)

        # Target: the correction from where the rollout actually is back onto
        # the ground truth, in AR-OT's normalized-delta space.
        target_delta = graph.y_seq[:, step, :] - state[:, :output_var]
        target = (target_delta - ctx.delta_mean) / ctx.delta_std

        loss, loss_sum, loss_count = loss_fn(prediction, target)
        total_loss = loss if total_loss is None else total_loss + loss
        total_sum = loss_sum if total_sum is None else total_sum + loss_sum
        total_count += loss_count

        if step < steps - 1:
            advanced = state[:, :output_var] + (prediction * ctx.delta_std + ctx.delta_mean)
            if ctx.input_var > output_var:
                # Channels the model does not predict are carried unchanged,
                # matching inference.
                state = torch.cat([advanced, state[:, output_var:]], dim=1)
            else:
                state = advanced

    # Mean over steps: one AR-RT step costs exactly what one AR-OT step costs,
    # so learning rates carry over between the two schemes.
    return total_loss / steps, total_sum, total_count
