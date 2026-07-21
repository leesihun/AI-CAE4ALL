"""Autoregressive rollout training (AR-RT) for variational MeshGraphNets.

Implements the scheme from NVIDIA/GM's crash-dynamics study
(arXiv:2510.15201) as its reference implementation does
(`physicsnemo/examples/structural_mechanics/crash/rollout.py`):

  * unroll the full trajectory, `num_time_steps - 1` steps;
  * feed each prediction straight into the next step with no detach, so
    gradients flow through the entire unroll;
  * gradient-checkpoint every step while training, which is what makes that
    affordable;
  * inject no noise and use no rollout-length curriculum.

Two things differ from the reference, both deliberate:

1.  Integration order. The reference predicts acceleration and integrates
    twice (`vel = dt*acc + vel; y = dt*vel + y`), which only makes sense for
    kinematic channels. This repository predicts a first-order state delta,
    which also covers non-kinematic outputs such as stress. AR-RT is
    orthogonal to that choice, so the first-order integrator is kept.

2.  Loss space. Each step's target is the correction back onto the ground
    truth from wherever the rollout currently is, `s_gt[k+1] - s_pred[k]`,
    expressed in the same normalized-delta space AR-OT uses. Since
    `s_pred[k+1] = s_pred[k] + denorm(pred)`, the residual is exactly
    `(s_pred[k+1] - s_gt[k+1]) / delta_std`, i.e. a per-channel-scaled state
    loss over the trajectory — the same quantity the reference minimizes, on
    a scale that keeps AR-OT learning rates transferable and makes a
    single-step trajectory numerically identical to AR-OT.

Geometry is rebuilt from the predicted state at every step, exactly as
inference does: mesh edge features, world (contact) edges, and multiscale
coarse-level edge features are all recomputed on-device, so a contact that
forms mid-rollout is seen during training.

Variational specifics
---------------------

Every unrolled step is a full forward through the model, so **the latent is
resampled at each step**: the rollout explores the posterior along the
trajectory instead of committing to one draw for all of it.

The **loss composition is unchanged** from the one-step path. Each step
produces the same weighted sum of reconstruction, MMD, aux and prior-density
terms that AR-OT produces, and the rollout averages those composed losses over
the trajectory. This module therefore never reads the VAE weights itself; the
training loop passes in a `compose_loss(graph)` callback and stays the single
place where the objective is defined.

The posterior encoder conditions on `graph.y`, so the rollout writes each
step's target there before the forward — the encoder sees the correction the
model is actually being asked to make from its own current state, not the
ground-truth transition it would have seen under teacher forcing.

Per-step gradient checkpointing preserves RNG state across recomputation, so
the latent drawn during backward is the one drawn during forward.
"""

import torch
from torch.utils.checkpoint import checkpoint

from general_modules.edge_features import deformed_edge_attr_torch
from general_modules.time_integration import AR_RT, resolve_time_integration
from general_modules.world_edges import compute_world_edges_torch


def ar_rt_enabled(config) -> bool:
    return resolve_time_integration(config) == AR_RT


def describe_ar_rt(window: int) -> str:
    return (f"Time integration: AR-RT ({window}-step rollout, BPTT through the "
            f"full unroll, per-step gradient checkpointing)")


class RolloutContext:
    """Device-resident constants for an unroll: normalization stats and switches.

    Built once per epoch and reused across batches — every tensor here is a
    constant of the run, so rebuilding them per batch would just add H2D
    copies to the hot loop.
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
        # node-type one-hot are geometry-static and reused from the dataloader).
        self.node_mean = to_tensor(stats['node_mean'])[:self.input_var]
        self.node_std = to_tensor(stats['node_std'])[:self.input_var]
        # Edge features are [deformed_dx, dy, dz, dist | ref_dx, dy, dz, dist];
        # only the deformed half moves during a rollout.
        self.edge_mean_def = to_tensor(stats['edge_mean'])[:4]
        self.edge_std_def = to_tensor(stats['edge_std'])[:4]
        self.edge_mean = to_tensor(stats['edge_mean'])
        self.edge_std = to_tensor(stats['edge_std'])
        self.delta_mean = to_tensor(stats['delta_mean'])
        self.delta_std = to_tensor(stats['delta_std'])

        self.coarse_edge_means = [to_tensor(m)[:4] for m in stats.get('coarse_edge_means', [])]
        self.coarse_edge_stds = [to_tensor(s)[:4] for s in stats.get('coarse_edge_stds', [])]

        self.use_world_edges = bool(config.get('use_world_edges', False))
        self.world_edge_radius = stats.get('world_edge_radius', None)
        self.world_max_num_neighbors = int(config.get('world_max_num_neighbors', 64))

        self.use_multiscale = bool(config.get('use_multiscale', False))
        self.multiscale_levels = int(config.get('multiscale_levels', 1))
        if self.use_multiscale and not self.coarse_edge_means:
            # Silently skipping this normalization would feed the coarse
            # processor raw-scale edge features for the whole rollout.
            raise RuntimeError(
                "AR-RT with use_multiscale needs per-level coarse edge stats in "
                "config['_norm_stats']['coarse_edge_means'/'coarse_edge_stds']."
            )
        if self.use_multiscale and bool(config.get('coarse_world_edges', False)):
            raise ValueError(
                "time_integration ar_rt does not support coarse_world_edges True: "
                "lifted contact edges would have to be re-derived per level per step. "
                "Set coarse_world_edges False, or train this config with ar_ot."
            )


def _coarse_positions(fine_pos, graph, level, num_coarse_total):
    """Positions of one coarse level, derived from the level below.

    Mirrors `multiscale_helpers.attach_coarse_levels_to_graph`: seed-anchored
    levels take their anchor's position (exported as `coarse_anchor_idx_{l}`),
    every other level takes the arithmetic centroid of its cluster.
    """
    anchors = graph.get(f'coarse_anchor_idx_{level}', None)
    if anchors is not None:
        return fine_pos[anchors]

    fine_to_coarse = graph[f'fine_to_coarse_{level}']
    summed = fine_pos.new_zeros((num_coarse_total, fine_pos.shape[1]))
    summed.index_add_(0, fine_to_coarse, fine_pos)
    counts = fine_pos.new_zeros((num_coarse_total, 1))
    counts.index_add_(0, fine_to_coarse, torch.ones_like(fine_pos[:, :1]))
    return summed / counts.clamp(min=1.0)


def _refresh_multiscale(graph, deformed_pos, ctx):
    """Recompute every level's deformed edge features from `deformed_pos`.

    Cluster topology, reference anchors and the reference half of each level's
    edge features are trajectory-invariant, so only the deformed half is
    rebuilt — the same split the fine mesh uses.
    """
    current_pos = deformed_pos
    for level in range(ctx.multiscale_levels):
        num_coarse = int(graph[f'num_coarse_{level}'].sum())
        coarse_pos = _coarse_positions(current_pos, graph, level, num_coarse)

        coarse_edge_index = graph[f'coarse_edge_index_{level}']
        previous_attr = graph[f'coarse_edge_attr_{level}']
        if coarse_edge_index.shape[1] > 0:
            deformed_half = deformed_edge_attr_torch(coarse_pos, coarse_edge_index)
            if level < len(ctx.coarse_edge_means):
                deformed_half = ((deformed_half - ctx.coarse_edge_means[level])
                                 / ctx.coarse_edge_stds[level])
            graph[f'coarse_edge_attr_{level}'] = torch.cat(
                [deformed_half, previous_attr[:, 4:]], dim=1
            )
        current_pos = coarse_pos


def _apply_state(graph, state, ctx, static_node_features, reference_edge_attr):
    """Write the features implied by `state` onto `graph`, on-device.

    This is the training-time twin of the per-step feature construction in
    `inference_profiles/rollout.py`; keeping the two in step is what makes
    AR-RT train the model under the conditions it is actually deployed in.
    """
    physical = state[:, :ctx.input_var]
    normalized = (physical - ctx.node_mean) / ctx.node_std
    graph.x = torch.cat([normalized, static_node_features], dim=1)

    deformed_pos = graph.pos + state[:, :3]

    deformed_half = deformed_edge_attr_torch(deformed_pos, graph.edge_index)
    deformed_half = (deformed_half - ctx.edge_mean_def) / ctx.edge_std_def
    graph.edge_attr = torch.cat([deformed_half, reference_edge_attr], dim=1)

    if ctx.use_world_edges and ctx.world_edge_radius is not None:
        world_edge_index, world_edge_attr = compute_world_edges_torch(
            graph.pos, deformed_pos, graph.edge_index,
            radius=float(ctx.world_edge_radius),
            max_num_neighbors=ctx.world_max_num_neighbors,
            batch=getattr(graph, 'batch', None),
            ptr=getattr(graph, 'ptr', None),
            edge_mean=ctx.edge_mean, edge_std=ctx.edge_std,
        )
        graph.world_edge_index = world_edge_index
        graph.world_edge_attr = world_edge_attr

    if ctx.use_multiscale:
        _refresh_multiscale(graph, deformed_pos, ctx)

    return graph


def rollout_loss(model, graph, ctx, compose_loss, training=True):
    """Unroll the model over the trajectory and average the composed losses.

    `compose_loss(graph) -> (prediction, loss, recon_sum, *extra_terms)` is
    supplied by the training loop; it applies the full VAE objective to one
    forward pass. Every element it returns must be a tensor so the step can be
    gradient-checkpointed.

    Returns `(loss, prediction, recon_sum, recon_count, extra_terms)`, with
    `loss` and every extra term averaged over the trajectory and the
    reconstruction statistics summed, exactly as the one-step path reports them
    for a single step.
    """
    if getattr(graph, 'y_seq', None) is None:
        raise RuntimeError(
            "AR-RT expects graph.y_seq from the dataset; the dataloader was "
            "built with time_integration ar_ot. Rebuild it with ar_rt."
        )

    steps = int(graph.y_seq.shape[1])
    state = graph.state0
    static_node_features = graph.x[:, ctx.input_var:]
    reference_edge_attr = graph.edge_attr[:, 4:]
    output_var = ctx.output_var

    totals = None
    total_recon_sum = None
    total_recon_count = 0
    prediction = None

    for step in range(steps):
        target_states = graph.y_seq[:, step, :]

        def run_step(current_state):
            _apply_state(graph, current_state, ctx, static_node_features, reference_edge_attr)
            # Target: the correction from where the rollout actually is back
            # onto the ground truth, in AR-OT's normalized-delta space. The
            # posterior encoder reads it off the graph, so it must be attached
            # before the forward rather than compared after it.
            target_delta = target_states - current_state[:, :output_var]
            graph.y = (target_delta - ctx.delta_mean) / ctx.delta_std
            return compose_loss(graph)

        # Checkpoint while training (as the reference does): only the per-step
        # state tensors stay live, and each step's activations -- and its
        # latent draw, since RNG state is preserved -- are recomputed during
        # backward.
        if training:
            outputs = checkpoint(run_step, state, use_reentrant=False)
        else:
            outputs = run_step(state)

        prediction, step_loss, recon_sum = outputs[0], outputs[1], outputs[2]
        extras = outputs[3:]

        if totals is None:
            totals = [step_loss] + list(extras)
        else:
            totals = [running + current for running, current
                      in zip(totals, [step_loss] + list(extras))]
        total_recon_sum = recon_sum if total_recon_sum is None else total_recon_sum + recon_sum
        total_recon_count += prediction.shape[0]

        if step < steps - 1:
            advanced = state[:, :output_var] + (prediction * ctx.delta_std + ctx.delta_mean)
            if ctx.input_var > output_var:
                # Channels the model does not predict are carried unchanged,
                # matching inference.
                state = torch.cat([advanced, state[:, output_var:]], dim=1)
            else:
                state = advanced

    # Mean over steps: one AR-RT step then costs exactly what one AR-OT step
    # costs, so learning rates carry over between the two schemes.
    averaged = [total / steps for total in totals]
    return averaged[0], prediction, total_recon_sum, total_recon_count, averaged[1:]
