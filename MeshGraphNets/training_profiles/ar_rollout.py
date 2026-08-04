"""Autoregressive rollout training (AR-RT) for MeshGraphNets.

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

2.  Loss space. The reference computes `MSELoss` on **raw absolute positions**
    (`node_target = pos_seq[1:]`), so its magnitude is set by the physical
    position spread and does not depend on trajectory length. This repository
    cannot use raw units — it predicts four heterogeneous channels
    (displacement ~1e1, stress ~1e-7 on ex2), and a raw MSE would ignore
    stress entirely. So the state error is normalized, and the scale is the
    **state** spread (`node_std`), which is the multi-channel analogue of the
    reference's implicit position scale.

    It is deliberately *not* the one-step `delta_std`. Dividing an
    error that accumulates over `k` steps by a one-step spread makes the loss
    grow linearly with the step index: measured on ex2, `|target|` went from
    0.34 at step 0 to 27.5 (peak 129) at step 48, an initial MSE of 260
    against AR-OT's 0.31, and initial gradient norms ~700x AR-OT's — which
    `clip_grad_norm_(3.0)` then flattened into pure direction. `node_std /
    delta_std` is 20-31x per channel on ex2, i.e. exactly the trajectory-length
    factor. That earlier scaling also made a single-step trajectory numerically
    identical to AR-OT; that property is gone, and it was the reason the bug
    was invisible at K=1. A single step now equals AR-OT's loss rescaled per
    channel by `(delta_std / node_std)**2` — see tests/test_ar_rollout.py.

Geometry is rebuilt from the predicted state at every step, exactly as
inference does: mesh edge features, world (contact) edges, and multiscale
coarse-level edge features are all recomputed on-device, so a contact that
forms mid-rollout is seen during training.
"""

import torch
from torch.utils.checkpoint import checkpoint

from general_modules.edge_features import (
    DEFORMED_SLICE,
    REFERENCE_SLICE,
    deformed_edge_attr_torch,
)
from general_modules.time_integration import AR_RT, resolve_time_integration
from general_modules.world_edges import world_edge_attr_torch, world_edge_index_torch


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
        # Only the deformed half of an edge feature moves during a rollout
        # (see general_modules/edge_features.py for the layout).
        self.edge_mean_def = to_tensor(stats['edge_mean'])[DEFORMED_SLICE]
        self.edge_std_def = to_tensor(stats['edge_std'])[DEFORMED_SLICE]
        self.edge_mean = to_tensor(stats['edge_mean'])
        self.edge_std = to_tensor(stats['edge_std'])
        self.delta_mean = to_tensor(stats['delta_mean'])
        self.delta_std = to_tensor(stats['delta_std'])
        # Scale for the rollout loss: the STATE spread over the trajectory, not
        # the one-step delta spread. See the module docstring -- using
        # delta_std here makes the loss grow linearly with the step index.
        self.state_std = to_tensor(stats['node_std'])[:self.output_var]

        # A multi-partition level (ATTENTION_TRANSFER_DESIGN.md Part II) stores
        # a LIST of per-branch stat arrays where every other level stores one
        # array. DEFORMED_SLICE selects channels, so it must be applied to each
        # array individually -- applied to the list it would slice the branch
        # axis instead and silently produce a wrongly-shaped constant.
        def to_stat(value):
            if isinstance(value, list):
                return [to_tensor(v)[DEFORMED_SLICE] for v in value]
            return to_tensor(value)[DEFORMED_SLICE]

        self.coarse_edge_means = [to_stat(m) for m in stats.get('coarse_edge_means', [])]
        self.coarse_edge_stds = [to_stat(s) for s in stats.get('coarse_edge_stds', [])]

        self.use_world_edges = bool(config.get('use_world_edges', False))
        self.world_edge_radius = stats.get('world_edge_radius', None)
        self.world_max_num_neighbors = int(config.get('world_max_num_neighbors', 64))
        # The rollout used to hardcode torch_cluster whenever it was importable
        # on CUDA, silently ignoring this key (which the dataloader path does
        # honor). On a large mesh with a small contact radius that is the slower
        # backend -- see world_edge_index_torch.
        self.world_edge_backend = str(
            config.get('world_edge_backend', 'scipy_kdtree')
        ).strip().lower()

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
    def coarse_edge_stat(self, level, branch=None):
        """Deformed-half normalization stats for one coarse partition.

        Returns (None, None) past the end of the stats list, which is the
        signal to skip normalization -- the same condition the single-partition
        code expressed inline as `if level < len(ctx.coarse_edge_means)`.
        """
        if level >= len(self.coarse_edge_means):
            return None, None
        mean = self.coarse_edge_means[level]
        std = self.coarse_edge_stds[level]
        if isinstance(mean, list):          # multi-partition level
            idx = 0 if branch is None else branch
            return mean[idx], std[idx]
        return mean, std


def _coarse_positions(fine_pos, graph, key, num_coarse_total):
    """Positions of one coarse partition, derived from the level below.

    Mirrors `multiscale_helpers.attach_coarse_levels_to_graph`: seed-anchored
    levels take their anchor's position (exported as `coarse_anchor_idx_{key}`),
    every other level takes the arithmetic centroid of its cluster.

    `key` is `str(level)` for a normal level, or `f'{level}_{branch}'` for one
    branch of a multi-partition level -- the same keying the dataset writes.
    """
    anchors = graph.get(f'coarse_anchor_idx_{key}', None)
    if anchors is not None:
        return fine_pos[anchors]

    fine_to_coarse = graph[f'fine_to_coarse_{key}']
    summed = fine_pos.new_zeros((num_coarse_total, fine_pos.shape[1]))
    summed.index_add_(0, fine_to_coarse, fine_pos)
    counts = fine_pos.new_zeros((num_coarse_total, 1))
    counts.index_add_(0, fine_to_coarse, torch.ones_like(fine_pos[:, :1]))
    return summed / counts.clamp(min=1.0)


def _refresh_one_partition(graph, key, current_pos, mean, std):
    """Rebuild one coarse partition's deformed edge features in place.

    Returns that partition's coarse positions so a single-partition caller can
    chain them into the next level.
    """
    num_coarse = int(graph[f'num_coarse_{key}'].sum())
    coarse_pos = _coarse_positions(current_pos, graph, key, num_coarse)

    coarse_edge_index = graph[f'coarse_edge_index_{key}']
    previous_attr = graph[f'coarse_edge_attr_{key}']
    if coarse_edge_index.shape[1] > 0:
        deformed_half = deformed_edge_attr_torch(coarse_pos, coarse_edge_index)
        if mean is not None:
            deformed_half = (deformed_half - mean) / std
        # .detach() on the reference half: it is trajectory-invariant, so its
        # gradient contribution is already zero (the chain terminates at the
        # dataloader's non-grad tensor). Without the detach every step's `cat`
        # node stays reachable from the next step's, so a 49-step unroll keeps
        # 49 of them alive for nothing.
        graph[f'coarse_edge_attr_{key}'] = torch.cat(
            [deformed_half, previous_attr[:, REFERENCE_SLICE].detach()], dim=1
        )
    return coarse_pos


def _refresh_multiscale(graph, deformed_pos, ctx):
    """Recompute every level's deformed edge features from `deformed_pos`.

    Cluster topology, reference anchors and the reference half of each level's
    edge features are trajectory-invariant, so only the deformed half is
    rebuilt — the same split the fine mesh uses.

    A multi-partition level (ATTENTION_TRANSFER_DESIGN.md Part II) writes no
    unsuffixed `*_{level}` attributes at all, only `*_{level}_{branch}`, so it
    is detected by probing for branch 0. Every branch partitions the same
    upstream node set, so they all refresh from the same `current_pos`; such a
    level is always terminal, so there is nothing to chain afterwards.
    """
    current_pos = deformed_pos
    for level in range(ctx.multiscale_levels):
        if graph.get(f'num_coarse_{level}_0', None) is not None:
            branch = 0
            while graph.get(f'num_coarse_{level}_{branch}', None) is not None:
                mean, std = ctx.coarse_edge_stat(level, branch)
                _refresh_one_partition(graph, f'{level}_{branch}', current_pos, mean, std)
                branch += 1
            return
        if graph.get(f'num_coarse_{level}', None) is None:
            # Coarsening saturated before multiscale_levels (n_c <= 1 or no
            # coarse edges), so the dataset stopped emitting levels here.
            return
        mean, std = ctx.coarse_edge_stat(level)
        current_pos = _refresh_one_partition(graph, str(level), current_pos, mean, std)


def _world_edge_search(graph, state, ctx):
    """Contact connectivity for `state` — the part that must NOT be checkpointed.

    The radius search is `no_grad` and returns indices, so re-running it during
    backward yields the identical set at full cost. On ex2 it is the single
    most expensive operation in a rollout step (measured: ~200 ms per call at
    200k nodes), so keeping it outside the checkpoint halves the rollout's
    dominant cost. Returns None when world edges are off.
    """
    if not (ctx.use_world_edges and ctx.world_edge_radius is not None):
        return None
    with torch.no_grad():
        deformed_pos = graph.pos + state[:, :3]
        return world_edge_index_torch(
            deformed_pos, graph.edge_index,
            radius=float(ctx.world_edge_radius),
            max_num_neighbors=ctx.world_max_num_neighbors,
            batch=getattr(graph, 'batch', None),
            ptr=getattr(graph, 'ptr', None),
            backend=ctx.world_edge_backend,
        )


def _apply_state(graph, state, ctx, static_node_features, reference_edge_attr,
                 world_edge_index=None):
    """Write the features implied by `state` onto `graph`, on-device.

    This is the training-time twin of the per-step feature construction in
    `inference_profiles/rollout.py`; keeping the two in step is what makes
    AR-RT train the model under the conditions it is actually deployed in.

    `world_edge_index` is an optional contact set from `_world_edge_search`,
    which the rollout computes outside its gradient checkpoint so backward does
    not redo the search. Omit it and the search happens here instead: the
    optimization is opt-in, so a caller that does not know about it still gets
    correct contact edges rather than silently stale ones. Either way only the
    edge *features* are differentiable in `deformed_pos`, which is what lets a
    contact that forms mid-rollout influence the loss.
    """
    physical = state[:, :ctx.input_var]
    normalized = (physical - ctx.node_mean) / ctx.node_std
    graph.x = torch.cat([normalized, static_node_features], dim=1)

    deformed_pos = graph.pos + state[:, :3]

    deformed_half = deformed_edge_attr_torch(deformed_pos, graph.edge_index)
    deformed_half = (deformed_half - ctx.edge_mean_def) / ctx.edge_std_def
    graph.edge_attr = torch.cat([deformed_half, reference_edge_attr], dim=1)

    if ctx.use_world_edges and ctx.world_edge_radius is not None:
        if world_edge_index is None:
            world_edge_index = _world_edge_search(graph, state, ctx)
        graph.world_edge_index = world_edge_index
        graph.world_edge_attr = world_edge_attr_torch(
            graph.pos, deformed_pos, world_edge_index,
            edge_mean=ctx.edge_mean, edge_std=ctx.edge_std,
        )

    if ctx.use_multiscale:
        _refresh_multiscale(graph, deformed_pos, ctx)

    return graph


def rollout_loss(model, graph, ctx, loss_fn, training=True):
    """Unroll the model over the trajectory and average the per-step losses.

    `loss_fn(prediction, target) -> (loss, loss_sum, loss_count)` is supplied
    by the training loop so feature weighting and the sync-free accumulation
    contract stay in one place.

    Returns the same `(loss, loss_sum, loss_count)` triple as the one-step
    path, so callers need no branch beyond choosing this function.
    """
    if getattr(graph, 'y_seq', None) is None:
        raise RuntimeError(
            "AR-RT expects graph.y_seq from the dataset; the dataloader was "
            "built with time_integration ar_ot. Rebuild it with ar_rt."
        )

    steps = int(graph.y_seq.shape[1])
    state = graph.state0
    # Everything past the state block is constant over the unroll: the
    # input-only conditioning rows (cond_var), the positional features and the
    # node-type one-hot. The dataloader orders x as [state | cond | pos |
    # onehot] precisely so this one slice captures all of them.
    static_node_features = graph.x[:, ctx.input_var:]
    reference_edge_attr = graph.edge_attr[:, REFERENCE_SLICE]
    output_var = ctx.output_var

    total_loss = None
    total_sum = None
    total_count = 0

    def run_step(current_state, world_ei):
        _apply_state(graph, current_state, ctx, static_node_features,
                     reference_edge_attr, world_edge_index=world_ei)
        prediction, _ = model(graph, add_noise=False)
        return prediction

    for step in range(steps):
        # The contact search runs OUTSIDE the checkpoint: it is no_grad and
        # returns indices, so recomputing it in backward would cost the same
        # again for an identical result. On ex2 that search dominates the step.
        world_ei = _world_edge_search(graph, state, ctx)

        # Checkpoint while training (as the reference does): only the per-step
        # state tensors stay live, and each step's activations are recomputed
        # during backward.
        if training:
            prediction = checkpoint(run_step, state, world_ei, use_reentrant=False)
        else:
            prediction = run_step(state, world_ei)

        # Integrate the prediction into the next state, in physical units.
        advanced = state[:, :output_var] + (prediction * ctx.delta_std + ctx.delta_mean)

        # Loss on the STATE, normalized by the state spread -- the reference
        # compares raw positions; this is that comparison made scale-free
        # across heterogeneous channels. Normalizing by delta_std instead
        # would make the loss grow with the step index (module docstring).
        loss, loss_sum, loss_count = loss_fn(
            advanced / ctx.state_std, graph.y_seq[:, step, :] / ctx.state_std,
        )
        total_loss = loss if total_loss is None else total_loss + loss
        total_sum = loss_sum if total_sum is None else total_sum + loss_sum
        total_count += loss_count

        if step < steps - 1:
            if ctx.input_var > output_var:
                # Channels the model does not predict are carried unchanged,
                # matching inference (`current_state[:, :output_dim]` there).
                state = torch.cat([advanced, state[:, output_var:]], dim=1)
            else:
                state = advanced

    # Mean over steps: one AR-RT step then costs exactly what one AR-OT step
    # costs, so learning rates carry over between the two schemes.
    return total_loss / steps, total_sum, total_count
