"""Conditional flow matching in field space.

The generative contract of this repo, in three lines:

    z0  ~ N(0, I)                        noise field, same shape as the target
    y_t = (1 - s*t) * z0 + t * y1        a point on the straight path, s = 1 - sigma_min
    u   = y1 - s * z0                    the velocity that path travels at

The network regresses `u` from `(y_t, t, graph)`. Its regression optimum is the
marginal velocity E[u | y_t, t, g], whose ODE flow transports N(0, I) exactly
onto p(y | g) (Lipman et al. 2023, Liu et al. 2023). Nothing else is needed:
no posterior encoder, no learned prior, no MMD, no scoring rule.

Two facts about cost that are easy to get wrong:

  * TRAINING never integrates. The path is closed-form, so a training step
    teleports to one random t and asks for the velocity there — one forward,
    no chain. The number of integration steps K does not appear.
  * K is chosen at SAMPLING time and can change without retraining. Model error
    accumulates as the integral of a fixed interval, not K times over; only the
    discretisation error is O(1/K) (O(1/K^2) with Heun).
"""
import math

import torch
import torch.nn as nn

# Path endpoint noise floor. sigma_min = 0 would make t=1 a Dirac endpoint.
SIGMA_MIN = 1e-4


class TimeEmbedding(nn.Module):
    """Fourier features of the flow time t: [sin(2^k pi t), cos(2^k pi t)].

    The velocity field is smooth in t but its useful frequency content is not
    known ahead of time, so a multi-octave basis is safer than a single scale.
    Buffers are non-persistent: the embedding is a fixed function, not a
    learned parameter, so it must not enter the checkpoint.
    """

    def __init__(self, num_freqs: int = 16):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer('freqs', freqs, persistent=False)
        self.dim = 2 * int(num_freqs)

    def forward(self, t):
        """t: [B, 1] in [0, 1] -> [B, 2*num_freqs]."""
        ang = t.float() * self.freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def draw_times(num_graphs, device, dtype, sampling='uniform', logit_scale=1.0):
    """Draw the flow times t in [0, 1], one per graph.

    'uniform' weights every t equally. 'logitnormal' draws logit(t) ~ N(0, s^2)
    and concentrates on the middle of the path, where the velocity field is
    hardest to predict and therefore where gradient steps buy the most: near
    t=0 the target is nearly pure noise and near t=1 it is nearly the data, so
    both ends are comparatively easy. This is the schedule Stable Diffusion 3
    adopted for rectified flow; it changes only WHERE the training budget is
    spent, never what the optimum is, so the two are directly comparable and it
    is a legitimate sweep axis rather than a change of objective.
    """
    mode = str(sampling).lower().strip()
    if mode == 'uniform':
        return torch.rand(num_graphs, 1, device=device, dtype=dtype)
    if mode == 'logitnormal':
        u = torch.randn(num_graphs, 1, device=device, dtype=dtype) * float(logit_scale)
        return torch.sigmoid(u)
    raise ValueError(f"flow_t_sampling must be 'uniform' or 'logitnormal', got '{sampling}'")


def loss_weight(t, weighting='uniform'):
    """Per-graph weight on the velocity regression term.

    The network always emits a VELOCITY -- that keeps the ODE exact and avoids
    the division by (1 - s*t) that an explicit data-prediction head needs and
    that blows up as t -> 1. Parameterization is expressed as a loss weight
    instead, which is mathematically the same objective:

        x0_hat = s*y_t + (1 - s*t) * v_hat          (exact, see predict_x0)
        ||x0_hat - y1||^2 = (1 - s*t)^2 * ||v_hat - u||^2

    because s*y_t is fixed given the input. So training the velocity head under
    the weight (1 - s*t)^2 IS x0-prediction ("data prediction" in the diffusion
    literature), with no numerical hazard.

    'uniform'  every t weighted equally -- v-prediction. Best for sampling: it
               spends budget where the ODE actually needs accuracy.
    'x0'       (1 - s*t)^2. Weight 1 at t=0 falling to sigma_min^2 at t=1, so
               the budget concentrates on the DETERMINISTIC end of the path.
               This is what makes a flow model a competitive conditional-mean
               predictor -- and it is a genuine trade: the t~1 detail that
               sharpens individual samples is exactly what it gives up.

    Note the tension with `flow_t_sampling logitnormal`, which de-emphasises
    BOTH endpoints. Combining logitnormal with x0 weighting cancels much of the
    latter; they are best treated as alternatives, not as a stack.
    """
    mode = str(weighting).lower().strip()
    if mode == 'uniform':
        return None
    if mode == 'x0':
        return (1.0 - (1.0 - SIGMA_MIN) * t) ** 2
    raise ValueError(f"flow_loss_weighting must be 'uniform' or 'x0', got '{weighting}'")


def sample_path(y1, batch, num_graphs, sampling='uniform', logit_scale=1.0,
                det_prob=0.0):
    """Draw one random (t, z0) per graph and build the training pair.

    Args:
        y1:         [N, F] target field, already normalised.
        batch:      [N] PyG graph assignment, or None for a single graph.
        num_graphs: B.
        sampling:   't' schedule -- see draw_times.
        det_prob:   probability of pinning t to exactly 0 for a given graph.
                    At t=0 the path point IS the noise field, which carries no
                    information about y, so the term collapses to

                        || (z0 + v_hat) - y ||^2   (under x0 weighting, w(0)=1)

                    i.e. a PURE DETERMINISTIC REGRESSION on E[y|g]. This is how
                    the deterministic mode gets trained rather than merely read
                    out: it is one slice of the same objective, given an
                    explicit share of the budget.

    Returns:
        t   [B, 1]  the sampled flow times (one per graph)
        y_t [N, F]  the point on the path the network is shown
        u   [N, F]  the velocity target it must regress

    t is drawn PER GRAPH, not per node: the velocity field is a function of the
    whole field state, so mixing times inside one graph would ask the network to
    denoise a state that lies on no path at all.
    """
    device, dtype = y1.device, y1.dtype
    t = draw_times(num_graphs, device, dtype, sampling, logit_scale)
    if det_prob > 0.0:
        pin = torch.rand(num_graphs, 1, device=device, dtype=dtype) < float(det_prob)
        t = torch.where(pin, torch.zeros_like(t), t)
    t_n = t[batch] if batch is not None else t.expand(y1.shape[0], 1)
    z0 = torch.randn_like(y1)
    s = 1.0 - SIGMA_MIN
    y_t = (1.0 - s * t_n) * z0 + t_n * y1
    u = y1 - s * z0
    return t, y_t, u


def predict_x0(y_t, t, v_hat):
    """The clean-field readout implied by a velocity prediction.

        y_t = (1 - s*t) z0 + t y1 ,  u = y1 - s z0
        =>  y1 = s*y_t + (1 - s*t) * u

    Exact algebra, no approximation. At t=0 it reduces to z0 + v_hat, which is
    E[y|g] in the optimum -- see predict_mean.
    """
    s = 1.0 - SIGMA_MIN
    return s * y_t + (1.0 - s * t) * v_hat


@torch.no_grad()
def integrate(velocity, y, steps: int, solver: str = 'heun'):
    """Transport `y` from t=0 to t=1 along the learned velocity field.

    Args:
        velocity: callable (y, t_scalar) -> v with the same shape as y. It must
            close over a FIXED graph and a FIXED coarsening hierarchy — see the
            warning below.
        y:      [N, F] the initial noise field, y ~ N(0, I).
        steps:  K, the number of integration steps.
        solver: 'heun' (2nd-order trapezoid, 2 evaluations/step) or 'euler'.

    WARNING — the hierarchy must be held fixed across all K steps, and across
    every sample drawn for one geometry. Rebuilding the Voronoi partition inside
    the loop makes each step a different vector field, so the ODE integrates a
    discontinuous object and the result is meaningless. This constraint does not
    exist in a one-shot model, which is why it has never bitten this codebase
    before.
    """
    if solver not in ('heun', 'euler'):
        raise ValueError(f"solver must be 'heun' or 'euler', got '{solver}'")
    dt = 1.0 / int(steps)
    for k in range(int(steps)):
        t = k * dt
        v1 = velocity(y, t)
        if solver == 'euler':
            y = y + dt * v1
            continue
        v2 = velocity(y + dt * v1, min(t + dt, 1.0))
        y = y + 0.5 * dt * (v1 + v2)
    return y


@torch.no_grad()
def predict_mean(velocity, z0):
    """The conditional mean E[y | g] in ONE forward pass.

    Not an approximation -- it falls out of the objective. At t=0 the path point
    IS the noise field, `y_0 = z0`, so conditioning on it makes z0 known and the
    regression optimum is

        v*(z0, 0, g) = E[y1 - s*z0 | z0, g] = E[y1 | g] - s*z0

    because y1 and z0 are independent. A single Euler step of size dt=1 then
    gives

        z0 + v*(z0, 0, g) = E[y1 | g] + (1 - s)*z0 = E[y1 | g] + sigma_min*z0

    i.e. the conditional mean to within sigma_min (1e-4) times a unit-variance
    field. So the same checkpoint that samples the distribution also serves a
    deterministic prediction at exactly the cost of a deterministic model --
    one network forward, no integration.

    Two caveats worth stating:
      * this leans entirely on the network being accurate AT t=0, whereas the
        training budget was spread over all t. A deterministic model spends
        everything on this one query. Whether that costs accuracy is empirical.
      * the ensemble mean of M integrated draws estimates the same quantity and
        may beat this when the t=0 slice is under-trained, at M*K times the cost.
        `misc/eval_prediction_modes.py` measures both.

    Args:
        velocity: callable (y, t_scalar) -> v, closing over a fixed graph.
        z0:       [N, F] noise field. Its choice is immaterial in exact
                  arithmetic; the residual sigma_min*z0 is the only trace it
                  leaves.
    """
    return z0 + velocity(z0, 0.0)


def resolve_flow_config(config):
    """Read and validate the flow-matching keys from a parsed config."""
    steps = int(config.get('flow_steps', 30))
    if steps < 1:
        raise ValueError(f"flow_steps must be >= 1, got {steps}")
    solver = str(config.get('flow_solver', 'heun')).lower().strip()
    if solver not in ('heun', 'euler'):
        raise ValueError(f"flow_solver must be 'heun' or 'euler', got '{solver}'")
    freqs = int(config.get('flow_time_freqs', 16))
    if freqs < 1:
        raise ValueError(f"flow_time_freqs must be >= 1, got {freqs}")
    t_sampling = str(config.get('flow_t_sampling', 'uniform')).lower().strip()
    if t_sampling not in ('uniform', 'logitnormal'):
        raise ValueError(
            f"flow_t_sampling must be 'uniform' or 'logitnormal', got '{t_sampling}'")
    logit_scale = float(config.get('flow_t_logit_scale', 1.0))
    if logit_scale <= 0:
        raise ValueError(f"flow_t_logit_scale must be > 0, got {logit_scale}")
    weighting = str(config.get('flow_loss_weighting', 'uniform')).lower().strip()
    if weighting not in ('uniform', 'x0'):
        raise ValueError(
            f"flow_loss_weighting must be 'uniform' or 'x0', got '{weighting}'")
    det_prob = float(config.get('flow_det_prob', 0.0))
    if not 0.0 <= det_prob < 1.0:
        raise ValueError(f"flow_det_prob must be in [0, 1), got {det_prob}")
    predict = str(config.get('flow_predict', 'sample')).lower().strip()
    if predict not in ('sample', 'mean', 'ensemble_mean'):
        raise ValueError(
            f"flow_predict must be 'sample', 'mean' or 'ensemble_mean', got '{predict}'")
    return {'steps': steps, 'solver': solver, 'time_freqs': freqs,
            't_sampling': t_sampling, 'logit_scale': logit_scale,
            'weighting': weighting, 'det_prob': det_prob, 'predict': predict}
