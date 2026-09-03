"""Time-integration scheme selection shared by the dataset and the training loop.

Two schemes, following the naming in NVIDIA/GM's crash-dynamics study
(arXiv:2510.15201, "Automotive Crash Dynamics Modeling Accelerated with
Machine Learning"):

    ar_ot  Autoregressive with One-step Training. The model is trained on
           ground-truth consecutive pairs (teacher forcing) and only ever
           consumes its own predictions at inference time. This is what this
           repository has always done, and it stays the default so existing
           configs reproduce bit-for-bit.

    ar_rt  Autoregressive with Rollout Training. The model is unrolled over the
           whole trajectory during training, consuming its own predictions
           exactly as it will at inference, so it learns to correct its own
           accumulated error instead of relying on injected noise as a proxy.

`time_integration` is the only knob. Everything else about AR-RT follows the
reference implementation (`physicsnemo/examples/structural_mechanics/crash/
rollout.py`): unroll `num_time_steps - 1` steps, backpropagate through all of
them, gradient-checkpoint each step, inject no noise.
"""

AR_OT = 'ar_ot'
AR_RT = 'ar_rt'
_VALID_SCHEMES = (AR_OT, AR_RT)


def resolve_time_integration(config) -> str:
    """Normalize the `time_integration` config value to `ar_ot` / `ar_rt`.

    Accepts the paper's hyphenated spelling and any capitalization, so
    `AR-RT`, `ar_rt` and `ARRT` all select rollout training.
    """
    raw = str(config.get('time_integration', AR_OT)).strip().lower()
    scheme = raw.replace('-', '_').replace(' ', '_')
    if scheme == 'arot':
        scheme = AR_OT
    elif scheme == 'arrt':
        scheme = AR_RT
    if scheme not in _VALID_SCHEMES:
        raise ValueError(
            f"time_integration must be one of {list(_VALID_SCHEMES)} "
            f"(AR-OT / AR-RT also accepted), got '{config.get('time_integration')}'"
        )
    return scheme


def resolve_rollout_window(config, num_timesteps: int) -> int:
    """Return the number of steps each training item spans.

    AR-OT spans one step (a ground-truth pair). AR-RT spans the full
    trajectory, `num_timesteps - 1`, as the reference implementation does.
    """
    if resolve_time_integration(config) == AR_OT:
        return 1

    if num_timesteps <= 1:
        raise ValueError(
            "time_integration ar_rt requires a temporal dataset (num_timesteps > 1); "
            f"this dataset has num_timesteps={num_timesteps}. Use ar_ot for static data."
        )
    return num_timesteps - 1
