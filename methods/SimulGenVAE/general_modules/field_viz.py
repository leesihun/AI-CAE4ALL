"""Periodic training visualization for SimulGenVAE.

Two pictures, one per stage:

* ``plot_field_reconstruction`` (VAE) -- truth / reconstruction / |error| as
  [channel, time] images on a shared colour scale, plus a few channel traces
  over time. A hierarchical VAE that has collapsed to the dataset mean produces
  a reconstruction panel that is flat where the truth is not, which is exactly
  what the side-by-side makes visible.
* ``plot_latent_parity`` (latent conditioner) -- predicted against true latent
  code, main and hierarchical stacks separately. The conditioner is a
  regression onto a frozen target, so parity is the honest view of it.

Both denormalize before plotting, and both import matplotlib lazily so a
missing install degrades to a printed note rather than killing training.
"""

import os

import numpy as np

_MISSING_BACKEND_WARNED = False


def _load_backend():
    global _MISSING_BACKEND_WARNED
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        if not _MISSING_BACKEND_WARNED:
            _MISSING_BACKEND_WARNED = True
            print(f'  [viz] matplotlib unavailable ({exc}); skipping plots.')
        return None


def _denormalize(field_ct, normalization):
    """[C, T] scaled -> [C, T] physical. invert_minmax works channel-last."""
    if normalization is None:
        return np.asarray(field_ct, dtype=np.float64)
    from general_modules.fom_dataset import invert_minmax
    return invert_minmax(np.asarray(field_ct).T, normalization).T.astype(np.float64)


def _rel_l2(truth, pred):
    denominator = float(np.linalg.norm(truth))
    if denominator < 1e-30:
        return float('nan')
    return float(np.linalg.norm(pred - truth)) / denominator


def plot_field_reconstruction(truths, reconstructions, path, *, epoch=None,
                              sample_labels=None, normalization=None,
                              num_traces=3, dpi=140):
    """Write one row per sample: truth, reconstruction, |error|, channel traces.

    truths / reconstructions: sequences of [C, T] arrays in scaled units.
    """
    plt = _load_backend()
    if plt is None:
        return None
    if not len(truths):
        return None

    rows = len(truths)
    fig, axes = plt.subplots(rows, 4, figsize=(19.0, 3.6 * rows), dpi=dpi,
                             squeeze=False, facecolor='white')

    for row, (truth_s, recon_s) in enumerate(zip(truths, reconstructions)):
        truth = _denormalize(truth_s, normalization)
        recon = _denormalize(recon_s, normalization)
        error = np.abs(recon - truth)

        # Truth and reconstruction share one scale; comparing two images drawn
        # on independent scales is the classic way to make a bad fit look good.
        low = float(min(truth.min(), recon.min()))
        high = float(max(truth.max(), recon.max()))
        if high - low < 1e-30:
            high = low + 1e-30

        label = sample_labels[row] if sample_labels is not None and row < len(sample_labels) else f'sample {row}'
        panels = (
            (truth, f'{label}\ntruth', low, high, 'viridis'),
            (recon, f'reconstruction\nrel L2 = {_rel_l2(truth, recon):.3e}', low, high, 'viridis'),
            (error, f'|error|\nmax = {error.max():.3e}', 0.0, float(max(error.max(), 1e-30)), 'magma'),
        )
        for col, (image, title, vmin, vmax, cmap) in enumerate(panels):
            ax = axes[row][col]
            handle = ax.imshow(image, aspect='auto', origin='lower', cmap=cmap,
                               vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('time index')
            ax.set_ylabel('channel')
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)

        # Traces: evenly spaced channels so the picture is not dominated by one
        # region of the (num_var x num_nodes) flattening.
        ax = axes[row][3]
        channels = np.linspace(0, truth.shape[0] - 1, min(num_traces, truth.shape[0]), dtype=int)
        time_axis = np.arange(truth.shape[1])
        for channel in channels:
            line, = ax.plot(time_axis, truth[channel], lw=1.4, label=f'ch {channel} truth')
            ax.plot(time_axis, recon[channel], lw=1.2, ls='--', color=line.get_color(),
                    label=f'ch {channel} recon')
        ax.set_title('channel traces (solid = truth)', fontsize=10)
        ax.set_xlabel('time index')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)

    header = 'SimulGenVAE reconstruction'
    if epoch is not None:
        header += f' -- epoch {epoch}'
    fig.suptitle(header, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_latent_parity(true_main, pred_main, true_hier, pred_hier, path, *,
                       epoch=None, dpi=140, max_points=20000, seed=0):
    """Predicted vs true latent code for the conditioner, main and hierarchical.

    Arrays are [N, Dm] and [N, L, Dh]; every coordinate is one point.
    """
    plt = _load_backend()
    if plt is None:
        return None

    rng = np.random.default_rng(seed)

    def flatten(array):
        flat = np.asarray(array, dtype=np.float64).ravel()
        if max_points > 0 and flat.size > max_points:
            return flat[rng.choice(flat.size, max_points, replace=False)]
        return flat

    def r2(truth, pred):
        denominator = float(((truth - truth.mean()) ** 2).sum())
        if denominator < 1e-30:
            return float('nan')
        return 1.0 - float(((truth - pred) ** 2).sum()) / denominator

    # One shared subsample per stack so truth and prediction stay paired.
    pairs = []
    for name, truth, pred in (('main latent', true_main, pred_main),
                              ('hierarchical latents', true_hier, pred_hier)):
        truth_flat = np.asarray(truth, dtype=np.float64).ravel()
        pred_flat = np.asarray(pred, dtype=np.float64).ravel()
        if truth_flat.size == 0 or truth_flat.size != pred_flat.size:
            continue
        if max_points > 0 and truth_flat.size > max_points:
            picked = rng.choice(truth_flat.size, max_points, replace=False)
            truth_flat, pred_flat = truth_flat[picked], pred_flat[picked]
        pairs.append((name, truth_flat, pred_flat))

    if not pairs:
        return None

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 5.0),
                             dpi=dpi, squeeze=False, facecolor='white')
    for col, (name, truth_flat, pred_flat) in enumerate(pairs):
        ax = axes[0][col]
        ax.scatter(truth_flat, pred_flat, s=4, alpha=0.3, linewidths=0, color='#3B82C4')
        low = float(min(truth_flat.min(), pred_flat.min()))
        high = float(max(truth_flat.max(), pred_flat.max()))
        pad = 0.04 * (high - low) if high > low else 1.0
        ax.plot([low - pad, high + pad], [low - pad, high + pad],
                color='#111111', lw=1.0, ls='--', alpha=0.7)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect('equal', adjustable='box')
        rmse = float(np.sqrt(((pred_flat - truth_flat) ** 2).mean()))
        ax.set_title(f'{name}\nR2={r2(truth_flat, pred_flat):.4f}  RMSE={rmse:.4g}', fontsize=11)
        ax.set_xlabel('true (scaled)')
        ax.set_ylabel('predicted (scaled)')
        ax.grid(True, alpha=0.25)

    header = 'SimulGenVAE latent conditioner'
    if epoch is not None:
        header += f' -- epoch {epoch}'
    fig.suptitle(header, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path
