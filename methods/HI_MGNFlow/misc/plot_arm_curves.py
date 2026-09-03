"""Plot the ex9 arm curves on one axis so the convergence question is visible.

The point of the figure is NOT to compare loss VALUES across arms -- the
deterministic tree regresses the field while the flow arms regress a velocity,
so their training losses live on different scales and are not comparable.

What IS comparable, and what the figure is for:

  * left panel  -- each arm's own validation curve, normalised to its own first
    value. This shows the SHAPE: has the curve flattened, or is it still
    falling? That decides whether the budget was long enough.
  * right panel -- the one quantity that is genuinely on a common scale: the
    field-space MSE of a single-forward prediction. HI-MGN's validation loss and
    the flow arms' det(1fwd) readout both measure "how far is the predicted
    field from the truth", so they can share an axis.

    python misc/plot_arm_curves.py --logs det=path fm_x0=path fm_v=path \\
        --out curves.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/")
DET_VALID_RE = re.compile(r"Valid(?:ation)?[:=]\s*([0-9.eE+-]+)")
FLOW_VALID_RE = re.compile(r"Valid\s+fm=([0-9.eE+-]+)")
DIAG_RE = re.compile(
    r"crps=([0-9.eE+-]+)\s+det\(1fwd\) mse=([0-9.eE+-]+)\s+"
    r"1-draw mse=([0-9.eE+-]+)\s+spread/gt=([0-9.eE+-]+)")


def as_float(m, group=1):
    """`m` is None on every line missing the field, and None.group raises
    AttributeError -- which a (TypeError, ValueError) guard sails past."""
    if m is None:
        return None
    try:
        return float(m.group(group))
    except (TypeError, ValueError):
        return None


def parse(path):
    valid, diag = [], []
    if not os.path.exists(path):
        return valid, diag
    epoch = 0
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'Epoch' in line:
                e = as_float(EPOCH_RE.search(line))
                if e is not None:
                    epoch = int(e)
                    v = as_float(FLOW_VALID_RE.search(line))
                    if v is None:
                        v = as_float(DET_VALID_RE.search(line))
                    if v is not None:
                        valid.append((epoch, v))
            m = DIAG_RE.search(line)
            if m:
                diag.append((epoch, float(m.group(1)), float(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    return valid, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', required=True, help='name=path ...')
    ap.add_argument('--out', default='arm_curves.png')
    ap.add_argument('--baseline', default='det',
                    help='arm whose final value is drawn as a reference line')
    args = ap.parse_args()

    arms = {}
    for spec in args.logs:
        name, path = spec.split('=', 1)
        arms[name] = parse(path)

    colors = {'det': '#A2650D', 'fm_x0': '#0B7B85', 'fm_v': '#7A5AA8'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: own-scale normalised, shows SHAPE ─────────────────────────────
    for name, (valid, _diag) in arms.items():
        if len(valid) < 2:
            continue
        xs = [e for e, _ in valid]
        ys = [v / valid[0][1] for _, v in valid]
        ax1.plot(xs, ys, 'o-', label=name, color=colors.get(name), lw=2, ms=5)
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('validation loss / its own first value')
    ax1.set_title('Convergence SHAPE (each arm normalised to itself)\n'
                  'flat = converged; still falling = budget was too short',
                  fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ── right: field-space MSE, genuinely comparable ────────────────────────
    base_final = None
    for name, (valid, diag) in arms.items():
        if name == args.baseline and valid:
            xs = [e for e, _ in valid]
            ys = [v for _, v in valid]
            base_final = ys[-1]
            ax2.plot(xs, ys, 'o-', label=f'{name} (HI-MGN valid MSE)',
                     color=colors.get(name), lw=2, ms=5)
        elif diag:
            xs = [d[0] for d in diag]
            ys = [d[2] for d in diag]          # det(1fwd) mse
            ax2.plot(xs, ys, 'o-', label=f'{name} det(1fwd) MSE',
                     color=colors.get(name), lw=2, ms=5)
    if base_final is not None:
        ax2.axhline(base_final, ls='--', color=colors.get(args.baseline),
                    alpha=0.6, lw=1.2)
        ax2.annotate(f'HI-MGN final {base_final:.2e}',
                     xy=(0.98, base_final), xycoords=('axes fraction', 'data'),
                     ha='right', va='bottom', fontsize=9,
                     color=colors.get(args.baseline))
    ax2.set_yscale('log')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('field MSE (1 forward)')
    ax2.set_title('Single-forward field MSE -- the one common scale\n'
                  'HI-MGN validation vs the flow arms\' det(1fwd) readout',
                  fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle('ex9 (Geo-FNO plasticity) -- 900 samples x 19 steps x 3,131 nodes, '
                 'identical backbone and budget', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f'saved {os.path.abspath(args.out)}')

    # ── the numbers the figure is arguing from ──────────────────────────────
    print('\nlast three validation points per arm (is it flat?):')
    for name, (valid, diag) in arms.items():
        tail = valid[-3:]
        if tail:
            s = '  '.join(f'ep{e}={v:.3e}' for e, v in tail)
            drop = ((tail[0][1] - tail[-1][1]) / tail[0][1] * 100) if len(tail) > 1 else 0
            print(f'  {name:<7} {s}   -> {drop:+.1f}% over that span')
        if diag:
            d = diag[-1]
            print(f'          spread/gt={d[4]:.3f}  crps={d[1]:.3e}  '
                  f'1-draw={d[3]:.3e}')


if __name__ == '__main__':
    main()
