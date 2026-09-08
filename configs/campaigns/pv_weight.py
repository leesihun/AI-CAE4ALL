#!/usr/bin/env python3
"""Pick `beta_aux` from a probe log instead of guessing it.

    python configs/campaigns/pv_weight.py <probe log> --config <train cfg> \
        [--fraction 0.30]

`beta_aux` weights an MSE on the per-graph PEAK-TO-VALLEY of the decoded field.
Its natural scale is not knowable in advance: it depends on the target's
normalization and on how far the untrained decoder's extremes sit from the
truth's. Guessing produced the previous failure -- the head it replaces ran at
`beta_aux 1.0` against `alpha_recon 1000` and contributed on the order of 1% of
the objective, so the hypothesis was never actually tested at strength.

The probe reports both terms every epoch:

    Train  recon=1.23e-03 mmd=4.5e-02 aux=6.7e-02 total=...

and the objective is

    alpha_recon * recon  +  lambda_mmd * mmd  +  beta_aux * aux

so the weight that gives the peak-to-valley term a share `f` of the two-term
total (recon + pv, ignoring mmd, which is separately tuned) is

    beta_aux = f / (1 - f)  *  alpha_recon * recon / aux

Arm pvA reports `aux=` even at `beta_aux 0` -- `_pv_loss` runs whenever the VAE
path does and is simply multiplied by zero -- so the CONTROL arm supplies this
measurement for free, without a run whose objective is already contaminated by
the weight being chosen.

Prints one number to stdout, nothing else, so a shell can capture it.
"""
import argparse
import pathlib
import re
import sys

# Train  recon=1.23e-03 mmd=4.56e-02 aux=7.89e-02 total=1.35e+00
TRAIN_RE = re.compile(
    r"Train\s+recon=([0-9.eE+-]+)\s+mmd=([0-9.eE+-]+)\s+aux=([0-9.eE+-]+)")
KEY_RE = re.compile(r"^(\w+)[\t ]+([^#\n]+)")


def config_value(path, key, default):
    """Read one key out of the flat `key value` config, case-insensitively.

    The native parser lowercases keys, and this file writes `Batch_size` /
    `alpha_recon` with inconsistent case, so matching exactly would silently
    fall through to the default.
    """
    want = key.lower()
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('%') or not line.strip():
            continue
        m = KEY_RE.match(line)
        if m and m.group(1).lower() == want:
            try:
                return float(m.group(2).strip())
            except ValueError:
                return default
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--config', required=True)
    ap.add_argument('--fraction', type=float, default=0.30)
    a = ap.parse_args()

    text = pathlib.Path(a.log).read_text(encoding='utf-8', errors='replace')
    hits = TRAIN_RE.findall(text)
    if not hits:
        print("no 'Train  recon=... aux=...' line in " + a.log, file=sys.stderr)
        return 1

    # Last epoch: the first is dominated by initialization transients.
    recon, _mmd, aux = (float(v) for v in hits[-1])
    if aux <= 0.0:
        print(f"aux is {aux} -- the peak-to-valley term is not being computed. "
              f"Check that use_vae is True and the model carries _pv_loss.",
              file=sys.stderr)
        return 1

    alpha = config_value(a.config, 'alpha_recon', 1.0)
    f = a.fraction
    beta = (f / (1.0 - f)) * alpha * recon / aux

    # Round to 2 significant figures: the estimate is from one epoch of an
    # untrained model, so trailing digits would be false precision.
    if beta > 0:
        import math
        mag = math.floor(math.log10(beta))
        beta = round(beta, -(mag - 1))
    print(f"probe: recon={recon:.3e} aux={aux:.3e} alpha_recon={alpha:g} "
          f"-> beta_aux={beta:g} for {f:.0%} of (recon + pv)", file=sys.stderr)
    print(f"{beta:g}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
