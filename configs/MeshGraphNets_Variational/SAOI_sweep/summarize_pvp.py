#!/usr/bin/env python3
"""Collapse every posterior_vs_prior.py run of the sweep into one table.

    python configs/MeshGraphNets_Variational/SAOI_sweep/summarize_pvp.py
    python configs/MeshGraphNets_Variational/SAOI_sweep/summarize_pvp.py <diag_dir>

run_sweep.sh writes one JSON per (arm, eval set) under `<out_root>/diag/<arm>/
<tag>/`. Each decomposes ONE part into four ensembles of the same statistic:
its 125 true realizations, the decoded posterior mean, the decoded posterior
sample, and the decoded FM prior. rank_arms.py scores only the last against the
truth, which tells you an arm is bad but not which stage is at fault. Reading
all four is what separates a prior defect from a decoder defect, so this exists
to put that decomposition for all 24 runs on one screen.

Columns (ideal value in parentheses):

  pmean     sd(decoded posterior mean)   / sd(truth)       (1)
  psamp     sd(decoded posterior sample) / sd(truth)       (1)
  prior     sd(decoded FM prior)         / sd(truth)       (1)
  b_pm      (mean pmean ensemble - mean truth) / sd(truth) (0)
  b_ps      same for the posterior sample                  (0)
  b_pr      same for the prior                             (0)
  z_sd      latent-space sd(prior) / sd(posterior), rms over slots
  pca       fraction of the prior's latent variance lying in the posterior
            cloud's own top-k principal directions
  verdict   posterior_vs_prior.py's own reading, abbreviated

How to read it, in the order that closes questions fastest:

  psamp ~ 1, prior << 1      the decoder can carry the width and the prior is
                             the bottleneck. Prior-side work is the lever.
  psamp << 1                 the decoder squashes the width before the prior is
                             ever consulted. No prior can put back what
                             encoding removed -- the prior-architecture axis is
                             closed and beta_aux / latent capacity is the lever.
  b_pm ~ 1                   even decoding the posterior -- which encoded the
                             answer -- lands in the wrong place. That defect is
                             not distributional, and no width fix repairs it.
  z_sd ~ 1 while prior << 1  the prior is right in latent space and the decoder
                             compresses it: the psamp << 1 verdict again,
                             reached through an independent quantity.

A missing (arm, tag) is reported by name with its log path rather than silently
shrinking the table -- a partial run reads exactly like a complete one here.
"""
import glob
import json
import os
import sys

# run_sweep.sh's own defaults. Absences are only visible against an expectation.
ARMS = [str(i) for i in range(1, 9)]
TAGS = ['s26fe_main', 's26fe_sec', 'sm_l345u']
DEFAULT_DIAG = 'output/meshgraphnets-v/saoi_fm_v2_sweep/diag'

VERDICT_TAGS = (
    ('PRIOR MISMATCH', 'PRIOR'),
    ('ENCODE/DECODE LOSS', 'DECODER'),
    ('MIXED', 'MIXED'),
    ('prior path', 'PRIOR-WIDE'),
)


def config_value(path, key):
    """One value from a native flat config, or None.

    Deliberately minimal: `%` comments the line, `#` comments the tail, and the
    two keys read here (fm_variant, dataset_dir) are single tokens.
    """
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.split('#', 1)[0].strip()
                if not line or line.startswith('%'):
                    continue
                parts = line.split()
                if parts[0] == key and len(parts) > 1:
                    return parts[1]
    except OSError:
        return None
    return None


def arm_label(arm, infer_cfg):
    """`1 bot/P0` -- the ablation cell, not just its number.

    The half comes from the inference config's own dataset_dir; fm_variant is
    a training-only key, so it is read from the sibling train config.
    """
    half = variant = '?'
    if infer_cfg:
        ds = config_value(infer_cfg, 'dataset_dir') or ''
        for candidate in ('bot', 'top'):
            if candidate in os.path.basename(ds):
                half = candidate
        train_cfg = os.path.join(os.path.dirname(infer_cfg), f'config_train_{arm}.txt')
        variant = config_value(train_cfg, 'fm_variant') or '?'
    return f'{arm} {half}/{variant}'


def short_verdict(lines):
    head = (lines or [''])[0]
    for needle, tag in VERDICT_TAGS:
        if needle in head:
            return tag
    return head.split(':', 1)[0][:10]


def read_run(path):
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    e = d['ensembles']
    lat = d.get('latent', {})
    nan = float('nan')
    return {
        'label': arm_label(os.path.basename(os.path.dirname(os.path.dirname(path))),
                           d.get('config')),
        'pmean': e['posterior_mean']['sd_ratio'],
        'psamp': e['posterior_sample']['sd_ratio'],
        'prior': e['prior']['sd_ratio'],
        'b_pm': e['posterior_mean']['dmean_over_sd'],
        'b_ps': e['posterior_sample']['dmean_over_sd'],
        'b_pr': e['prior']['dmean_over_sd'],
        'z_sd': lat.get('overall_sd_ratio_rms', nan),
        'pca': lat.get('pca', {}).get('prior_var_frac_in_top_k', nan),
        'verdict': short_verdict(d.get('verdict')),
    }


def main(argv):
    diag = argv[1] if len(argv) > 1 else DEFAULT_DIAG
    if not os.path.isdir(diag):
        print(f'no such diagnostic directory: {diag}', file=sys.stderr)
        return 1
    out_root = os.path.dirname(os.path.normpath(diag))

    found, missing, broken = {}, [], []
    for arm in ARMS:
        for tag in TAGS:
            hits = glob.glob(os.path.join(diag, arm, tag, 'posterior_vs_prior_*.json'))
            if not hits:
                missing.append((arm, tag))
                continue
            try:
                found[(arm, tag)] = read_run(hits[0])
            except (OSError, ValueError, KeyError) as exc:
                broken.append((arm, tag, f'{type(exc).__name__}: {exc}'))

    header = (f"{'arm':<10} {'eval set':<11} {'pmean':>6} {'psamp':>6} {'prior':>6}"
              f" {'b_pm':>7} {'b_ps':>7} {'b_pr':>7} {'z_sd':>6} {'pca':>5}  verdict")
    print(f'{len(found)} / {len(ARMS) * len(TAGS)} runs in {diag}')
    print(header)
    print('-' * len(header))
    for arm in ARMS:
        for tag in TAGS:
            r = found.get((arm, tag))
            if r is None:
                continue
            print(f"{r['label']:<10} {tag:<11} {r['pmean']:6.3f} {r['psamp']:6.3f}"
                  f" {r['prior']:6.3f} {r['b_pm']:+7.3f} {r['b_ps']:+7.3f}"
                  f" {r['b_pr']:+7.3f} {r['z_sd']:6.3f} {r['pca']:5.2f}  {r['verdict']}")
        if any((arm, t) in found for t in TAGS):
            print()

    # Absences belong in the same stream as the table: this output gets copied
    # out of a terminal, and a warning on stderr is exactly what does not
    # survive that trip.
    for arm, tag, why in broken:
        print(f'UNREADABLE  arm {arm} {tag}: {why}')
    for arm, tag in missing:
        log = os.path.join(out_root, 'run_logs', f'{arm}.posterior_vs_prior_{tag}.log')
        print(f'MISSING     arm {arm} {tag} -- {log}')
    return 1 if (missing or broken) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
