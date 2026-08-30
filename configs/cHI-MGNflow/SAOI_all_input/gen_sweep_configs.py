"""Generate the cHI-MGNflow SAOI Wave-B sweep: a 2^4 full factorial, 16 arms.

    python configs/cHI-MGNflow/SAOI_all_input/gen_sweep_configs.py

Every arm is config_train_bot.txt with four keys overridden and the run-scoped
keys (gpu_ids, batch, epochs, paths) retargeted for a 16-way parallel sweep.
Regenerate rather than hand-editing 16 files.

WHY THESE FOUR AND NOT THE OBVIOUS ONES
  The variational tree's wave-3 swept z_conditioning / prior_grad_to_encoder /
  vae_latent_dim / lambda_mmd. None of those exist here -- there is no posterior
  and no learned prior. And the two knobs that look most tempting, flow_steps
  and flow_solver, must NOT be here: they are SAMPLING-TIME choices. The same
  checkpoint integrates at K=4 or K=100, so sweeping them over training runs
  would burn the entire 16-arm budget on something inference answers for free.
  See docs/SWEEP_PLAN.md -- they belong to Wave A.

  batch_size        16 | 32    Gradient variance is the real cost driver of flow
                               matching: the target y-z0 carries irreducible
                               noise of size Var(y|g). Per-step forward cost
                               dropped 33 -> 8 (no posterior encoder, no prior
                               trunk), so a larger batch now fits the same VRAM.
                               Spending that headroom on variance is the most
                               direct answer to the method's main expense.
  flow_t_sampling   uniform    Concentrates the budget on mid-path, where the
                  | logitnorm  velocity is hardest to predict (t=0 is nearly
                               pure noise, t=1 nearly the data -- both ends are
                               easy). Changes only WHERE budget is spent, never
                               the optimum, so the two levels stay comparable.
  voronoi_clusters  1000,100   FM-specific hypothesis: at t~0 the input is white
                  | 2000,250   noise plus geometry, and recovering global
                               structure is the coarsest level's job. This axis
                               should matter MORE here than for deterministic
                               regression. Worth 8 arms to find out.
  learningr        1e-4 | 3e-4 A noisier target may want a different LR, and it
                               is the classic partner of batch size.

GPU PACKING
  Arm index i = 8*B + 4*T + 2*C + L lands on GPU min(i, 15-i), pairing every arm
  with its BITWISE COMPLEMENT. Each GPU therefore hosts one of each level of
  every factor: no factor is confounded with GPU, and each card gets exactly one
  batch-16 and one batch-32 arm so VRAM is balanced.

  CHECK THE FIRST ARM'S `VRAM peak=` LINE. batch 32 x voronoi 2000,250 is the
  worst corner. If it does not fit, drop BATCH_LEVELS to ('8', '16') and
  regenerate -- every axis survives.
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / 'config_train_bot.txt'
OUT_PREFIX = 'config_'          # arm names already start with 'sweep_'

# ── the four swept factors ───────────────────────────────────────────────────
FACTORS = [
    ('batch_size',       [('b16', '16'),       ('b32', '32')]),
    ('flow_t_sampling',  [('tu', 'uniform'),   ('tl', 'logitnormal')]),
    ('voronoi_clusters', [('c1k', '1000, 100'), ('c2k', '2000, 250')]),
    ('learningr',        [('lr1', '0.0001'),   ('lr3', '0.0003')]),
]

NOTES = {
    'batch_size': {'16': 'production per-rank value',
                   '32': 'double batch -- lower gradient variance'},
    'flow_t_sampling': {'uniform': 'flat t schedule (baseline)',
                        'logitnormal': 'budget concentrated on mid-path'},
    'voronoi_clusters': {'1000, 100': 'production hierarchy',
                         '2000, 250': 'finer coarse levels -- more global capacity at t~0'},
    'learningr': {'0.0001': 'production LR',
                  '0.0003': '3x LR'},
}

# ── held fixed for every arm, overriding the production config ───────────────
# Training_epochs MUST come from Wave 0 (docs/SWEEP_PLAN.md). The placeholder
# below is deliberately small so that launching this sweep before measuring the
# multiple is obviously wrong rather than silently under-trained.
FIXED = {
    'training_epochs':      ('2000', 'PLACEHOLDER -- set from the Wave 0 measurement'),
    'num_workers':          ('2',    '16 concurrent jobs; 4 workers each would be 64 processes'),
    'val_interval':         ('100',  'CRPS is the selection metric; 20 evals over the run'),
    'test_interval':        ('500',  'periodic plots; 4 over the run'),
    'val_flow_steps':       ('12',   'set from Wave A; validation integrates the ODE every val_interval'),
    'val_num_samples':      ('8',    'FIXED across arms -- the CRPS estimator is unbiased at any S, '
                                     'so varying it would only add noise to the comparison'),
    'flow_steps':           ('30',   'recorded for provenance only; Wave A picks the production K '
                                     'and config wins over the checkpoint at inference'),
    'hierarchy_cache_keep': ('True', 'REQUIRED: 16 arms share one cache; a finishing arm must not delete it'),
    'best_by':              ('crps', 'select on the sampling metric, not the one-step regression loss'),
    'gpu_ids':              (None,   None),   # filled per arm
}

HEADER = """%   ============================================================
%   cHI-MGNflow SAOI Wave B -- 2^4 full factorial, 16 arms, 2 per GPU.
%   GENERATED by gen_sweep_configs.py; do not hand-edit, regenerate.
%
%   ARM: {arm}
{factor_lines}%     gpu {gpu} -- shared with the bitwise-complement arm {partner}
%
%   Everything else is config_train_bot.txt, except the sweep-scoped keys
%   listed inline below with the reason on each line.
%
%   NOT SWEPT HERE ON PURPOSE: flow_steps and flow_solver are sampling-time
%   choices -- the same checkpoint integrates at any K. They belong to Wave A,
%   which costs zero training runs. See docs/SWEEP_PLAN.md.
%   ============================================================
"""


def arm_name(levels):
    return 'sweep_' + '_'.join(tag for tag, _ in levels)


def main():
    base_lines = BASE.read_text(encoding='utf-8').splitlines()
    overrides = {}
    written = []

    for i in range(16):
        bits = [(i >> (3 - k)) & 1 for k in range(4)]
        levels = [FACTORS[k][1][bits[k]] for k in range(4)]
        keys = [FACTORS[k][0] for k in range(4)]
        name = arm_name(levels)
        gpu = min(i, 15 - i)
        partner_bits = [1 - b for b in bits]
        partner = arm_name([FACTORS[k][1][partner_bits[k]] for k in range(4)])

        overrides = {k: v for k, (_tag, v) in zip(keys, levels)}
        overrides.update({k: v for k, (v, _n) in FIXED.items() if v is not None})
        overrides['gpu_ids'] = str(gpu)
        overrides['log_file_dir'] = f'../../output/chi-mgnflow/saoi_sweepB/{name}.log'
        overrides['modelpath'] = f'../output/chi-mgnflow/saoi_sweepB/{name}.pth'

        factor_lines = ''.join(
            f"%     {k:<18}{tag:<6} {NOTES[k][v]}\n"
            for k, (tag, v) in zip(keys, levels)
        )
        out = [HEADER.format(arm=name, factor_lines=factor_lines,
                             gpu=gpu, partner=partner)]

        seen = set()
        for line in base_lines:
            if line.startswith('%') or not line.strip():
                out.append(line)
                continue
            key = line.split('\t', 1)[0].split()[0] if '\t' in line else line.split()[0]
            if key in overrides:
                note = ''
                if key in [k for k in keys]:
                    note = '  # SWEPT AXIS'
                elif key in FIXED and FIXED[key][1]:
                    note = f'  # {FIXED[key][1]}'
                out.append(f'{key}\t{overrides[key]}{note}')
                seen.add(key)
            else:
                out.append(line)

        # keys not present in the base config
        missing = [k for k in overrides if k not in seen]
        if missing:
            out.append('')
            out.append('%   Sweep-only keys (absent from the production config)')
            for k in missing:
                if k in keys:
                    note = '  # SWEPT AXIS'
                elif k in FIXED and FIXED[k][1]:
                    note = f'  # {FIXED[k][1]}'
                else:
                    note = ''
                out.append(f'{k}\t{overrides[k]}{note}')

        path = HERE / f'{OUT_PREFIX}{name}.txt'
        path.write_text('\n'.join(out) + '\n', encoding='utf-8')
        written.append((gpu, name))

    print(f'Wrote {len(written)} arm configs to {HERE}')
    by_gpu = {}
    for gpu, name in written:
        by_gpu.setdefault(gpu, []).append(name)
    for gpu in sorted(by_gpu):
        print(f'  gpu {gpu}: ' + '  '.join(by_gpu[gpu]))
    print('\nBEFORE LAUNCHING:')
    print('  1. Set training_epochs from the Wave 0 measurement (docs/SWEEP_PLAN.md).')
    print('  2. Set val_flow_steps from Wave A.')
    print('  3. Delete any stale mscache -- the cache signature pins the source '
          'h5 mtime, which write_preprocessing_to_hdf5 bumps every run.')
    print('  4. Warm the cache with ONE arm before launching all 16 into a miss.')
    print('  5. Check the first arm\'s "VRAM peak=" line (b32 x c2k is the worst corner).')


if __name__ == '__main__':
    main()
