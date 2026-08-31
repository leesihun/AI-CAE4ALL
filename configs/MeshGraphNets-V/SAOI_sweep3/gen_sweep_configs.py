"""Generate the SAOI wave-3 sweep: a 2^(4-1) resolution-IV half fraction, 8 arms.

    python configs/MeshGraphNets-V/SAOI_sweep3/gen_sweep_configs.py

Every arm is ../SAOI_all_input/config_train_bot.txt with the swept keys
overridden and the run-scoped keys (gpu_ids, epochs, paths) retargeted for an
8-way parallel sweep, ONE ARM PER GPU; the per-arm inference configs are
derived the same way from that folder's *_bot infer configs. Nothing here is
hand-edited -- regenerate instead, and change the production config when a
NON-swept key needs to move.

THE DESIGN
  Four factors in eight runs, defining relation  I = ABCD  (resolution IV):
  D is set to A xor B xor C rather than run freely. All four main effects are
  estimable clean; the six 2-factor interactions come in three CONFOUNDED
  PAIRS (AB=CD, AC=BD, AD=BC) -- a large one cannot be attributed to a single
  pair without a follow-up run. With one run per cell there was never
  replication anyway, so 3-factor terms were never trustworthy either way --
  this is the shape that keeps main effects clean at the lowest run count.

  This replaced an earlier 2^(5-1) x 16-arm design (two arms per GPU, 2000
  epochs). A measured epoch cost of ~576 s/epoch meant that budget was
  multiple GPU-weeks (24h of wall time only reached epoch 150), and
  two-per-GPU sharing was the source of the OOM risk documented below. Eight
  arms at one-per-GPU and 500 epochs is a
  576 s/epoch x 500 epochs ~= 3.3-day, BUDGET-LIMITED comparison instead of a
  converged one -- see the twin cHI-MGNflow sweep (SAOI_sweepB), which made
  the identical trade for the identical reason.

  A  z_conditioning         cc  concat (legacy fuser) | ad  AdaLN-Zero
  B  prior_grad_to_encoder  g0  detached | g1  end-to-end (CVAE rate term)
  C  capacity               c0  128 / 4,6,8,6,4 | c1  128 / 6,8,12,8,6 (+prior/VAE depth)
  D  regularizer scale (generated: D = A xor B xor C)
                            r001 lambda_mmd 1 + prior_nll_weight 1
                            r100 lambda_mmd 100 + prior_nll_weight 100

  `vae_latent_dim` was DROPPED from the earlier 5-factor grid and is pinned at
  16 (see FIXED below) -- the lowest-priority of the five. Wave-1's small-z
  win was measured with MMD statistically dead (batch 4, per-rank), so
  re-measuring it matters, but less than the three architectural axes kept
  here, and it costs almost nothing in VRAM or wall time -- unlike `capacity`,
  it was never going to be what made this budget unaffordable, so it is the
  natural one to cut rather than degrading the others.

WHY THE REMAINING THREE
  z_conditioning  AdaLN-Zero is provably identity-at-init and shrinks the
                  residual stream ~8.6x at 12 blocks, but that is an INIT
                  property; whether it converges better is open, and it is the
                  change that would silently propagate into every future run.
  prior_grad_to_encoder  g1 restores the CVAE rate term KL(q||p) that the old
                  detached objective was missing entirely. It also adds pressure
                  toward z = h(g); beta_aux (1.0, fixed) is the I(z;y) floor that
                  guards it. Untested -- must be an axis.
  capacity        The SAOI config was scaled DOWN from the b8 winner for mesh
                  size, so under-capacity is genuinely open. More importantly
                  this factor carries the sharpest predicted INTERACTION in the
                  grid: under `cc`, every extra block compounds the fuser's ~1.33x
                  gain, so depth should HURT; under `ad` the residual highway is
                  intact, so depth should HELP. z_conditioning x capacity is
                  estimable here (confounded only with prior_grad_to_encoder x
                  regularizer-scale, not with itself).
  regularizer scale  alpha_recon is 1000 while both regularizers sit at ~1, so
                  each is ~0.1% of the objective -- plausibly off. That matters
                  most for prior_grad_to_encoder: the rate term it opens competes
                  with recon INSIDE the encoder and loses 1000:1. lambda_mmd and
                  prior_nll_weight therefore move TOGETHER; "are the regularizers
                  on at all" has to be answered before "which one matters more".
                  CHECK THE FIRST EPOCHS' tqdm postfix: `mmd` and `fm_p` vs `total`.

  None of the swept keys enter the coarsening cache signature (that is
  multiscale_levels / coarsening_type / voronoi_clusters / hierarchy_variants /
  positional_features + the source file), so all 8 arms still share ONE cache.

GPU PACKING
  One arm per GPU (gpu = arm index) -- no card sharing, so the OOM exposure
  the old two-per-GPU design carried disappears along with the wall time.
"""
import itertools
# newline='\n' on EVERY write: the default (None) translates to CRLF on
# Windows, and a CRLF run_sweep.sh dies on Linux with `bad interpreter: ^M`.
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
# The sweep lives in its own folder; the configs it derives FROM are the
# production ones next door, which is the whole point -- an arm is
# config_train_bot.txt with a handful of keys overridden, so the production
# config stays the single source of truth for everything not swept.
PROD = HERE.parent / 'SAOI_all_input'
BASE = PROD / 'config_train_bot.txt'
TRAIN_PREFIX = 'config_train_'   # run_sweep.sh: config_train_${arm}.txt
INFER_PREFIX = 'config_infer_'   # run_sweep.sh: config_infer_${arm}_${tag}.txt

# ── factors, in the order the arm name encodes them: (label, [(tag, {k: v})*2])
# A level may set several keys; `capacity` and `regularizer scale` are composite
# on purpose (see the module docstring).
FACTORS = [
    ('z_conditioning', [
        ('cc', {'z_conditioning': 'concat'}),
        ('ad', {'z_conditioning': 'adaln'}),
    ]),
    ('prior_grad_to_encoder', [
        ('g0', {'prior_grad_to_encoder': '0.0'}),
        ('g1', {'prior_grad_to_encoder': '1.0'}),
    ]),
    ('capacity', [
        ('c0', {'Latent_dim': '128', 'mp_per_level': '4, 6, 8, 6, 4',
                'vae_mp_layers': '5', 'prior_mp_layers': '5', 'prior_hidden_dim': '256'}),
        ('c1', {'Latent_dim': '128', 'mp_per_level': '6, 8, 12, 8, 6',
                'vae_mp_layers': '7', 'prior_mp_layers': '7', 'prior_hidden_dim': '384'}),
    ]),
    # GENERATED, not free: D = A xor B xor C  (defining relation I = ABCD).
    ('regularizer scale', [
        ('r001', {'lambda_mmd': '1',   'prior_nll_weight': '1.0'}),
        ('r100', {'lambda_mmd': '100', 'prior_nll_weight': '100'}),
    ]),
]

# ── held fixed for every arm, overriding the production config ───────────────
# Batch_size 16 matches production per-rank. ONE arm per GPU now, so there is
# no sharing and the c1 (40-block) arms no longer risk OOM the way they did
# when paired with another arm on the same card.
FIXED = {
    'Batch_size':           ('16',   'production per-rank value; one arm per GPU, no sharing'),
    'num_workers':          ('4',    'matches production; one arm per GPU, no need to economize'),
    'Training_epochs':      ('500',  'measured budget: 576 s/epoch x 500 = ~3.3 days per arm at '
                                     'one arm per GPU. NOT a converged comparison -- see the README'),
    'vae_latent_dim':       ('16',   'FIXED, not swept -- see the module docstring for why'),
    'val_interval':         ('100',  'CRPS is the selection metric; 5 evals over the run'),
    'test_interval':        ('500',  'periodic plots; fires once, at the last epoch'),
    'hierarchy_cache_keep': ('True', 'REQUIRED: all 8 arms share one cache; a finishing arm must not delete it'),
    'best_by':              ('crps', 'select on the GENERATIVE metric, not the posterior recon loss'),
}

HEADER = """%   ============================================================
%   SAOI wave 3 -- 2^(4-1) resolution-IV half fraction, 8 arms, ONE PER GPU.
%   Defining relation I = ABCD: the regularizer scale is A xor B xor C, not
%   free. All 4 main effects are clean; the 6 two-factor interactions come in
%   3 CONFOUNDED PAIRS (AB=CD, AC=BD, AD=BC).
%   GENERATED by gen_sweep_configs.py; do not hand-edit, regenerate.
%
%   ARM: {arm}
{axis_lines}
%     gpu {gpu} -- one arm per card ({mate})
%
%   Everything else is config_train_bot.txt, except the sweep-scoped keys
%   (batch, workers, epochs, intervals, cache-keep, best_by, vae_latent_dim)
%   listed inline below with the reason on each line.
%   ============================================================
"""

TAG_NOTE = {
    'cc':   'legacy Linear([x, z]) fuser',
    'ad':   'AdaLN-Zero modulation, identity at init',
    'g0':   'detached: no CVAE rate term (legacy)',
    'g1':   'end-to-end: rate term restored',
    'c0':   'current production size (28 processor blocks)',
    'c1':   'DEEPER only, same width (40 blocks vs 28) -- width was dropped back to 128 for VRAM, and depth is the sharper test anyway: the concat fuser gain compounds PER BLOCK',
    'r001': 'both regularizers ~0.1% of the objective (alpha_recon is 1000)',
    'r100': 'both regularizers ~10% of the objective',
}


def arms():
    """The 8 runs of the half fraction: (index, name, {key: value}, [tags])."""
    out = []
    for i in range(8):
        free = [(i >> (2 - k)) & 1 for k in range(3)]      # A B C
        bits = free + [free[0] ^ free[1] ^ free[2]]         # D = A^B^C
        tags, values = [], {}
        for (_, levels), b in zip(FACTORS, bits):
            tag, kv = levels[b]
            tags.append(tag)
            values.update(kv)
        out.append((i, '_'.join(tags), values, tags))
    return out


def check_design(table):
    """Assert the properties that make this a usable resolution-IV design.

    Main effects 4v4, and every PAIR of factors sees all four combinations
    twice -- so each 2-factor effect is estimable, just confounded with the
    complementary pair (AB=CD, AC=BD, AD=BC). Every arm is on its own GPU, so
    there is no pairing constraint to check.
    """
    tags = [t for _, _, _, t in table]
    for k, (label, lv) in enumerate(FACTORS):
        counts = [sum(1 for t in tags if t[k] == lv[b][0]) for b in (0, 1)]
        assert counts == [4, 4], (label, counts)
    for a, b in itertools.combinations(range(len(FACTORS)), 2):
        cells = {}
        for t in tags:
            cells[(t[a], t[b])] = cells.get((t[a], t[b]), 0) + 1
        assert sorted(cells.values()) == [2, 2, 2, 2], (a, b, cells)


# ── per-arm inference configs ───────────────────────────────────────────────
# Only the *_bot infer configs apply: the sweep trains on saoi_train_bot.h5, so
# a *_top checkpoint does not exist for these arms. tag -> source config.
INFER_SOURCES = {
    's26fe_main':  'config_infer_s26fe_main_bot.txt',
    's26fe_sec':   'config_infer_s26fe_sec_bot.txt',
    'sm_l345u':    'config_infer_sm_l345u_main_bot.txt',
}
# Draws per scene for the spread histogram (production uses 5000).
# THIS IS THE SWEEP'S DOMINANT COST: total forwards = arms x eval sets x scenes
# x INFER_SAMPLES. No trajectory files are written (save_rollouts False), so it
# costs compute and nothing else. 8 arms (not 16) halves this stage's cost
# relative to the earlier design at the same INFER_SAMPLES.
INFER_SAMPLES = '2000'

INFER_OVERRIDES = {
    'num_vae_samples': (INFER_SAMPLES, 'sweep draws per scene (production uses 5000)'),
    'save_rollouts':   ('False', 'write NO trajectory HDF5s -- scene x draws would be '
                                 'thousands of files across the grid; the histogram '
                                 'and spread_values.npz still get written'),
    'make_histogram':  ('True',  'GT vs generated z_disp spread (max - min) per realization'),
    'show_histogram':  ('False', 'headless node: save the PNG, do not try to open a viewer'),
    'histogram_bins':  ('60',    'bins in the overlaid GT/generated histogram'),
    # ONE arm per GPU during inference too now, so the automatic VAE batch
    # sizer can use the full default fraction of free VRAM -- no more halving
    # for a card-sharing partner that no longer exists.
    'vae_batch_vram_fraction': ('0.70', 'one arm per GPU; this is the native default'),
}


def render_infer(src_lines, arm, tag, gpu, values):
    """One inference config: same eval set, this arm's checkpoint.

    `eval_dataset` is set to the SAME file as `infer_dataset`: that HDF5 carries
    the true fields, and _eval_dataset_spreads reads channel 5 (z_disp) at the
    final timestep out of it. Without it the runtime skips the comparison.

    The architecture keys are written out even though the checkpoint's
    model_config overrides them -- it keeps the file self-describing and stops
    the rollout log from printing a wall of "overridden by checkpoint" lines.
    prior_fm_solver is NOT in model_config, so its value here is the live one.
    """
    out, skip_pct = [], False
    seen = set()
    over = {k: v for k, (v, _) in INFER_OVERRIDES.items()}
    notes = {k: n for k, (_, n) in INFER_OVERRIDES.items()}
    over.update(values)
    for line in src_lines:
        if skip_pct and line.startswith('%'):
            continue
        skip_pct = False
        if line.startswith('%') or not line.strip():
            out.append(line)
            continue
        key = line.split('\t')[0].split()[0] if '\t' in line else line.split()[0]
        if key in over:
            seen.add(key)
            note = notes.get(key, 'matches the training arm')
            out.append(f"{key}\t{over[key]}  # {note}" if note else f"{key}\t{over[key]}")
            skip_pct = key in values
            continue
        if key == 'gpu_ids':
            out.append(f"gpu_ids\t{gpu}  # the GPU this arm trained on")
            continue
        if key == 'log_file_dir':
            out.append(f"log_file_dir\t../../output/meshgraphnets-v/saoi_sweep3/"
                       f"{arm}.infer_{tag}.log")
            continue
        if key == 'modelpath':
            out.append(f"modelpath\t../output/meshgraphnets-v/saoi_sweep3/{arm}.pth")
            continue
        if key == 'inference_output_dir':
            out.append(f"inference_output_dir\t../output/meshgraphnets-v/saoi_sweep3/infer/{arm}/{tag}")
            continue
        if key in ('dataset_dir', 'infer_dataset'):
            # The six production infer configs spell the directory `SAOI` while
            # every training config spells it `saoi`. Both keys are PATH_KEYS, so
            # the parser preserves case and exactly one spelling can resolve on a
            # case-sensitive filesystem -- the training configs' lowercase one,
            # since that is what produced the checkpoints.
            val = line.split('\t', 1)[1].split('#')[0].strip()
            val = val.replace('/dataset/SAOI/', '/dataset/saoi/')
            if key == 'dataset_dir':
                val = '../dataset/saoi/saoi_train_bot.h5'
            out.append(f"{key}\t{val}")
            if key == 'infer_dataset':
                out.append(f"eval_dataset\t{val}  # ground truth for the spread histogram")
                seen.add('eval_dataset')
            continue
        out.append(line)

    missing = [k for k in over if k not in seen]
    if missing:
        out += ['', '%   Sweep-only keys (absent from the production infer config)']
        out += [f"{k}\t{over[k]}  # {notes.get(k, 'matches the training arm')}"
                for k in missing]
    return '\n'.join(out)


INFER_HEADER = """%   ============================================================
%   SAOI wave 3 -- INFERENCE for one sweep arm on one eval set.
%   GENERATED by gen_sweep_configs.py; do not hand-edit, regenerate.
%
%   arm        {arm}
%   eval set   {tag}   (also used as `eval_dataset`: it carries the true fields)
%   gpu        {gpu}   (the GPU this arm trained on)
%
%   Writes NO trajectory files. What it produces, per arm and eval set:
%     histogram_compare.png   GT vs generated z_disp spread (max - min)
%     spread_values.npz       the raw gt/gen arrays behind that plot
%   score_sweep.py reads the npz to tabulate every arm on one axis.
%   ============================================================
"""


def render(base_lines, values, arm, gpu, mate):
    swept = set(values)
    overrides = dict(values)
    overrides.update({k: v for k, (v, _) in FIXED.items()})
    notes = {k: n for k, (_, n) in FIXED.items()}
    seen = set()

    out, skip_pct = [], False
    for line in base_lines:
        if skip_pct and line.startswith('%'):
            continue                      # drop the comment block of a replaced key
        skip_pct = False
        if line.startswith('%') or not line.strip():
            out.append(line)
            continue
        key = line.split('\t')[0].split()[0] if '\t' in line else line.split()[0]
        if key in overrides:
            seen.add(key)
            note = notes.get(key, 'SWEPT AXIS' if key in swept else '')
            out.append(f"{key}\t{overrides[key]}  # {note}" if note
                       else f"{key}\t{overrides[key]}")
            skip_pct = key in swept       # its comment described the other level
            continue
        if key == 'gpu_ids':
            out.append(f"gpu_ids\t{gpu}  # one GPU; {mate}")
            continue
        if key == 'log_file_dir':
            out.append(f"log_file_dir\t../../output/meshgraphnets-v/saoi_sweep3/{arm}.log")
            continue
        if key == 'modelpath':
            out.append(f"modelpath\t../output/meshgraphnets-v/saoi_sweep3/{arm}.pth")
            continue
        out.append(line)

    # Keys the production config does not carry at all get appended.
    missing = [k for k in overrides if k not in seen]
    assert not (set(missing) & swept), f"{arm}: a swept key is missing from the base"
    if missing:
        out += ['', '%   Sweep-only keys (absent from the production config)']
        out += [f"{k}\t{overrides[k]}  # {notes.get(k, '')}" for k in missing]
    return '\n'.join(out)


def main():
    base_lines = BASE.read_text(encoding='utf-8').split('\n')
    # Strip the base config's own banner; each arm gets its own.
    while base_lines and base_lines[0].startswith('%'):
        base_lines.pop(0)

    table = arms()
    check_design(table)
    for i, arm, values, tags in table:
        gpu = i                    # one arm per GPU: no sharing, no OOM exposure
        mate = 'no card-sharing arm (one arm per GPU)'
        axis_lines = []
        for (label, levels), tag in zip(FACTORS, tags):
            kv = dict(levels)[tag]
            axis_lines.append(f"%     {label:<22}{tag:<6}{TAG_NOTE[tag]}")
            for k, v in kv.items():
                axis_lines.append(f"%     {'':<28}{k} {v}")
        header = HEADER.format(arm=arm, gpu=gpu, mate=mate,
                               axis_lines='\n'.join(axis_lines))
        (HERE / f'{TRAIN_PREFIX}{arm}.txt').write_text(
            header + render(base_lines, values, arm, gpu, mate), encoding='utf-8', newline='\n')
        n_inf = 0
        for tag, src in INFER_SOURCES.items():
            src_lines = (PROD / src).read_text(encoding='utf-8').split('\n')
            while src_lines and src_lines[0].startswith('%'):
                src_lines.pop(0)
            (HERE / f'{INFER_PREFIX}{arm}_{tag}.txt').write_text(
                INFER_HEADER.format(arm=arm, tag=tag, gpu=gpu)
                + render_infer(src_lines, arm, tag, gpu, values),
                encoding='utf-8', newline='\n')
            n_inf += 1
        print(f"  gpu {gpu}  {arm}  (+{n_inf} inference configs)")
    print(f"\n{len(table)} training + {len(table) * len(INFER_SOURCES)} "
          f"inference configs written to {HERE}")
    print("design checks passed: 4v4 on all 4 main effects, "
          "2/2/2/2 on all 6 factor pairs, one arm per GPU")
    print("ARMS=\"" + ' '.join(n for _, n, _, _ in table) + "\"")
    print("INFER_TAGS=\"" + ' '.join(INFER_SOURCES) + "\"")


if __name__ == '__main__':
    main()
