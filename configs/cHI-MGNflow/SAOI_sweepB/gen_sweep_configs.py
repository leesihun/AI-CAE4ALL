"""Generate the cHI-MGNflow SAOI Wave-B sweep: 2^(5-1) resolution V, 16 arms.

    python configs/cHI-MGNflow/SAOI_sweepB/gen_sweep_configs.py

Emits, from the production configs in ../SAOI_all_input/:
    config_train_<arm>.txt          16   one training arm
    config_infer_<arm>_<tag>.txt    48   one arm x one held-out eval set

Deliberately the same shape as configs/MeshGraphNets-V/SAOI_sweep3/ so the two
methods are compared on identical data with identical machinery. Nothing here is
hand-edited -- regenerate, and change ../SAOI_all_input/ when a NON-swept key
has to move.

THE DESIGN
  Five factors in sixteen runs, defining relation  I = ABCDE  (resolution V):
  E is set to A xor B xor C xor D rather than run freely. All five main effects
  and all ten 2-factor interactions are estimable clean; only 3-factor and
  higher alias. With one run per cell there is no replication either way, so the
  half fraction gives up nothing real and buys a whole extra factor over the
  2^4 design this replaces.

WHY THESE FIVE, AND WHY NOT THE OBVIOUS ONES
  The variational tree's wave-3 swept z_conditioning / prior_grad_to_encoder /
  vae_latent_dim / regularizer scale. NONE of those exist here -- there is no
  posterior and no learned prior, so there is nothing to transfer.

  And the two knobs that look most tempting, `flow_steps` and `flow_solver`,
  must NOT be here: they are SAMPLING-TIME choices. The same checkpoint
  integrates at K=4 or K=100, so sweeping them across training runs would burn
  the entire 16-arm budget on a question inference answers for free. They belong
  to Wave A (docs/SWEEP_PLAN.md), which costs zero training runs.

  A  batch_size        16 | 32     Gradient variance is flow matching's real cost
                                   driver: the target y - z0 carries irreducible
                                   noise of size Var(y|g). Per-step forward cost
                                   dropped (no posterior encoder, no prior trunk),
                                   so a larger batch now fits the same VRAM --
                                   spending that headroom on variance is the most
                                   direct answer to the method's main expense.
  B  flow_t_sampling   tu | tl     logit-normal concentrates the budget on
                                   mid-path, where the velocity is hardest: t=0
                                   is nearly pure noise and t=1 nearly the data,
                                   both easy. Changes only WHERE budget is spent,
                                   never the optimum, so the levels stay
                                   comparable.
  C  voronoi_clusters  c1k | c2k   FM-specific hypothesis: at t~0 the input is
                                   white noise plus geometry, and recovering
                                   global structure is the coarsest level's job.
                                   This should matter MORE here than for
                                   deterministic regression.
                                   COSTS: this is the ONLY swept key that enters
                                   the coarsening cache signature, so the grid
                                   builds TWO caches, not one, and run_sweep.sh
                                   warms one arm per level.
  D  capacity          k0 | k1     latent_dim + mp_per_level together. Added to
                                   match wave-3's capacity axis so the two
                                   studies can be read side by side, and because
                                   a velocity field is a harder function than a
                                   point prediction -- whether it wants more
                                   width is open.
  E  learningr         lr1 | lr3   A noisier target may want a different LR, and
                                   it is the classic partner of batch size.
                                   GENERATED: E = A xor B xor C xor D.

GPU PACKING
  Arms are paired by complementing the four free factors; E is then unchanged
  (four flips leave the parity alone). Every GPU therefore hosts one of each
  level of batch_size, flow_t_sampling, voronoi_clusters and capacity -- which
  is what balances VRAM, since batch and capacity are what move memory. The LR
  is constant within a pair, which costs nothing: it changes no memory or
  runtime, and eight identical cards in one node carry no batch/day/operator
  effect for it to confound with.
"""
import itertools
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PROD = HERE.parent / 'SAOI_all_input'
BASE = PROD / 'config_train_bot.txt'
TRAIN_PREFIX = 'config_train_'
INFER_PREFIX = 'config_infer_'

# ── factors, in the order the arm name encodes them ──────────────────────────
FACTORS = [
    ('batch_size', [
        ('b16', {'batch_size': '16'}),
        ('b32', {'batch_size': '32'}),
    ]),
    ('flow_t_sampling', [
        ('tu', {'flow_t_sampling': 'uniform'}),
        ('tl', {'flow_t_sampling': 'logitnormal'}),
    ]),
    ('voronoi_clusters', [
        ('c1k', {'voronoi_clusters': '1000, 100'}),
        ('c2k', {'voronoi_clusters': '2000, 250'}),
    ]),
    ('capacity', [
        ('k0', {'latent_dim': '128', 'mp_per_level': '4, 6, 8, 6, 4'}),
        ('k1', {'latent_dim': '192', 'mp_per_level': '6, 8, 12, 8, 6'}),
    ]),
    # GENERATED, not free: E = A xor B xor C xor D  (defining relation I = ABCDE)
    ('learningr', [
        ('lr1', {'learningr': '0.0001'}),
        ('lr3', {'learningr': '0.0003'}),
    ]),
]

TAG_NOTE = {
    'b16': 'production per-rank batch',
    'b32': 'double batch -- lower gradient variance on a noisy target',
    'tu':  'flat t schedule (baseline)',
    'tl':  'logit-normal: budget concentrated on mid-path',
    'c1k': 'production hierarchy',
    'c2k': 'finer coarse levels -- more global capacity at t~0 (SEPARATE CACHE)',
    'k0':  'production size (28 processor blocks)',
    'k1':  'wider + deeper (40 blocks)',
    'lr1': 'production LR',
    'lr3': '3x LR',
}

# ── training keys held fixed across every arm ───────────────────────────────
# training_epochs MUST come from Wave 0 (docs/SWEEP_PLAN.md): flow matching needs
# more steps than deterministic regression because the target carries
# irreducible noise of size Var(y|g), and that multiple has never been measured
# on this data. Launching at a guessed budget makes the ranking a function of
# convergence speed rather than of the factors.
FIXED_TRAIN = {
    'training_epochs':      ('6000', 'PLACEHOLDER -- set from the Wave 0 measurement'),
    'num_workers':          ('2',    '16 concurrent jobs; 4 workers each would be 64 processes'),
    'val_interval':         ('100',  'CRPS is the selection metric; validation integrates the ODE'),
    'test_interval':        ('500',  'periodic plots'),
    'val_flow_steps':       ('12',   'cheap integration for the periodic validation score'),
    'val_num_samples':      ('8',    'FIXED across arms -- the CRPS estimator is unbiased at any S, '
                                     'so varying it would only add noise to the comparison'),
    'flow_steps':           ('30',   'provenance only; the inference configs set the K actually used'),
    'flow_solver':          ('heun', '2nd-order trapezoid'),
    'hierarchy_cache_keep': ('True', 'REQUIRED: arms share a cache per voronoi level; '
                                     'a finishing arm must not delete it'),
    'best_by':              ('crps', 'select on the sampling metric, not the one-step regression loss'),
}

# ── inference ───────────────────────────────────────────────────────────────
INFER_SOURCES = {
    's26fe_main':    'config_infer_s26fe_main_bot.txt',
    's26fe_sec':     'config_infer_s26fe_sec_bot.txt',
    'sm_l345u_main': 'config_infer_sm_l345u_main_bot.txt',
}

# COST. Unlike the variational method, ONE DRAW IS NOT ONE FORWARD here: a draw
# integrates the ODE, so it costs flow_steps x 2 forwards under Heun. Total for
# the stage is  arms x eval sets x scenes x INFER_SAMPLES x INFER_STEPS x 2.
# At K=30 that is 60 forwards per draw, ~60x the variational method's per-draw
# cost, and it is what makes this stage the sweep's dominant expense.
#
# INFER_STEPS 12 rather than 30 is the deliberate lever: K is a sampling-time
# choice, the spread statistic (max - min over nodes) is far less sensitive to
# integration error than a per-node field, and Wave A measures on ONE checkpoint
# what K the production runs should use. If Wave A says the spread histogram
# still moves between K=12 and K=30, raise this and re-run the stage only --
# no retraining.
INFER_SAMPLES = '2000'
INFER_STEPS = '12'

INFER_OVERRIDES = {
    'num_vae_samples': (INFER_SAMPLES, 'draws per scene; the histogram carries scenes x this'),
    'flow_steps':      (INFER_STEPS,   'ODE steps per draw. Sampling-time -- see Wave A. '
                                       'Each draw costs 2x this in forwards under heun'),
    'flow_solver':     ('heun',        '2nd-order; euler halves the cost at higher tail error'),
    'save_rollouts':   ('False',       'write NO trajectory HDF5s -- scene x draws would be tens '
                                       'of thousands of files across the grid; the histogram and '
                                       'spread_values.npz still get written'),
    'make_histogram':  ('True',        'GT vs generated z_disp spread (max - min) per realization'),
    'show_histogram':  ('False',       'headless node: save the PNG, do not open a viewer'),
    'histogram_bins':  ('60',          'bins in the overlaid GT/generated histogram'),
    'vae_batch_vram_fraction': ('0.35', 'two arms share the GPU; 2 x 0.35 = the usual 0.70'),
}

TRAIN_HEADER = """%   ============================================================
%   cHI-MGNflow SAOI Wave B -- 2^(5-1) resolution-V half fraction, 16 arms.
%   Defining relation I = ABCDE: learningr is A xor B xor C xor D, not free.
%   All 5 main effects and all 10 two-factor interactions are clean.
%   GENERATED by gen_sweep_configs.py; do not hand-edit, regenerate.
%
%   ARM: {arm}
{axis_lines}
%     gpu {gpu} -- shared with {mate} (flips all four free factors, so this
%     card carries both batch and both capacity levels; VRAM stays balanced)
%
%   Everything else is ../SAOI_all_input/config_train_bot.txt.
%
%   NOT SWEPT ON PURPOSE: flow_steps and flow_solver are sampling-time choices.
%   The same checkpoint integrates at any K, so putting them here would burn
%   training runs on what inference answers for free. See docs/SWEEP_PLAN.md.
%   ============================================================
"""

INFER_HEADER = """%   ============================================================
%   cHI-MGNflow SAOI Wave B -- INFERENCE for one arm on one eval set.
%   GENERATED by gen_sweep_configs.py; do not hand-edit, regenerate.
%
%   arm        {arm}
%   eval set   {tag}   (also `eval_dataset`: it carries the true fields)
%   gpu        {gpu}   (the GPU this arm trained on)
%
%   Writes NO trajectory files. Per arm and eval set it produces:
%     histogram_compare.png   GT vs generated z_disp spread (max - min)
%     spread_values.npz       the raw gt/gen arrays behind that plot
%   score_sweep.py reads the npz to put every arm on one axis.
%   ============================================================
"""


def arms():
    """The 16 runs of the half fraction: (index, name, {key: value}, [tags])."""
    out = []
    for i in range(16):
        free = [(i >> (3 - k)) & 1 for k in range(4)]
        bits = free + [free[0] ^ free[1] ^ free[2] ^ free[3]]
        tags, values = [], {}
        for (_, levels), b in zip(FACTORS, bits):
            tag, kv = levels[b]
            tags.append(tag)
            values.update(kv)
        out.append((i, '_'.join(tags), values, tags))
    return out


def check_design(table):
    """Assert the properties that make this a usable resolution-V design."""
    tags = [t for _, _, _, t in table]
    for k, (label, lv) in enumerate(FACTORS):
        counts = [sum(1 for t in tags if t[k] == lv[b][0]) for b in (0, 1)]
        assert counts == [8, 8], (label, counts)
    for a, b in itertools.combinations(range(len(FACTORS)), 2):
        cells = {}
        for t in tags:
            cells[(t[a], t[b])] = cells.get((t[a], t[b]), 0) + 1
        assert sorted(cells.values()) == [4, 4, 4, 4], (a, b, cells)
    for i in range(8):
        for k in range(4):
            assert tags[i][k] != tags[15 - i][k], (i, k)


def _key_of(line):
    return line.split('\t')[0].split()[0] if '\t' in line else line.split()[0]


def render(base_lines, values, arm, gpu, mate):
    swept = set(values)
    overrides = dict(values)
    overrides.update({k: v for k, (v, _) in FIXED_TRAIN.items()})
    notes = {k: n for k, (_, n) in FIXED_TRAIN.items()}
    seen, out, skip_pct = set(), [], False
    for line in base_lines:
        if skip_pct and line.startswith('%'):
            continue
        skip_pct = False
        if line.startswith('%') or not line.strip():
            out.append(line)
            continue
        key = _key_of(line)
        if key in overrides:
            seen.add(key)
            note = notes.get(key, 'SWEPT AXIS' if key in swept else '')
            out.append(f"{key}\t{overrides[key]}  # {note}" if note
                       else f"{key}\t{overrides[key]}")
            skip_pct = key in swept
            continue
        if key == 'gpu_ids':
            out.append(f"gpu_ids\t{gpu}  # one GPU; {mate} shares it")
            continue
        if key == 'log_file_dir':
            out.append(f"log_file_dir\t../../output/chi-mgnflow/saoi_sweepB/{arm}.log")
            continue
        if key == 'modelpath':
            out.append(f"modelpath\t../output/chi-mgnflow/saoi_sweepB/{arm}.pth")
            continue
        out.append(line)
    # A swept key the production config never carried (flow_t_sampling) is
    # APPENDED rather than substituted. That is expected, not an error — the
    # only thing that must hold is that it ends up in the file.
    missing = [k for k in overrides if k not in seen]
    if missing:
        out += ['', '%   Sweep-only keys (absent from the production config)']
        out += [f"{k}\t{overrides[k]}  # "
                f"{'SWEPT AXIS' if k in swept else notes.get(k, '')}"
                for k in missing]
    written = {_key_of(l) for l in out if l.strip() and not l.startswith('%')}
    assert swept <= written, f"{arm}: swept keys lost: {sorted(swept - written)}"
    return '\n'.join(out)


def render_infer(src_lines, arm, tag, gpu):
    """One inference config: same eval set, this arm's checkpoint.

    The architecture comes from the checkpoint's model_config, so the swept
    training keys are NOT rewritten here -- only the run-scoped and
    sampling-time ones, which model_config does not carry.
    """
    over = {k: v for k, (v, _) in INFER_OVERRIDES.items()}
    notes = {k: n for k, (_, n) in INFER_OVERRIDES.items()}
    seen, out = set(), []
    for line in src_lines:
        if line.startswith('%') or not line.strip():
            out.append(line)
            continue
        key = _key_of(line)
        if key in over:
            seen.add(key)
            out.append(f"{key}\t{over[key]}  # {notes[key]}")
            continue
        if key == 'gpu_ids':
            out.append(f"gpu_ids\t{gpu}  # the GPU this arm trained on")
            continue
        if key == 'modelpath':
            out.append(f"modelpath\t../output/chi-mgnflow/saoi_sweepB/{arm}.pth")
            continue
        if key == 'log_file_dir':
            out.append(f"log_file_dir\t../../output/chi-mgnflow/saoi_sweepB/"
                       f"{arm}.infer_{tag}.log")
            continue
        if key == 'inference_output_dir':
            out.append(f"inference_output_dir\t../output/chi-mgnflow/saoi_sweepB/"
                       f"infer/{arm}/{tag}")
            continue
        out.append(line)
    missing = [k for k in over if k not in seen]
    if missing:
        out += ['', '%   Sweep-only keys (absent from the production infer config)']
        out += [f"{k}\t{over[k]}  # {notes[k]}" for k in missing]
    return '\n'.join(out)


def main():
    base_lines = BASE.read_text(encoding='utf-8').split('\n')
    while base_lines and base_lines[0].startswith('%'):
        base_lines.pop(0)
    infer_src = {}
    for tag, src in INFER_SOURCES.items():
        lines = (PROD / src).read_text(encoding='utf-8').split('\n')
        while lines and lines[0].startswith('%'):
            lines.pop(0)
        infer_src[tag] = lines

    table = arms()
    check_design(table)
    by_index = {i: name for i, name, _, _ in table}
    for i, arm, values, tags in table:
        gpu = min(i, 15 - i)
        mate = by_index[15 - i]
        axis_lines = []
        for (label, levels), tag in zip(FACTORS, tags):
            kv = dict(levels)[tag]
            axis_lines.append(f"%     {label:<18}{tag:<5}{TAG_NOTE[tag]}")
            for k, v in kv.items():
                axis_lines.append(f"%     {'':<23}{k} {v}")
        (HERE / f'{TRAIN_PREFIX}{arm}.txt').write_text(
            TRAIN_HEADER.format(arm=arm, gpu=gpu, mate=mate,
                                axis_lines='\n'.join(axis_lines))
            + render(base_lines, values, arm, gpu, mate), encoding='utf-8')
        for tag in INFER_SOURCES:
            (HERE / f'{INFER_PREFIX}{arm}_{tag}.txt').write_text(
                INFER_HEADER.format(arm=arm, tag=tag, gpu=gpu)
                + render_infer(infer_src[tag], arm, tag, gpu), encoding='utf-8')
        print(f"  gpu {gpu}  {arm}  (+{len(INFER_SOURCES)} inference configs)")

    print(f"\n{len(table)} training + {len(table) * len(INFER_SOURCES)} inference "
          f"configs written to {HERE}")
    print("design checks passed: 8v8 on all 5 main effects, "
          "4/4/4/4 on all 10 factor pairs, VRAM-balanced GPU pairs")
    print("ARMS=\"" + ' '.join(n for _, n, _, _ in table) + "\"")
    print("INFER_TAGS=\"" + ' '.join(INFER_SOURCES) + "\"")
    c1k = [n for _, n, _, t in table if t[2] == 'c1k']
    print("WARM_ARMS=\"" + f"{c1k[0]} " +
          next(n for _, n, _, t in table if t[2] == 'c2k') +
          "\"   # one per voronoi level: the two caches must both be built")


if __name__ == '__main__':
    main()
