"""Render a sweep's configs from a method's production profile.

Both SAOI sweeps derive every arm from `SAOI_all_input/config_train_<half>.txt`
and that folder's `config_infer_<tag>_<half>.txt`, with a small override table
on top. Keeping the derivation here means a change to the production profile
reaches both sweeps by regenerating, and the two trees cannot drift in how they
build a path.

WHAT THE RENDERER GUARANTEES, because each has cost a run before:

  * `%` comment blocks that describe a value are dropped when that value is
    overridden, so a config never documents the opposite of what it sets.
  * `/dataset/SAOI/` is forced to that exact case. A lowercase `saoi/` twin
    exists on disk holding a different vintage; it opens without error and
    yields wrong numbers. Path-valued keys are exempt from the native parser's
    value lowercasing (PATH_KEYS), so the case written here is the case opened.
  * `inference_output_dir` is `<out_root>/infer/<arm>/<tag>`. rank_arms.py and
    write_report.py read the arm and eval set from those two path components;
    any other layout makes the scorer silently group the wrong dumps.
  * an override whose key is absent from the production profile is appended
    rather than dropped, so a new key cannot be lost in silence.
"""
import pathlib
import re

NL = chr(10)
SAOI_DIR = re.compile(r'/dataset/SAOI/', re.IGNORECASE)


def _key_of(line):
    if '\t' in line:
        return line.split('\t')[0].strip()
    parts = line.split()
    return parts[0] if parts else ''


def render(prod_lines, overrides, arm, gpu, out_root, slug, mode,
           tag=None, extra_tail=None, det=False):
    """One config file's text.

    overrides : {key: (value, comment)} applied on top of the production file
    mode      : 'train' or 'infer' -- decides which run-scoped paths are set
    tag       : eval-set name, for inference
    det       : the deterministic-control variant of an inference config. Its
                dumps go to a separate `infer/det/` subtree, which rank_arms.py
                recognises as a different run mode -- one draw per scene is a
                bias check, not a distribution, and must never be pooled with
                the stochastic ensemble of the same arm.
    """
    out, skip_pct, seen = [], False, set()
    for line in prod_lines:
        if skip_pct and line.startswith('%'):
            continue
        skip_pct = False
        if line.startswith('%') or not line.strip():
            out.append(line)
            continue
        key = _key_of(line)

        if key in overrides:
            seen.add(key)
            value, note = overrides[key]
            out.append(f"{key}\t{value}  # {note}" if note else f"{key}\t{value}")
            skip_pct = True           # its comment described the old value
            continue
        if key == 'gpu_ids':
            out.append(f"gpu_ids\t{gpu}  # one arm per card, nothing shared")
            continue
        if key == 'log_file_dir':
            if mode == 'train':
                name = f"{arm}.log"
            else:
                name = f"{arm}.infer_{tag}{'_det' if det else ''}.log"
            out.append(f"log_file_dir\t{out_root}/{name}")
            continue
        if key == 'modelpath':
            out.append(f"modelpath\t{out_root}/{arm}.pth")
            continue
        if key == 'inference_output_dir':
            # rank_arms.py / write_report.py read <arm>/<eval set> off this path,
            # and a `det/` component before them marks the deterministic run.
            sub = 'infer/det' if det else 'infer'
            out.append(f"inference_output_dir\t{out_root}/{sub}/{arm}/{tag}")
            continue
        if key in ('dataset_dir', 'infer_dataset', 'eval_dataset'):
            out.append(SAOI_DIR.sub('/dataset/SAOI/', line))
            continue
        out.append(line)

    missing = [k for k in overrides if k not in seen]
    if missing:
        out += ['', '%   Sweep-only keys (absent from the production profile)']
        for k in missing:
            value, note = overrides[k]
            out.append(f"{k}\t{value}  # {note}" if note else f"{k}\t{value}")
    if extra_tail:
        out += ['', *extra_tail]
    return NL.join(out).rstrip(NL) + NL


def strip_banner(path):
    lines = pathlib.Path(path).read_text(encoding='utf-8').split(NL)
    while lines and lines[0].startswith('%'):
        lines.pop(0)
    return lines


def write_sweep(spec):
    """Generate every config of one sweep and print the roster."""
    here = pathlib.Path(spec['here'])
    prod = pathlib.Path(spec['prod_dir'])
    out_root = spec['out_root']
    banner = spec['banner']

    n_train = n_infer = 0
    for arm, half, over, note in spec['arms']:
        gpu = spec['gpu_of'](arm)
        train_over = dict(spec['train_fixed'])
        train_over.update(over)
        head = banner.format(arm=arm, half=half, gpu=gpu, note=note,
                             axis='\n'.join(f"%     {k:<22}{v[0]}" for k, v in over.items()))
        (here / f"config_train_{arm}.txt").write_text(
            head + render(strip_banner(prod / f"config_train_{half}.txt"),
                          train_over, arm, gpu, out_root, spec['slug'], 'train'),
            encoding='utf-8', newline='\n')
        n_train += 1

        for tag, src in spec['infer_sources'].items():
            infer_over = dict(spec['infer_fixed'])
            # architecture keys must mirror training or the checkpoint will not
            # load; rollout.py overrides them from model_config anyway, but a
            # config that contradicts the checkpoint misdocuments the run.
            infer_over.update({k: v for k, v in over.items()
                               if k in spec['arch_keys']})
            src_lines = strip_banner(prod / src.format(half=half))
            (here / f"config_infer_{arm}_{tag}.txt").write_text(
                spec['infer_banner'].format(arm=arm, half=half, tag=tag, gpu=gpu,
                                            note=note)
                + render(src_lines, infer_over, arm, gpu, out_root, spec['slug'],
                         'infer', tag=tag),
                encoding='utf-8', newline='\n')
            n_infer += 1

            # Deterministic control: one draw per scene with the sampling noise
            # off. It is what separates a shrunk conditional MEAN (a regression
            # failure no width fix can repair) from an over-tight ensemble (a
            # sampler failure). Costs one forward per scene against the
            # stochastic run's thousands, so it always ships alongside.
            if spec.get('det_fixed'):
                det_over = dict(infer_over)
                det_over.update(spec['det_fixed'])
                (here / f"config_infer_{arm}_{tag}_det.txt").write_text(
                    spec['infer_banner'].format(
                        arm=arm, half=half, gpu=gpu, note=note,
                        tag=f"{tag} -- DETERMINISTIC CONTROL, one draw, no noise")
                    + render(src_lines, det_over, arm, gpu, out_root, spec['slug'],
                             'infer', tag=tag, det=True),
                    encoding='utf-8', newline='\n')
                n_infer += 1

    print(spec['title'])
    for arm, half, over, note in spec['arms']:
        axis = '  '.join(f"{k} {v[0]}" for k, v in over.items())
        print(f"  arm {arm:<3} gpu {spec['gpu_of'](arm):<3} {half:<4} {axis:<44}{note}")
    print(f"  {n_train} training + {n_infer} inference configs -> {here}")
    print('ARMS="' + ' '.join(a for a, _, _, _ in spec['arms']) + '"')
    print('INFER_TAGS="' + ' '.join(spec['infer_sources']) + '"')
