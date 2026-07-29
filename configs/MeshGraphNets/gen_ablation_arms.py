"""Generate the p1/p2/p12 arm configs from each ex's ablation baseline.

Deriving them mechanically guarantees the arms differ from the baseline ONLY
by the feature block appended near the end -- which is the entire premise of
the comparison. Hand-editing four near-identical files invites exactly the
silent drift the study is trying to measure against.

Re-run after editing either config_train_himgn_base.txt:
    python gen_arms.py
"""
import re
from pathlib import Path

CFG = Path(__file__).resolve().parent

ARM_HEADER = """%   ABLATION ARM {arm} for the transfer-operator / multi-partition study.
%
%   GENERATED from config_train_himgn_base.txt -- edit that file and re-run the
%   generator rather than editing this one, so the arms cannot drift apart.
%   This file is the baseline verbatim plus the single feature block near the
%   bottom (marked "ARM DIFFERENCE"). Read the baseline for why the shared
%   settings are what they are, including which time_integration this ex uses.
%   See MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md."""

P1_BLOCK = """'
%   ==== ARM DIFFERENCE: Part I -- learned inter-level transfer operators ====
%   Both reduce EXACTLY to the baseline's fixed mean-pool / sum-unpool at
%   initialization (zero-init score heads), so this run starts from the
%   baseline model and departs only if the gradient says to.
%   {param_note}
pool_type\tattention   # mean (baseline) | attention
unpool_type\tattention   # sum (baseline) | attention
pool_heads\t4           # latent_dim 128 / 4 heads = 32 dims per head
"""

P2_BLOCK = """'
%   ==== ARM DIFFERENCE: Part II -- multi-partition coarse representation ====
%   4 parallel Voronoi partitions of the coarsest (100-cluster) level instead
%   of 1. Level 0 stays single-partition; only the LAST level may branch.
%   Measured on ex2 sample 0 at k=100: 4 partitions carry ~32% more field
%   information than 1 (broadcast-recon RMSE 0.4831 -> 0.3277) while the
%   coarse-graph diameter only goes 5 -> 6. K=4 over K=8/16 because it is the
%   cheapest setting with a real gain, so a win is hard to dismiss as capacity.
%   {param_note}
%   NOTE: changes the hierarchy-cache signature -> a separate .mscache.*.h5 is
%   built on first run (shared with the _p12 arm).
voronoi_branches\t1, 4
"""

P12_BLOCK = """'
%   ==== ARM DIFFERENCE: Part I + Part II together ====
%   Lowest priority of the four arms -- only interpretable once you know what
%   each feature does alone. But this is what Part III predicts should win:
%   multi-partition supplies information a single partition cannot carry at any
%   head count, and attention is the mechanism that extracts it.
%   {param_note}
%   Shares the _p2 arm's hierarchy cache (same voronoi_branches signature).
pool_type\tattention   # mean (baseline) | attention
unpool_type\tattention   # sum (baseline) | attention
pool_heads\t4           # latent_dim 128 / 4 heads = 32 dims per head
voronoi_branches\t1, 4    # 4 partitions of the coarsest level; level 0 unbranched
"""

ARMS = {'p1': P1_BLOCK, 'p2': P2_BLOCK, 'p12': P12_BLOCK}

# One GPU per config on an 8-GPU node, so all ten runs can be launched at once.
# `orig` and `base` deliberately share a GPU (they are the two reference runs,
# and 5 configs x 2 datasets does not fit 8 lanes otherwise); run_ablation.sh
# runs those two sequentially within their shared lane rather than stacking
# them on the same device.
GPU = {
    'ex1': {'orig': 0, 'base': 0, 'p1': 1, 'p2': 2, 'p12': 3},
    'ex2': {'orig': 4, 'base': 4, 'p1': 5, 'p2': 6, 'p12': 7},
}


def _set_gpu(text, gpu, label):
    """Rewrite the gpu_ids line (single value = single GPU, no DDP)."""
    out, found = [], False
    for line in text.split('\n'):
        if line.split('\t')[0].strip().lower() == 'gpu_ids' or line.startswith('gpu_ids'):
            out.append(f'gpu_ids\t{gpu}  # {label}')
            found = True
        else:
            out.append(line)
    assert found, 'no gpu_ids line to rewrite'
    return '\n'.join(out)

# Measured with model/MeshGraphNets.py at each ex's own mp_per_level.
PARAMS = {
    'ex1': {'base': 4_667_652, 'p1': 4_835_598, 'p2': 4_716_804, 'p12': 4_884_750},
    'ex2': {'base': 3_164_292, 'p1': 3_332_238, 'p2': 3_213_444, 'p12': 3_381_390},
}


def _swap_header(text, arm):
    """Replace the leading `%` comment block (the run just above dataset_dir)."""
    lines = text.split('\n')
    end = next(i for i, l in enumerate(lines) if l.startswith('dataset_dir'))
    start = end
    while start > 0 and lines[start - 1].lstrip().startswith('%'):
        start -= 1
    assert start < end, 'no leading comment block found above dataset_dir'
    return '\n'.join(lines[:start] + ARM_HEADER.format(arm=arm).split('\n') + lines[end:])


def build(ex):
    base_path = CFG / ex / 'config_train_himgn_base.txt'
    base = base_path.read_text(encoding='utf-8')

    # The time_integration block is always last; arms insert their feature
    # block before it so the file still ends on the scheme selector.
    marker = "'\n%   Time integration"
    assert marker in base, f'{base_path}: expected a trailing time-integration block'
    head, tail = base.split(marker, 1)
    tail = marker + tail

    for arm, block in ARMS.items():
        p = PARAMS[ex]
        note = (f'+{100 * (p[arm] / p["base"] - 1):.2f}% parameters vs baseline '
                f'({p[arm]:,} vs {p["base"]:,}).')
        out = head + block.format(param_note=note) + tail
        out = out.replace('train_himgn_base.log', f'train_himgn_{arm}.log')
        out = out.replace('model_himgn_base.pth', f'model_himgn_{arm}.pth')
        out = _swap_header(out, arm)
        out = _set_gpu(out, GPU[ex][arm], f'ablation arm {arm} ({ex}) -- one GPU, no DDP')

        assert 'ABLATION BASELINE' not in out, f'{ex}/{arm}: stale BASELINE label'
        assert f'train_himgn_{arm}.log' in out and f'model_himgn_{arm}.pth' in out
        assert 'ARM DIFFERENCE' in out
        # newline='' so '\n' is written verbatim: the checked-in configs are
        # LF, and Path.write_text would translate to CRLF on Windows.
        dest = CFG / ex / f'config_train_himgn_{arm}.txt'
        with open(dest, 'w', encoding='utf-8', newline='') as fh:
            fh.write(out)
        print(f'  wrote config_train_himgn_{arm}.txt')


INFER_HEADER = """%   INFERENCE for ablation arm {arm} ({ex}).
%
%   GENERATED by gen_ablation_arms.py from config_infer_himgn.txt +
%   config_train_himgn_{arm}.txt -- do not edit; re-run the generator.
%
%   Architecture keys below are informational: inference_profiles/rollout.py
%   overwrites them from the checkpoint's `model_config` at load time, which
%   is what actually determines the branch layout and pooling operators. They
%   are written here anyway so the file states what checkpoint it expects.
%   The hierarchy IS rebuilt per sample at inference exactly as in training,
%   including the K parallel partitions -- skip_projs is sized for (1 + K)
%   merge inputs, so a mismatch here does not silently degrade, it fails to
%   multiply."""

# Keys copied from the matching train arm so infer cannot disagree with it.
MIRROR_FROM_TRAIN = (
    'use_world_edges', 'use_multiscale', 'coarsening_type', 'voronoi_clusters',
    'multiscale_levels', 'mp_per_level', 'voronoi_branches',
    'pool_type', 'unpool_type', 'pool_heads', 'use_node_types',
    'positional_features', 'input_var', 'output_var', 'edge_var', 'latent_dim',
)


def _read_keys(path):
    out = {}
    for line in path.read_text(encoding='utf-8').split('\n'):
        s = line.strip()
        if not s or s.startswith('%') or s.startswith("'"):
            continue
        parts = s.split('\t', 1) if '\t' in s else s.split(None, 1)
        if len(parts) == 2:
            out[parts[0].strip().lower()] = parts[1].split('#')[0].strip()
    return out


def build_infer(ex):
    tmpl_path = CFG / ex / 'config_infer_himgn.txt'
    if not tmpl_path.exists():
        print(f'  (no {tmpl_path.name}; skipping infer configs)')
        return
    tmpl = tmpl_path.read_text(encoding='utf-8')

    for arm in ['base'] + list(ARMS):
        train = _read_keys(CFG / ex / f'config_train_himgn_{arm}.txt')
        lines = tmpl.split('\n')
        out, seen = [], set()

        for line in lines:
            stripped = line.strip()
            key = None
            if stripped and not stripped.startswith('%') and not stripped.startswith("'"):
                parts = stripped.split('\t', 1) if '\t' in stripped else stripped.split(None, 1)
                key = parts[0].strip().lower() if parts else None

            if key in MIRROR_FROM_TRAIN and key in train:
                out.append(f'{parts[0].strip()}\t{train[key]}  # mirrored from config_train_himgn_{arm}.txt')
                seen.add(key)
                continue
            if key == 'modelpath':
                out.append(f'modelpath\t../output/meshgraphnets/{ex}/model_himgn_{arm}.pth')
                continue
            if key == 'log_file_dir':
                out.append(f'log_file_dir\t../../output/meshgraphnets/{ex}/infer_himgn_{arm}.log')
                continue
            if key == 'inference_output_dir':
                out.append(f'inference_output_dir\t../output/meshgraphnets/rollout/{ex}/model_himgn_{arm}')
                continue
            out.append(line)

        # Append any arm-only keys the template never had (pool_*, branches).
        extra = [k for k in MIRROR_FROM_TRAIN if k in train and k not in seen]
        if extra:
            out.append("'")
            out.append(f'%   Arm-specific keys absent from the template (checkpoint still wins):')
            for k in extra:
                out.append(f'{k}\t{train[k]}  # mirrored from config_train_himgn_{arm}.txt')

        text = '\n'.join(out)
        # Swap the leading comment block for an arm-specific one.
        body = text.split('\n')
        end = next(i for i, l in enumerate(body) if l.startswith('dataset_dir'))
        start = end
        while start > 0 and body[start - 1].lstrip().startswith('%'):
            start -= 1
        text = '\n'.join(body[:start]
                         + INFER_HEADER.format(arm=arm, ex=ex).split('\n')
                         + body[end:])

        dest = CFG / ex / f'config_infer_himgn_{arm}.txt'
        with open(dest, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        print(f'  wrote config_infer_himgn_{arm}.txt')


def set_gpu_in_place(ex, stem, arm):
    """Pin gpu_ids on a file the generator does not otherwise own.

    The baseline and the pre-existing `config_train_himgn.txt` are hand-edited
    files, but their GPU assignment has to come from the same table as the
    generated arms or the sweep would double-book a device.
    """
    p = CFG / ex / f'{stem}.txt'
    if not p.exists():
        print(f'  (no {p.name}; skipping gpu pin)')
        return
    text = p.read_text(encoding='utf-8')
    label = f'ablation {arm} ({ex}) -- one GPU, no DDP'
    if stem == 'config_train_himgn':
        label = f'ablation reference "orig" ({ex}) -- shares GPU with base'
    new = _set_gpu(text, GPU[ex][arm], label)
    if new != text:
        with open(p, 'w', encoding='utf-8', newline='') as fh:
            fh.write(new)
        print(f'  pinned gpu_ids={GPU[ex][arm]} in {p.name}')


for ex in ('ex1', 'ex2'):
    if not (CFG / ex / 'config_train_himgn_base.txt').exists():
        print(f'{ex}: no baseline, skipping')
        continue
    print(f'{ex}:')
    set_gpu_in_place(ex, 'config_train_himgn_base', 'base')
    set_gpu_in_place(ex, 'config_train_himgn', 'orig')
    build(ex)
    build_infer(ex)
