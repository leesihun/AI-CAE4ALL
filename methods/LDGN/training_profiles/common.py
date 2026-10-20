"""Helpers shared by the two training stages and the sampler."""
import os
import random
import time

import numpy as np
import torch

from general_modules import mesh_dataset as md


def as_list(value, cast=int):
    """The config parser gives a bare scalar for one value and a list for
    several (`vae_num_layers 2` -> 2, `sample_steps 20, 11, 2` -> [20, 11, 2])."""
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(value)]


def as_float(value, default=None):
    """`1e-4` stays a string in the config parser; convert explicitly."""
    if value is None:
        return default
    return float(value)


def resolve_device(config):
    """Single-process: the first entry of `gpu_ids`; -1 (or no CUDA) -> CPU."""
    gpu_ids = config.get('gpu_ids', 0)
    if isinstance(gpu_ids, (list, tuple)):
        if len(gpu_ids) > 1:
            print(f'gpu_ids {gpu_ids}: LDGN trains and samples on one GPU; using {gpu_ids[0]}')
        gpu_ids = gpu_ids[0]
    gid = int(gpu_ids)
    if gid >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gid)
        return torch.device(f'cuda:{gid}')
    if gid >= 0:
        print('CUDA is not available; running on CPU')
    return torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_generator(config):
    """Seeds the global RNG and returns the loader's shuffle generator.

    Unset `training_seed` -> nothing is seeded and None is returned. (Upstream's
    scripts always seeded, with `--seed` defaulting to 0; the ex3 configs set
    `training_seed 42` like the MGN-V arms.)"""
    seed = config.get('training_seed')
    if seed is None:
        return None
    seed = int(seed)
    seed_everything(seed)
    return torch.Generator().manual_seed(seed)


def stage_log_path(config, stage):
    """`<stage>_log_file_dir`, else `log_file_dir` with `_<stage>` inserted."""
    explicit = config.get(f'{stage}_log_file_dir')
    if explicit:
        return os.path.normpath(str(explicit))
    base = config.get('log_file_dir')
    if not base:
        return None
    root, ext = os.path.splitext(os.path.normpath(str(base)))
    return f'{root}_{stage}{ext or ".log"}'


def init_stage_log(config, config_filename, stage, resumed_at=None):
    """Create the stage's epoch log (config embedded), like MGN-V's init_log_file.

    Records `config['log_dir']` so the periodic pictures land beside the log.
    A resumed stage appends a resume line instead of starting a new file.
    """
    path = stage_log_path(config, stage)
    if not path:
        return None
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    config['log_dir'] = directory
    if resumed_at is not None and os.path.exists(path):
        from general_modules.resume_state import resume_log_line
        with open(path, 'a', encoding='utf-8') as f:
            f.write(resume_log_line(resumed_at) + '\n')
        return path
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'Training epoch log file ({stage.upper()} stage)\n')
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f'Log file absolute path: {os.path.abspath(path)}\n')
        if config_filename and os.path.exists(config_filename):
            with open(config_filename, 'r', encoding='utf-8') as fc:
                f.write(fc.read())
        f.write('\n')
    return path


def append_log(path, line):
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def checkpoint_config(config):
    """The config as plain values (weights_only-loadable)."""
    out = {}
    for key, value in config.items():
        if str(key).startswith('_'):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
            out[key] = list(value)
    return out


def save_checkpoint(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f'Saved checkpoint: {os.path.abspath(path)}')


def load_checkpoint(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)


def check_normalization_matches(stats, config, source):
    """The LDGN stage and the sampler reuse the VAE's train statistics; the data
    keys that shape them must not have changed in between."""
    for key in ('output_var', 'cond_var'):
        want = int(config.get(key, 0) or 0)
        if int(stats[key]) != want:
            raise ValueError(f'{source} was trained with {key} {stats[key]}, config says {want}')
    want_types = bool(config.get('use_node_types', True))
    if bool(stats['use_node_types']) != want_types:
        raise ValueError(f'{source} was trained with use_node_types {stats["use_node_types"]}, '
                         f'config says {want_types}')
    if config.get('rel_pos_ratios') is not None:
        trained = list(stats.get('rel_pos_ratios') or md.REL_POS_RATIOS)
        want = md.config_rel_pos_ratios(config)
        if len(trained) != len(want) or not np.allclose(trained, want):
            raise ValueError(f'{source} was trained with rel_pos_ratios {trained}, config says {want}')
