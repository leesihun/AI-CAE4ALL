"""Config keys renamed on 2026-10-22, and the two places that must know.

A config FILE that still spells an old key is refused (`check_config_file`): the
native parsers would otherwise drop it and fall back to the default, e.g. one
inference draw instead of sixteen, or displaced geometry instead of fixed.
A CHECKPOINT saved before the rename carries the old names in `model_config`, so
`migrate_checkpoint_config` translates them; the checkpoint stays the source of
truth, so a translated key replaces the config file's value.
"""

RENAMED_CONFIG_KEYS = {
    'geometry_state_mode': 'use_deformed_edge (True = displacement, False = fixed)',
    'displacement_state_indices': 'displacement_indices',
    'vae_mp_layers': 'vae_encoder_mp_layers',
    'num_vae_samples': 'inference_samples',
    'vae_batch_size': 'inference_batch_size',
    'vae_batch_size_min': 'inference_batch_size_min',
    'vae_batch_size_max': 'inference_batch_size_max',
    'vae_batch_vram_fraction': 'inference_batch_vram_fraction',
}


def migrate_checkpoint_config(model_config):
    """Return a copy of a checkpoint `model_config` with the renamed keys translated."""
    out = dict(model_config)
    if 'geometry_state_mode' in out:
        out['use_deformed_edge'] = str(out.pop('geometry_state_mode')).lower() != 'fixed'
    for old, new in RENAMED_CONFIG_KEYS.items():
        if old in out:
            out[new.split(' ')[0]] = out.pop(old)
    return out


def check_config_file(config, source='config'):
    """Raise on a config FILE that still uses a renamed key (never on a checkpoint)."""
    stale = sorted(k for k in RENAMED_CONFIG_KEYS if k in config)
    if stale:
        raise ValueError(f"{source} uses renamed key(s): "
                         + "; ".join(f"{k} -> {RENAMED_CONFIG_KEYS[k]}" for k in stale))
