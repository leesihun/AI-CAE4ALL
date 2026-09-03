#!/usr/bin/env python3


# Values for these keys are filesystem paths and keep their original case;
# every other string value is still lowercased (see parse_value). Mirrored in
# cae_suite/config_parser.py::PATH_KEYS -- keep the two in sync. The vae_/lc_
# log prefixes are stripped by train_pipeline.build_stage_config *after*
# parsing, so both the prefixed and bare spellings are listed.
PATH_KEYS = frozenset({
    'dataset_dir',
    'param_dir',
    'output_dir',
    'init_vae_modelpath',
    'vae_modelpath',
    'lc_modelpath',
    'log_file_dir',
    'vae_log_file_dir',
    'lc_log_file_dir',
    'pipeline_log_file',
})

REMOVED_NOOP_KEYS = frozenset({'load_all', 'plot_mode', 'recon_iter'})


def load_config(config_path):
    """Load configuration from a key/value text file (MeshGraphNets convention)."""
    config = {}

    print(f"Loading configuration from {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith('%'):
                continue

            # Handle comments
            if '#' in line:
                line = line.split('#')[0].strip()

            if not line:
                continue

            # Handle both tab and space separation
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()

            if len(parts) >= 2:
                key = parts[0].strip().lower()
                value = ' '.join(parts[1:]).strip()

                if key == 'reserved':
                    continue

                parsed = parse_value(value, preserve_case=key in PATH_KEYS)
                config[key] = parsed

    removed = sorted(REMOVED_NOOP_KEYS.intersection(config))
    if removed:
        raise ValueError(
            "Retired standalone/pickle settings have no HDF5-native runtime effect: "
            + ", ".join(removed)
        )

    print(f"Configuration loaded with {len(config)} parameters")

    return config


def parse_value(value_str, preserve_case=False):
    """Parse string value to appropriate type

    preserve_case (set for PATH_KEYS) skips only the string-lowercasing; the
    bool/int/float/list typing is identical either way.
    """
    value_str = value_str.strip()

    def _text(part):
        return part if preserve_case else part.lower()

    # Handle comma-separated values (e.g., gpu_ids, cond_values)
    if ',' in value_str:
        parts = [part.strip() for part in value_str.split(',')]
        try:
            return [int(part) if '.' not in part else float(part) for part in parts]
        except ValueError:
            return [_text(part) for part in parts]

    # Handle arrays (space-separated values)
    if ' ' in value_str:
        parts = value_str.split()
        if len(parts) > 1:
            try:
                return [int(part) if '.' not in part else float(part) for part in parts]
            except ValueError:
                return [_text(part) for part in parts]

    # Handle boolean values
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'

    # Handle numeric values
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return _text(value_str)
