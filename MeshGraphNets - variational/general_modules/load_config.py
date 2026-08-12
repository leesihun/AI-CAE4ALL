#!/usr/bin/env python3

# Values for these keys are filesystem paths and keep their original case;
# every other string value is still lowercased (see parse_value). Mirrored in
# cae_suite/config_parser.py::PATH_KEYS -- keep the two in sync.
PATH_KEYS = frozenset({
    'dataset_dir',
    'infer_dataset',
    'modelpath',
    'inference_output_dir',
    'hierarchy_cache_dir',
    'log_file_dir',
})

def load_config(config_path):
    """Load configuration config.txt"""
    config = {}

    print(f"Loading configuration from {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
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

                config[key] = parse_value(value, preserve_case=key in PATH_KEYS)

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

    # Handle comma-separated values (e.g., gpu_ids)
    if ',' in value_str:
        parts = [part.strip() for part in value_str.split(',')]
        try:
            # Try to parse as array of numbers
            return [int(part) if '.' not in part else float(part) for part in parts]
        except ValueError:
            # Return as array of strings (lowercase unless a path key)
            return [_text(part) for part in parts]

    # Handle arrays (space-separated values)
    if ' ' in value_str:
        parts = value_str.split()
        if len(parts) > 1:
            try:
                # Try to parse as array of numbers
                return [int(part) if '.' not in part else float(part) for part in parts]
            except ValueError:
                # Return as array of strings (lowercase unless a path key)
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
        # Return string value as lowercase (path keys keep their case)
        return _text(value_str)

