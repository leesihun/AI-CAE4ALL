"""
Condition-name registry for SDFFlow's conditional flow-matching stage.

Two families of names can appear in a dataset's condition vector:

* ``GEOMETRIC_NAMES`` -- the five descriptors ``build_dataset.py`` measures on
  the normalized mesh (longest side 1.8) and stores in every
  ``shapes/<i>/cond`` row. They are scale-free: ``bbox_y`` is exactly 1.8 for
  every DeepJEB shape and ``bbox_x`` varies by 0.45%, so only ``volume`` and
  ``area`` carry information (about 1.5 effective degrees of freedom).
* ``FEA_CONDITIONS`` -- DeepJEB's per-design FEA labels
  (``bracket_labels.csv``: mass, per-load-case max von Mises stress and max
  displacement magnitude, the first two eigenfrequencies, ...). They are
  appended to an existing HDF5 by ``add_fea_conditions.py`` as the root
  ``cond_extra`` sidecar (constants ``SIDECAR_*`` below) and merged after the
  geometric names by ``general_modules.sdf_dataset.SDFShapeDataset``.

Every FEA entry records the CSV column it comes from, the ``transform`` under
which it is STORED (``'identity'`` or natural ``'log'`` -- stress, displacement,
and frequency are right-skewed and the log makes them roughly Gaussian), its
physical unit, a ``kind`` (``mass|stress|displacement|frequency|area|volume|
inertia|cg``), and the GE bracket-challenge ``load_case`` it belongs to
(``vertical|horizontal|diagonal|torsion``; ``None`` for global scalars). The
load-case names match ``design_loop/problem.py::LOAD_CASES`` so a condition
can be re-measured on a decoded mesh.

``to_stored`` / ``from_stored`` apply and invert the transform; both are the
identity for geometric names. Conditions in an FM checkpoint, ``cond_values``
in a sample config, and the dataset's ``cond`` rows are all in STORED space;
convert with ``from_stored`` before quoting a value in MPa / mm / Hz.

All names are lowercase.
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Sidecar layout (HDF5 root). Append-only and backward compatible: a file
# without SIDECAR_DATASET is read exactly as before.
# ---------------------------------------------------------------------------

SIDECAR_DATASET = 'cond_extra'                 # float32 [num_shapes, k]; row i = shape i
SIDECAR_NAMES_ATTR = 'cond_extra_names'        # array of str, length k
SIDECAR_SOURCE_ATTR = 'cond_extra_source'      # csv path the values came from
SIDECAR_TRANSFORMS_ATTR = 'cond_extra_transforms'  # JSON {name: transform}
SIDECAR_CREATED_ATTR = 'cond_extra_created'    # ISO-8601 timestamp
# Optional provenance written by add_fea_conditions.py (not required by readers):
SIDECAR_CSV_COLUMNS_ATTR = 'cond_extra_csv_columns'   # JSON {name: csv column}
SIDECAR_MISSING_ATTR = 'cond_extra_missing_count'     # int: rows written as NaN
SIDECAR_MISSING_ROWS_ATTR = 'cond_extra_missing_rows'  # int32 shape indices of those rows

# ---------------------------------------------------------------------------
# Geometric descriptors (build_dataset.py / sdf_sampling.mesh_descriptors)
# ---------------------------------------------------------------------------

GEOMETRIC_NAMES = ('bbox_x', 'bbox_y', 'bbox_z', 'volume', 'area')

TRANSFORMS = ('identity', 'log')
KINDS = ('mass', 'stress', 'displacement', 'frequency', 'area', 'volume', 'inertia', 'cg')
LOAD_CASES = ('vertical', 'horizontal', 'diagonal', 'torsion')

# CSV short tag -> design_loop/problem.py load-case name.
_LOAD_CASE_TAGS = (('ver', 'vertical'), ('hor', 'horizontal'),
                   ('dia', 'diagonal'), ('tor', 'torsion'))


def _entry(csv_column, transform, unit, kind, load_case=None):
    assert transform in TRANSFORMS, transform
    assert kind in KINDS, kind
    assert load_case is None or load_case in LOAD_CASES, load_case
    return {'csv_column': csv_column, 'transform': transform, 'unit': unit,
            'kind': kind, 'load_case': load_case}


# ---------------------------------------------------------------------------
# FEA label registry (DeepJEB bracket_labels.csv)
# ---------------------------------------------------------------------------

FEA_CONDITIONS = {}

# Global scalars (load-case independent).
FEA_CONDITIONS['mass_kg'] = _entry('mass(kg)', 'identity', 'kg', 'mass')
FEA_CONDITIONS['log_mass_kg'] = _entry('mass(kg)', 'log', 'kg', 'mass')
FEA_CONDITIONS['volume_mm3'] = _entry('volume(mm3)', 'identity', 'mm3', 'volume')
FEA_CONDITIONS['surface_area_mm2'] = _entry('surface_area(mm2)', 'identity', 'mm2', 'area')
for _axis in 'xyz':
    FEA_CONDITIONS[f'cg_{_axis}_mm'] = _entry(f'CG_{_axis}(mm)', 'identity', 'mm', 'cg')
for _k in (1, 2, 3):
    FEA_CONDITIONS[f'log_i_{_k}_kgmm2'] = _entry(f'I_{_k}(kg*mm2)', 'log', 'kg*mm2', 'inertia')

# Per load case: max von Mises stress and max displacement magnitude, log-stored.
for _tag, _case in _LOAD_CASE_TAGS:
    FEA_CONDITIONS[f'log_max_{_tag}_stress_mpa'] = _entry(
        f'max_{_tag}_stress(MPa)', 'log', 'MPa', 'stress', _case)
    FEA_CONDITIONS[f'log_max_{_tag}_magdisp_mm'] = _entry(
        f'abs_max_{_tag}_magdisp(mm)', 'log', 'mm', 'displacement', _case)

# Eigenfrequencies.
FEA_CONDITIONS['log_first_mode_freq_hz'] = _entry('1st_mode_freq(Hz)', 'log', 'Hz', 'frequency')
FEA_CONDITIONS['log_second_mode_freq_hz'] = _entry('2nd_mode_freq(Hz)', 'log', 'Hz', 'frequency')

del _axis, _k, _tag, _case

assert all(name == name.lower() for name in FEA_CONDITIONS)
assert not set(FEA_CONDITIONS) & set(GEOMETRIC_NAMES)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def normalize_name(name):
    """Config/CSV spelling -> registry spelling (stripped, lowercase)."""
    return str(name).strip().lower()


def is_geometric(name):
    return normalize_name(name) in GEOMETRIC_NAMES


def is_fea(name):
    return normalize_name(name) in FEA_CONDITIONS


def is_known(name):
    return is_geometric(name) or is_fea(name)


def all_fea_names():
    """Registry order, as a list (the builder's default ``--names``)."""
    return list(FEA_CONDITIONS.keys())


def describe(name):
    """The FEA registry entry for ``name`` (a copy, with ``'name'`` added), or
    ``None`` for geometric and unknown names.

    Use ``is_geometric`` for the five descriptors; ``describe(...)['load_case']``
    is the key into ``design_loop/problem.py::LOAD_CASES`` when re-measuring an
    FEA condition on a decoded mesh (``None`` for mass/CG/inertia/frequency).
    """
    key = normalize_name(name)
    entry = FEA_CONDITIONS.get(key)
    if entry is None:
        return None
    out = dict(entry)
    out['name'] = key
    return out


def transform_of(name):
    """``'identity'`` or ``'log'``; geometric names are identity. Raises
    ValueError for a name in neither registry."""
    key = normalize_name(name)
    if key in GEOMETRIC_NAMES:
        return 'identity'
    entry = FEA_CONDITIONS.get(key)
    if entry is None:
        raise ValueError(
            f'Unknown condition name {name!r}; geometric names are {list(GEOMETRIC_NAMES)} '
            f'and FEA names are {all_fea_names()}')
    return entry['transform']


def _as_float_array(value):
    return np.asarray(value, dtype=np.float64)


def _return_like(result, value):
    return float(result) if np.ndim(value) == 0 else result


def to_stored(name, raw_value):
    """Raw (physical-unit) value -> stored condition value.

    ``log`` names return ``ln(raw)``; a non-positive raw value maps to NaN (no
    exception) so the caller can count and report it. Scalars in, ``float``
    out; arrays in, float64 arrays out. Identity for geometric names.
    """
    value = _as_float_array(raw_value)
    if transform_of(name) == 'log':
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(value > 0, np.log(np.where(value > 0, value, 1.0)), np.nan)
    else:
        result = value
    return _return_like(result, raw_value)


def from_stored(name, stored_value):
    """Inverse of ``to_stored``: ``exp`` for log names, identity otherwise."""
    value = _as_float_array(stored_value)
    if transform_of(name) == 'log':
        result = np.exp(value)
    else:
        result = value
    return _return_like(result, stored_value)


def unit_of(name):
    """Physical unit string for an FEA name, ``'normalized'`` for geometric."""
    key = normalize_name(name)
    if key in GEOMETRIC_NAMES:
        return 'normalized'
    return FEA_CONDITIONS[key]['unit'] if key in FEA_CONDITIONS else None


# ---------------------------------------------------------------------------
# Join key between the HDF5 and the CSV
# ---------------------------------------------------------------------------

def item_name_from_source(source):
    """CSV ``item_name`` of a shape from its builder ``source`` attr.

    ``'DeepJEB\\\\SurfaceMesh\\\\101_428.stl' -> '101_428'``: the basename
    without its extension, both path separators accepted (the DeepJEB file was
    built on Windows and stores backslashes). Synthetic shapes
    (``source='synthetic_7'``) map to themselves. Returns ``None`` when the
    source is missing or empty.
    """
    if source is None:
        return None
    if isinstance(source, bytes):
        source = source.decode('utf-8', 'replace')
    name = str(source).replace('\\', '/').rsplit('/', 1)[-1].strip()
    if not name:
        return None
    stem = os.path.splitext(name)[0]
    return stem or None
