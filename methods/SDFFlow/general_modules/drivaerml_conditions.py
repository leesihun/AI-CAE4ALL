"""
Condition-name registry for the DrivAerML dataset's per-run geometry and
aerodynamic labels.

This is deliberately a SEPARATE, independent registry from
``condition_names.py``'s ``FEA_CONDITIONS`` -- that one is DeepJEB-specific
(GE bracket-challenge load cases, ``kind`` enum tied to structural FEA). A new
dataset gets its own name registry and its own join key rather than being
squeezed into DeepJEB's, per SDFFlow/CLAUDE.md "Conditioning datasets beyond
DeepJEB". Both write the exact same ``cond_extra`` sidecar layout (the
``SIDECAR_*`` constants below are re-exported from ``condition_names`` so
``general_modules.sdf_dataset.SDFShapeDataset`` merges either one identically.

Two CSV sources, joined on run number:
    geo_parameters_all.csv  -- 16 continuous design parameters (mm / deg)
    force_mom_all.csv       -- 5 aerodynamic force/moment coefficients (cd, cl,
                                clf, clr, cs), dimensionless

All 21 values are stored with an identity transform: unlike DeepJEB's
stress/displacement/frequency labels, neither the geometry deviations (roughly
symmetric perturbations around a baseline DrivAer shape) nor the force
coefficients are strongly right-skewed, so no log transform is warranted.
"""

import os

from general_modules.condition_names import (  # noqa: F401  (re-exported)
    SIDECAR_CREATED_ATTR, SIDECAR_CSV_COLUMNS_ATTR, SIDECAR_DATASET,
    SIDECAR_MISSING_ATTR, SIDECAR_MISSING_ROWS_ATTR, SIDECAR_NAMES_ATTR,
    SIDECAR_SOURCE_ATTR, SIDECAR_TRANSFORMS_ATTR,
)

GEO_COLUMNS = (
    'Vehicle_Length', 'Vehicle_Width', 'Vehicle_Height', 'Front_Overhang',
    'Front_Planview', 'Hood_Angle', 'Approach_Angle', 'Windscreen_Angle',
    'Greenhouse_Tapering', 'Backlight_Angle', 'Decklid_Height',
    'Rearend_tapering', 'Rear_Overhang', 'Rear_Diffusor_Angle',
    'Vehicle_Ride_Height', 'Vehicle_Pitch',
)
AERO_COLUMNS = ('cd', 'cl', 'clf', 'clr', 'cs')

DRIVAERML_CONDITIONS = {}
for _col in GEO_COLUMNS:
    DRIVAERML_CONDITIONS[_col.lower()] = {
        'csv_column': _col, 'transform': 'identity', 'unit': 'mm_or_deg', 'kind': 'geometry_param'}
for _col in AERO_COLUMNS:
    DRIVAERML_CONDITIONS[_col] = {
        'csv_column': _col, 'transform': 'identity', 'unit': 'dimensionless', 'kind': 'aero_coeff'}
del _col

assert all(name == name.lower() for name in DRIVAERML_CONDITIONS)


def all_names():
    """Registry order, as a list (the builder's default ``--names``)."""
    return list(DRIVAERML_CONDITIONS.keys())


def describe(name):
    key = str(name).strip().lower()
    entry = DRIVAERML_CONDITIONS.get(key)
    if entry is None:
        return None
    out = dict(entry)
    out['name'] = key
    return out


def to_stored(name, raw_value):
    """Identity for every DrivAerML name -- kept for symmetry with
    condition_names.to_stored so the sidecar writer's code reads the same."""
    return float(raw_value)


def from_stored(name, stored_value):
    return float(stored_value)


def run_id_from_source(source):
    """Builder ``source`` attr -> DrivAerML run id string.

    ``'D:/CAE_datasets_raw/drivaerml/run_123/drivaer_123.stl' -> '123'``: the
    file stem's trailing run of digits (``drivaer_123`` -> ``123``), so it
    works whether the join key comes from the filename or, as a fallback, the
    parent ``run_<id>`` directory. Returns ``None`` when no digits are found.
    """
    if source is None:
        return None
    if isinstance(source, bytes):
        source = source.decode('utf-8', 'replace')
    name = str(source).replace('\\', '/').rstrip('/')
    stem = os.path.splitext(name.rsplit('/', 1)[-1])[0]
    digits = ''
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits:
        return digits
    parent = name.rsplit('/', 2)[-2] if '/' in name else ''
    digits = ''
    for ch in reversed(parent):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return digits or None
