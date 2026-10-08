"""designs.h5: every screened design of an optimize run, shape plus fields.

`screening.csv` gives each design's numbers; this file is what the numbers were
read off. It uses the shared mesh HDF5 layout (`data/{id}/nodal_data`
[F, 1, N] + `mesh_edge` [2, E], `metadata/feature_names`), so any mesh viewer
in the suite -- the Studio's sample viewer above all -- opens it as is.

- One sample per analysed design, keyed by its screening.csv id (`screen_0000`),
  so a table row and its shape meet by name.
- Rows: x, y, z (mm), von Mises (MPa), |u| (mm), u_z, u_x, u_y (mm).
- The fields are the vertical load case when it was analysed, else the worst
  case; the group attribute `load_case` names which.
- HI-MGN runs store the surrogate's prediction on its own surface nodes
  (`source` "surrogate"). FEA runs store the solve, restricted to the boundary
  nodes of the tet mesh (`source` "fea").
- Scalar group attributes (mass, peak stress, u_z, feasible) are the design's
  screening.csv numbers. The viewer lists them as the sample's parameters.
"""

import os

import numpy as np

DESIGNS_FILE = 'designs.h5'
FEATURE_NAMES = ('x_coord', 'y_coord', 'z_coord', 'von_mises', 'disp_mag', 'u_z', 'u_x', 'u_y')


def _pick_case(names, worst=None):
    for name in ('vertical', 'ver'):
        if name in names:
            return name
    return worst if worst in names else next(iter(names))


def _edges_from_faces(faces):
    """Both directions of every triangle edge, as the mesh contract stores them."""
    faces = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return np.concatenate([pairs, pairs[:, ::-1]]).T


class DesignFieldsWriter:
    """Append designs to `<out_dir>/designs.h5`, one group per design."""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, DESIGNS_FILE)
        if os.path.isfile(self.path):
            os.remove(self.path)
        self.count = 0
        self.failures = 0

    def _write(self, design_id, coords_mm, vm_mpa, u_mm, edges, attrs,
               disp_mag=None, u_z=None):
        import h5py

        n = coords_mm.shape[0]
        u = np.full((n, 3), np.nan) if u_mm is None else np.asarray(u_mm, dtype=np.float64)
        rows = np.stack([coords_mm[:, 0], coords_mm[:, 1], coords_mm[:, 2], vm_mpa,
                         np.linalg.norm(u, axis=1) if disp_mag is None else disp_mag,
                         u[:, 2] if u_z is None else u_z, u[:, 0], u[:, 1]])
        with h5py.File(self.path, 'a') as f:
            if 'metadata' not in f:
                f.create_group('metadata').create_dataset(
                    'feature_names', data=np.array(FEATURE_NAMES, dtype='S'))
                f.attrs['num_features'] = len(FEATURE_NAMES)
                f.attrs['num_timesteps'] = 1
            group = f.require_group('data').create_group(design_id)
            group.create_dataset('nodal_data', data=rows[:, None, :].astype(np.float32),
                                 compression='gzip', compression_opts=4)
            group.create_dataset('mesh_edge', data=np.asarray(edges, dtype=np.int32),
                                 compression='gzip', compression_opts=4)
            for key, value in attrs.items():
                if value is not None:
                    group.attrs[key] = value
            self.count += 1
            f.attrs['num_samples'] = self.count

    def annotate(self, design_id, attrs):
        """Set scalar attributes on a design already written (scores come later)."""
        import h5py

        if not os.path.isfile(self.path):
            return
        with h5py.File(self.path, 'a') as f:
            group = f.get(f'data/{design_id}')
            if group is None:
                return
            for key, value in attrs.items():
                if value is not None:
                    group.attrs[key] = value

    def add_fea(self, design_id, fields, length_scale, attrs):
        """A solved FEA record's `fields` (Bracket.analyze, return_fields=True)."""
        case = _pick_case(list(fields['cases']), fields.get('worst_case'))
        faces = np.asarray(fields['faces'])
        surface, local = np.unique(faces, return_inverse=True)
        coords = np.asarray(fields['nodes_norm'])[surface] * length_scale * 1e3
        solved = fields['cases'][case]
        self._write(design_id, coords,
                    np.asarray(solved['von_mises_nodal'])[surface] / 1e6,
                    np.asarray(solved['displacement'])[surface] * 1e3,
                    _edges_from_faces(local.reshape(faces.shape)),
                    {**attrs, 'source': 'fea', 'load_case': case})

    def add_surrogate(self, design_id, rollouts, layout, attrs):
        """A surrogate record's rollout file(s), {case: path}, as HI-MGN wrote them."""
        import h5py

        rollouts = {case: path for case, path in (rollouts or {}).items()
                    if path and os.path.isfile(path)}
        if not rollouts:
            self.failures += 1
            return
        case = _pick_case(list(rollouts))
        with h5py.File(rollouts[case], 'r') as f:
            sample = f['data'][next(iter(f['data']))]
            pred = np.asarray(sample['nodal_data'][:, -1, :], dtype=np.float64)
            edges = np.asarray(sample['mesh_edge'][:])
        coords = pred[0:3].T
        attrs = {**attrs, 'source': 'surrogate', 'load_case': case}
        if layout == 'ver':
            # VER_FEATURE_NAMES: u_x, u_y, u_z (mm), von Mises (MPa).
            self._write(design_id, coords, pred[6], pred[3:6].T, edges, attrs)
        else:
            # FEATURE_NAMES (build_deepjeb_mgn): von Mises, |u|, u_z; u_x/u_y are
            # not predicted and stay NaN.
            self._write(design_id, coords, pred[3], None, edges, attrs, disp_mag=pred[4],
                        u_z=pred[5] if pred.shape[0] > 5 else None)


def screening_attrs(record, limits):
    """The scalar numbers a design carries beside its fields (screening.csv units)."""
    f = record['fea']
    uz = f.get('vertical_displacement')
    penalty = record.get('penalty') or {}
    allow = limits.get('vertical_disp_allow')
    return {
        'mass_kg': float(f['mass']),
        'peak_von_mises_mpa': float(f['peak_von_mises']) / 1e6,
        'max_displacement_mm': float(f['max_displacement']) * 1e3,
        'vertical_displacement_mm': None if uz is None else float(uz) * 1e3,
        'vertical_limit_mm': None if allow is None else float(allow) * 1e3,
        'feasible': int(bool(penalty.get('feasible'))) if penalty else None,
        'score': None if record.get('score') is None else float(record['score']),
    }
