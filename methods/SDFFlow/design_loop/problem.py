"""The structural problem posed on a generated DeepJEB bracket.

DeepJEB derives from the GE jet-engine-bracket challenge, and the dataset's
shapes are rigidly aligned: y is the long axis (extent 1.8 for every sample,
train-split std 0), the mounting plate is the low-z slab, and the loaded
interface is the lug that rises in +z near y = 0. A cross-shape occupancy study
over 600 samples confirmed both features are present in essentially every
shape, so a geometric rule -- not a per-shape label, which the dataset does not
carry -- picks the fixed and loaded node sets consistently.

Loads are the GE challenge cases in SI (see `Bracket`): DeepJEB's labels were
generated with them, and `fea.py` works in N, m and Pa throughout.
"""

import numpy as np

from . import fea

# Normalized-coordinate rules for the two interfaces (see module docstring).
MOUNT_ABS_Y = 0.60          # end pads start here
MOUNT_Z_BAND = 0.06         # thickness of the bolted-down bottom face
LUG_ABS_Y = 0.32            # central interface half-width along y
LUG_Z_BAND = 0.10           # depth below the lug crown that carries the load

# Unit conversions for the GE challenge's imperial load statement. `fea.py`
# works in N, m and Pa, so every load has to enter in SI. Both factors are the
# exact definitions (1 lbf = 0.45359237 kg * 9.80665 m/s^2; 1 in = 0.0254 m).
LBF_TO_N = 4.4482216152605               # pound-force -> newton
LBIN_TO_NM = 0.1129848290276167          # pound-force inch -> newton metre

# GE jet-engine-bracket challenge loads as stated (imperial) and as DeepJEB's
# FEA labels were produced (SI, arXiv 2406.09047): vertical 8,000 lbf = 35.6 kN,
# horizontal 8,500 lbf = 37.8 kN, diagonal 9,500 lbf = 42.3 kN at 42 deg,
# torsion 5,000 lb*in = 565 N*m. The paper's kN / N*m figures are these
# conversions rounded to three significant digits.
VERTICAL_LBF = 8000.0
HORIZONTAL_LBF = 8500.0
DIAGONAL_LBF = 9500.0
DIAGONAL_ANGLE_DEG = 42.0
TORSION_LBIN = 5000.0


class Bracket:
    """Load cases from the GE bracket challenge, scaled to this geometry.

    Magnitudes are SI (N and N*m), converted explicitly from the challenge's
    imperial statement so they match the loads DeepJEB's labels were computed
    under (35.6 / 37.8 / 42.3 kN, 565 N*m). This table used to hold the bare
    imperial numbers 8000 / 8500 / 9500 and 5000 and `fea.py` read them as N
    and N*m: the three forces were 4.448x too small and the torsion moment
    8.85x too large, so the four cases were mis-weighted against each other
    and absolute stresses were understated. Linear statics means a common
    force factor rescales every stress and displacement exactly (compliance by
    its square); absolute numbers recorded before this change are not
    comparable with numbers recorded after it.

    Axes are this repo's, derived from an occupancy study of the normalized
    frame (module docstring): vertical is +z, horizontal is +y (the long
    axis), torsion is about y, and the interface rules above depend on them.
    The DeepJEB paper labels its horizontal case +x and its torsion about -z in
    *its* frame -- a frame-convention difference, not a load difference, and
    nothing is rotated here. The diagonal vector likewise keeps its historical
    decomposition (F cos 42 along y, F sin 42 along z, i.e. 42 deg from the
    horizontal axis), whereas the challenge statement reads "42 deg from
    vertical"; that is a deliberate no-rotation choice, flagged, not fixed.
    """

    LOAD_CASES = {
        'vertical': dict(kind='force',
                         vector=(0.0, 0.0, VERTICAL_LBF * LBF_TO_N)),               # 35.6 kN
        'horizontal': dict(kind='force',
                           vector=(0.0, HORIZONTAL_LBF * LBF_TO_N, 0.0)),           # 37.8 kN
        'diagonal': dict(kind='force',                                             # 42.3 kN
                         vector=(0.0,
                                 DIAGONAL_LBF * LBF_TO_N * np.cos(np.deg2rad(DIAGONAL_ANGLE_DEG)),
                                 DIAGONAL_LBF * LBF_TO_N * np.sin(np.deg2rad(DIAGONAL_ANGLE_DEG)))),
        'torsion': dict(kind='moment', axis=(0.0, 1.0, 0.0),
                        magnitude=TORSION_LBIN * LBIN_TO_NM),                      # 565 N*m
    }

    def __init__(self, material=None, length_scale=0.19 / 1.8,
                 load_cases=('vertical', 'diagonal'), stress_percentile=99.5):
        self.material = material or fea.Material()
        self.length_scale = float(length_scale)
        self.load_cases = tuple(load_cases)
        self.stress_percentile = float(stress_percentile)

    # ---------------------------------------------------------------- #
    # Interface detection
    # ---------------------------------------------------------------- #

    def interfaces(self, nodes_norm, faces):
        """Fixed (mount) and loaded (lug) node sets in normalized coordinates."""
        y, z = nodes_norm[:, 1], nodes_norm[:, 2]
        on_surface = np.zeros(len(nodes_norm), dtype=bool)
        on_surface[np.unique(faces)] = True

        pad = np.abs(y) >= MOUNT_ABS_Y
        if not pad.any():
            raise ValueError('no mounting pad region found')
        z_base = z[pad].min()
        mount = np.flatnonzero(pad & on_surface & (z <= z_base + MOUNT_Z_BAND))

        lug_band = np.abs(y) <= LUG_ABS_Y
        if not lug_band.any():
            raise ValueError('no central lug region found')
        z_crown = z[lug_band].max()
        lug = np.flatnonzero(lug_band & on_surface & (z >= z_crown - LUG_Z_BAND))

        if len(mount) < 12:
            raise ValueError(f'mount region too small ({len(mount)} nodes)')
        if len(lug) < 6:
            raise ValueError(f'lug region too small ({len(lug)} nodes)')
        # Both ends must be gripped, otherwise the part is a cantilever off one pad.
        if (y[mount] > 0).sum() < 4 or (y[mount] < 0).sum() < 4:
            raise ValueError('mounting pads found on only one end')
        return mount, lug

    # ---------------------------------------------------------------- #
    # Load assembly
    # ---------------------------------------------------------------- #

    def _load_vector(self, name, nodes, faces, lug, ndof):
        spec = self.LOAD_CASES[name]
        weights = fea.face_area_weights(nodes, faces, lug)
        f = np.zeros(ndof)
        if spec['kind'] == 'force':
            vec = np.asarray(spec['vector'], dtype=float)
            contribution = weights[lug][:, None] * vec[None, :]
        else:
            axis = np.asarray(spec['axis'], dtype=float)
            axis = axis / np.linalg.norm(axis)
            r = nodes[lug] - np.average(nodes[lug], axis=0, weights=weights[lug])
            r = r - np.outer(r @ axis, axis)                  # radial part only
            tangential = np.cross(axis[None, :], r)
            denom = float((weights[lug] * (np.linalg.norm(tangential, axis=1) ** 2)).sum())
            if denom <= 0:
                raise ValueError('torsion load case has no lever arm')
            contribution = (spec['magnitude'] / denom) * weights[lug][:, None] * tangential
        np.add.at(f, (lug[:, None] * 3 + np.arange(3)[None, :]).ravel(),
                  contribution.ravel())
        return f

    # ---------------------------------------------------------------- #
    # Analysis
    # ---------------------------------------------------------------- #

    def analyze(self, nodes_norm, tets, return_fields=False):
        """Run every configured load case. Returns a result dict in SI units.

        With `return_fields`, the worst case's nodal von Mises field, the outer
        triangles and the interface node sets are attached under 'fields' for
        plotting. They are megabytes per design, so the search never asks for them.
        """
        nodes = nodes_norm * self.length_scale
        K, B, vol, tets = fea.assemble(nodes, tets, self.material)
        faces = fea.boundary_faces(tets)
        mount, lug = self.interfaces(nodes_norm, faces)

        total_volume = float(np.abs(vol).sum())
        mass = total_volume * self.material.rho
        ndof = nodes.shape[0] * 3
        fixed = (mount[:, None] * 3 + np.arange(3)[None, :]).ravel()
        D = self.material.constitutive()

        solver = fea.LinearSolver(K, ndof, fixed, nodes)
        cases = {}
        fields = {}
        for name in self.load_cases:
            f = self._load_vector(name, nodes, faces, lug, ndof)
            u, solve_info = solver.solve(f)
            _, vm = fea.element_stress(B, D, u, tets)
            vm_nodal = fea.nodal_average(vm, tets, vol, nodes.shape[0])
            disp = np.linalg.norm(u, axis=1)
            if return_fields:
                fields[name] = {'von_mises_nodal': vm_nodal, 'displacement': u}
            cases[name] = {
                'max_von_mises': float(vm.max()),
                'peak_von_mises': float(np.percentile(vm_nodal, self.stress_percentile)),
                'max_displacement': float(disp.max()),
                'compliance': float(f @ u.ravel()),
                'solver': solve_info,
            }

        worst_name = max(cases, key=lambda n: cases[n]['peak_von_mises'])
        worst = cases[worst_name]
        result = {
            'mass': mass,
            'volume': total_volume,
            'num_nodes': int(nodes.shape[0]),
            'num_tets': int(tets.shape[0]),
            'mount_nodes': int(len(mount)),
            'lug_nodes': int(len(lug)),
            'cases': cases,
            'peak_von_mises': worst['peak_von_mises'],
            'max_von_mises': max(c['max_von_mises'] for c in cases.values()),
            'max_displacement': max(c['max_displacement'] for c in cases.values()),
            'max_compliance': max(c['compliance'] for c in cases.values()),
            'worst_case': worst_name,
        }
        if return_fields:
            result['fields'] = {
                'nodes_norm': nodes_norm, 'faces': faces, 'mount': mount, 'lug': lug,
                'worst_case': worst_name,
                'von_mises_nodal': fields[worst_name]['von_mises_nodal'],
                'displacement': fields[worst_name]['displacement'],
            }
        return result


class MassObjective:
    """Minimize mass subject to a peak-stress and a deflection limit.

    Scalarized with quadratic exterior penalties so an infeasible design still
    gives a population search a direction to move in.
    """

    def __init__(self, mass_ref, stress_allow, disp_allow,
                 stress_weight=6.0, disp_weight=3.0, failure_score=10.0):
        self.mass_ref = float(mass_ref)
        self.stress_allow = float(stress_allow)
        self.disp_allow = float(disp_allow)
        self.stress_weight = float(stress_weight)
        self.disp_weight = float(disp_weight)
        self.failure_score = float(failure_score)

    def __call__(self, result):
        mass_term = result['mass'] / self.mass_ref
        gs = max(0.0, result['peak_von_mises'] / self.stress_allow - 1.0)
        gd = max(0.0, result['max_displacement'] / self.disp_allow - 1.0)
        score = mass_term + self.stress_weight * gs ** 2 + self.disp_weight * gd ** 2
        return float(score), {
            'mass_term': float(mass_term),
            'stress_violation': float(gs),
            'disp_violation': float(gd),
            'feasible': bool(gs <= 0.0 and gd <= 0.0),
        }
