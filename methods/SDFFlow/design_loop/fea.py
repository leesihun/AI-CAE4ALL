"""Linear-static structural FEA on 4-node tetrahedra (constant strain).

Self-contained: scipy sparse assembly + an algebraic-multigrid solve whose
near-nullspace is seeded with the six rigid-body modes, which is what makes
elasticity converge in a handful of cycles. Element stress is constant per tet
and volume-averaged to the nodes for reporting.
"""

import numpy as np
import scipy.sparse as sp

VOIGT = 6


class SolveError(RuntimeError):
    pass


class Material:
    def __init__(self, name='Ti-6Al-4V', E=113.8e9, nu=0.342, rho=4430.0, yield_stress=903e6):
        self.name, self.E, self.nu, self.rho, self.yield_stress = name, E, nu, rho, yield_stress

    def constitutive(self):
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.E / (2.0 * (1.0 + self.nu))
        D = np.zeros((VOIGT, VOIGT))
        D[:3, :3] = lam
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D


def element_gradients(nodes, tets):
    """Shape-function gradients (E,4,3) and signed volumes (E,)."""
    p = nodes[tets]                                    # (E,4,3)
    J = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    det = np.linalg.det(J)
    vol = det / 6.0
    if np.any(np.abs(vol) < 1e-18):
        raise SolveError('degenerate tetrahedron (zero volume)')
    Jinv = np.linalg.inv(J)
    grads = np.empty((len(tets), 4, 3))
    grads[:, 1:, :] = Jinv                             # rows of J^-1 are grad N1..N3
    grads[:, 0, :] = -Jinv.sum(axis=1)
    return grads, vol


def strain_displacement(grads):
    """Voigt B matrices (E,6,12) with engineering shear strains."""
    E = grads.shape[0]
    B = np.zeros((E, VOIGT, 12))
    for i in range(4):
        gx, gy, gz = grads[:, i, 0], grads[:, i, 1], grads[:, i, 2]
        B[:, 0, 3 * i + 0] = gx
        B[:, 1, 3 * i + 1] = gy
        B[:, 2, 3 * i + 2] = gz
        B[:, 3, 3 * i + 0] = gy
        B[:, 3, 3 * i + 1] = gx
        B[:, 4, 3 * i + 1] = gz
        B[:, 4, 3 * i + 2] = gy
        B[:, 5, 3 * i + 0] = gz
        B[:, 5, 3 * i + 2] = gx
    return B


def assemble(nodes, tets, material):
    """Global stiffness (CSR), B matrices, element volumes."""
    grads, vol = element_gradients(nodes, tets)
    if vol.mean() < 0:                                  # normalize orientation
        tets = tets[:, [0, 2, 1, 3]]
        grads, vol = element_gradients(nodes, tets)
    flipped = vol < 0
    if flipped.any():
        tets = tets.copy()
        tets[flipped] = tets[flipped][:, [0, 2, 1, 3]]
        grads, vol = element_gradients(nodes, tets)

    B = strain_displacement(grads)
    D = material.constitutive()
    Ke = np.einsum('e,eki,kl,elj->eij', vol, B, D, B, optimize=True)

    dof = (tets[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(-1, 12)
    rows = np.repeat(dof, 12, axis=1).ravel()
    cols = np.tile(dof, (1, 12)).ravel()
    ndof = nodes.shape[0] * 3
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, B, vol, tets


def boundary_faces(tets):
    """Triangles belonging to exactly one tet (the closed outer surface)."""
    f = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                        tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]], axis=0)
    key = np.sort(f, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[counts == 1]]


def face_area_weights(nodes, faces, node_subset):
    """Consistent nodal weights for a traction spread over `node_subset`."""
    mask = np.zeros(nodes.shape[0], dtype=bool)
    mask[node_subset] = True
    keep = mask[faces].all(axis=1)
    w = np.zeros(nodes.shape[0])
    if keep.any():
        tri = faces[keep]
        a = nodes[tri[:, 1]] - nodes[tri[:, 0]]
        b = nodes[tri[:, 2]] - nodes[tri[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        np.add.at(w, tri.ravel(), np.repeat(area / 3.0, 3))
    if w.sum() <= 0:                                    # fall back to an equal split
        w[node_subset] = 1.0
    return w / w.sum()


class LinearSolver:
    """AMG hierarchy for one constrained stiffness matrix, reused across load cases.

    The hierarchy setup dominates the solve, and every load case on a given
    geometry shares K and the same constraints -- so it is built once here and
    each right-hand side is just a few CG cycles.
    """

    def __init__(self, K, ndof, fixed_dofs, nodes):
        import pyamg

        free = np.ones(ndof, dtype=bool)
        free[fixed_dofs] = False
        self.free_idx = np.flatnonzero(free)
        if self.free_idx.size == 0:
            raise SolveError('every degree of freedom is constrained')

        Kff = K[self.free_idx][:, self.free_idx].tocsr()
        if not np.isfinite(Kff.data).all():
            raise SolveError('non-finite stiffness entries')
        self.ndof = ndof
        self.ml = pyamg.smoothed_aggregation_solver(
            Kff, B=rigid_body_modes(nodes)[self.free_idx], max_coarse=500)

    def solve(self, force, tol=1e-9, maxiter=500, accept=1e-6):
        """Returns (displacement (N,3), solver info).

        pyamg returns its last iterate whether or not it converged, so the
        residual is checked explicitly: a stalled solve on a badly conditioned
        mesh would otherwise be scored as a perfectly good design.
        """
        residuals = []
        u_free = self.ml.solve(force[self.free_idx], tol=tol, maxiter=maxiter,
                               accel='cg', residuals=residuals)
        if not np.isfinite(u_free).all():
            raise SolveError('solver diverged')
        relative = residuals[-1] / max(residuals[0], 1e-300)
        if relative > accept:
            raise SolveError(f'CG stalled at relative residual {relative:.2e} '
                             f'after {len(residuals) - 1} iterations')
        u = np.zeros(self.ndof)
        u[self.free_idx] = u_free
        return u.reshape(-1, 3), {
            'iterations': len(residuals) - 1,
            'relative_residual': float(residuals[-1] / max(residuals[0], 1e-300)),
        }


def rigid_body_modes(nodes):
    """Six rigid-body modes as the AMG near-nullspace (N*3, 6)."""
    n = nodes.shape[0]
    rbm = np.zeros((3 * n, 6))
    for d in range(3):
        rbm[d::3, d] = 1.0
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    rbm[0::3, 3], rbm[1::3, 3] = -y, x        # rotation about z
    rbm[1::3, 4], rbm[2::3, 4] = -z, y        # rotation about x
    rbm[0::3, 5], rbm[2::3, 5] = z, -x        # rotation about y
    return rbm


def element_stress(B, D, u, tets):
    """Constant per-tet Cauchy stress in Voigt order and von Mises (E,)."""
    ue = u[tets].reshape(len(tets), 12)
    strain = np.einsum('ekj,ej->ek', B, ue)
    stress = strain @ D.T
    sx, sy, sz, txy, tyz, tzx = stress.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                 + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))
    return stress, vm


def nodal_average(values, tets, vol, num_nodes):
    """Volume-weighted element->node averaging."""
    num = np.zeros(num_nodes)
    den = np.zeros(num_nodes)
    w = np.abs(vol)
    np.add.at(num, tets.ravel(), np.repeat(values * w, 4))
    np.add.at(den, tets.ravel(), np.repeat(w, 4))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
