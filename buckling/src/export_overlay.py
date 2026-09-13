"""Export several CONVERGED deformed shapes on one shared lattice, for overlay.

All samples share the same nominal geometry and the same mesh, so only the
displacement field differs between them -- which is exactly the one-to-many
claim. Nominal coordinates are therefore stored once, not per sample.
"""
import sys, os, json, glob, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from analyse import radial, mode_label
from diag_settle import read_all_frd_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RT, LR = 110, 1.0
m = ShellMesh(RT, LR, elems_per_half_wave=3.0)

NI, NJ = 2 * m.n_circ, 2 * m.n_axial + 1
I_even = list(range(0, NI, 2))
J_even = list(range(0, NJ, 2))
nodes = np.array([[m.node_index[i, j] for j in J_even] for i in I_even])
assert (nodes >= 0).all()
nc, na = nodes.shape

samples = []

# seed 1 comes from the original long-settle diagnostic
WANT_DAMP_EARLY = float(os.environ.get("OVERLAY_DAMP", "0.3"))
diag = os.path.join(ROOT, "work", "diag", "diag_rt110_lr100_a0.3",
                    "diag_rt110_lr100_a0.3.frd")
# this diagnostic was run at alpha=0.3 only -- do not mix it into another set
if os.path.exists(diag) and abs(WANT_DAMP_EARLY - 0.3) < 1e-9:
    blocks = read_all_frd_blocks(diag)
    good = [b for b in blocks if b.shape[0] >= m.n_nodes]
    if good:
        samples.append(("seed 1", good[-1][:m.n_nodes]))

# seeds 2..N from the converged batch
# Runs made before the damping suffix was added carry no "_d" tag at all; those
# are the alpha=0.3 baseline. Parse the damping out of the name and filter on it.
WANT_DAMP = float(os.environ.get("OVERLAY_DAMP", "0.3"))
for d in sorted(glob.glob(os.path.join(ROOT, "work", "converged", "cv_*"))):
    base = os.path.basename(d)
    f = os.path.join(d, "disp3d.npy")
    if not os.path.exists(f):
        continue
    mm = re.search(r"_s(\d+)", base)
    if not mm:
        continue
    dm = re.search(r"_d([0-9.]+)", base)
    damp = float(dm.group(1)) if dm else 0.3
    if abs(damp - WANT_DAMP) > 1e-9:
        continue
    seed = int(mm.group(1))
    samples.append(("seed %d" % seed, np.load(f)[:m.n_nodes]))

print("samples found:", len(samples))

out_samples = []
for label, disp_full in samples:
    disp = disp_full[nodes]                      # (nc, na, 3)
    w = radial(m, disp_full) / m.t
    ml = mode_label(m, w)
    out_samples.append(dict(
        label=label,
        n_dominant=ml["n"],
        rms_over_t=round(float(np.sqrt(np.mean(w ** 2))), 4),
        max_over_t=round(float(np.abs(w).max()), 4),
        disp=np.round(disp, 7).ravel().tolist(),
    ))
    print("  %-8s n=%-3d rms/t=%.4f max/t=%.4f"
          % (label, ml["n"], np.sqrt(np.mean(w ** 2)), np.abs(w).max()))

nom = m.coords[nodes]
out = dict(nc=nc, na=na, R=float(m.R), t=float(m.t), L=float(m.L),
           r_over_t=RT, l_over_r=LR,
           nominal=np.round(nom, 6).ravel().tolist(),
           samples=out_samples)
p = os.path.join(ROOT, "work", "overlay_110_100.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
print("wrote %s  %.2f MB  (%d samples)" % (p, os.path.getsize(p) / 1e6, len(out_samples)))
