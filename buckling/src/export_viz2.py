"""Export EVERY pilot draw (not a curated 12) for an interactive per-sample viewer.

Fields are quantized to uint8 (symmetric range about 0) and base64-encoded so
320 real fields fit in a small payload -- this is a display compression, the
analysis upstream (gates.py) always runs on the float32 field.
"""
import sys, os, json, base64, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from analyse import to_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "work", "pilot")

recs = [r for r in json.load(open(os.path.join(B, "results.json"))) if r.get("ok")]
by = defaultdict(list)
for r in recs:
    by[(r["r_over_t"], r["l_over_r"])].append(r)
corners = sorted(by)
meshes = {c: ShellMesh(c[0], c[1], elems_per_half_wave=3.0) for c in corners}


def field_grid(mesh, w, max_circ=100):
    g = to_grid(mesh, w)
    NI, NJ = g.shape
    cols = list(range(0, NJ, 2))
    g = g[:, cols]
    step = max(1, NI // max_circ)
    g = g[::step]
    for i in range(g.shape[0]):
        row = g[i]
        if np.isnan(row).any():
            idx = np.where(~np.isnan(row))[0]
            if len(idx):
                row[np.isnan(row)] = np.interp(np.where(np.isnan(row))[0], idx, row[idx])
    return g


def quantize(g):
    ext = float(max(np.abs(np.nanmin(g)), np.abs(np.nanmax(g)))) or 1.0
    q = np.clip(np.round((g / ext + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return q, ext


out = {"corners": [], "samples": []}

for c in corners:
    m = meshes[c]
    rs = by[c]
    ws = [np.load(os.path.join(B, r["key"], "w.npy")) for r in rs]

    corner_key = "%d_%.1f" % c
    out["corners"].append(dict(
        key=corner_key, r_over_t=c[0], l_over_r=c[1], Z=round(m.batdorf_Z, 1),
        n_nodes=int(m.n_nodes), n_draws=len(rs),
    ))

    for r, w in zip(rs, ws):
        g = field_grid(m, w)
        q, ext = quantize(g)
        rows, cols = q.shape
        b64 = base64.b64encode(q.tobytes()).decode("ascii")
        out["samples"].append(dict(
            corner=corner_key, seed=r["seed"], n_dominant=r["n_dominant"],
            n_share=round(r.get("n_share", 0), 3),
            f_over_lin=round(r["f_end_over_lin"], 4),
            rms_over_t=round(r["rms_over_t"], 4),
            max_over_t=round(r["max_over_t"], 4),
            rows=rows, cols=cols, ext=round(ext, 4), grid_b64=b64,
        ))

path = os.path.join(ROOT, "work", "viz_data2.json")
json.dump(out, open(path, "w"), separators=(",", ":"))
sz = os.path.getsize(path)
print("wrote %s  (%.2f MB), %d samples across %d corners"
      % (path, sz / 1e6, len(out["samples"]), len(out["corners"])))
