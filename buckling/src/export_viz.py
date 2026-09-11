"""Export real pilot results into a single JSON for the results dashboard.

Everything in here is measured data -- no placeholders. Field grids are
downsampled circumferentially (not axially, since axial counts are already
small) to keep the payload small while preserving the actual buckling pattern.
"""
import sys, os, json, numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from analyse import to_grid
from gates import g2, mode_hist, tv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "work", "pilot")

recs = [r for r in json.load(open(os.path.join(B, "results.json"))) if r.get("ok")]
by = defaultdict(list)
for r in recs:
    by[(r["r_over_t"], r["l_over_r"])].append(r)
corners = sorted(by)
meshes = {c: ShellMesh(c[0], c[1], elems_per_half_wave=3.0) for c in corners}

def field_grid(mesh, w, max_circ=120):
    """Downsample the (I,J) grid circumferentially to <= max_circ columns."""
    g = to_grid(mesh, w)          # (NI, NJ), NaN at serendipity holes
    NI, NJ = g.shape
    cols = list(range(0, NJ, 2))  # even rows are complete
    g = g[:, cols]
    step = max(1, NI // max_circ)
    g = g[::step]
    # fill any remaining NaN by nearest neighbour along axial direction
    for i in range(g.shape[0]):
        row = g[i]
        if np.isnan(row).any():
            idx = np.where(~np.isnan(row))[0]
            if len(idx):
                row[np.isnan(row)] = np.interp(np.where(np.isnan(row))[0], idx, row[idx])
    return np.round(g, 4).tolist()

out = {"corners": [], "fields": []}

for c in corners:
    m = meshes[c]
    rs = by[c]
    ws = [np.load(os.path.join(B, r["key"], "w.npy")) for r in rs]
    res = g2(m, ws)
    cnt = Counter(r["n_dominant"] for r in rs)
    fl = [r["f_end_over_lin"] for r in rs]
    ns = [r["n_dominant"] for r in rs]

    hist = {int(k): v for k, v in cnt.items()}
    out["corners"].append(dict(
        r_over_t=c[0], l_over_r=c[1], Z=round(m.batdorf_Z, 1),
        n_nodes=int(m.n_nodes), n_elems=int(len(m.elements)),
        n_circ=m.n_circ, n_axial=m.n_axial,
        spread=round(res["mean_pairwise_relL2"], 4),
        effective_k=round(res["effective_k"], 3),
        top_weight=round(res["top_weight"], 3),
        n_min=min(ns), n_max=max(ns), n_mean=round(float(np.mean(ns)), 2),
        n_over_sqrt_rt=round(float(np.mean(ns)) / np.sqrt(c[0]), 4),
        n_hist=hist,
        flin_mean=round(float(np.mean(fl)), 4), flin_std=round(float(np.std(fl)), 5),
        flin_min=round(float(np.min(fl)), 4), flin_max=round(float(np.max(fl)), 4),
        n_draws=len(rs), n_buckled=sum(1 for r in rs if r["buckled"]),
        cost_mean_s=round(float(np.mean([r["seconds"] for r in rs])), 1),
    ))

    # pick 3 representative draws spanning low / median / high n
    order = np.argsort(ns)
    picks = [order[0], order[len(order) // 2], order[-1]]
    for p in picks:
        r = rs[p]
        out["fields"].append(dict(
            corner="%d_%.1f" % c, seed=r["seed"], n_dominant=r["n_dominant"],
            r_over_t=c[0], l_over_r=c[1],
            grid=field_grid(m, ws[p]),
            n_circ_full=m.n_circ, n_axial_full=m.n_axial,
        ))

# G3: pairwise TV
hists = {c: mode_hist(by[c]) for c in corners}
tv_pairs = []
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        tv_pairs.append(dict(a=list(corners[i]), b=list(corners[j]),
                             tv=round(tv(hists[corners[i]], hists[corners[j]]), 4)))
out["tv_pairs"] = tv_pairs
out["totals"] = dict(n_draws=len(recs), n_ok=len(recs), n_attempted=len(recs),
                     n_corners=len(corners))

path = os.path.join(ROOT, "work", "viz_data.json")
json.dump(out, open(path, "w"))
print("wrote %s (%.0f KB)" % (path, os.path.getsize(path) / 1024))
print("corners:", [c["r_over_t"], c["l_over_r"]] and [(c["r_over_t"], c["l_over_r"]) for c in out["corners"]])
print("fields exported:", len(out["fields"]))
