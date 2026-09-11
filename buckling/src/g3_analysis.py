import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
from geometry import ShellMesh
from gates import g2, mode_hist, tv

B = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "pilot")
recs = [r for r in json.load(open(os.path.join(B, "results.json"))) if r.get("ok")]
by = defaultdict(list)
for r in recs:
    by[(r["r_over_t"], r["l_over_r"])].append(r)
corners = sorted(by)

meshes = {c: ShellMesh(c[0], c[1], elems_per_half_wave=3.0) for c in corners}
g2res, cnts, flins = {}, {}, {}
for c in corners:
    rs = by[c]
    ws = [np.load(os.path.join(B, r["key"], "w.npy")) for r in rs]
    g2res[c] = g2(meshes[c], ws)
    cnts[c] = Counter(r["n_dominant"] for r in rs)
    flins[c] = [r["f_end_over_lin"] for r in rs]

print("=== per-corner summary ===")
for c in corners:
    m, res, cnt, fl = meshes[c], g2res[c], cnts[c], flins[c]
    ns = [r["n_dominant"] for r in by[c]]
    print("R/t=%-4d L/R=%-4.1f Z=%-6.1f nodes=%-6d | spread=%.3f effK=%.2f topw=%.3f | n:[%d..%d] mean=%.1f | F/Flin=%.3f+-%.4f | buckled=%d/%d"
          % (c[0], c[1], m.batdorf_Z, m.n_nodes, res["mean_pairwise_relL2"], res["effective_k"],
             res["top_weight"], min(ns), max(ns), np.mean(ns), np.mean(fl), np.std(fl),
             sum(1 for r in by[c] if r["buckled"]), len(by[c])))

print()
print("=== dominant-n histograms ===")
for c in corners:
    cnt = cnts[c]
    tot = sum(cnt.values())
    print(c)
    for n in sorted(cnt):
        print("   n=%-3d %3d %5.1f%% %s" % (n, cnt[n], 100 * cnt[n] / tot, "#" * cnt[n]))

print()
print("=== G3: pairwise TV of mode-weight histograms ===")
hists = {c: mode_hist(by[c]) for c in corners}
tvs = []
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        t = tv(hists[corners[i]], hists[corners[j]])
        tvs.append(t)
        print("  %s vs %s : TV=%.3f" % (corners[i], corners[j], t))
ge035 = sum(1 for t in tvs if t >= 0.35)
ge015 = sum(1 for t in tvs if t >= 0.15)
print("G3 result: %d/6 pairs TV>=0.35 (need >=5), %d/6 pairs TV>=0.15 (need 6)" % (ge035, ge015))
print("G3 PASS" if (ge035 >= 5 and ge015 == 6) else "G3 marginal/fail on strict threshold")

np.save(os.path.join(B, "..", "pilot_hists.npy"), np.stack([hists[c] for c in corners]))
json.dump({"corners": [list(c) for c in corners]}, open(os.path.join(B, "..", "pilot_meta.json"), "w"))
print("\nsaved pilot_hists.npy, pilot_meta.json")
