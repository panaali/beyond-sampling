"""B1/B2/B3 driver: cells = distinct k-prefixes of points sampled in tri's ordering cone (and event points);
compare T390's float relax decision (relax.py, read from origin/claude/t373-event-cone, path in argv) with
verify_prune. usage: python b_test.py RELAX_DIR p1,p2,p3 k n_cells"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import verify as V  # noqa: E402
import relax  # noqa: E402  (T390, float)

tri = tuple(int(v) for v in sys.argv[2].split(",")); k = int(sys.argv[3]); ncell = int(sys.argv[4])
p1, p2, p3 = tri
rng = np.random.default_rng(4000 + k)
lows = [i for i in range(10) if i not in tri]
X = rng.standard_normal((200000, 10)); X.sort(1); X = X[:, ::-1].copy()
perm = np.empty((len(X), 10), int); perm[:, :3] = tri
perm[:, 3:] = np.array(lows)[np.argsort(rng.random((len(X), 7)), 1)]
Y = np.zeros_like(X); np.put_along_axis(Y, perm, X, 1)
hs = V.common.run(Y)[:, 1:k + 1] > 0
ev = np.load("data/regions_%s.npz" % "-".join(map(str, tri)))["gates"][:, :k]
cells = {}
for g in ev:
    cells.setdefault(g.astype(np.int8).tobytes(), ("event", g))
for g in hs:
    if len(cells) >= ncell + len(ev):
        break
    cells.setdefault(g.astype(np.int8).tobytes(), ("rand", g.astype(int)))
out = []
for key, (kind, g) in cells.items():
    if kind == "event" and len([o for o in out if o["kind"] == "event"]) >= 50:
        continue
    gates = np.full((10, 16), -1); gates[:k] = g
    G, EG, A, EA = V.cell_rows(gates, k)
    t = time.time(); fl = bool(relax.relax_infeasible(G, A, k, tri)); tf = time.time() - t
    t = time.time(); ok, info = V.verify_prune(tri, gates, k, return_info=True); tv = time.time() - t
    out.append(dict(kind=kind, float_pruned=fl, verified=ok, t_float=tf, t_verify=tv, **{kk: vv for kk, vv in info.items()}))
    print(out[-1], flush=True)
json.dump(out, open("results_b_%s_k%d.json" % ("-".join(map(str, tri)), k), "w"), indent=0)
ev_ = [o for o in out if o["kind"] == "event"]; rd = [o for o in out if o["kind"] == "rand"]
print("B1 event cells verified (must be 0):", sum(o["verified"] for o in ev_), "/", len(ev_))
fp = [o for o in rd if o["float_pruned"]]
print("B2 float-pruned", len(fp), "verified", sum(o["verified"] for o in fp), "; verified but not float-pruned",
      sum(o["verified"] and not o["float_pruned"] for o in rd))
print("B3 verify s median/p90", np.median([o["t_verify"] for o in rd]), np.percentile([o["t_verify"] for o in rd], 90),
      "float relax median", np.median([o["t_float"] for o in rd]))
