"""B1/B2/B3 on T398's benchmark (nodes/bench_182_k4.npz, branch claude/t373-event-cone @ fce7aaa8).
B1: the 47 k=4 survivors with probe_best > 0 (a hit-and-run point with L_p3 >= max L_j) must NOT verify.
B2/B3: children at k=5 of the first N survivors; T398's float relax2.prune_stage decides; verify_prune (exact-
safe) must confirm each float prune (stage 3 leaf by leaf). usage: python b_bench.py EC_DIR N"""
import json
import sys
import time

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, sys.argv[1])
import verify as V  # noqa: E402
import relax2  # noqa: E402
import cone_count as cc  # noqa: E402

d = np.load("data/bench_182_k4.npz", allow_pickle=True)
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"]); N = int(sys.argv[2])
gates_of = lambda h: np.unpackbits(np.frombuffer(h, np.uint8)).reshape(k, 16).astype(int)  # noqa: E731
out = dict(B1=[], B2=[])
for i in (np.where(d["probe_best"] > 0)[0] if len(sys.argv) < 5 else []):
    g = np.full((10, 16), -1); g[:k] = gates_of(d["hist"][i])
    t = time.time(); ok, info = V.verify_prune(tri, g, k, event_aware=True, return_info=True)
    out["B1"].append(dict(i=int(i), verified=ok, t=time.time() - t, margin_lb=info.get("margin_lb")))
print("B1 event-feasible cells verified (must be 0):", sum(o["verified"] for o in out["B1"]), "/", len(out["B1"]), flush=True)


class Rec(relax2.Relax):
    def feasible(self, fix=None):
        f = super().feasible(fix)
        if not f:
            self.leaves.append(dict(fix or {}))
        return f


def interior(G):
    n = np.linalg.norm(G, axis=1)
    r = linprog(np.r_[np.zeros(10), -1], A_ub=np.c_[-G, n], b_ub=np.zeros(len(G)), bounds=[(-1, 1)] * 10 + [(0, 1)], method="highs")
    return r.x[:10]


Wi, Wh = V.Wi, V.Wh
for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 0, N):
    g4 = gates_of(d["hist"][i]); m = int(d["m"][i]); G = d["G"][i][:m]; A = d["A"][i]
    Z = Wh @ A; Z[:, k] += Wi
    pieces = [(G, interior(G), [])]
    for j in range(16):
        nxt = []
        for Gp, xp, gg in pieces:
            dd, xpos, xneg = cc.split_fast(Gp, xp, Z[j])
            if dd == "off":
                nxt.append((Gp, xp, gg + [0]))
            elif dd == "on":
                nxt.append((Gp, xp, gg + [1]))
            else:
                nxt.append((np.vstack([Gp, Z[j]]), xpos, gg + [1])); nxt.append((np.vstack([Gp, -Z[j]]), xneg, gg + [0]))
        pieces = nxt
    for Gp, xp, gg in pieces:
        gg = np.array(gg); A5 = gg[:, None] * Z
        R = Rec(Gp, A5, k + 1, tri); R.leaves = []
        t = time.time()
        if not R.forward_bounds():
            st = 0
        elif not R.feasible():
            st = 1
        elif not R.event_tighten() or not R.feasible():
            st = 2
        else:
            R.leaves = []
            st = 3 if R.split_prune(R.split_order(), relax2.DEPTH) else -1
        tf = time.time() - t
        g = np.full((10, 16), -1); g[:k] = g4; g[k] = gg
        t = time.time()
        if st in (0, 1):
            ok = V.verify_prune(tri, g, k + 1)
            nleaf = 1
        elif st == 2:
            ok = V.verify_prune(tri, g, k + 1, event_aware=True)
            nleaf = 1
        elif st == 3:
            leaves = []
            for fx in R.leaves:
                gl = g.copy()
                for (s, j), f in fx.items():
                    gl[k + 1 + s][j] = f
                leaves.append(gl)
            ok = V.verify_leaves(tri, leaves, k + 1, event_aware=True)
            if not ok and len(sys.argv) > 4:
                base = leaves[0].copy(); base[k + 1:] = -1
                Rv = V.Relax(tri, base, k + 1); inf_ = dict(lps=0); V._bounds(Rv, True, inf_)
                print("lo/hi diff vs relax2", np.nanmax(np.abs(Rv.lo - R.lo)), np.nanmax(np.abs(Rv.hi - R.hi)))
                for gl, fx in zip(leaves, R.leaves):
                    Rv.gates = gl; V._final(Rv, inf_)
                    print("leaf", fx, "margin_lb", inf_["margin_lb"], "per-leaf", V.verify_prune(tri, gl, k + 1, event_aware=True, return_info=True)[1].get("margin_lb"))
            nleaf = len(leaves)
        else:
            ok = None; nleaf = 0
        rec = dict(parent=i, stage=st, verified=ok, leaves=nleaf, t_float=tf, t_verify=time.time() - t)
        out["B2"].append(rec); print(rec, flush=True)
json.dump(out, open("results_b_bench_%s.json" % "_".join(sys.argv[2:]), "w"), indent=0)
pr = [o for o in out["B2"] if o["stage"] >= 0]
print("B2 float-pruned", len(pr), "verified", sum(bool(o["verified"]) for o in pr),
      {s: (sum(o["stage"] == s for o in pr), sum(bool(o["verified"]) for o in pr if o["stage"] == s)) for s in range(4)})
print("B3 verify s median/p90", np.median([o["t_verify"] for o in pr]), np.percentile([o["t_verify"] for o in pr], 90),
      "float median", np.median([o["t_float"] for o in pr]))
