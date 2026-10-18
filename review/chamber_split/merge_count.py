"""Count estT leaves that would merge if the start partition were coarser (geometry only; masses skipped).
usage: python merge_count.py H T seed mode   (mode exact | recipe108)"""
import sys, os, time, json
R = "/home/user/clean-neurons/m16"
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), R + "/approaches/estT_fix", R + "/approaches/arc_m43_repro"]
import numpy as np
import estT
from m43 import Ops
from pthload import zoo
H, T, seed, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
m = zoo(H, T, seed)
HIST = []; LEAVES = []; CUR = [None]
f0 = estT._finish
def fin(c, g, A, t, *a):
    HIST.append(g.astype(np.uint8).tobytes()); r = f0(c, g, A, t, *a); HIST.pop(); return r
def ce(c, g, A, Wo, truth, ops):
    LEAVES.append((CUR[0], tuple(HIST) + (g.astype(np.uint8).tobytes(),), c)); return 0.0
estT._finish = fin; estT._cell_error = ce
estT.MODE[0] = mode; exact = mode == "exact"
t0 = time.process_time(); o = Ops()
Wi, Wh, Wo = m
STRIDE, OFF = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1, 0)
HV = estT.halves(T); TC = sorted({(h["sigma"], h["perm"][0], h["perm"][1]) for h in HV})
KEEP = {tc for i, tc in enumerate(TC) if i % STRIDE == OFF}
for hi, h in enumerate(HV):
    if (h["sigma"], h["perm"][0], h["perm"][1]) not in KEEP: continue
    side = 0 if h["n_true"][h["perm"][0]] != 0 else 1
    CUR[0] = (h["sigma"], h["perm"], side)
    g1 = (h["sigma"] * Wi > 0).astype(float); A1 = np.outer(Wi, np.eye(T)[0])
    estT._finish(h["cone"], g1, A1, 1, m, h["truth"], h["n_true"], 0.1, o, exact)
geo = time.process_time() - t0
keys = {
    "fine": lambda s, p, sd: (s, p, sd),
    "chamber (merge across 2x_b=x_a+x_c)": lambda s, p, sd: (s, p),
    "top3 ordered + side": lambda s, p, sd: (s, p[:3], sd),
    "truth cone (a,b) x sign": lambda s, p, sd: (s, p[0], p[1]),
    "truth class b x sign (lb)": lambda s, p, sd: (s, p[1]),
    "history only (lb)": lambda s, p, sd: (),
}
out = dict(H=H, T=T, seed=seed, mode=mode, cones=f"{len(KEEP)}/{len(TC)}", leaves=len(LEAVES), geo_cpu=round(geo, 1), ops=o.n)
for name, kf in keys.items():
    out[name] = len({(kf(*k), hst) for k, hst, _ in LEAVES})
# adjacency-checked components for the truth-cone merge (upper bound on merged count)
groups = {}
for i, (k, hst, c) in enumerate(LEAVES):
    groups.setdefault(((k[0], k[1][0], k[1][1]), hst), []).append(i)
par = list(range(len(LEAVES)))
def fnd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
def adjacent(P, Q):
    for n in P.N:
        nn = n / np.linalg.norm(n)
        for q in Q.N:
            if np.allclose(nn, -q / np.linalg.norm(q), atol=1e-9):
                for X, Y in ((P, Q), (Q, P)):
                    on = np.abs(X.R @ nn) < 1e-9
                    if on.sum() >= T - 1:
                        p = X.R[on].mean(0)
                        if np.all(Y.N @ p / np.linalg.norm(Y.N, axis=1) >= -1e-9):
                            return True
    return False
for idx in groups.values():
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if fnd(idx[a]) != fnd(idx[b]) and adjacent(LEAVES[idx[a]][2], LEAVES[idx[b]][2]):
                par[fnd(idx[a])] = fnd(idx[b])
out["truth cone, adjacency components (ub)"] = len({fnd(i) for i in range(len(LEAVES))})
out["cpu"] = round(time.process_time() - t0, 1)
print(json.dumps(out))
