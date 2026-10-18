"""T287 (pre-registered in README). E1 witness-class estimators D0..D3 on all 30 LB3 models + frozen-gate ideal bias (paired
sampling, diagnostic) + exact-first-steps diagnostic on D1. usage (repo root): python m16/approaches/frozen_cheap/cheap.py [n]
-> cheap.json"""
import itertools, json, os, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "m16"))
import numpy as np
import leaderboard as lb
T = 10; N = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000; CH = 100_000

# ---------------- model-free tables: conditional means of descending order statistics given k (and the 2-3 gap bin)
Z = -np.sort(-np.random.default_rng(1234).standard_normal((4_000_000, T)), 1); KZ = (Z > 0).sum(1); GZ = Z[:, 1] - Z[:, 2]
COND = np.array([Z[KZ == k].mean(0) for k in range(T + 1)])
GMED = np.array([np.median(GZ[KZ == k]) for k in range(T + 1)])
COND_G = np.array([[Z[(KZ == k) & ((GZ > GMED[k]) == bool(b))].mean(0) for b in (0, 1)] for k in range(T + 1)])   # (11, 2, 10)
PK = np.array([comb(T, k) for k in range(T + 1)]) / 2.0 ** T


def place(top, vals, req_pos=(), req_neg=()):
    """Witness: ranks 1..len(top) at the given positions; remaining ranks to the free positions, with required-positive
    positions taking the highest remaining ranks and required-negative the lowest, others in increasing index order."""
    x = np.empty(T); free = [p for p in range(T) if p not in top]; r = len(top)
    for i, p in enumerate(top):
        x[p] = vals[i]
    order = [p for p in free if p in req_pos] + [p for p in free if p not in req_pos and p not in req_neg] + [p for p in free if p in req_neg]
    for i, p in enumerate(order):
        x[p] = vals[r + i]
    return x


def classes(design):
    """-> list of (weight, witness, top positions, extra key)."""
    out = []
    if design == "D0":
        for a, b in itertools.permutations(range(T), 2):
            for k in range(T + 1):
                out.append((PK[k] / 90, place((a, b), COND[k]), (a, b)))
    elif design in ("D1", "D2", "D3"):
        for tri in itertools.permutations(range(T), 3):
            for k in range(T + 1):
                w = PK[k] / 720
                if design == "D1":
                    out.append((w, place(tri, COND[k]), tri))
                elif design == "D3":
                    for b in (0, 1):
                        out.append((w / 2, place(tri, COND_G[k, b]), tri))
                else:   # D2: signs of positions 8, 9
                    j = max(0, k - 3)                     # positive ranks among the 7 non-top-3 positions
                    pos_known = {p: (rk < k) for rk, p in enumerate(tri)}
                    free89 = [p for p in (8, 9) if p not in tri]
                    for signs in itertools.product((True, False), repeat=len(free89)):
                        # probability that the free positions among {8, 9} have these signs: draw without replacement from 7 ranks
                        npos = sum(signs); nneg = len(signs) - npos
                        if npos > j or nneg > 7 - j:
                            continue
                        pr = 1.0; jj, rem = j, 7
                        for s in signs:
                            pr *= (jj / rem) if s else ((rem - jj) / rem); jj -= s; rem -= 1
                        rp = [p for p, s in zip(free89, signs) if s]; rn = [p for p, s in zip(free89, signs) if not s]
                        out.append((w * pr, place(tri, COND[k], rp, rn), tri))
    return out


def verdicts(m, X, top):
    Wi, Wh, Wo = m; h = np.zeros((len(X), len(Wi)))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) == np.array([t[1] for t in top])


def run(m, X, G=None, f=0, mm=T):
    """Forward with steps < f exact, steps f..T-mm-1 frozen at G, the rest exact (G None = the net)."""
    Wi, Wh, Wo = m; h = np.zeros((len(X), len(Wi)))
    for t in range(T):
        z = h @ Wh.T + X[:, t:t + 1] * Wi[None]
        h = z * G[t] if (G is not None and f <= t < T - mm) else np.maximum(z, 0)
    return h @ Wo.T


def gates(m, C):
    Wi, Wh, _ = m; h = np.zeros((len(C), len(Wi))); G = []
    for t in range(T):
        z = h @ Wh.T + C[:, t:t + 1] * Wi[None]; G.append(z > 0); h = np.maximum(z, 0)
    return G


CL = {d: classes(d) for d in ("D0", "D1", "D2", "D3")}
for d, c in CL.items():
    print(d, len(c), "classes, weight sum - 1 = %.1e" % (sum(x[0] for x in c) - 1), flush=True)
models = lb.models(); out = {}
for name, m in models.items():
    r = {}
    for d, c in CL.items():
        W = np.array([x[0] for x in c]); Xw = np.array([x[1] for x in c]); top = [x[2] for x in c]
        r[f"E1_{d}"] = dict(est=float((W * verdicts(m, Xw, top)).sum()), cost_fpe=len(c))
    # frozen-gate ideal (paired sampling diagnostic) for D1 witnesses: each input's (triple, k) witness gates
    rng = np.random.default_rng(287); acc = {}; net_ok = 0; n = 0
    tri_index = {t: i for i, t in enumerate(itertools.permutations(range(T), 3))}
    for _ in range(N // CH):
        X = rng.standard_normal((CH, T)); order = np.argsort(-X, 1); y = order[:, 1]; k = (X > 0).sum(1)
        # T282 protocol: move the input's ranks 4..10 into the witness arrangement (free positions in increasing index order),
        # score the frozen approximation against the net on the moved input
        Xs = -np.sort(-X, 1); tri_order = np.concatenate([order[:, :3], np.sort(order[:, 3:], 1)], 1)
        X = np.empty_like(Xs); np.put_along_axis(X, tri_order, Xs, 1)
        net_ok += (run(m, X).argmax(1) == y).sum(); n += CH
        ti = np.array([tri_index[tuple(o[:3])] for o in order])
        Cw = np.array([x[1] for x in CL["D1"]]).reshape(720, 11, T)[ti, k]
        G = gates(m, Cw)
        for f in (0, 1, 2, 3, 4):
            for mm in (0, 2, 4):
                acc[(f, mm)] = acc.get((f, mm), 0) + (run(m, X, G, f, mm).argmax(1) == y).sum()
    pn = net_ok / n
    r["ideal_D1"] = {f"f{f}_m{mm}": float(v / n - pn) for (f, mm), v in acc.items()}
    r["net_mc"] = float(pn); out[name] = r
    print(name, {k: round(100 * (v["est"] - pn), 2) for k, v in r.items() if k.startswith("E1")}, "ideal f0m0 %.2f" % (100 * r["ideal_D1"]["f0_m0"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "cheap.json"), "w"), indent=1)
