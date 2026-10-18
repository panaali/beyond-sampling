"""T282 (pre-registered in README): paired bias of frozen-canonical-gate approximations with exact last m steps, own code.
usage (repo root): python m16/approaches/frozen_check/frozen.py [n=400000] -> frozen.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "m16"))
import numpy as np
import leaderboard as lb
T, N, CH, MS = 10, int(sys.argv[1]) if len(sys.argv) > 1 else 400_000, 100_000, range(0, 9)

# model-free tables (descending order statistics), from 4e6 i.i.d. draws
Z = -np.sort(-np.random.default_rng(1234).standard_normal((4_000_000, T)), 1); KZ = (Z > 0).sum(1)
EOS = Z.mean(0)
COND = np.array([Z[KZ == k].mean(0) for k in range(T + 1)])                                  # (11, 10)
QUANT = np.array([[np.quantile(Z[KZ == k], q, axis=0) for k in range(T + 1)] for q in (.2, .4, .6, .8)])   # (4, 11, 10)


def canon(order, vals):
    """vals (n, T) in rank order -> inputs with vals[:, r] at position order[:, r]."""
    C = np.empty_like(vals); np.put_along_axis(C, order, vals, 1); return C


def gates(m, C):
    Wi, Wh, _ = m; h = np.zeros((len(C), len(Wi))); G = []
    for t in range(T):
        z = h @ Wh.T + C[:, t:t + 1] * Wi[None]; G.append(z > 0); h = np.maximum(z, 0)
    return G


def frozen_correct(m, X, y, G, k):
    Wi, Wh, Wo = m; h = np.zeros((len(X), len(Wi)))
    for t in range(T):
        z = h @ Wh.T + X[:, t:t + 1] * Wi[None]
        h = z * G[t] if t < T - k else np.maximum(z, 0)
    return (h @ Wo.T).argmax(1) == y


models = lb.models(); out = {}
for name, m in models.items():
    rng = np.random.default_rng(282); acc = {"net": 0.0}
    for _ in range(N // CH):
        X = rng.standard_normal((CH, T)); order = np.argsort(-X, 1); y = order[:, 1]; kpos = (X > 0).sum(1)
        net = frozen_correct(m, X, y, [np.ones((CH, len(m[0])), bool)] * T, T)              # all exact = the net
        acc["net"] += net.sum()
        # triple: x' = x with its ranks 4..10 moved into the fixed arrangement (increasing index order of the free positions);
        # the approximation is scored on x' against the net on x'; design_bias = net(x') - net(x) (T274b's protocol)
        rest = np.sort(order[:, 3:], 1); tri_order = np.concatenate([order[:, :3], rest], 1)
        Xs = -np.sort(-X, 1); X2 = canon(tri_order, Xs); net2 = frozen_correct(m, X2, y, [np.ones((CH, len(m[0])), bool)] * T, T)
        acc["net2"] = acc.get("net2", 0.0) + net2.sum()
        designs = {"full": [canon(order, np.broadcast_to(EOS, X.shape).copy())],
                   "triple": [canon(tri_order, np.broadcast_to(EOS, X.shape).copy())],
                   "V1_sign": [canon(order, COND[kpos])],
                   "V2_mix": [canon(order, QUANT[q][kpos]) for q in range(4)]}
        for d, Cs in designs.items():
            Gs = [gates(m, C) for C in Cs]
            Xd = X2 if d == "triple" else X
            for k in MS:
                c = np.mean([frozen_correct(m, Xd, y, G, k) for G in Gs], 0)
                acc[(d, k)] = acc.get((d, k), 0.0) + c.sum()
    n = (N // CH) * CH; pn = acc["net"] / n; pn2 = acc["net2"] / n
    out[name] = dict(net=pn, design_bias=pn2 - pn,
                     bias={f"{d}_m{k}": (v / n - (pn2 if d == "triple" else pn)) for (d, k), v in ((kk, vv) for kk, vv in acc.items() if kk not in ("net", "net2"))})
    print(name, "net %.4f" % pn, {kk: round(100 * v, 2) for kk, v in out[name]["bias"].items() if kk.endswith(("m0", "m4", "m8"))}, flush=True)
    json.dump(out, open(os.path.join(HERE, "frozen.json"), "w"), indent=1)
