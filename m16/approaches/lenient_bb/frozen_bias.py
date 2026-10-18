"""T274b fallback feasibility (DIAGNOSTIC, uses random inputs; not an estimator): bias of rank_exact's frozen-gate
approximation when the last m steps are run with the true ReLU, on all 30 LB3 models.

a_m(x): steps 1..T-m use the gates of the canonical input of x's class (expected order statistics given K placed
by rank; rank_exact's design), as an exact linear map; steps T-m+1..T apply the true ReLU to that state.
Designs: "hash" = rank_exact's 7,920 classes (top-3 triple x K, other ranks in a hashed arrangement);
"full" = the canonical input of x's own full ordering (ideal; infeasible as an estimator).
bias(m) = mean(correct_approx) - mean(correct_net), paired. For "hash" the input is x' (x's values with ranks 4..10
moved into the hashed arrangement, as rank_exact integrates); design_bias = net(x') - net(x) is reported separately.  usage: python frozen_bias.py [n] -> results/frozen_bias.json
"""
import json
import math
import os
import sys
from itertools import permutations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T, H = 10, 16
MS = (0, 1, 2, 3, 4, 6, 8)
SIGMAS = (0.0025, 0.005, 0.01, 0.02); N_NOISY = 6


def lb3_models():
    """Same construction as m16/leaderboard.py models(16, 10) (v3)."""
    from alg_zoo import example_2nd_argmax, zoo_2nd_argmax
    f = lambda m: (m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64),
                   m.rnn.weight_hh_l0.detach().numpy().astype(np.float64), m.linear.weight.detach().numpy().astype(np.float64))
    base = f(example_2nd_argmax()); ms = {"orig": base}
    for sd in range(5):
        ms[f"seed{sd}"] = f(zoo_2nd_argmax(16, 10, seed=sd))
    for sg in SIGMAS:
        rng = np.random.default_rng(0)
        for k in range(N_NOISY):
            ms[f"s{sg}_{k}"] = tuple(W + sg * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in base)
    return ms


_ref = np.sort(np.random.default_rng(0).standard_normal((2_000_000, T)), 1)[:, ::-1]      # model-free table
_K = (_ref > 0).sum(1)
EOSK = np.array([_ref[_K == k].mean(0) for k in range(T + 1)])                            # rank-ordered means
del _ref, _K
HASH_REST = {}
for j, (p1, p2, p3) in enumerate(permutations(range(T), 3)):
    rest = [i for i in range(T) if i not in (p1, p2, p3)]
    HASH_REST[(p1, p2, p3)] = list(np.random.default_rng(1000 + j).permutation(rest))       # as rank_exact.design


def rearranged(x):
    """x' = x's values with ranks 4..10 moved into rank_exact's hashed arrangement for x's top-3 triple."""
    order = np.argsort(-x, 1); vals = -np.sort(-x, 1); xp = np.empty_like(x)
    for i in range(len(x)):
        o = list(order[i, :3]) + HASH_REST[tuple(order[i, :3])]
        xp[i, o] = vals[i]
    return xp


def canonical(x, design):
    """Canonical input per row: EOS[K] placed by rank, ordering = full (own) or hash (top-3 own, rest hashed)."""
    order = np.argsort(-x, 1)                         # order[:, r] = position of rank r
    K = (x > 0).sum(1)
    if design == "hash":
        order = order.copy()
        for i in range(len(x)):
            order[i, 3:] = HASH_REST[tuple(order[i, :3])]
    xc = np.empty_like(x)
    np.put_along_axis(xc, order, EOSK[K], 1)
    return xc


def gates(model, xc):
    Wi, Wh, _ = model
    h = np.zeros((len(xc), H)); G = []
    for t in range(T):
        z = h @ Wh.T + np.outer(xc[:, t], Wi); g = z > 0; G.append(g); h = np.where(g, z, 0)
    return G


def approx_pred(model, x, G, m):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), H))
    for t in range(T):
        z = h @ Wh.T + np.outer(x[:, t], Wi)
        h = np.where(G[t], z, 0.0) if t < T - m else np.maximum(z, 0)
    return (h @ Wo.T).argmax(1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    ms = lb3_models(); out = {}
    for name, model in ms.items():
        rng = np.random.default_rng(274_000 + sum(map(ord, name)))
        acc = {}
        for s in range(0, n, 100_000):
            x = rng.standard_normal((min(100_000, n - s), T)); y = np.argsort(-x, 1)[:, 1]
            Wi, Wh, Wo = model; h = np.zeros((len(x), H))
            for t in range(T):
                h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
            ok_net = (h @ Wo.T).argmax(1) == y
            xp = rearranged(x); h = np.zeros((len(x), H))
            for t in range(T):
                h = np.maximum(h @ Wh.T + np.outer(xp[:, t], Wi), 0)
            ok_netp = (h @ Wo.T).argmax(1) == y                     # same 2nd-largest position by construction
            blind = (x > 0).sum(1) <= 1
            acc.setdefault("net", []).append(ok_net.sum())
            acc.setdefault("design_bias", []).append((ok_netp.astype(int) - ok_net).sum())
            for design, xx, ref in (("hash", xp, ok_netp), ("full", x, ok_net)):
                G = gates(model, canonical(xx, design))
                for m in MS:
                    d = (approx_pred(model, xx, G, m) == y).astype(int) - ref
                    acc.setdefault(f"{design}_m{m}", []).append(d.sum())
                    acc.setdefault(f"{design}_m{m}_blind", []).append(d[blind].sum())
        out[name] = {k: float(np.sum(v) / n) for k, v in acc.items()}
        print(name, {k: round(100 * v, 3) for k, v in out[name].items() if "blind" not in k}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "frozen_bias.json"), "w"), indent=1)
    grp = lambda m: "orig" if m == "orig" else "seed" if m.startswith("seed") else m.rsplit("_", 1)[0]
    print("\nper-group RMS bias (pp):")
    for key in ["design_bias"] + [f"{d}_m{m}" for d in ("hash", "full") for m in MS]:
        row = {}
        for g in ["orig", "seed", "s0.0025", "s0.005", "s0.01", "s0.02"]:
            v = [out[mn][key] for mn in out if grp(mn) == g]
            row[g] = round(100 * math.sqrt(np.mean(np.square(v))), 3)
        print(key, row)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    main()
