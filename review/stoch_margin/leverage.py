"""T79 check 2: leverage Var(E[f|S]) / Var(f) for several skeletons S (DIAGNOSIS ONLY, random inputs).
f = 1[model correct]. 1/leverage = p(1-p)/Var(E[f|S]) is the conditional-MC variance-reduction ceiling: the most
any method that computes E[f|S] EXACTLY and samples S can gain over plain sampling, per sample of S, at zero cost.
Independent estimator from E31's: the PAIR trick, Var(E[f|S]) = Cov(f(S,U1), f(S,U2)) with two independent
completions U1, U2 of the same S (no per-skeleton q estimates). Discrete S (record times) use an unbiased
grouped estimator instead.
  topK      positions + values of the K largest inputs, K = 1..5
  posK      positions (in rank order) of the K largest inputs only
  pos3v12   positions of the top 3 + values of the top 2 (v3 and the low inputs integrated)
  rec1      set of times where the running max is broken (upper record times)
  rec2      sequence of events (t, new max | new 2nd max) of the running top-2: what a 'max/2nd-max tracker' sees
usage: python review/stoch_margin/leverage.py MODEL [N]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import truncnorm

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
model = rv.dev_model(name)
rng = np.random.default_rng(202)
T = rv.T
CH = 50000


def correct(x):
    return (rv.run(model, x)[0].argmax(1) == np.argsort(-x, 1)[:, 1]).astype(float)


def complete_topK(pos, val, K):
    free = rv.free_positions(pos)
    low = rv.low_draws(val[:, -1], 1, T - K, rng)
    return rv.assemble(pos, val, free, low)[:, 0]


def complete_posK(pos, K):
    n = len(pos)
    s = -np.sort(-rng.standard_normal((n, T)), 1)
    free = rv.free_positions(pos)
    perm = np.argsort(rng.uniform(size=(n, T - K)), 1)
    x = np.empty((n, T))
    x[np.arange(n)[:, None], pos] = s[:, :K]
    x[np.arange(n)[:, None], np.take_along_axis(free, perm, 1)] = s[:, K:]
    return x


def complete_pos3v12(pos, val):
    n = len(pos)
    rest = truncnorm.rvs(-np.inf, np.broadcast_to(val[:, 1:2], (n, 8)), size=(n, 8), random_state=rng)
    rest = -np.sort(-rest, 1)
    x = np.empty((n, T))
    x[np.arange(n)[:, None], pos[:, :2]] = val[:, :2]
    x[np.arange(n), pos[:, 2]] = rest[:, 0]
    free = rv.free_positions(pos)
    x[np.arange(n)[:, None], np.take_along_axis(free, np.argsort(rng.uniform(size=(n, 7)), 1), 1)] = rest[:, 1:]
    return x


out = dict(model=name, N=N)
fs = {}
for c in range(N // CH):
    x = rng.standard_normal((CH, T))
    order = np.argsort(-x, 1)
    f0 = correct(x)
    fs.setdefault("p", []).append(f0)
    for K in range(1, 6):
        pos = order[:, :K]; val = np.take_along_axis(x, pos, 1)
        fs.setdefault(f"top{K}", []).append(f0 * correct(complete_topK(pos, val, K)))
    for K in (2, 3, 4):
        fs.setdefault(f"pos{K}", []).append(f0 * correct(complete_posK(order[:, :K], K)))
    pos = order[:, :3]
    fs.setdefault("pos3v12", []).append(f0 * correct(complete_pos3v12(pos, np.take_along_axis(x, pos, 1))))
    print(c, flush=True)
p = np.concatenate(fs["p"]).mean()
out["acc"] = float(p)
for k, v in fs.items():
    if k == "p":
        continue
    v = np.concatenate(v)
    var = v.mean() - p ** 2
    out[k] = dict(leverage=float(var / (p * (1 - p))), cmc_gain_ceiling=float(p * (1 - p) / var),
                  se_leverage=float(v.std() / np.sqrt(len(v)) / (p * (1 - p))))

# record-time skeletons: grouped unbiased estimator of E[q_S^2]
NR = 4 * N
cnt = {k: defaultdict(lambda: [0, 0]) for k in ("rec1", "rec2")}
for c in range(NR // CH):
    x = rng.standard_normal((CH, T))
    f = correct(x)
    run_max = np.maximum.accumulate(x, 1)
    rec1 = np.concatenate([np.ones((CH, 1), bool), x[:, 1:] > run_max[:, :-1]], 1)
    k1 = rec1 @ (1 << np.arange(T))
    # running 2nd max events: 0 nothing, 1 new max, 2 new 2nd max
    ev = np.zeros((CH, T), np.int64)
    m1 = x[:, 0].copy(); m2 = np.full(CH, -np.inf); ev[:, 0] = 1
    for t in range(1, T):
        a = x[:, t] > m1; b = (~a) & (x[:, t] > m2)
        ev[a, t] = 1; ev[b, t] = 2
        m2 = np.where(a, m1, np.where(b, x[:, t], m2)); m1 = np.where(a, x[:, t], m1)
    k2 = ev @ (3 ** np.arange(T))
    for key, kk in (("rec1", k1), ("rec2", k2)):
        u, inv = np.unique(kk, return_inverse=True)
        n_g = np.bincount(inv); s_g = np.bincount(inv, weights=f)
        d = cnt[key]
        for uu, nn, ss in zip(u, n_g, s_g):
            d[uu][0] += nn; d[uu][1] += ss
pr = None
for key in ("rec1", "rec2"):
    n_g = np.array([v[0] for v in cnt[key].values()], float); s_g = np.array([v[1] for v in cnt[key].values()])
    tot = n_g.sum(); pr = s_g.sum() / tot
    ok = n_g >= 2
    Eq2 = np.sum(n_g[ok] / tot * (s_g[ok] ** 2 - s_g[ok]) / (n_g[ok] * (n_g[ok] - 1)))
    Eq2 += np.sum(n_g[~ok] / tot * (s_g[~ok] / n_g[~ok]) * pr)       # singletons: shrink to p (tiny mass)
    var = Eq2 - pr ** 2
    out[key] = dict(leverage=float(var / (pr * (1 - pr))), cmc_gain_ceiling=float(pr * (1 - pr) / var),
                    n_groups=int(len(n_g)), singleton_mass=float(n_g[~ok].sum() / tot))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/lev_{name}.json", "w"), indent=1)
