"""T398 diagnostic: search for event points inside a surviving cell by hit-and-run in
P = {G x >= 0, |x_i| <= 1, g23 >= DELTA} (uniform), plus a local ascent on margin = logit_p3 - max other logit.
A cell where an event point is found is genuinely event-feasible: no sound relaxation can prune it."""
import sys

import numpy as np

sys.path.insert(0, "../..")
import common  # noqa: E402

DELTA = 0.1


def polytope(G, tri):
    p1, p2, p3 = tri; rows = [g for g in G]
    r = np.zeros(10); r[p2] = 1; r[p3] = -1; rows.append(r); b = [0.] * len(G) + [DELTA]
    for i in range(10):
        e = np.zeros(10); e[i] = 1; rows.append(-e); b.append(-1.); rows.append(e); b.append(-1.)
    return np.array(rows), np.array(b)                             # R x >= b


def margin(X, p3):
    L = common.run(X)[:, -1] @ common.Wo.T
    o = np.delete(L, p3, 1).max(1)
    return L[:, p3] - o


def hit_and_run(R, b, x0, n, rng, thin=5):
    x = x0.copy(); out = []
    for i in range(n * thin):
        d = rng.standard_normal(10); d /= np.linalg.norm(d)
        Rd = R @ d; s = R @ x - b                                   # s >= 0; need s + t Rd >= 0
        with np.errstate(divide="ignore"):
            tmax = np.min(np.where(Rd < 0, -s / Rd, np.inf)); tmin = np.max(np.where(Rd > 0, -s / Rd, -np.inf))
        x = x + rng.uniform(tmin, tmax) * d
        if i % thin == 0:
            out.append(x.copy())
    return np.array(out)


def chebyshev(R, b):
    from scipy.optimize import linprog
    nr = np.linalg.norm(R, axis=1)
    r = linprog(np.r_[np.zeros(10), -1], A_ub=np.c_[-R, nr], b_ub=-b, bounds=[(None, None)] * 10 + [(0, None)], method="highs")
    return r.x[:10], r.x[10]


def probe(G, tri, n=4000, seed=0):
    rng = np.random.default_rng(seed); R, b = polytope(G, tri)
    x0, rad = chebyshev(R, b)
    if rad <= 1e-12:
        return dict(radius=rad, found=False, best=None)
    X = hit_and_run(R, b, x0, n, rng)
    m = margin(X, tri[2]); best = float(m.max())
    # local ascent from the best 20 points: random steps inside P, keep improvements
    for i in np.argsort(-m)[:20]:
        x = X[i]; mx = m[i]
        for _ in range(200):
            y = x + rng.standard_normal(10) * 0.02
            if np.all(R @ y >= b):
                my = margin(y[None], tri[2])[0]
                if my > mx:
                    x, mx = y, my
        best = max(best, float(mx))
    return dict(radius=float(rad), found=best > 0, best=best)
