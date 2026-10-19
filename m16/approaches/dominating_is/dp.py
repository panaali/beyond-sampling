"""Seeded local dominating points of E1 (t0 = 1 normalisation; scale by t0 afterwards).
Pilot: x ~ N(0, I), keep cone points (argmax logits = position of x_(3)), rank by kappa = g23 / ||x||.
Local descent: in the linear region of the current point (fixed gate pattern and triple), solve the convex QP
  min ||x||^2  s.t. gates as in the pattern, x_a >= x_b, x_b - x_c >= 1, x_c >= x_k (k not in triple), l_c >= l_j;
if gate rows are active at the optimum, flip them and re-solve in the neighbouring region; stop when no decrease."""
import time
import numpy as np
from scipy.optimize import minimize
from net import Wi, Wh, Wo, H, T, logits, top3


def affine(x):
    """gate pattern G (T, H) and the linear maps: pre-activation rows A[t,k] (a = A x), logit rows Lg (10, 10)."""
    G = np.zeros((T, H), bool); A = np.zeros((T, H, 10)); M = np.zeros((H, 10))        # h_t = M x
    for t in range(T):
        pre = Wh @ M; pre[:, t] += Wi
        a = pre @ x; G[t] = a > 0; A[t] = pre
        M = pre * G[t][:, None]
    return G, A, Wo @ M


def region_rows(A, G, Lg, tri, flip=()):
    """inequality rows R x >= 0 / >= 1 for the region; returns (R, b) with R x >= b."""
    a, b_, c = tri; rows, rhs = [], []
    G = G.copy()
    for (t, k) in flip:
        G[t, k] = ~G[t, k]
    # recompute A with the flipped pattern (later steps depend on gates)
    M = np.zeros((H, 10)); As = np.zeros_like(A)
    for t in range(T):
        pre = Wh @ M; pre[:, t] += Wi; As[t] = pre; M = pre * G[t][:, None]
    Lg = Wo @ M
    for t in range(T):
        for k in range(H):
            rows.append(As[t, k] if G[t, k] else -As[t, k]); rhs.append(0.0)
    e = np.eye(10)
    rows.append(e[a] - e[b_]); rhs.append(0.0)
    rows.append(e[b_] - e[c]); rhs.append(1.0)
    for k in range(10):
        if k not in tri:
            rows.append(e[c] - e[k]); rhs.append(0.0)
    for j in range(10):
        if j != c:
            rows.append(Lg[c] - Lg[j]); rhs.append(0.0)
    return np.array(rows), np.array(rhs), G


def qp(R, b, x0):
    cons = [{"type": "ineq", "fun": lambda x: R @ x - b, "jac": lambda x: R}]
    r = minimize(lambda x: x @ x, x0, jac=lambda x: 2 * x, constraints=cons, method="SLSQP",
                 options=dict(maxiter=500, ftol=1e-12))
    return r.x, (R @ r.x - b).min() > -1e-7


def descend(x0, max_iter=60):
    top, v = top3(x0[None]); tri = tuple(top[0]); x = x0 / (v[0, 1] - v[0, 2])          # normalise to g23 = 1
    G, A, Lg = affine(x); flip = (); best = x @ x; path = 0
    for _ in range(max_iter):
        R, b, Gp = region_rows(A, G, Lg, tri, flip)
        xn, ok = qp(R, b, x)
        if not ok or xn @ xn >= best - 1e-10:
            break
        x, best = xn, xn @ xn; path += 1
        G, A, Lg = affine(x)
        act = [(t, k) for t in range(T) for k in range(H) if abs(A[t, k] @ x) < 1e-7]
        flip = tuple(act)
    # validate: event at the point (nudged slightly outward along x to break ties) holds
    xt = x * (1 + 1e-9)
    tp, vv = top3(xt[None]); ok = (logits(xt[None]).argmax(1)[0] == tp[0, 2]) and vv[0, 1] - vv[0, 2] >= 1 - 1e-6
    return x, float(best), tri, bool(ok), path


def pilot(n, seed, keep=50):
    rng = np.random.default_rng(seed); X = rng.standard_normal((n, 10))
    top, v = top3(X); g23 = v[:, 1] - v[:, 2]
    K = logits(X).argmax(1) == top[:, 2]
    kap = np.where(K, g23 / np.linalg.norm(X, axis=1), -1)
    idx = np.argsort(-kap)[:keep]
    return X[idx], kap[idx], int(K.sum())


if __name__ == "__main__":
    import sys
    t = time.time(); S, kap, nK = pilot(int(float(sys.argv[1])), 1, 40); print("pilot cone points", nK, "best kappa", kap[:5].round(3), "-> 1/kappa^2", (1 / kap[:5] ** 2).round(2), "%.0fs" % (time.time() - t), flush=True)
    res = []
    for i, s in enumerate(S):
        t = time.time(); x, r2, tri, ok, path = descend(s)
        res.append((r2, tri, ok)); print(i, "seed 1/k^2 %.3f -> r2 %.4f tri %s ok %s steps %d (%.1fs)" % (1 / kap[i] ** 2, r2, tri, ok, path, time.time() - t), flush=True)
