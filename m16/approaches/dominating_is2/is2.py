"""T453: dominating-point IS v2 for E1(t0) on any 16x10 2nd-argmax RNN without biases (README 'Pre-registration').
Changes from T436 (../dominating_is): cheap pilot (net only on the top 1e5 of 2e6 draws by g23/||x||), a per-triple
cap on mixture points, and a two-stage adaptive mixture: stage 1 fits each component's mean and covariance to its own
weighted hits (cross-entropy step; anisotropic across the tie sliver), stage 2 scores."""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

US_PER_FPE = 4.0
A0 = 0.05            # defensive N(0, I) weight
CAP_TRI, CAP_ALL = 3, 60


class Net:
    def __init__(self, Wi, Wh, Wo):
        self.Wi, self.Wh, self.Wo = Wi, Wh, Wo; self.H = len(Wi); self.T = Wo.shape[0]

    def logits(self, X):
        h = np.zeros((len(X), self.H))
        for t in range(self.T):
            h = np.maximum(h @ self.Wh.T + X[:, t:t + 1] * self.Wi[None], 0)
        return h @ self.Wo.T


def top3(X):
    top = np.argpartition(-X, 3, axis=1)[:, :3]; v = np.take_along_axis(X, top, 1)
    o = np.argsort(-v, 1); return np.take_along_axis(top, o, 1), np.take_along_axis(v, o, 1)


def event(net, X, t0):
    top, v = top3(X); sel = v[:, 1] - v[:, 2] >= t0; hit = np.zeros(len(X), bool)
    if sel.any():
        hit[sel] = net.logits(X[sel]).argmax(1) == top[sel, 2]
    return hit, top, int(sel.sum())


# ------------------------------------------------------------------ local dominating points (as T436 dp.py)
def affine(net, x):
    T, H = net.T, net.H; G = np.zeros((T, H), bool); A = np.zeros((T, H, 10)); M = np.zeros((H, 10))
    for t in range(T):
        pre = net.Wh @ M; pre[:, t] += net.Wi; a = pre @ x; G[t] = a > 0; A[t] = pre; M = pre * G[t][:, None]
    return G, A


def region_rows(net, G, tri, flip=()):
    T, H = net.T, net.H; a, b_, c = tri; G = G.copy()
    for (t, k) in flip:
        G[t, k] = ~G[t, k]
    M = np.zeros((H, 10)); rows, rhs = [], []
    for t in range(T):
        pre = net.Wh @ M; pre[:, t] += net.Wi
        for k in range(H):
            rows.append(pre[k] if G[t, k] else -pre[k]); rhs.append(0.0)
        M = pre * G[t][:, None]
    Lg = net.Wo @ M; e = np.eye(10)
    rows += [e[a] - e[b_], e[b_] - e[c]]; rhs += [0.0, 1.0]
    for k in range(10):
        if k not in tri:
            rows.append(e[c] - e[k]); rhs.append(0.0)
    for j in range(10):
        if j != c:
            rows.append(Lg[c] - Lg[j]); rhs.append(0.0)
    return np.array(rows), np.array(rhs)


def descend(net, x0, max_iter=60):
    top, v = top3(x0[None]); tri = tuple(int(i) for i in top[0]); x = x0 / (v[0, 1] - v[0, 2])
    G, A = affine(net, x); flip = (); best = x @ x
    for _ in range(max_iter):
        R, b = region_rows(net, G, tri, flip)
        r = minimize(lambda z: z @ z, x, jac=lambda z: 2 * z, method="SLSQP", options=dict(maxiter=500, ftol=1e-12),
                     constraints=[{"type": "ineq", "fun": lambda z: R @ z - b, "jac": lambda z: R}])
        if (R @ r.x - b).min() < -1e-7 or r.x @ r.x >= best - 1e-10:
            break
        x, best = r.x, r.x @ r.x; G, A = affine(net, x)
        flip = tuple((t, k) for t in range(net.T) for k in range(net.H) if abs(A[t, k] @ x) < 1e-7)
    return x, float(best), tri


def find_points(net, seed, n_draw=2_000_000, n_net=100_000, n_seed=60):
    t = time.time(); rng = np.random.default_rng(seed); cand = []
    for _ in range(n_draw // 500_000):
        X = rng.standard_normal((500_000, 10)); top, v = top3(X)
        k0 = (v[:, 1] - v[:, 2]) / np.linalg.norm(X, axis=1); idx = np.argsort(-k0)[:n_net * 500_000 // n_draw]
        cand.append((X[idx], k0[idx]))
    X = np.concatenate([c[0] for c in cand]); top, v = top3(X)
    K = net.logits(X).argmax(1) == top[:, 2]; kap = np.where(K, (v[:, 1] - v[:, 2]) / np.linalg.norm(X, axis=1), -1)
    seeds = X[np.argsort(-kap)[:n_seed]]; seeds = seeds[kap[np.argsort(-kap)[:n_seed]] > 0]
    t_pilot = time.time() - t; t = time.time(); ends = []
    for s in seeds:
        x, r2, tri = descend(net, s); ends.append(dict(x=x.tolist(), r2=r2, tri=list(tri)))
    t_desc = time.time() - t; ends.sort(key=lambda e: e["r2"]); pts = []
    for e in ends:
        if all(np.linalg.norm(np.array(e["x"]) - np.array(p["x"])) >= 0.05 for p in pts):
            pts.append(e)
    return dict(points=pts, pilot_fpe=len(X), pilot_draws=n_draw, pilot_sec=t_pilot, descent_sec=t_desc,
                n_cone_seeds=int(len(seeds)))


def select(pts, t0):
    rmin = pts[0]["r2"]; cut = rmin + 2 * np.log(1000) / t0 ** 2; per = {}; out = []
    for p in pts:
        k = tuple(p["tri"])
        if p["r2"] <= cut and per.get(k, 0) < CAP_TRI and len(out) < CAP_ALL:
            per[k] = per.get(k, 0) + 1; out.append(p)
    return out


# ------------------------------------------------------------------ mixture of Gaussians
class Mix:
    def __init__(self, means, covs, w):
        self.m = np.array(means); self.w = np.asarray(w) / np.sum(w)
        self.L = [np.linalg.cholesky(C) for C in covs]
        self.P = [np.linalg.inv(C) for C in covs]
        self.ld = np.array([2 * np.log(np.diag(L)).sum() for L in self.L])

    def sample(self, rng, n):
        q = np.r_[A0, (1 - A0) * self.w]; comp = rng.choice(len(q), size=n, p=q)
        X = rng.standard_normal((n, 10))
        for k in range(len(self.w)):
            s = comp == k + 1
            if s.any():
                X[s] = self.m[k] + X[s] @ self.L[k].T
        return X, comp

    def logq(self, X):
        l0 = np.log(A0) - 0.5 * (X ** 2).sum(1); ls = [l0]
        for k in range(len(self.w)):
            D = X - self.m[k]
            ls.append(np.log((1 - A0) * self.w[k]) - 0.5 * self.ld[k] - 0.5 * np.einsum("ij,jk,ik->i", D, self.P[k], D))
        Lm = np.stack(ls, 1); mx = Lm.max(1)
        return mx + np.log(np.exp(Lm - mx[:, None]).sum(1))


def draw(net, mix, t0, n, seed, chunk=200_000):
    rng = np.random.default_rng(seed); Xh, Wh_, fwd, tri = [], [], 0, []
    for s in range(0, n, chunk):
        m = min(chunk, n - s); X, _ = mix.sample(rng, m); hit, top, k = event(net, X, t0); fwd += k
        if hit.any():
            w = np.exp(-0.5 * (X[hit] ** 2).sum(1) - mix.logq(X[hit])); Xh.append(X[hit]); Wh_.append(w); tri.append(top[hit])
    Xh = np.concatenate(Xh) if Xh else np.zeros((0, 10)); W = np.concatenate(Wh_) if Wh_ else np.zeros(0)
    T3 = np.concatenate(tri) if tri else np.zeros((0, 3), int)
    return Xh, W, T3, fwd


def estimate(net, pts_all, t0, seed, n1=1_000_000, n2=1_000_000, extend=True):
    sel = select(pts_all["points"], t0); t = time.time()
    mu = [t0 * np.array(p["x"]) for p in sel]; lw = np.array([-t0 ** 2 * p["r2"] / 2 for p in sel])
    w0 = np.exp(lw - lw.max())
    mix1 = Mix(mu, [np.eye(10)] * len(mu), w0)
    Xh, W, T3, f1 = draw(net, mix1, t0, n1, seed)                                   # stage 1 (discarded for the estimate)
    means, covs, contrib = [], [], np.zeros(len(mu))
    if len(W):
        near = np.argmin(((Xh[:, None, :] - np.array(mu)[None]) ** 2).sum(2), 1)
    for k in range(len(mu)):
        s = (near == k) if len(W) else np.zeros(0, bool)
        contrib[k] = W[s].sum() if len(W) else 0.0
        if s.sum() >= 30:
            ww = W[s] / W[s].sum(); m = ww @ Xh[s]; D = Xh[s] - m; C = (D * ww[:, None]).T @ D
            ev, V = np.linalg.eigh(1.5 * C); ev = np.clip(ev, 0.01, 4.0); C = (V * ev) @ V.T
            means += [m, mu[k]]; covs += [C, np.eye(10)]; split = (0.8, 0.2)
        else:
            means += [mu[k]]; covs += [np.eye(10)]; split = (1.0,)
    share = contrib / contrib.sum() if contrib.sum() > 0 else np.ones(len(mu)) / len(mu)
    wts = []
    for k in range(len(mu)):
        base = max(share[k], 0.01)
        n_fit = ((near == k).sum() >= 30) if len(W) else False
        wts += [0.8 * base, 0.2 * base] if n_fit else [base]
    mix2 = Mix(means, covs, wts)
    rng_seed = seed + 1
    Xh2, W2, T32, f2 = draw(net, mix2, t0, n2, rng_seed); n = n2
    def stats(W2, n):
        s1 = W2.sum(); s2 = (W2 ** 2).sum(); p = s1 / n; var = (s2 / n - p ** 2) * n / (n - 1); return p, np.sqrt(max(var, 0) / n)
    p, se = stats(W2, n); ext = False
    if extend and (p == 0 or se / p > 0.1):
        Xe, We, Te, fe = draw(net, mix2, t0, 9 * n2, seed + 2); W2 = np.r_[W2, We]; T32 = np.r_[T32, Te]; f2 += fe
        n = 10 * n2; p, se = stats(W2, n); ext = True
    tri_hits = {}
    for tr in map(tuple, T32.tolist()):
        tri_hits[str(tr)] = tri_hits.get(str(tr), 0) + 1
    tri_contrib = {}
    for tr, w in zip(map(tuple, T32.tolist()), W2):
        tri_contrib[str(tr)] = tri_contrib.get(str(tr), 0) + w / n
    desc_fpe = pts_all["descent_sec"] * 1e6 / US_PER_FPE
    C = pts_all["pilot_fpe"] + desc_fpe + f1 + f2
    C_draws = pts_all["pilot_draws"] + desc_fpe + n1 + n
    return dict(t0=t0, p=float(p), se=float(se), rel_se=float(se / p) if p > 0 else None, n2=n, extended=ext,
                hits=int(len(W2)), fwd_stage1=f1, fwd_stage2=f2, C=C, C_draws=C_draws,
                F=float(se ** 2 * C / (p * (1 - p))) if p > 0 else None,
                F_draws=float(se ** 2 * C_draws / (p * (1 - p))) if p > 0 else None,
                n_points=len(sel), n_components=len(means), triples=[p_["tri"] for p_ in sel],
                tri_hits=tri_hits, tri_contrib=tri_contrib, sec=time.time() - t)
