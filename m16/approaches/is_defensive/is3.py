"""T454: dominating-point IS v3 = v2 (../dominating_is2/is2.py) + a defensive component (Hesterberg 1995): the
gap-prefiltered nominal law q_def(x) = phi(x) 1{g23(x) >= t0} / Z(t0), sampled exactly (v3 from its density,
tops / lows from truncated normals, random positions) with Z(t0) by quadrature. Since the event lies inside
{g23 >= t0}, every IS weight on a hit is <= Z / a_d: per draw, the variance is at most (Z / a_d) p, i.e. never worse
than the prefiltered MC (variance p Z per network pass ... per draw p(1-p)) by more than a factor ~1/a_d."""
import os, sys, time
import numpy as np
from scipy import integrate
from scipy.special import ndtr, ndtri, log_ndtr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dominating_is2"))
import is2
from is2 import Net, top3, event, select, find_points

AD = 0.2                    # defensive weight (pre-registered)


def phi(v):
    return np.exp(-0.5 * v * v) / np.sqrt(2 * np.pi)


def Z(t0):
    f = lambda v: 360 * phi(v) * ndtr(v) ** 7 * ndtr(-(v + t0)) ** 2
    return integrate.quad(f, -12, 12, epsabs=1e-16, epsrel=1e-12, limit=400)[0]


class Defensive:
    def __init__(self, t0, ngrid=200_001):
        self.t0 = t0; self.Z = Z(t0)
        v = np.linspace(-9, 9, ngrid)
        lg = np.log(360) - 0.5 * v * v - 0.5 * np.log(2 * np.pi) + 7 * log_ndtr(v) + 2 * log_ndtr(-(v + t0))
        d = np.exp(lg - lg.max()); c = np.concatenate([[0], np.cumsum((d[1:] + d[:-1]) / 2)]); c /= c[-1]
        self.v, self.c = v, c

    def sample(self, rng, n):
        v3 = np.interp(rng.random(n), self.c, self.v)
        a = v3 + self.t0
        tops = -ndtri(rng.random((n, 2)) * ndtr(-a)[:, None])            # N | > a  (upper tail via ndtr(-a))
        lows = ndtri(rng.random((n, 7)) * ndtr(v3)[:, None])             # N | < v3
        vals = np.concatenate([tops, v3[:, None], lows], 1)
        perm = np.argsort(rng.random((n, 10)), 1)                         # random positions
        X = np.empty((n, 10)); np.put_along_axis(X, perm, vals, 1)
        return X

    def logq(self, X):              # on the gap set only (all hits lie there)
        return -0.5 * (X ** 2).sum(1) - np.log(self.Z)


class Mix3:
    """AD * q_def + (1 - AD) * sum_k w_k N(m_k, C_k); densities without the common (2 pi)^-5."""
    def __init__(self, dfn, means, covs, w):
        self.d = dfn; self.m = np.array(means); self.w = np.asarray(w) / np.sum(w)
        self.L = [np.linalg.cholesky(C) for C in covs]; self.P = [np.linalg.inv(C) for C in covs]
        self.ld = np.array([2 * np.log(np.diag(L)).sum() for L in self.L])

    def sample(self, rng, n):
        q = np.r_[AD, (1 - AD) * self.w]; comp = rng.choice(len(q), size=n, p=q); X = rng.standard_normal((n, 10))
        s0 = comp == 0
        if s0.any():
            X[s0] = self.d.sample(rng, int(s0.sum()))
        for k in range(len(self.w)):
            s = comp == k + 1
            if s.any():
                X[s] = self.m[k] + X[s] @ self.L[k].T
        return X, comp

    def logq(self, X):              # evaluated on hits (gap set)
        ls = [np.log(AD) + self.d.logq(X)]
        for k in range(len(self.w)):
            D = X - self.m[k]
            ls.append(np.log((1 - AD) * self.w[k]) - 0.5 * self.ld[k] - 0.5 * np.einsum("ij,jk,ik->i", D, self.P[k], D))
        Lm = np.stack(ls, 1); mx = Lm.max(1); return mx + np.log(np.exp(Lm - mx[:, None]).sum(1))


def estimate(net, pts_all, t0, seed, n1=1_000_000, n2=1_000_000, extend=True):
    """as is2.estimate, with the defensive component in BOTH stages; stage-1 hits drawn by the defensive part are
    assigned to their nearest centre like any other hit."""
    sel = select(pts_all["points"], t0); t = time.time(); dfn = Defensive(t0)
    mu = [t0 * np.array(p["x"]) for p in sel]; lw = np.array([-t0 ** 2 * p["r2"] / 2 for p in sel]); w0 = np.exp(lw - lw.max())
    is2.Mix = None
    mix1 = Mix3(dfn, mu, [np.eye(10)] * len(mu), w0)
    Xh, W, T3, f1 = is2.draw(net, mix1, t0, n1, seed)
    means, covs, wts = [], [], []; contrib = np.zeros(len(mu))
    near = np.argmin(((Xh[:, None, :] - np.array(mu)[None]) ** 2).sum(2), 1) if len(W) else np.zeros(0, int)
    for k in range(len(mu)):
        contrib[k] = W[near == k].sum() if len(W) else 0.0
    share = contrib / contrib.sum() if contrib.sum() > 0 else np.ones(len(mu)) / len(mu)
    for k in range(len(mu)):
        s = near == k; base = max(share[k], 0.01)
        if s.sum() >= 30:
            ww = W[s] / W[s].sum(); m = ww @ Xh[s]; D = Xh[s] - m; C = (D * ww[:, None]).T @ D
            ev, V = np.linalg.eigh(1.5 * C); ev = np.clip(ev, 0.01, 4.0)
            means += [m, mu[k]]; covs += [(V * ev) @ V.T, np.eye(10)]; wts += [0.8 * base, 0.2 * base]
        else:
            means += [mu[k]]; covs += [np.eye(10)]; wts += [base]
    mix2 = Mix3(dfn, means, covs, wts)
    Xh2, W2, T32, f2 = is2.draw(net, mix2, t0, n2, seed + 1); n = n2
    def stats(W2, n):
        p = W2.sum() / n; var = ((W2 ** 2).sum() / n - p ** 2) * n / (n - 1); return p, np.sqrt(max(var, 0) / n)
    p, se = stats(W2, n); ext = False
    if extend and (p == 0 or se / p > 0.1):
        Xe, We, Te, fe = is2.draw(net, mix2, t0, 9 * n2, seed + 2); W2 = np.r_[W2, We]; T32 = np.r_[T32, Te]; f2 += fe
        n = 10 * n2; p, se = stats(W2, n); ext = True
    wmax = Z(t0) / AD
    desc_fpe = pts_all["descent_sec"] * 1e6 / is2.US_PER_FPE
    C = pts_all["pilot_fpe"] + desc_fpe + f1 + f2; C_draws = pts_all["pilot_draws"] + desc_fpe + n1 + n
    tri_hits = {}
    for tr in map(tuple, T32.tolist()):
        tri_hits[str(tr)] = tri_hits.get(str(tr), 0) + 1
    return dict(t0=t0, p=float(p), se=float(se), rel_se=float(se / p) if p > 0 else None, n2=n, extended=ext,
                hits=int(len(W2)), max_weight=float(W2.max()) if len(W2) else 0.0, weight_cap=wmax, Z=dfn.Z,
                fwd_stage1=f1, fwd_stage2=f2, C=C, C_draws=C_draws,
                F=float(se ** 2 * C / (p * (1 - p))) if p > 0 else None,
                F_draws=float(se ** 2 * C_draws / (p * (1 - p))) if p > 0 else None,
                n_points=len(sel), n_components=len(means), tri_hits=tri_hits, sec=time.time() - t)
