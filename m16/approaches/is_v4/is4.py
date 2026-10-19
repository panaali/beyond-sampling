"""T456: dominating-point IS v4 = v3 (../is_defensive/is3.py) + (i) a discovery stage: stage 1 draws half from the
defensive gap-conditional law, and its hits seed new descents (<= 5 per triple, <= 100 in all), so the defensive part
finds regions the pilot missed; (ii) a heavy-tail gate on the reported interval, fixed before any truth."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "dominating_is2"), os.path.join(HERE, "..", "is_defensive")]
import is2
from is2 import top3, descend, select
from is3 import Defensive, Z, Mix3
import is3

AD1, AD2 = 0.5, 0.2           # defensive weight in stage 1 (discovery) and stage 2 (estimate)
G_MAXW, G_CAPSHARE = 0.05, 0.2


class Mix4(Mix3):
    def __init__(self, dfn, means, covs, w, ad):
        super().__init__(dfn, means, covs, w); self.ad = ad

    def sample(self, rng, n):
        q = np.r_[self.ad, (1 - self.ad) * self.w]; comp = rng.choice(len(q), size=n, p=q); X = rng.standard_normal((n, 10))
        s0 = comp == 0
        if s0.any():
            X[s0] = self.d.sample(rng, int(s0.sum()))
        for k in range(len(self.w)):
            s = comp == k + 1
            if s.any():
                X[s] = self.m[k] + X[s] @ self.L[k].T
        return X, comp

    def logq(self, X):
        ls = [np.log(self.ad) + self.d.logq(X)]
        for k in range(len(self.w)):
            D = X - self.m[k]
            ls.append(np.log((1 - self.ad) * self.w[k]) - 0.5 * self.ld[k] - 0.5 * np.einsum("ij,jk,ik->i", D, self.P[k], D))
        Lm = np.stack(ls, 1); mx = Lm.max(1); return mx + np.log(np.exp(Lm - mx[:, None]).sum(1))


def merge(pts, new):
    allp = sorted(pts + new, key=lambda e: e["r2"]); out = []
    for e in allp:
        if all(np.linalg.norm(np.array(e["x"]) - np.array(p["x"])) >= 0.05 for p in out):
            out.append(e)
    return out


def estimate(net, pts_all, t0, seed, n1=2_000_000, n2=1_000_000, extend=True):
    t = time.time(); dfn = Defensive(t0); pts = list(pts_all["points"]); sel = select(pts, t0)
    mu = [t0 * np.array(p["x"]) for p in sel]; lw = np.array([-t0 ** 2 * p["r2"] / 2 for p in sel]); w0 = np.exp(lw - lw.max())
    mix1 = Mix4(dfn, mu, [np.eye(10)] * len(mu), w0, AD1)
    Xh, W, T3, f1 = is2.draw(net, mix1, t0, n1, seed)
    # (i) discovery: seed descents from stage-1 hits, best (smallest ||x||/g23) first, <= 5 per triple, <= 100 total
    td = time.time(); new = []; per = {}
    if len(W):
        _, v = top3(Xh); r = np.linalg.norm(Xh, axis=1) / (v[:, 1] - v[:, 2])
        for i in np.argsort(r):
            k = tuple(int(j) for j in T3[i])
            if per.get(k, 0) >= 5 or len(new) >= 100:
                continue
            per[k] = per.get(k, 0) + 1; x, r2, tri = descend(net, Xh[i]); new.append(dict(x=x.tolist(), r2=r2, tri=list(tri)))
    t_disc = time.time() - td
    pts2 = merge(pts, new); sel2 = select(pts2, t0); mu2 = [t0 * np.array(p["x"]) for p in sel2]
    # CE fit on stage-1 hits (weights relative to mix1 are valid IS weights for the fit), assigned to nearest new centre
    means, covs, wts = [], [], []; contrib = np.zeros(len(mu2))
    near = np.argmin(((Xh[:, None, :] - np.array(mu2)[None]) ** 2).sum(2), 1) if len(W) else np.zeros(0, int)
    for k in range(len(mu2)):
        contrib[k] = W[near == k].sum() if len(W) else 0.0
    share = contrib / contrib.sum() if contrib.sum() > 0 else np.ones(len(mu2)) / len(mu2)
    for k in range(len(mu2)):
        s = near == k; base = max(share[k], 0.01)
        if s.sum() >= 30:
            ww = W[s] / W[s].sum(); m = ww @ Xh[s]; D = Xh[s] - m; C = (D * ww[:, None]).T @ D
            ev, V = np.linalg.eigh(1.5 * C); ev = np.clip(ev, 0.01, 4.0)
            means += [m, mu2[k]]; covs += [(V * ev) @ V.T, np.eye(10)]; wts += [0.8 * base, 0.2 * base]
        else:
            means += [mu2[k]]; covs += [np.eye(10)]; wts += [base]
    mix2 = Mix4(dfn, means, covs, wts, AD2)
    Xh2, W2, T32, f2 = is2.draw(net, mix2, t0, n2, seed + 1); n = n2

    def stats(W2, n):
        p = W2.sum() / n; var = ((W2 ** 2).sum() / n - p ** 2) * n / (n - 1); return p, np.sqrt(max(var, 0) / n)
    p, se = stats(W2, n); ext = False
    if extend and (p == 0 or se / p > 0.1):
        Xe, We, Te, fe = is2.draw(net, mix2, t0, 9 * n2, seed + 2); W2 = np.r_[W2, We]; T32 = np.r_[T32, Te]; f2 += fe
        n = 10 * n2; p, se = stats(W2, n); ext = True
    cap = dfn.Z / AD2
    g_maxw = float(W2.max() / W2.sum()) if len(W2) else 1.0
    capped = W2 > 0.5 * cap; n_capped = int(capped.sum())
    g_cap = float(W2[capped].sum() / W2.sum()) if len(W2) else 1.0
    # flag: one hit > 5% of the estimate, or FEW (< 30) near-cap defensive hits carrying > 20% of it, or no hits
    flag = bool(g_maxw > G_MAXW or (n_capped < 30 and g_cap > G_CAPSHARE) or p == 0)
    desc_fpe = (pts_all["descent_sec"] + t_disc) * 1e6 / is2.US_PER_FPE
    C = pts_all["pilot_fpe"] + desc_fpe + f1 + f2; C_draws = pts_all["pilot_draws"] + desc_fpe + n1 + n
    tri1 = sorted({str(tuple(p_["tri"])) for p_ in sel}); tri2 = sorted({str(tuple(p_["tri"])) for p_ in sel2})
    return dict(t0=t0, p=float(p), se=float(se), rel_se=float(se / p) if p > 0 else None, n2=n, extended=ext,
                hits=int(len(W2)), gate_maxw=g_maxw, gate_capshare=g_cap, n_capped=n_capped, flag=flag, Z=dfn.Z, cap=cap,
                C=C, C_draws=C_draws, F=float(se ** 2 * C / (p * (1 - p))) if p > 0 else None,
                F_draws=float(se ** 2 * C_draws / (p * (1 - p))) if p > 0 else None,
                n_new_descents=len(new), triples_before=tri1, triples_after=tri2,
                n_points=len(sel2), n_components=len(means), sec=time.time() - t)
