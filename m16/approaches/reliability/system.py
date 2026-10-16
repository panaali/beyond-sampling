"""Full FORM + system-reliability estimate: all chambers x all wrong-rank modes, both sign regions."""
import numpy as np, torch
from scipy.stats import norm, multivariate_normal
from form import design_points, chambers, tmodel, C, T
from rosen import G0


def bvn_lower(b1, b2, rho):
    """P(Z1 < b1, Z2 < b2) for correlation rho (vectorised via Owen-free scipy loop over unique pairs)."""
    out = np.empty(len(b1))
    for i in range(len(b1)):
        r = float(np.clip(rho[i], -0.999999, 0.999999))
        out[i] = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([b1[i], b2[i]])
    return out


def mode_table(model, n_arr=1, modes=(0, 2, 3, 4, 5, 6, 7, 8, 9), positive=True, iters=6, sub=None, seed=0):
    tm = tmodel(model)
    perms = torch.as_tensor(chambers(n_arr, seed=seed))
    if sub is not None:
        perms = perms[sub]
    Cn = len(perms)
    P = perms.repeat(len(modes), 1)
    ks = torch.as_tensor(np.repeat(modes, Cn))
    b, a, u = design_points(tm, P, ks, positive, iters=iters)
    return b.reshape(len(modes), Cn), a.reshape(len(modes), Cn, T), u.reshape(len(modes), Cn, T)


def combine(b, a, pairwise=True):
    """b: (m, C) betas, a: (m, C, T) alphas. Returns per-chamber failure:
    sum (first-order union bound), Ditlevsen lower/upper bounds (modes ordered by beta), and the
    'exclusive' estimate. Pairwise intersections from bivariate normals of the linearised modes."""
    m, Cn = b.shape
    pf = norm.cdf(-b)
    s = pf.sum(0)
    if not pairwise:
        return dict(sum=s)
    lo = np.zeros(Cn); up = np.zeros(Cn)
    for c in range(Cn):
        o = np.argsort(b[:, c]); bb = b[o, c]; aa = a[o, c]; p = pf[o, c]
        keep = bb < 5.5
        bb, aa, p = bb[keep], aa[keep], p[keep]
        if len(bb) == 0:
            continue
        R = aa @ aa.T
        lo_c = p[0]; up_c = p[0]
        for i in range(1, len(bb)):
            pij = np.array([multivariate_normal(mean=[0, 0], cov=[[1, R[i, j]], [R[i, j], 1]]).cdf([-bb[i], -bb[j]])
                            if abs(R[i, j]) < 0.999999 else min(p[i], p[j]) for j in range(i)])
            lo_c += max(0.0, p[i] - pij.sum())
            up_c += p[i] - pij.max()
        lo[c] = lo_c; up[c] = up_c
    return dict(sum=s, lo=lo, up=up)
