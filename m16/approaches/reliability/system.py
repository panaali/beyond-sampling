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


def hr_mvn(b, a, rng_seed=0):
    """Hohenbichler-Rackwitz: P(correct | chamber) = Phi_m(beta; R) with R = alpha alpha^T (linearised modes).
    Modes with beta >= 5.5 are dropped (P(g<0) < 2e-8)."""
    m, Cn = b.shape; ps = np.zeros(Cn)
    for c in range(Cn):
        keep = b[:, c] < 5.5
        if keep.sum() == 0:
            ps[c] = 1.0; continue
        bb = b[keep, c]; aa = a[keep, c]
        if len(bb) == 1:
            ps[c] = norm.cdf(bb[0]); continue
        R = aa @ aa.T + 1e-9 * np.eye(len(bb))
        ps[c] = multivariate_normal(mean=np.zeros(len(bb)), cov=R, allow_singular=True).cdf(bb, rng=rng_seed)
    return ps


from scipy.special import owens_t


def bvn(h, k, rho):
    """Vectorised P(Z1 < h, Z2 < k), corr rho, via Owen's T."""
    h = np.asarray(h, float) + 1e-12; k = np.asarray(k, float) + 1.3e-12
    rho = np.clip(rho, -0.9999999, 0.9999999); s = np.sqrt(1 - rho ** 2)
    a1 = (k - rho * h) / (h * s); a2 = (h - rho * k) / (k * s)
    beta = np.where(h * k < 0, 0.5, 0.0)
    return 0.5 * norm.cdf(h) + 0.5 * norm.cdf(k) - owens_t(h, a1) - owens_t(k, a2) - beta


def ditlevsen(b, a):
    """Vectorised Ditlevsen bounds on P(union of {g_k<0}) per chamber; modes sorted by beta."""
    m, Cn = b.shape
    o = np.argsort(b, 0)
    bs = np.take_along_axis(b, o, 0); as_ = np.take_along_axis(a, o[:, :, None], 0)
    p = norm.cdf(-bs)
    lo = p[0].copy(); up = p[0].copy()
    for i in range(1, m):
        pij = np.stack([bvn(-bs[i], -bs[j], (as_[i] * as_[j]).sum(1)) for j in range(i)])
        pij = np.clip(pij, 0, np.minimum(p[i], p[:i]))
        lo += np.maximum(0, p[i] - pij.sum(0)); up += p[i] - pij.max(0)
    return lo, up
