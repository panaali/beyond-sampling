"""ARC's two mechanistic estimators for M_4,3 (J. Hilton, "Heuristic explanations for 2nd argmax models",
2024-12-21, section "The case d = 4, n = 3"), re-implemented from the notes.

est_A(model): "Original version". Symmetric-average decision boundaries for h2 (4 wedge regions in (x0,x1)),
  x 6 orderings = 14 likely + 4 unlikely subregions; in each likely subregion the last ReLU's gates are replaced by
  random variables independent of z3, coupled by a latent Z ~ Bernoulli(q), matching each gate's mean p and
  variance p(1-p); unlikely subregions get accuracy 1/3. All volumes are exact solid angles (m43.cone_frac).
  ARC's op count: 5(18*4^3 + 56*5^3 + 28*6^3) = 71,000 (5 k^3 per k-halfspace volume, fixed per model).

est_B(model): "Version based on George's approach". Reduced symmetric model W' (least squares onto the U/V
  symmetric subspace), exact reduced logits W'_oh |y3(x)|; error terms eps propagated by covariance propagation
  under the presumption of independence (every variable treated as zero-mean Gaussian, so every P(positive) = 1/2);
  eps_l ~ N(0, Sigma_l) independent of x. The 24 six-dimensional volumes (4 sign quadrants of y3 x 6 orderings)
  are estimated by sampling (x, eps) as ARC did ("we ... just use sampling"); n_vol samples, fixed seed.
  ARC's op count: 5 * 24 * 4^3 = 7,680 (sampling of the 6-d volumes not charged, as in the notes).
"""
import numpy as np

from m43 import Ops, PERMS, clip, frac, order_normals, simplicial

ARC_OPS_A = 5 * (18 * 4**3 + 56 * 5**3 + 28 * 6**3)    # 71,040 ("around 71000")
ARC_OPS_B = 5 * 24 * 4**3                               # 7,680


def cone_vol(normals, ops=None):
    """Gaussian measure of {x: n.x >= 0 all n} for a pointed cone; picks an independent starting triple."""
    normals = [np.asarray(n, float) for n in normals]
    idx = None
    k = len(normals)
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                if abs(np.linalg.det(np.array([normals[i], normals[j], normals[l]]))) > 1e-12:
                    idx = (i, j, l); break
            if idx: break
        if idx: break
    assert idx is not None, "cone not pointed"
    poly = simplicial(np.array([normals[i] for i in idx]))
    for m, n in enumerate(normals):
        if m in idx:
            continue
        poly, _ = clip(poly, n, ops)
        if not poly:
            return 0.0
    return frac(poly, ops)


def _gate_values(p, q):
    """Two-point gate (values for Z=0, Z=1) with mean p and variance p(1-p) (Hilton's latent-Bernoulli)."""
    s = np.sqrt(p * (1 - p))
    lo, hi = np.sqrt(q / (1 - q)) * s, np.sqrt((1 - q) / q) * s
    g0 = np.where(p <= 0.5, p - lo, p + lo)
    g1 = np.where(p <= 0.5, p + hi, p - hi)
    return g0, g1


def est_A(model, q=0.2, ops=None, info=None):
    Wi, Wh, Wo = model
    e = np.eye(3)
    a = Wh[:, 0] * Wi[0] + Wh[:, 1] * Wi[1]
    b = Wh[:, 2] * Wi[2] + Wh[:, 3] * Wi[3]
    c = Wi.copy()
    al3, al2 = (a[3] - a[0]) / 2, (a[2] - a[1]) / 2
    be3, be2 = (b[3] - b[0]) / 2, (b[2] - b[1]) / 2
    ga0, ga1 = (c[0] - c[3]) / 2, (c[1] - c[2]) / 2
    k0, k1, k0p, k1p = ga0 / al3, ga1 / al2, ga0 / be3, ga1 / be2
    if ops is not None:
        ops.add(16 + 6 + 4)                               # a, b; the six averages; four ratios
    if info is not None:
        info["ks"] = (k0, k1, k0p, k1p)
        info["checks_ok"] = bool(max(k0, k0p) < 1 < min(k1, k1p))
    rowa = lambda j: np.array([a[j], c[j], 0.0])
    rowb = lambda j: np.array([b[j], c[j], 0.0])
    Z = np.zeros(3)
    # (region normals, h2 rows) for the 4 regions of the second approximation
    regions = [
        ([(-1, k0, 0), (-1, k1p, 0)], [rowb(0), rowb(1), Z, Z]),          # x0 <= min(k0 x1, k1' x1)
        ([(1, -k0p, 0), (1, -k1, 0)], [Z, Z, rowa(2), rowa(3)]),          # x0 >= max(k0' x1, k1 x1)
        ([(1, -k0, 0), (-1, k1, 0)], [Z, rowa(1), Z, rowa(3)]),           # k0 x1 <= x0 <= k1 x1
        ([(1, -k1p, 0), (-1, k0p, 0)], [rowb(0), Z, rowb(2), Z]),         # k1' x1 <= x0 <= k0' x1
    ]
    orders = [[(1, 0, 2), (1, 2, 0), (2, 1, 0)], [(0, 1, 2), (0, 2, 1), (2, 0, 1)], PERMS, PERMS]
    unlikely = {(0, 2, 1), (1, 2, 0)}                     # x2 between x0 and x1 in regions 3, 4
    est = 0.0; n_like = n_unl = 0
    for r, ((rn, H2), perms) in enumerate(zip(regions, orders)):
        H2 = np.array(H2)
        A3 = Wh @ H2 + np.outer(Wi, e[2])
        if ops is not None:
            ops.add(4 * 2 * 2 + 4)                        # two active columns of H2, two input coords
        for perm in perms:
            base = [np.array(n, float) for n in rn] + order_normals(perm)
            V = cone_vol(base, ops)
            if r >= 2 and perm in unlikely:
                est += V / 3; n_unl += 1
                continue
            n_like += 1
            if V <= 0:
                continue
            p = np.array([cone_vol(base + [A3[j]], ops) for j in range(4)]) / V
            p = np.clip(p, 0, 1)
            g0, g1 = _gate_values(p, q)
            if ops is not None:
                ops.add(4 + 4 * 6)                        # divisions by V, gate values
            cc = perm[1]
            accs = []
            for g in (g0, g1):
                L = Wo @ (g[:, None] * A3)
                if ops is not None:
                    ops.add(4 * 3 + 3 * 4 * 3 + 6 + 1)    # scale rows, W_oh, two differences, divide
                accs.append(cone_vol(base + [L[cc] - L[o] for o in range(3) if o != cc], ops) / V)
            est += V * ((1 - q) * accs[0] + q * accs[1])
            if ops is not None:
                ops.add(3)
    if info is not None:
        info["n_likely"], info["n_unlikely"] = n_like, n_unl
    return est


# ------------------------------------------------------------------------------------------- George / B
U = np.array([[1, 0, 0, -1], [0, 1, -1, 0]], float)
V = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], float)


def reduce_model(model):
    """Least-squares projection onto Whi = U^T w', Whh = U^T W' U, Woh = W'_oh V."""
    Wi, Wh, Wo = model
    wp = U @ Wi / 2                                    # U U^T = 2 I
    Whp = U @ Wh @ U.T / 4
    Wop = Wo @ V.T / 2
    return wp, Whp, Wop


def est_B(model, n_vol=1 << 18, seed=0, info=None, return_parts=False):
    Wi, Wh, Wo = model
    wp, Whp, Wop = reduce_model(model)
    e_hi = Wi - U.T @ wp
    e_hh = Wh - U.T @ Whp @ U
    e_oh = Wo - Wop @ V
    Whh_full = U.T @ Whp @ U
    wi_full = U.T @ wp
    # covariance propagation, all variables zero-mean, P(positive) = 1/2 for every unit
    Ch = np.zeros((4, 4)); Ce = np.zeros((4, 4))
    for t in range(3):
        Cz = Whh_full @ Ch @ Whh_full.T + np.outer(wi_full, wi_full)          # z'_{t+1}
        Cez = Wh @ Ce @ Wh.T + e_hh @ Ch @ e_hh.T + np.outer(e_hi, e_hi)    # eps_z, independence of eps_h, h', x
        Ch = Cz / 4                                                          # h' ~ (1/2) z'
        Ce = Cez / 4                                                         # eps_h ~ (1/2) eps_z
    Sig = Wo @ Ce @ Wo.T + e_oh @ Ch @ e_oh.T
    # reduced logits: l' = W'_oh |y3|, y_{t+1} = W'_hh y_t + w' x_t
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_vol, 3))
    y = np.zeros((n_vol, 2))
    for t in range(3):
        y = y @ Whp.T + x[:, t:t + 1] * wp[None]
    lp = np.abs(y) @ Wop.T
    Lc = np.linalg.cholesky(Sig + 1e-12 * np.eye(3))
    eps = rng.standard_normal((n_vol, 3)) @ Lc.T
    lab = np.argsort(x, 1)[:, -2]
    acc_noeps = float((lp.argmax(1) == lab).mean())
    acc = float(((lp + eps).argmax(1) == lab).mean())
    if info is not None:
        info["sigma_l"] = np.sqrt(np.diag(Sig)); info["acc_reduced_only"] = acc_noeps
    return (acc, acc_noeps) if return_parts else acc
