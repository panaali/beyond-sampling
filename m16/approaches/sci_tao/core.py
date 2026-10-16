"""Co-area (tie-face) decomposition of the accuracy of a bias-free ReLU RNN 2nd-argmax model, any T.

Exact identity (model-free geometry, see README):
    P(wrong) = Z23 * E_face23[F] + Z12 * E_face12[F]
R23 = {v2-v3 < v1-v2}, R12 = complement. Every input lies on exactly one segment x = y + t d,
d = (e_a - e_b)/sqrt2, where y is a point of the tie face {x_a = x_b} (a,b = the pair at ranks 2,3 in R23,
ranks 1,2 in R12). F(y) = int_{window} 1[model wrong at y + t d] phi(t) dt, computed EXACTLY by walking the
network's linear pieces (no biases + ReLU => piecewise linear in t).
Face laws (u = tie value):
  R23: u ~ e^{-u^2}(1-Phi(u))Phi(u)^{T-3}; max ~ N|>u; T-3 lower ~ N|<u; positions uniform; window
       |t| < min(sqrt2 (o1-u)/3, sqrt2 (u - o2))
  R12: w ~ e^{-w^2} Phi(w)^{T-2}; T-2 lower ~ N|<w; window |t| < sqrt2 (w - o2)/3
"""
import itertools
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

SQ2 = np.sqrt(2.0)
_g = np.linspace(-8, 8, 160001)
_dg = _g[1] - _g[0]


def _face_tables(T):
    Phi = ndtr(_g)
    e = np.exp(-_g ** 2) / np.sqrt(np.pi)
    d23 = e * (1 - Phi) * Phi ** (T - 3) if T >= 3 else None
    d12 = e * Phi ** (T - 2)
    out = {}
    for k, d in (("23", d23), ("12", d12)):
        if d is None:
            continue
        c = np.cumsum(d) * _dg
        out[k] = (c[-1], (c - c[0]) / (c[-1] - c[0]))
    npos23 = T * (T - 1) // 2 * (T - 2)
    npos12 = T * (T - 1) // 2
    Z23 = out["23"][0] * npos23 if T >= 3 else 0.0
    Z12 = out["12"][0] * npos12
    return out, Z23, Z12


_cache = {}


def tables(T):
    if T not in _cache:
        _cache[T] = _face_tables(T)
    return _cache[T]


def configs(T, kind):
    """Position configurations: R23 -> (a,b,p1), R12 -> (a,b,-1). All equally likely (exchangeability)."""
    out = []
    for a, b in itertools.combinations(range(T), 2):
        if kind == "12":
            out.append((a, b, -1))
        else:
            for p in range(T):
                if p not in (a, b):
                    out.append((a, b, p))
    return np.array(out)


def face_points(T, kind, U):
    """Map uniform points U (n, T) -> face points. U[:,0] chooses the configuration (stratified),
    U[:,1] the tie value, the rest the other inputs. Returns y, a, b, target-side info, window tau."""
    tab, _, _ = tables(T)
    cfg = configs(T, kind)
    n = len(U)
    ci = np.minimum((U[:, 0] * len(cfg)).astype(int), len(cfg) - 1)
    a, b, p1 = cfg[ci, 0], cfg[ci, 1], cfg[ci, 2]
    u = np.interp(U[:, 1], tab[kind][1], _g)
    Pu = ndtr(u)
    y = np.empty((n, T)); rows = np.arange(n)
    y[rows, a] = u; y[rows, b] = u
    col = 2
    if kind == "23":
        o1 = ndtri(np.clip(Pu + U[:, col] * (1 - Pu), 1e-300, 1 - 1e-16)); col += 1
        y[rows, p1] = o1
    nlow = T - 3 if kind == "23" else T - 2
    o2 = np.full(n, -np.inf)
    if nlow > 0:
        low = ndtri(np.clip(U[:, col:col + nlow] * Pu[:, None], 1e-300, 1 - 1e-16))
        o2 = low.max(1)
        # fill remaining positions in index order
        mask = np.ones((n, T), bool); mask[rows, a] = False; mask[rows, b] = False
        if kind == "23":
            mask[rows, p1] = False
        idx = np.nonzero(mask)[1].reshape(n, nlow)
        y[rows[:, None], idx] = low
    if kind == "23":
        tau = np.minimum(SQ2 * (o1 - u) / 3, SQ2 * (u - o2))
    else:
        tau = SQ2 * (u - o2) / 3
    tau = np.minimum(tau, 8.5)
    return y, a, b, tau


def walk_mass(model, y, a, b, tau, kind, eps=1e-10, max_cells=2000):
    """Exact misclassified phi-mass on each segment y + t d, |t| < tau.
    Target: R23 -> a for t>0 (x_a is rank 2), b for t<0. R12 -> b for t>0 (x_b is then rank 2), a for t<0.
    Returns mass (n,), cost (n,) in forward-pass units (value+tangent = 2 per recomputed step fraction),
    cells (n,)."""
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]; rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    first = np.minimum(a, b)
    t = -tau.copy()
    mass = np.zeros(n); cost = first / T + 0.0; cells = np.zeros(n, int)
    restart = first.copy()
    act = np.nonzero(t < tau)[0]
    it = 0
    while len(act) and it < max_cells:
        it += 1
        X0, D = y[act], d[act]
        tc = t[act] + eps * (1 + np.abs(t[act]))
        h = np.zeros((len(act), H)); dh = np.zeros_like(h)
        nxt = tau[act].copy(); rstep = np.full(len(act), T)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tc * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nxt
            nxt = np.where(upd, r, nxt); rstep = np.where(upd, s, rstep)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = h @ Wo.T; dL = dh @ Wo.T                 # logits at tc, slope in t
        lo = t[act]; hi = nxt
        cost[act] += 2 * (T - restart[act]) / T
        cells[act] += 1
        # two target pieces: [lo, min(hi,0)] target tneg ; [max(lo,0), hi] target tpos
        for side in (0, 1):
            plo = lo if side else lo
            if side == 0:
                p_lo, p_hi, k = lo, np.minimum(hi, 0.0), tneg[act]
            else:
                p_lo, p_hi, k = np.maximum(lo, 0.0), hi, tpos[act]
            ok = p_hi > p_lo
            if not ok.any():
                continue
            rr = np.arange(len(act))
            # correct set: L_k(t) - L_j(t) >= 0 for all j, with L(t) = L + (t - tc) dL
            c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
            e = dL[rr, k][:, None] - dL
            with np.errstate(divide="ignore", invalid="ignore"):
                root = -c / e
            lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
            bad0 = (e == 0) & (c < 0)
            lb[rr, k] = -np.inf; ub[rr, k] = np.inf
            clo = np.maximum(lb.max(1), p_lo); chi = np.minimum(ub.min(1), p_hi)
            corr = np.where((chi > clo) & ~bad0.any(1), ndtr(chi) - ndtr(clo), 0.0)
            tot = ndtr(p_hi) - ndtr(p_lo)
            mass[act] += np.where(ok, tot - corr, 0.0)
        t[act] = nxt
        restart[act] = rstep
        act = act[nxt < tau[act]]
    return mass, cost, cells


def sobol(nd, n, seed):
    """Fixed (model-independent) scrambled Sobol rule; the seed is part of the rule, not model input."""
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(nd, scramble=True, seed=seed).random_base2(m)[:n]


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for s in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, s:s + 1] * Wi[None], 0)
    return h @ Wo.T


def zoo(H, T, seed=None):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(H, T, seed=seed)
    f = lambda w: w.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def true_acc(model, n=2_000_000, seed=5, chunk=250_000):
    rng = np.random.default_rng(seed); ok = 0; T = model[2].shape[0]
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T))
        ok += int((forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    p = ok / n
    return p, np.sqrt(p * (1 - p) / n)
