"""Array-RQMC for the M_16,10 RNN viewed as a Markov chain, plus exact last-step integration.

Chain state after t inputs: h_t (16), running top-2 values (m1, m2) and positions (p1, p2).
The target (2nd argmax of x_0..x_9) is a function of (m1, m2, p1, p2, x_9), so the state is Markov.

Last step (conditional MC): given h_9 and the top-2 of x_0..x_8, the logits are piecewise linear in
s = x_9 (16 gate breakpoints), the target is piecewise constant (breakpoints m2, m1). So
P(correct | state_9) is a sum of Gaussian masses of at most 19 intervals, computed in closed form.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

T = 10


# ------------------------------------------------------------------ chain
def init(n, H=16):
    return dict(h=np.zeros((n, H)), m1=np.full(n, -np.inf), m2=np.full(n, -np.inf),
                p1=np.zeros(n, int), p2=np.zeros(n, int), t=0)


def step(model, S, x):
    Wi, Wh, Wo = model
    t = S["t"]
    S["h"] = np.maximum(S["h"] @ Wh.T + x[:, None] * Wi[None], 0)
    up1 = x > S["m1"]; up2 = (~up1) & (x > S["m2"])
    S["m2"] = np.where(up1, S["m1"], np.where(up2, x, S["m2"]))
    S["p2"] = np.where(up1, S["p1"], np.where(up2, t, S["p2"]))
    S["m1"] = np.where(up1, x, S["m1"]); S["p1"] = np.where(up1, t, S["p1"])
    S["t"] = t + 1
    return S


def last_step_value(model, S):
    """Exact P(correct | state after 9 inputs), integrating x_9 ~ N(0,1)."""
    Wi, Wh, Wo = model
    n, H = S["h"].shape
    a = S["h"] @ Wh.T                                      # pre-activation = a + Wi*s
    with np.errstate(divide="ignore"):
        bp = -a / Wi[None]
    pts = np.sort(np.concatenate([bp, S["m1"][:, None], S["m2"][:, None]], 1), 1)
    lo = np.concatenate([np.full((n, 1), -np.inf), pts], 1)       # (n, 19)
    hi = np.concatenate([pts, np.full((n, 1), np.inf)], 1)
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2,
                   np.where(np.isfinite(lo), lo + 1, hi - 1))
    G = (a[:, None, :] + Wi[None, None, :] * mid[..., None]) > 0       # (n, K, H)
    K = G.shape[1]
    c = ((G * a[:, None, :]).reshape(-1, H) @ Wo.T).reshape(n, K, -1)  # logits = c + d*s
    d = ((G * Wi[None, None, :]).reshape(-1, H) @ Wo.T).reshape(n, K, -1)
    m1, m2 = S["m1"][:, None], S["m2"][:, None]
    tgt = np.where(mid > m1, S["p1"][:, None], np.where(mid > m2, T - 1, S["p2"][:, None]))
    ck = np.take_along_axis(c, tgt[..., None], 2); dk = np.take_along_axis(d, tgt[..., None], 2)
    dd = dk - d; dc = c - ck                                            # need dc <= dd*s for all j
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dc / dd
        L = np.where(dd > 0, r, -np.inf).max(2)
        U = np.where(dd < 0, r, np.inf).min(2)
    bad = ((dd == 0) & (dc > 0)).any(2)
    L = np.maximum(L, lo); U = np.minimum(U, hi)
    mass = np.where((U > L) & ~bad, ndtr(U) - ndtr(L), 0.0)
    return mass.sum(1)


# ------------------------------------------------------------------ importance (sorting) functions
def feats(S):
    """Cheap state features for the learned value function (~45 numbers)."""
    m1 = np.where(np.isfinite(S["m1"]), S["m1"], -3.0)
    m2 = np.where(np.isfinite(S["m2"]), S["m2"], -3.0)
    g = m1 - m2
    cols = [np.ones_like(m1), m1, m2, g, g * g, np.exp(-4 * g), np.exp(-20 * g), np.maximum(m2, 0),
            np.exp(-4 * np.maximum(m2, 0)), m1 * m1]
    t = S["t"]
    oh1 = np.eye(T)[S["p1"]][:, :t] if t > 0 else np.zeros((len(m1), 0))
    oh2 = np.eye(T)[S["p2"]][:, :t] if t > 1 else np.zeros((len(m1), 0))
    return np.column_stack(cols + [S["h"], oh1, oh2])


class ValueFn:
    """V_t(state) ~ P(correct | state_t), fitted by least squares on a pilot of the ORIGINAL model."""
    def __init__(self, coefs):
        self.coefs = coefs

    def __call__(self, S):
        return feats(S) @ self.coefs[S["t"]]

    @staticmethod
    def fit(model, n=200_000, seed=7, ridge=1e-3):
        rng = np.random.default_rng(seed)
        S = init(n); F = {}
        for t in range(T - 1):
            step(model, S, rng.standard_normal(n))
            F[S["t"]] = feats(S)
        y = last_step_value(model, S)
        coefs = {}
        for t, X in F.items():
            A = X.T @ X + ridge * np.eye(X.shape[1])
            coefs[t] = np.linalg.solve(A, X.T @ y)
            r2 = 1 - ((X @ coefs[t] - y) ** 2).mean() / y.var()
            print(f"  V_{t}: R^2 = {r2:.3f}", flush=True)
        return ValueFn(coefs)


def sort_m1(S):
    return S["m1"]


def sort_gap(S):     # predicted difficulty: closeness of the top two (relative), then running max
    m2 = np.where(np.isfinite(S["m2"]), S["m2"], S["m1"] - 3)
    return (S["m1"] - m2) / (np.abs(S["m1"]) + 0.5)


# ------------------------------------------------------------------ estimators
def _scr(d, n, rng):
    return qmc.Sobol(d, scramble=True, seed=rng).random(n)


def _z(u):
    return ndtri(np.clip(u, 1e-15, 1 - 1e-15))


def est_mc(model, n, rng, cmc=True):
    """Plain MC; with cmc the last input is integrated exactly."""
    S = init(n)
    for t in range(T - 1):
        step(model, S, rng.standard_normal(n))
    if cmc:
        return last_step_value(model, S).mean()
    x = rng.standard_normal(n); m1, m2, p1, p2 = S["m1"], S["m2"], S["p1"], S["p2"]
    tgt = np.where(x > m1, p1, np.where(x > m2, T - 1, p2))
    step(model, S, x)
    return ((S["h"] @ model[2].T).argmax(1) == tgt).mean()


def est_rqmc(model, n, rng):
    """Classical RQMC: one scrambled 9-D Sobol set for x_0..x_8, exact integration of x_9."""
    X = _z(_scr(T - 1, n, rng))
    S = init(n)
    for t in range(T - 1):
        step(model, S, X[:, t])
    return last_step_value(model, S).mean()


def est_array(model, n, rng, sortfn, sort2=None):
    """Array-RQMC (L'Ecuyer-Lecot-Tuffin 2008). n must be a power of 2.
    Each step: sort chains by sortfn (or 2-D batch sort by (sort2, sortfn)), then advance with a fresh
    scrambled Sobol point set whose leading coordinate(s) are matched to the state order."""
    S = init(n)
    u = _scr(1, n, rng)[:, 0]
    step(model, S, _z(u))                                  # all chains identical at t=0
    for t in range(1, T - 1):
        if sort2 is None:
            P = _scr(2, n, rng)
            P = P[np.argsort(P[:, 0])]
            order = np.argsort(sortfn(S), kind="stable")
            x = np.empty(n); x[order] = _z(P[:, 1])
        else:   # batch sort: n1 groups by sort2, within each group by sortfn
            P = _scr(3, n, rng)
            k = int(np.log2(n)); n1 = 2 ** ((k + 1) // 2); n2 = n // n1
            P = P[np.argsort(P[:, 0])]
            P = P.reshape(n1, n2, 3)
            P = np.take_along_axis(P, np.argsort(P[:, :, 1], 1)[..., None], 1).reshape(n, 3)
            o1 = np.argsort(sort2(S), kind="stable").reshape(n1, n2)
            v = sortfn(S)[o1]
            order = np.take_along_axis(o1, np.argsort(v, 1, kind="stable"), 1).reshape(n)
            x = np.empty(n); x[order] = _z(P[:, 2])
        step(model, S, x)
    return last_step_value(model, S).mean()
