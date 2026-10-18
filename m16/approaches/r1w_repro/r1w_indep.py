"""T312: independent re-implementation of R1W V1 (claude/t218-tie-tail-labels r1_walk, frozen 684a22b9) from the
README text only (r1_walk README T304/T306 + r0_walk README T302). No code from those folders is used here.

Estimator (as described):
  per ordered top-3 position triple (720) and region (nonblind v2>0, blind v2<=0):
  * closure c: v4 at the latest free position, the 6 other lows at mu(v4) = -phi/Phi(v4);
  * transverse point (v2*, g12*, g23*) = exact conditional means given the region (numerical integration of the
    model-free top-3 law); v3* = v2* - g23*; v4 node = E[v4 | v4 < v3*] (K = 1 Gauss node of 7 phi Phi^6 / Phi(v3*)^7);
  * g23 line: v1 = v2*+g12*, v2 = v2*, v3 = v2* - g, v4 = v3 - (v3* - node), lows = mu(node) + mu'(node)(v4 - node);
  * g12 line (nonblind only): v1 = v2* + g, v3 = v3*, v4 = node, lows = mu(node);
  * exact segment walk on [0, tmax=.5] (every gate flip located; correct set per piece in closed form); beyond tmax
    assumed correct; blind g12 assumed correct;
  * mass = (1/720) sum_tri sum_region int 90 phi(v2) A12(v2) A23(v2) dv2 (Gauss-Legendre), A12 = sum [Phi(v2+b) -
    Phi(v2+a)], A23 = sum [Phi(v2-a)^8 - Phi(v2-b)^8]/8 over correct intervals.
Cost: walk first cell 2 FPE, later cells 2(T-s)/T (s = 0-based step of the flip); GL/Phi work charged in FLOPs.
"""
import itertools
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

T = 10
TRI = np.array(list(itertools.permutations(range(T), 3)))
TMAX = 0.5
EPS = 1e-10
FPE_FLOPS = 5948.0       # FLOPs per walk forward (T196 c_s); the frozen code uses its own 5,760 (see README)
FLOP_PHI = 20.0          # declared charge per Phi / phi evaluation (my choice; README says "charged")
N_GL = 48
MAX_CELLS = 10_000


def mu(v):
    return -norm.pdf(v) / ndtr(v)


def dmu(v):
    m = mu(v)
    return -v * m + m * m


# ---------------------------------------------------------------- model-free constants (numerical integration)
def _transverse_points():
    v = np.linspace(-9, 9, 360_001); dv = v[1] - v[0]
    f2 = 90 * norm.pdf(v) * (1 - ndtr(v)) * ndtr(v) ** 8                     # density of v2
    lam = norm.pdf(v) / np.maximum(1 - ndtr(v), 1e-300)
    eg12 = lam - v                                                             # E[v1 - v2 | v2]
    # E[v3 | v3 < v2] under phi Phi^7: cumulative integral of x phi(x) Phi(x)^7
    w = norm.pdf(v) * ndtr(v) ** 7
    num = np.cumsum(v * w) * dv
    den = np.cumsum(w) * dv
    ev3 = num / np.maximum(den, 1e-300)
    eg23 = v - ev3
    out = {}
    for name, mask in (("nb", v > 0), ("b", v <= 0)):
        p = (f2 * mask).sum() * dv
        out[name] = dict(P=p, v2=(v * f2 * mask).sum() * dv / p, g12=(eg12 * f2 * mask).sum() * dv / p,
                         g23=(eg23 * f2 * mask).sum() * dv / p)
    return out


def _v4_node(v3s):
    v = np.linspace(-12, v3s, 200_001); dv = v[1] - v[0]
    f = 7 * norm.pdf(v) * ndtr(v) ** 6
    return float((v * f).sum() / f.sum())


TP = _transverse_points()
for _r in TP.values():
    _r["v3"] = _r["v2"] - _r["g23"]
    _r["node"] = _v4_node(_r["v3"])
_t, _w = np.polynomial.legendre.leggauss(N_GL)
GL = {"nb": (3.25 + 3.25 * _t, 3.25 * _w), "b": (-3.25 + 3.25 * _t, 3.25 * _w)}   # v2 in (0, 6.5] / [-6.5, 0]


# ---------------------------------------------------------------- exact segment walk (vectorised over lines)
def walk(model, X0, D, label, tmax=TMAX):
    """Lines x(g) = X0 + g D, g in [0, tmax]. Returns per line a list of correct intervals in [0, tmax] and the
    charged cost in FPE (sum over lines)."""
    Wi, Wh, Wo = model
    n = len(X0)
    g = np.zeros(n)
    alive = np.ones(n, bool)
    first = np.ones(n, bool)
    restart = np.zeros(n, int)          # step of the flip that began the current cell
    cost = 0.0
    ints = [[] for _ in range(n)]
    cells = 0
    while alive.any() and cells < MAX_CELLS:
        cells += 1
        idx = np.where(alive)[0]
        ge = g[idx] + EPS
        x = X0[idx] + ge[:, None] * D[idx]
        dx = D[idx]
        h = np.zeros((len(idx), Wh.shape[0])); dh = np.zeros_like(h)
        nxt = np.full(len(idx), np.inf); sflip = np.zeros(len(idx), int)
        for t in range(T):
            z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
            dz = dh @ Wh.T + dx[:, t:t + 1] * Wi[None]
            on = z > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                r = ge[:, None] - z / dz
            cand = ((on & (dz < 0)) | (~on & (dz > 0))) & (r > ge[:, None])
            r = np.where(cand, r, np.inf).min(1)
            better = r < nxt
            nxt = np.where(better, r, nxt); sflip = np.where(better, t, sflip)
            h = np.where(on, z, 0.0); dh = np.where(on, dz, 0.0)
        L, dL = h @ Wo.T, dh @ Wo.T
        hi = np.minimum(nxt, tmax)
        lo = g[idx]
        # correct set on [lo, hi]: L_lab(g) - L_j(g) > 0 for all j != lab (affine around ge)
        lab = label[idx]
        rr = np.arange(len(idx))
        a = L[rr, lab][:, None] - L; b = dL[rr, lab][:, None] - dL           # margin_j(g) = a + b (g - ge)
        a[rr, lab] = np.inf; b[rr, lab] = 0.0
        L_ = np.full(len(idx), -np.inf); H_ = np.full(len(idx), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = ge[:, None] - a / b
        up = b > 0      # margin increasing: correct for g > root
        dn = b < 0      # decreasing: correct for g < root
        L_ = np.maximum(np.where(up, root, -np.inf).max(1), lo)
        H_ = np.minimum(np.where(dn, root, np.inf).min(1), hi)
        flat_bad = ((b == 0) & (a <= 0)).any(1)
        ok = (L_ < H_) & ~flat_bad
        for k in np.where(ok)[0]:
            ints[idx[k]].append((L_[k], H_[k]))
        # T302 README: first cell 2 FPE; a later cell restarts from the step s of the flip that began it
        cost += np.where(first[idx], 2.0, 2.0 * (T - restart[idx]) / T).sum()
        first[idx] = False
        restart[idx] = sflip
        g[idx] = hi
        alive[idx] = nxt < tmax
    return ints, cost, cells


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1e-12:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def est(model, budget=None):
    rr = np.arange(len(TRI))
    free = np.array([[p for p in range(T) if p not in t] for t in TRI])
    p4 = free[:, -1]                    # latest free position
    lows = free[:, :-1]
    flops = 0.0
    total = 0.0
    for reg in ("nb", "b"):
        tp = TP[reg]
        v2s, g12s, v3s, node = tp["v2"], tp["g12"], tp["v3"], tp["node"]
        delta = v3s - node
        m0, dm0 = mu(node), dmu(node)
        # g23 line
        X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
        X0[rr, TRI[:, 0]] = v2s + g12s; X0[rr, TRI[:, 1]] = v2s
        X0[rr, TRI[:, 2]] = v2s; D[rr, TRI[:, 2]] = -1.0
        X0[rr, p4] = v2s - delta; D[rr, p4] = -1.0
        # lows = mu(node) + dmu(node) (v4 - node), v4 = v2s - delta - g
        X0[rr[:, None], lows] = m0 + dm0 * (v2s - delta - node); D[rr[:, None], lows] = -dm0
        i23, c23, _ = walk(model, X0, D, TRI[:, 1])
        flops += c23 * FPE_FLOPS
        if reg == "nb":
            X0 = np.zeros((len(TRI), T)); D = np.zeros((len(TRI), T))
            X0[rr, TRI[:, 0]] = v2s; D[rr, TRI[:, 0]] = 1.0
            X0[rr, TRI[:, 1]] = v2s; X0[rr, TRI[:, 2]] = v3s; X0[rr, p4] = node
            X0[rr[:, None], lows] = m0
            i12, c12, _ = walk(model, X0, D, TRI[:, 1])
            flops += c12 * FPE_FLOPS
        x, w = GL[reg]
        dens = 720 * norm.pdf(x)                # top-3 law: 720 phi(v1) phi(v2) phi(v3) Phi(v3)^7; A23 carries the /8
        for k in range(len(TRI)):
            s23 = _merge(i23[k] + [(TMAX, np.inf)])
            A23 = sum((ndtr(x - a) ** 8 - (ndtr(x - b) ** 8 if np.isfinite(b) else 0.0)) / 8 for a, b in s23)
            if reg == "nb":
                s12 = _merge(i12[k] + [(TMAX, np.inf)])
                A12 = sum((1.0 if not np.isfinite(b) else ndtr(x + b)) - ndtr(x + a) for a, b in s12)
                nphi = 2 * len(s23) + 2 * len(s12) + 1
            else:
                A12 = 1 - ndtr(x)
                nphi = 2 * len(s23) + 2
            total += float((w * dens * A12 * A23).sum())
            flops += N_GL * (nphi * FLOP_PHI + 4 * (len(s23) + 2) + 6)
    return total / len(TRI), flops


if __name__ == "__main__":
    print({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in TP.items()})
    # validation: full intervals -> mass 1
    m = 0.0
    for reg in ("nb", "b"):
        x, w = GL[reg]
        A12 = 1 - ndtr(x)
        A23 = ndtr(x) ** 8 / 8
        m += float((w * 720 * norm.pdf(x) * A12 * A23).sum())
    print("total mass with full intervals", m)
