"""T347 frozen tilt formula (DEDUCTIVE given the candidate's walks; no sampling, no fitted constant).
D_pred(model) = predicted (product - joint) correct mass, in pp, of T317d's line-set product (nb region, V1 convention).
Per triple: theta* = the start of the first correct interval on each walked line (cont_c walk + merge); at that wall
crossing the binding rival's margin has slope alpha along the line and beta along the other line's direction
(forward-mode tangents, gates frozen), so the wall tilts by s = -beta/alpha. The mass difference uses the exact
order-statistic density. Also returns the extra cost (2 tangent evaluations per triple at the crossings)."""
import os, sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
CAND = os.environ.get("CAND", "/tmp/claude-0/-home-user-clean-neurons/d43de5b0-d4b8-5960-94ba-74926c630209/scratchpad/t317d/m16/approaches/r1_cont")
sys.path.insert(0, CAND)
import cont_c, cont_d, r0  # noqa: E402

TMAX = cont_c.TMAX
GQ, WQ = np.polynomial.legendre.leggauss(40); GQ = (GQ + 1) / 2 * TMAX; WQ = WQ * TMAX / 2       # [0, tmax] rule
NB = next(t for t in cont_d.TABLES if t["name"] == "nb")


def first_start(pieces, n):
    R, A, B, _ = cont_c.merge(pieces, n, TMAX)
    th = np.full(n, TMAX)
    np.minimum.at(th, R, np.where(A < TMAX, A, TMAX))
    return th                                         # 0 -> correct from the start (no wall); TMAX -> wrong throughout


def crossing_slopes(model, x, dx, cross, k, th):
    """alpha (along dx) and beta (along cross) of the binding rival's margin just inside the correct set at t = th."""
    xc = x + (th + 1e-9)[:, None] * dx
    L, dLa = r0.logits(model, xc, dx); _, dLb = r0.logits(model, xc, cross)
    rr = np.arange(len(x)); M = L[rr, k][:, None] - L; M[rr, k] = np.inf; riv = M.argmin(1)
    a = dLa[rr, k] - dLa[rr, riv]; b = dLb[rr, k] - dLb[rr, riv]
    return a, b


def D_pred(model):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = r0.TRIPLES[:, 1]; n = len(k)
    (x23, d23), (x12, d12) = NB["line23"], NB["line12"]
    p23, _ = cont_c.walk_counted(model, x23, d23, k, TMAX); p12, _ = cont_c.walk_counted(model, x12, d12, k, TMAX)
    th23, th12 = first_start(p23, n), first_start(p12, n)
    a23, b23 = crossing_slopes(model, x23, d23, d12, k, th23)       # cross direction of line 23 = moving g12
    a12, b12 = crossing_slopes(model, x12, d12, d23, k, th12)       # cross direction of line 12 = moving g23
    with np.errstate(divide="ignore", invalid="ignore"):
        s23 = np.where(np.abs(a23) > 1e-12, -b23 / a23, 0.0); s12 = np.where(np.abs(a12) > 1e-12, -b12 / a12, 0.0)
    w23 = (th23 > 0) & (th23 < TMAX); w12 = (th12 > 0) & (th12 < TMAX)
    v2s, g12s, g23s = NB["point"]; nodes, wts, dens = NB["nodes"], NB["wts"], NB["dens"]
    V = nodes[None, :, None]                                          # (1, G, 1)
    # 2~3 wall: theta(g12) over the g12 rule
    th = np.clip(th23[:, None, None] + s23[:, None, None] * (GQ[None, None] - g12s), 0, TMAX)
    Psi = lambda c: ndtr(V) ** 8 - ndtr(V - c) ** 8
    D23 = ((norm.pdf(V + GQ[None, None]) * (Psi(th) - Psi(th23[:, None, None]))) @ WQ) * w23[:, None]
    th = np.clip(th12[:, None, None] + s12[:, None, None] * (GQ[None, None] - g23s), 0, TMAX)
    Psi1 = lambda c: ndtr(V + c) - ndtr(V)
    f23 = 8 * norm.pdf(V - GQ[None, None]) * ndtr(V - GQ[None, None]) ** 7
    D12 = ((f23 * (Psi1(th) - Psi1(th12[:, None, None]))) @ WQ) * w12[:, None]
    D = ((D23 + D12) * dens[None]) @ wts
    extra_fpe = 2 * n * 3.0                                           # 2 crossings x (value + 2 tangents) ~ 3 FPE each
    return float(100 * D.mean()), dict(D23=float(100 * ((D23 * dens[None]) @ wts).mean()), D12=float(100 * ((D12 * dens[None]) @ wts).mean()),
                                        frac_wall23=float(w23.mean()), frac_wall12=float(w12.mean()), tilt23_med=float(np.median(s23[w23])) if w23.any() else 0.0,
                                        tilt12_med=float(np.median(s12[w12])) if w12.any() else 0.0, extra_fpe=extra_fpe)
