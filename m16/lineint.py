"""Exact 1-D integration of accuracy along coordinate lines, using the ReLU network's piecewise linearity.

For a base input x and a coordinate j, let s = x_j vary over R with the other coordinates fixed. The
network is exactly piecewise linear in s (no biases, ReLU): we walk the breakpoints where any
pre-activation crosses zero, solve within each piece for where the argmax changes (logits are affine in s),
intersect with the target (2nd argmax changes only where s crosses the other coordinates), and integrate the
N(0,1) density of s over the correct intervals in closed form.

P(correct | x_{-j}) is then exact, and averaging it over random x_{-j} (and j) is an unbiased, lower-variance
estimate of accuracy (conditional Monte Carlo / line sampling).
"""
import numpy as np
from scipy.special import ndtr

LO, HI = -9.0, 9.0


def _pre_acts(model, x, j, s):
    """Pre-activations z[t, i] for t >= j, with slopes dz/ds under the gates active at s. Also logits and slope."""
    Wi, Wh, Wo = model
    H = Wh.shape[0]
    h = np.zeros(H); dh = np.zeros(H)
    zs, dzs = [], []
    for t in range(x.shape[0]):
        xt = s if t == j else x[t]
        dxt = 1.0 if t == j else 0.0
        z = Wh @ h + Wi * xt
        dz = Wh @ dh + Wi * dxt
        if t >= j:
            zs.append(z); dzs.append(dz)
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return np.array(zs), np.array(dzs), Wo @ h, Wo @ dh


def _target_intervals(x, j):
    """[(lo, hi, target)] for s in [LO, HI]."""
    others = [(x[k], k) for k in range(len(x)) if k != j]
    others.sort(reverse=True)
    (o1, k1), (o2, k2) = others[0], others[1]
    return [(LO, o2, k2), (o2, o1, j), (o1, HI, k1)]


def line_accuracy(model, x, j, max_pieces=2000):
    """Exact P(correct | x_{-j}) under s ~ N(0,1) (truncated to [LO, HI]); returns (prob, forward_passes)."""
    s, n_fwd, correct_mass = LO, 0, 0.0
    tgt = _target_intervals(x, j)
    eps = 1e-9
    while s < HI and n_fwd < max_pieces:
        z, dz, lg, dlg = _pre_acts(model, x, j, s + eps)
        n_fwd += 1
        z, lg = z - dz * eps, lg - dlg * eps          # re-centre the affine pieces at s
        # next gate breakpoint: smallest root of z + dz*(u - s) = 0 with u > s
        with np.errstate(divide="ignore", invalid="ignore"):
            r = s - z / dz
        r = r[(r > s + 1e-12) & np.isfinite(r)]
        nxt = min(HI, r.min()) if r.size else HI
        # within [s, nxt] logits are lg + dlg*(u - s): walk the argmax envelope
        a = s
        while a < nxt:
            val = lg + dlg * (a - s)
            k = int(np.argmax(val + dlg * 1e-7))     # at a tie, the logit that is about to win
            # where does another logit overtake k?
            with np.errstate(divide="ignore", invalid="ignore"):
                c = a + (val[k] - val) / (dlg - dlg[k])
            c = c[(c > a + 1e-13) & (dlg > dlg[k])]
            b = min(nxt, c.min()) if c.size else nxt
            # accumulate mass where target == k on [a, b]
            for lo, hi, tk in tgt:
                if tk == k:
                    l, h_ = max(lo, a), min(hi, b)
                    if h_ > l:
                        correct_mass += ndtr(h_) - ndtr(l)
            a = b
        s = nxt
    return correct_mass, n_fwd
