"""T165 estimator: exact great-circle integration + deterministic quadrature over the circle family (deductive).
Bias-free ReLU RNN + 2nd-argmax label are invariant to x -> c x (c>0), so accuracy = uniform measure on S^{T-1}.
T=4: S^3 is the union of Hopf fibres x(phi) = a cos(phi) + b sin(phi); fibres are indexed by (u, d) in [0,1]x[0,2pi)
with uniform weight (u = sin^2 eta, d = phase gap), so acc = mean_{u,d} F(u,d), F = exact correct fraction of the fibre.
T=3: meridians through a pole, weight |sin phi| on the circle, uniform in longitude.
On a circle every pre-activation, logit difference and input difference is A cos + B sin on each linear piece, so its
zeros are closed form (one atan2); the circle is cut exactly into pieces of constant (gates, prediction, label) and the
correct arc measure is summed. F is continuous in the fibre parameters, so a midpoint grid converges ~N^-1 (no
random inputs). Ops: multiply-add / atan2 / compare / cos = 1 op (Ops convention of arc_m43_repro)."""
import numpy as np

TWO_PI = 2 * np.pi


def _cuts(lo, hi, A, B):
    """zeros of A cos + B sin (arrays (n,k)) inside (lo,hi) per row -> (row ids, angles)."""
    psi = np.arctan2(B, A)
    z = np.concatenate([(psi + np.pi / 2) % TWO_PI, (psi - np.pi / 2) % TWO_PI], 1)
    rid = np.repeat(np.arange(len(lo))[:, None], z.shape[1], 1)
    m = (z > lo[:, None]) & (z < hi[:, None]) & np.repeat((A != 0) | (B != 0), 2, 1)
    return rid[m], z[m]


def _split(lo, hi, rid, z):
    """split intervals at the given cut points; returns new lo, hi, parent index."""
    n = len(lo)
    ids = np.concatenate([np.arange(n), rid]); pts = np.concatenate([lo, z])
    o = np.lexsort((pts, ids)); ids, pts = ids[o], pts[o]
    nxt = np.empty_like(pts); same = ids[1:] == ids[:-1]
    nxt[:-1] = np.where(same, pts[1:], hi[ids[:-1]]); nxt[-1] = hi[ids[-1]]
    return pts, nxt, ids


def fibre(model, a, b, weight="uniform", ops=None):
    """exact correct measure on the circle a cos + b sin (normalised: fraction of total weight)."""
    Wi, Wh, Wo = model; H = len(Wi); T = Wo.shape[0]
    lo = np.array([0.0]); hi = np.array([TWO_PI]); C = np.zeros((1, H, 2)); n_ops = 0
    for t in range(T):
        pre = np.einsum("ij,njk->nik", Wh, C) + Wi[None, :, None] * np.array([a[t], b[t]])[None, None]
        n_ops += len(lo) * H * (2 * H + 2)
        rid, z = _cuts(lo, hi, pre[..., 0], pre[..., 1]); n_ops += len(lo) * H * 3
        lo, hi, par = _split(lo, hi, rid, z); pre = pre[par]
        mid = 0.5 * (lo + hi); s = pre[..., 0] * np.cos(mid)[:, None] + pre[..., 1] * np.sin(mid)[:, None]
        n_ops += len(lo) * (2 + 2 * H)
        C = pre * (s > 0)[..., None]
    L = np.einsum("ij,njk->nik", Wo, C); n_ops += len(lo) * T * H * 2
    iu, ju = np.triu_indices(T, 1)
    rid, z = _cuts(lo, hi, L[:, iu, 0] - L[:, ju, 0], L[:, iu, 1] - L[:, ju, 1]); n_ops += len(lo) * len(iu) * 4
    # label breakpoints x_i = x_j (global on the circle), and 0, pi for the |sin| weight
    zg = np.concatenate([(np.arctan2(b[iu] - b[ju], a[iu] - a[ju]) + s) % TWO_PI for s in (np.pi / 2, -np.pi / 2)])
    if weight != "uniform":
        zg = np.concatenate([zg, [np.pi]])
    n_ops += len(iu) * 4
    k = np.searchsorted(lo, zg, "right") - 1; ok = (zg > lo[k]) & (zg < hi[k])
    rid = np.concatenate([rid, k[ok]]); z = np.concatenate([z, zg[ok]]); n_ops += len(zg) * int(np.log2(len(lo) + 1) + 1)
    lo, hi, par = _split(lo, hi, rid, z); L = L[par]
    mid = 0.5 * (lo + hi); c, s = np.cos(mid), np.sin(mid)
    pred = (L[..., 0] * c[:, None] + L[..., 1] * s[:, None]).argmax(1)
    x = a[None] * c[:, None] + b[None] * s[:, None]; lab = np.argsort(-x, 1)[:, 1]
    n_ops += len(lo) * (2 + 2 * T + T + 2 * T + T * 2)
    w = (hi - lo) if weight == "uniform" else np.abs(np.cos(lo) - np.cos(hi))
    tot = TWO_PI if weight == "uniform" else 4.0
    if ops is not None:
        ops.add(n_ops + 2 * len(lo))
    return float(w[pred == lab].sum() / tot), len(lo)


def est_hopf(model, nu, ops=None, nd=None):
    """T=4: midpoint grid nu x nd (nd = 2 nu) over the Hopf base; T=3: nu meridians."""
    T = model[2].shape[0]; acc = 0.0
    if T == 3:
        for lam in (np.arange(nu) + 0.5) * np.pi / nu:
            a = np.array([0.0, 0.0, 1.0]); b = np.array([np.cos(lam), np.sin(lam), 0.0])
            acc += fibre(model, a, b, "abs_sin", ops)[0]; ops is not None and ops.add(4)
        return acc / nu
    nd = nd or 2 * nu
    for u in (np.arange(nu) + 0.5) / nu:
        ce, se = np.sqrt(1 - u), np.sqrt(u)
        for d in (np.arange(nd) + 0.5) * TWO_PI / nd:
            a = np.array([ce, 0.0, se * np.cos(d), se * np.sin(d)]); b = np.array([0.0, ce, -se * np.sin(d), se * np.cos(d)])
            acc += fibre(model, a, b, "uniform", ops)[0]; ops is not None and ops.add(10)
    return acc / (nu * nd)
