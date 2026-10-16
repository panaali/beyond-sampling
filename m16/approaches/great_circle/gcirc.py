"""Great-circle exact integration of the correct fraction for a bias-free ReLU RNN (M_16,10).

x(th) = cos(th) u + sin(th) v with (u, v) an orthonormal pair. Within one gate piece every pre-activation,
logit and input is a.cos + b.sin, so all breakpoints are closed form: the zeros of R cos(th - phi) are
phi + pi/2 + k pi. We walk gate pieces around [0, 2pi) for a whole batch of circles at once, and inside
each piece find the argmax switches and the target (2nd-largest input) switches, and sum the arc length
where argmax == target. By rotation invariance of N(0, I) and homogeneity, E[arc fraction] = accuracy.

Compute accounting: one 'piece' = one forward pass carrying the two coefficient vectors (A, B). We also
report 'partial' cost: a gate flip at step t only requires recomputing steps t..9, so the piece costs
(10 - t_flip)/10 of a pass (t_flip = step of the neuron that flipped at the previous breakpoint).
"""
import numpy as np

T = 10
TWO_PI = 2 * np.pi
_IU = np.triu_indices(T, 1)


def _next_root(a, b, th):
    """Smallest root > th (strictly) of a cos + b sin; th broadcast on leading axis. Returns th + delta."""
    phi = np.arctan2(b, a)
    d = np.mod(phi + np.pi / 2 - th[:, None], np.pi)
    d = np.where(d < 1e-12, d + np.pi, d)
    return d


def walk(model, U, V, max_pieces=5000, return_partial=True):
    """Exact correct fraction of each circle. U, V: (N, T) orthonormal pairs.
    Returns frac (N,), pieces (N,), partial_cost (N,)."""
    Wi, Wh, Wo = model
    H = Wh.shape[0]
    N = len(U)
    th = np.zeros(N)
    mass = np.zeros(N)
    pieces = np.zeros(N, int)
    pcost = np.zeros(N)
    tflip = np.zeros(N, int)              # step of last flip (0 at start => full pass)
    act = np.arange(N)
    eps = 1e-10
    while act.size:
        u, v, t0 = U[act], V[act], th[act]
        n = len(act)
        c, s = np.cos(t0 + eps), np.sin(t0 + eps)
        hA = np.zeros((n, H)); hB = np.zeros((n, H))
        zA = np.empty((n, T, H)); zB = np.empty((n, T, H))
        for t in range(T):
            a = hA @ Wh.T + u[:, t:t + 1] * Wi[None]
            b = hB @ Wh.T + v[:, t:t + 1] * Wi[None]
            zA[:, t], zB[:, t] = a, b
            g = (a * c[:, None] + b * s[:, None]) > 0
            hA = np.where(g, a, 0.0); hB = np.where(g, b, 0.0)
        pieces[act] += 1
        pcost[act] += (T - tflip[act]) / T
        # next gate breakpoint
        dz = _next_root(zA.reshape(n, -1), zB.reshape(n, -1), t0)       # (n, T*H)
        k = dz.argmin(1)
        dg = dz[np.arange(n), k]
        end = np.minimum(t0 + dg, TWO_PI)
        tflip[act] = k // H
        # within [t0, end]: logits and inputs
        LA, LB = hA @ Wo.T, hB @ Wo.T                                       # (n, 10)
        cand = [t0[:, None], end[:, None]]
        for (pa, pb) in ((LA, LB), (u, v)):
            da = pa[:, _IU[0]] - pa[:, _IU[1]]; db = pb[:, _IU[0]] - pb[:, _IU[1]]
            r1 = t0[:, None] + _next_root(da, db, t0)
            cand += [r1, r1 + np.pi]
        cand = np.concatenate(cand, 1)
        cand = np.minimum(np.maximum(cand, t0[:, None]), end[:, None])
        cand.sort(1)
        mid = 0.5 * (cand[:, 1:] + cand[:, :-1]); ln = cand[:, 1:] - cand[:, :-1]
        cm, sm = np.cos(mid), np.sin(mid)
        lg = LA[:, None, :] * cm[..., None] + LB[:, None, :] * sm[..., None]
        xs = u[:, None, :] * cm[..., None] + v[:, None, :] * sm[..., None]
        tgt = np.argsort(xs, -1)[..., -2]
        ok = lg.argmax(-1) == tgt
        mass[act] += (ln * ok).sum(1)
        th[act] = end
        keep = (end < TWO_PI) & (pieces[act] < max_pieces)
        act = act[keep]
    return mass / TWO_PI, pieces, pcost


def random_planes(n, rng):
    """n independent uniformly random 2-planes (orthonormal u, v)."""
    g = rng.standard_normal((n, T, 2))
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    return q[:, :, 0], q[:, :, 1]


def frame_planes(n_frames, rng):
    """Each Haar-random orthogonal frame gives 5 mutually orthogonal planes (each marginally uniform)."""
    g = rng.standard_normal((n_frames, T, T))
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    U = q[:, :, 0::2].transpose(0, 2, 1).reshape(-1, T)
    V = q[:, :, 1::2].transpose(0, 2, 1).reshape(-1, T)
    return U, V
