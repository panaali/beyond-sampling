"""Independent exact walker (T372; not rare.walk3): along lines x(v) = x0 + v d, v in [lo, hi], the ReLU RNN is affine
inside each gate cell. Every iteration re-runs the forward pass with tangents at the current v for all active lines
(gates: z > tol, or |z| <= tol and dz > 0 = the side the line moves into), takes the next pre-activation root, and
records the sub-interval of the cell on which logit k is the argmax.
argmax_pieces(model, x0, d, k, lo, hi) -> (rows, a, b) arrays of pieces (a < b) and executed FLOPs."""
import numpy as np


def _fwd_tan(model, X, D):
    Wi, Wh, Wo = model; n, T = X.shape; H = len(Wi)
    h = np.zeros((n, H)); dh = np.zeros((n, H)); Z = np.zeros((n, T, H)); DZ = np.zeros((n, T, H))
    for t in range(T):
        z = h @ Wh.T + X[:, t:t + 1] * Wi; dz = dh @ Wh.T + D[:, t:t + 1] * Wi
        tol = 1e-10 * (1.0 + np.abs(z).max(1, keepdims=True))
        g = (z > tol) | ((np.abs(z) <= tol) & (dz > 0))
        Z[:, t] = np.where(np.abs(z) > tol, z, 0.0); DZ[:, t] = dz          # a root at the current v is not "next"
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return Z, DZ, h @ Wo.T, dh @ Wo.T


def _interval(L, dL, k, lo, hi):
    r = np.arange(len(L)); c = L[r, k][:, None] - L; e = dL[r, k][:, None] - dL
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -c / e                                                            # offset from the cell start
    lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
    bad = (e == 0) & (c < 0)
    lb[r, k] = -np.inf; ub[r, k] = np.inf
    a = np.maximum(lb.max(1), lo); b = np.minimum(ub.min(1), hi)
    return a, b, ~bad.any(1)


def argmax_pieces(model, x0, d, k, lo, hi, max_cells=10000):
    Wi, Wh, Wo = model; n, T = x0.shape; H = len(Wi); C = len(Wo)
    v = lo.astype(float).copy(); act = np.arange(n); R, A, B = [], [], []; flops = 0.0; it = 0
    fl_cell = T * (4 * H * H + 6 * H + 6 * H) + 4 * C * H + 6 * C + T * H * 4
    while len(act) and it < max_cells:
        it += 1
        X = x0[act] + v[act, None] * d[act]; Z, DZ, L, dL = _fwd_tan(model, X, d[act])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -Z / DZ
        r = np.where(np.isfinite(r) & (r > 1e-12 * (1 + np.abs(v[act]))[:, None, None]), r, np.inf).reshape(len(act), -1).min(1)
        nxt = np.minimum(v[act] + r, hi[act])
        a, b, ok = _interval(L, dL, k[act], np.zeros(len(act)), nxt - v[act])
        m = ok & (b > a)
        R.append(act[m]); A.append(v[act][m] + a[m]); B.append(v[act][m] + b[m])
        flops += len(act) * fl_cell
        v[act] = nxt; act = act[nxt < hi[act]]
    if len(act):
        raise RuntimeError(f"{len(act)} lines exceeded max_cells")
    return np.concatenate(R), np.concatenate(A), np.concatenate(B), flops


def merge(rows, a, b, tol=1e-9):
    """Union of pieces per row -> (rows, a, b) of maximal intervals (touching pieces joined)."""
    o = np.lexsort((a, rows)); rows, a, b = rows[o].tolist(), a[o].tolist(), b[o].tolist()
    R, A, B = [], [], []
    for r, x, y in zip(rows, a, b):
        if R and R[-1] == r and x <= B[-1] + tol * (1 + abs(x)):
            B[-1] = max(B[-1], y)
        else:
            R.append(r); A.append(x); B.append(y)
    return np.array(R, int), np.array(A), np.array(B)
