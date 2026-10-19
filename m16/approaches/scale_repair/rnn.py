"""Per-input margin gradients for the RNN h_t = ReLU(Wh h_{t-1} + x_t Wi), logits = Wo h_T (any T, H, C).
margin_grad(model, X, k, j) -> (m, gw, gx): m = L_k - L_j at each row of X, gw (n, n_params) = dm/dW flattened as
concat(Wi, Wh row-major, Wo row-major), gx (n, T) = dm/dx. Gates are those of the forward pass at X (z > 0)."""
import numpy as np


def flat(model):
    return np.concatenate([np.ravel(w) for w in model])


def unflat(v, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return tuple(out)


def shapes(model):
    return [np.shape(w) for w in model]


def forward(model, X):
    Wi, Wh, Wo = model; n, T = X.shape; h = np.zeros((n, len(Wi)))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi, 0.0)
    return h @ Wo.T


def margin_grad(model, X, k, j):
    Wi, Wh, Wo = model; n, T = X.shape; H = len(Wi); r = np.arange(n)
    hs = [np.zeros((n, H))]; gs = []
    for t in range(T):
        z = hs[-1] @ Wh.T + X[:, t:t + 1] * Wi; g = z > 0; gs.append(g); hs.append(np.where(g, z, 0.0))
    L = hs[-1] @ Wo.T; m = L[r, k] - L[r, j]
    c = np.zeros_like(L); c[r, k] += 1.0; c[r, j] -= 1.0
    gWo = c[:, :, None] * hs[-1][:, None, :]
    dh = c @ Wo; gWh = np.zeros((n, H, H)); gWi = np.zeros((n, H)); gx = np.zeros((n, T))
    for t in range(T - 1, -1, -1):
        dz = np.where(gs[t], dh, 0.0)
        gWh += dz[:, :, None] * hs[t][:, None, :]; gWi += dz * X[:, t:t + 1]; gx[:, t] = dz @ Wi
        dh = dz @ Wh
    return m, np.concatenate([gWi, gWh.reshape(n, -1), gWo.reshape(n, -1)], 1), gx


def flops_per_grad(T, H, C):
    """Executed FLOPs per row of margin_grad: forward (T (2H^2 + 3H) + 2CH), backward (output outer CH + 2CH,
    per step mask H + outer-acc 2H^2 + gWi 2H + gx 2H + dh 2H^2)."""
    return T * (2 * H * H + 3 * H) + 2 * C * H + 3 * C * H + T * (H + 4 * H * H + 4 * H)
