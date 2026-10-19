"""Boundary-term gradient of exact line masses (T372).
Given merged argmax intervals [a, b] of target k on lines x(v) = x0 + v d, and a line density f(v) (per row), the
line mass M = int_a^b f(v) dv moves only through the endpoints that are roots of an argmax margin m = L_k - L_j:
dM/dW = sum_e f(e) (dm/dW)(x(e)) / |dm/dv (e)|. Endpoints flagged fixed (window ends, t0, T_hi) contribute 0.
endpoint_grads(model, x0, d, k, rows, e) -> per endpoint (dm/dW / |dm/dv|) (n_e, P), the competitor j, dm/dv, FLOPs."""
import numpy as np

import rnn


def endpoint_grads(model, x0, d, k, rows, e, chunk=20000):
    n = len(rows); P = sum(int(np.prod(np.shape(w))) for w in model); out = np.zeros((n, P)); J = np.zeros(n, int)
    S = np.zeros(n); T = x0.shape[1]; H = len(model[0]); C = len(model[2])
    for s in range(0, n, chunk):
        r = rows[s:s + chunk]; X = x0[r] + e[s:s + chunk, None] * d[r]; kk = k[r]
        L = rnn.forward(model, X); L2 = L.copy(); L2[np.arange(len(r)), kk] = -np.inf; jj = L2.argmax(1)
        m, gw, gx = rnn.margin_grad(model, X, kk, jj)
        dmdv = (gx * d[r]).sum(1)
        out[s:s + chunk] = gw / np.abs(dmdv)[:, None]; J[s:s + chunk] = jj; S[s:s + chunk] = dmdv
    return out, J, S, n * (rnn.flops_per_grad(T, H, C) + 2 * T + 2 * C * H)
