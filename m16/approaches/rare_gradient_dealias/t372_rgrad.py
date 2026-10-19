"""T372 primary: the frozen K3 estimate of E1(t0) (T364) and its exact weight-gradient by boundary terms.
est_grad(model, t0) -> dict(est, grad (432,), C_walk, C_grad (FPE), n_end, ...). Deductive (no random inputs).
est equals T354.est_seg(model, 3, (t0,), (), 6.0) (asserted in the checks); grad = d est / dW exactly (up to the
endpoint set being locally constant in W)."""
import numpy as np
from scipy.special import ndtr

import t372_k3src as K
import t372_lgrad as lgrad
import t372_walk as walk

TMAX = 6.0


def est_grad(model, t0, K3=3):
    model = tuple(np.asarray(w, float) for w in model); lines = K.TC.node_lines(K3, 0.0)
    est = 0.0; grad = 0.0; Cw = 0.0; Cg = 0.0; n_end = 0; n_tan = 0; dens_terms = []
    for ri, tab in enumerate(K.G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = lines[ri]; nrow = len(x)
        k = np.tile(K.TRI[:, 1], nrow // K.NT); k3 = np.tile(K.TRI[:, 2], nrow // K.NT)
        node_of_row = np.repeat(np.arange(len(nodes)), K3 * K.NT)
        pieces, _, f = K.T354.walk_seg(model, x, dx, k, k3, t0, TMAX); Cw += f
        W, f = K.rare.mass(pieces, nrow, node_of_row, nodes, t0); Cw += f
        wgt = tab["dens"] * tab["blind12"]
        est += float((W * om[:, None]).sum(0) * wgt @ wts) / K.NT; Cw += nrow * len(nodes) * 3
        rowfac = om * (wgt * wts)[node_of_row] / K.NT
        R = np.concatenate([p[0] for p in pieces]); A = np.concatenate([p[1] for p in pieces]); B = np.concatenate([p[2] for p in pieces])
        A = np.maximum(A, t0); m = B > A; R, A, B = walk.merge(R[m], A[m], B[m])
        v2 = nodes[node_of_row]
        for e, keep in ((A, A > t0 + 1e-9), (B, B < TMAX - 1e-9)):
            r, ee = R[keep], e[keep]
            if not len(r):
                continue
            Gm, J, S, f = lgrad.endpoint_grads(model, x, dx, k3, r, ee); Cg += f
            fd = 8 * ndtr(v2[r] - ee) ** 7 * np.exp(-(v2[r] - ee) ** 2 / 2) / np.sqrt(2 * np.pi)
            c = rowfac[r] * fd; grad = grad + (c[:, None] * Gm).sum(0); n_end += len(r)
            n_tan += int((np.abs(S) < 1e-3).sum()); dens_terms.append(c / np.abs(S))
    return dict(est=est, grad=np.asarray(grad), C_walk=Cw / K.FPE, C_grad=Cg / K.FPE, n_end=n_end, n_tangent=n_tan)
