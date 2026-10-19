"""T386: margin change between adjacent lines at the hard K3 endpoints, |dm/dx . dx/dv2| h_v2 and |dm/dx . dx/dg12| h_g12
(h = local node spacing), the scale that a line-spacing-tied bandwidth eps refers to. -> printed quantiles."""
import sys

import numpy as np
from scipy.special import ndtr

import common
import dealias
import t372_rnn as rnn
import t372_walk as walk

TRI, NT, G = dealias.TRI, dealias.NT, dealias.G


def spacing_scale(model, t0):
    out = []
    for name, lo, hi in G.REGIONS:
        vn, _ = dealias.v2_rule("gl20", lo, hi); hv = np.gradient(vn)
        for j, v2 in enumerate(vn):
            x, dx, om = dealias.lines(v2, ("gauss", 3)); k3 = np.tile(TRI[:, 2], len(x) // NT)
            pieces, _, f = dealias.K.T354.walk_seg(model, x, dx, np.tile(TRI[:, 1], len(x) // NT), k3, t0, 6.0)
            R = np.concatenate([p[0] for p in pieces]); A = np.maximum(np.concatenate([p[1] for p in pieces]), t0); B = np.concatenate([p[2] for p in pieces])
            mk = B > A
            if not mk.any():
                continue
            R, A, B = walk.merge(R[mk], A[mk], B[mk])
            xp, _, _ = dealias.lines(v2 + 1e-4, ("gauss", 3)); xm, _, _ = dealias.lines(v2 - 1e-4, ("gauss", 3)); dxv = (xp - xm) / 2e-4
            gn, _ = dealias.g12_rule(("gauss", 3), v2); hg = np.diff(gn).mean()
            for e, keep in ((A, A > t0 + 1e-9), (B, B < 6 - 1e-9)):
                r, ee = R[keep], e[keep]
                if not len(r):
                    continue
                X = x[r] + ee[:, None] * dx[r]; L = rnn.forward(model, X); L2 = L.copy(); L2[np.arange(len(r)), k3[r]] = -np.inf
                m, gw, gx = rnn.margin_grad(model, X, k3[r], L2.argmax(1))
                dg = np.zeros(10); 
                g1 = np.zeros((len(r), 10)); g1[np.arange(len(r)), TRI[r % NT, 0]] = 1.0     # dx/dg12 = e_{p1}
                out.append(np.c_[np.abs((gx * dxv[r]).sum(1)) * hv[j], np.abs((gx * g1).sum(1)) * hg, np.abs((gx * dx[r]).sum(1))])
    return np.concatenate(out)


if __name__ == "__main__":
    for nm in sys.argv[1].split(","):
        m, t0, U, D = common.case(nm); s = spacing_scale(m, t0)
        print(nm, "n_end", len(s), "quantiles 10/50/90 of |dm| per v2 spacing", np.round(np.quantile(s[:, 0], [.1, .5, .9]), 3),
              "per g12 spacing", np.round(np.quantile(s[:, 1], [.1, .5, .9]), 3), "|dm/dt|", np.round(np.quantile(s[:, 2], [.1, .5, .9]), 3), flush=True)
