"""T388 memory wrapper: t366.est_e1 with each region's lines walked in row chunks (<= CH rows at a time), to stay
under the 2 GB per-process cap. Lines are independent rows in walk_seg and rare.mass, so the result equals est_e1 up
to summation order; checked against the unchunked frozen run on models it completed (est_t388.py --check).
All frozen code (t366.py, t354.py, walkfix.py, t340c.py, rare.py, cont_g.py) is called unmodified."""
import numpy as np

CH = 40_000


def est_e1_chunked(t366, model, t0s, tmax=None):
    rare, G, S, NT, TRI = t366.rare, t366.G, t366.S, t366.NT, t366.TRI
    tmax = rare.TBIG if tmax is None else tmax
    model = tuple(np.asarray(w, np.float64) for w in model); res = {t0: 0.0 for t0 in t0s}; flops = 0.0; t_start = min(t0s)
    for ri, tab in enumerate(G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = t366.LINES[ri]; nrow = len(x)
        k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT); node_of_row = np.repeat(np.arange(len(nodes)), t366.NN * NT)
        wgt = tab["dens"] * tab["blind12"]; acc = {t0: np.zeros(len(nodes)) for t0 in t0s}
        for a in range(0, nrow, CH):
            sl = slice(a, min(a + CH, nrow)); n = sl.stop - sl.start
            pieces, _, f = S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f
            for t0 in t0s:
                W, f = rare.mass(pieces, n, node_of_row[sl], nodes, t0); flops += f
                acc[t0] += (W * om[sl][:, None]).sum(0)
        for t0 in t0s:
            res[t0] += float(acc[t0] * wgt @ wts) / NT; flops += nrow * len(nodes) * 3
    return res, flops
