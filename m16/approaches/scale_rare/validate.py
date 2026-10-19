"""T424 gate: gen.py at (16, 10) vs the frozen code (hashes asserted) on registry models.
  K3: frozen t354.est_seg(model, 3, t0s, t2s=(), tmax=6) (unchunked) vs gen.est_k3
  graded: frozen t366.est_e1 loop, rows chunked as rare_noise/t388_chunked.py (frozen walk_seg / mass unmodified;
          an unchunked run exceeds 2 GB) vs gen.est_graded
usage: python validate.py TAG[,TAG] -> results/validate.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import freeze  # noqa: E402
import gen  # noqa: E402
import models as M  # noqa: E402

T0S = (1.0, 1.5, 2.0, 2.5)
E, S, T366 = freeze.frozen()


def frozen_graded_chunked(model, t0s, CH=40_000):
    rare, G, NT, TRI, NN = T366.rare, T366.G, T366.NT, T366.TRI, T366.NN
    res = {t0: 0.0 for t0 in t0s}; flops = 0.0
    for ri, tab in enumerate(G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = T366.LINES[ri]; nrow = len(x)
        k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT); nor = np.repeat(np.arange(len(nodes)), NN * NT)
        wgt = tab["dens"] * tab["blind12"]; acc = {t0: np.zeros(len(nodes)) for t0 in t0s}
        for a in range(0, nrow, CH):
            sl = slice(a, min(a + CH, nrow)); n = sl.stop - sl.start
            pieces, _, f = S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], min(t0s), rare.TBIG); flops += f
            for t0 in t0s:
                W, f = rare.mass(pieces, n, nor[sl], nodes, t0); flops += f; acc[t0] += (W * om[sl][:, None]).sum(0)
        for t0 in t0s:
            res[t0] += float(acc[t0] * wgt @ wts) / NT; flops += nrow * len(nodes) * 3
    return res, flops


if __name__ == "__main__":
    fn = os.path.join(HERE, "results", "validate.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for tag in sys.argv[1].split(","):
        m = M.load(M.by_tag(tag)); rec = {}
        t = time.time(); a, fa = S.est_seg(m, 3, t0s=T0S, t2s=(), tmax=6.0)[0::2]; b, fb = gen.est_k3(m, T0S)
        rec["k3"] = dict(frozen={str(k): v for k, v in a.items()}, gen={str(k): v for k, v in b.items()},
                         maxrel=max(abs(b[k] / a[k] - 1) for k in T0S), C_frozen=fa / 5948, C_gen=fb / gen.fpe(16, 10), sec=time.time() - t)
        print(tag, "k3", rec["k3"], flush=True)
        t = time.time(); a, fa = frozen_graded_chunked(m, T0S); b, fb = gen.est_graded(m, T0S)
        rec["graded"] = dict(frozen={str(k): v for k, v in a.items()}, gen={str(k): v for k, v in b.items()},
                             maxrel=max(abs(b[k] / a[k] - 1) for k in T0S), C_frozen=fa / 5948, C_gen=fb / gen.fpe(16, 10), sec=time.time() - t)
        print(tag, "graded", rec["graded"], flush=True)
        out[tag] = rec; json.dump(out, open(fn, "w"), indent=1)
