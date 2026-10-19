"""T372 check (ii): analytic directional derivative of the K3 estimate vs central FD of the K3 estimate itself
(delta = 1e-5 along 3 random unit directions), on the 3 pre-registered cases. -> results/check_k3fd.json"""
import json
import os
import sys
import time

import numpy as np

import k3src as K
import rgrad
import rnn

HERE = os.path.dirname(os.path.abspath(__file__))


def main(names):
    ms = K.models(); out = {}
    for nm, t0 in K.CASES:
        if nm not in names:
            continue
        m = ms[nm]; sh = rnn.shapes(m); w0 = rnn.flat(m); t = time.time()
        r = rgrad.est_grad(m, t0); ref, _, _ = K.T354.est_seg(m, 3, t0s=(t0,), t2s=(), tmax=6.0)
        rng = np.random.default_rng(372900); rows = []
        for i in range(3):
            u = rng.normal(size=len(w0)); u /= np.linalg.norm(u); h = 1e-5
            ep = rgrad.est_grad(rnn.unflat(w0 + h * u, sh), t0)["est"]; em = rgrad.est_grad(rnn.unflat(w0 - h * u, sh), t0)["est"]
            fd = (ep - em) / (2 * h); rows.append(dict(analytic=float(r["grad"] @ u), fd=fd, rel=float(r["grad"] @ u / fd - 1)))
            print(nm, rows[-1], flush=True)
        out[nm] = dict(t0=t0, est=r["est"], est_ref=ref[t0], C_walk=r["C_walk"], C_grad=r["C_grad"], n_end=r["n_end"],
                       n_tangent=r["n_tangent"], dirs=rows, sec=time.time() - t)
        print(nm, {k: v for k, v in out[nm].items() if k != "dirs"}, flush=True)
        json.dump(out, open(os.path.join(HERE, "results", "check_k3fd_%s.json" % "_".join(names)), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
