"""T372 truth (SAMPLING, truth only): conditional-law MC of E1(t0) (T364 truthc sampler, copied unchanged as
truthc_copy.py) at the base weights and at W +- delta_u u for the 6 frozen directions, all on the SAME draws (common
random numbers). FD_u = p_gap (hits+ - hits-) / (2 delta_u N), s.e. from the paired per-draw differences.
usage: python truth_fd.py model N -> results/truth_fd_<model>.json (checkpointed per chunk)"""
import json
import os
import sys
import time

import numpy as np

import rnn
import truthc_copy as TCC

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = {"orig": 372100, "dev3_s526": 372101, "dev3_s528": 372102}


def hits(model, x, p3):
    return rnn.forward(model, x).argmax(1) == p3


def main(nm, N, chunk=250_000):
    D = json.load(open(os.path.join(HERE, "results", f"dirs_{nm}.json"))); t0 = D["t0"]
    w0 = np.array(D["w"]); sh = [(16,), (16, 16), (10, 16)]; U = np.array(D["dirs"]); dl = np.array(D["delta"])
    base = rnn.unflat(w0, sh); plus = [rnn.unflat(w0 + d * u, sh) for u, d in zip(U, dl)]; minus = [rnn.unflat(w0 - d * u, sh) for u, d in zip(U, dl)]
    draw = TCC.sampler(t0); rng = np.random.default_rng(SEED[nm]); pg = TCC.p_gap(t0)
    n = 0; h0 = 0; hp = np.zeros(len(U), int); hm = np.zeros(len(U), int); dd = np.zeros(len(U)); t = time.time()
    path = os.path.join(HERE, "results", f"truth_fd_{nm}.json")
    while n < N:
        m = min(chunk, N - n); x, p3 = draw(rng, m); b = hits(base, x, p3); h0 += int(b.sum())
        for i in range(len(U)):
            a = hits(plus[i], x, p3); c = hits(minus[i], x, p3)
            hp[i] += int(a.sum()); hm[i] += int(c.sum()); dd[i] += float((a != c).sum())     # sum of (1+ - 1-)^2
        n += m
        if n % (20 * chunk) == 0 or n == N:
            q = h0 / n; mean_d = (hp - hm) / n; var_d = dd / n - mean_d ** 2
            fd = pg * mean_d / (2 * dl); se = pg * np.sqrt(var_d / n) / (2 * dl)
            out = dict(model=nm, t0=t0, n=n, p_gap=pg, hits0=h0, P=pg * q, P_se=pg * np.sqrt(q * (1 - q) / n), hits_plus=hp.tolist(),
                       hits_minus=hm.tolist(), flips=dd.tolist(), P_plus=(pg * hp / n).tolist(), P_minus=(pg * hm / n).tolist(),
                       FD=fd.tolist(), FD_se=se.tolist(), sec=time.time() - t, forward_passes=n * (1 + 2 * len(U)))
            json.dump(out, open(path, "w"), indent=1)
            print(nm, n, "P %.4e" % out["P"], "FD/P", np.round(fd / out["P"], 3), "z", np.round(fd / se, 1), "%.0fs" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])))
