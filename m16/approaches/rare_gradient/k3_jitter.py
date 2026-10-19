"""T372 diagnostic (not pre-registered): is the K3 estimate's pointwise slope erratic? Analytic K3 derivative along
u_top at W + s u_top for small s, against the truth's slope. -> results/k3_jitter_<model>.json"""
import json
import os
import sys

import numpy as np

import rgrad
import rnn

HERE = os.path.dirname(os.path.abspath(__file__))
S = (-1e-3, -3e-4, -1e-4, 1e-4, 3e-4, 1e-3)


def main(nm):
    D = json.load(open(os.path.join(HERE, "results", f"dirs_{nm}.json"))); w0 = np.array(D["w"]); u = np.array(D["dirs"][5])
    sh = [(16,), (16, 16), (10, 16)]; rows = [dict(s=0.0, est=D["est"], slope=D["pred"][5], dlogP=D["pred"][5] / D["est"])]
    for s in S:
        r = rgrad.est_grad(rnn.unflat(w0 + s * u, sh), D["t0"]); g = float(r["grad"] @ u)
        rows.append(dict(s=s, est=r["est"], slope=g, dlogP=g / r["est"], n_end=r["n_end"])); print(nm, rows[-1], flush=True)
        json.dump(rows, open(os.path.join(HERE, "results", f"k3_jitter_{nm}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
