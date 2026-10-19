"""T405: per pre-registered case (model, t0): the K3 estimate + gradient (T372's direction definition: 5 random unit
directions + the top direction g_K3 / |g_K3|; step delta_u = 0.3 P_K3 / |g_K3 . u|, cap 1.0) and the frozen T5/gl20
estimate + gradient (T386, unchanged). -> results/case_<model>_<t0>.json
usage: python make_cases.py <model>[,<model>...]"""
import json
import os
import sys
import time

import numpy as np

import frozen

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("dev3_s529", "dev3_s530", "dev3_s531")             # pre-registered (README)
T0S = (2.35, 2.75)
CASES = [(m, t) for m in MODELS for t in T0S]


def path(nm, t0):
    return os.path.join(HERE, "results", f"case_{nm}_{t0}.json")


def main(names):
    for ci, (nm, t0) in enumerate(CASES):
        if nm not in names or os.path.exists(path(nm, t0)):
            continue
        m = frozen.dev3(nm); w = np.concatenate([np.ravel(a) for a in m])
        t = time.time(); k = frozen.k3(m, t0); tk = time.time() - t; g = k["grad"]
        rng = np.random.default_rng(405000 + ci); U = rng.normal(size=(5, len(g))); U /= np.linalg.norm(U, axis=1, keepdims=True)
        U = np.vstack([U, g / np.linalg.norm(g)]); delta = np.minimum(0.3 * k["est"] / np.abs(U @ g), 1.0)
        t = time.time(); r = frozen.t5(m, t0); tt = time.time() - t
        out = dict(model=nm, t0=t0, case=ci, w=w.tolist(), dirs=U.tolist(), delta=delta.tolist(),
                   k3=dict(est=k["est"], grad=g.tolist(), pred=(U @ g).tolist(), C_walk=k["C_walk"], C_grad=k["C_grad"], n_end=k["n_end"], sec=tk),
                   t5=dict(est=r["est"], grad=r["grad"].tolist(), pred=(U @ r["grad"]).tolist(), C_walk=r["C_walk"], C_grad=r["C_grad"],
                           n_end=r["n_end"], sec=tt))
        json.dump(out, open(path(nm, t0), "w"))
        print(nm, t0, "K3 %.4e T5 %.4e" % (k["est"], r["est"]), "C K3 %.0f T5 %.0f+%.0f" % (k["C_walk"], r["C_walk"], r["C_grad"]),
              "delta", np.round(delta, 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
