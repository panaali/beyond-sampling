"""T372: K3 estimate + gradient per pre-registered case, the 6 validation directions (5 random + top) and their
steps delta_u = 0.3 P_K3 / |g . u| (cap 1.0), frozen to results/dirs_<model>.json BEFORE any truth run.
usage: python make_dirs.py model[,model]"""
import json
import os
import sys
import time

import numpy as np

import k3src as K
import rgrad
import rnn

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = {nm: i for i, (nm, _) in enumerate(K.CASES)}


def main(names):
    ms = K.models()
    for nm, t0 in K.CASES:
        if nm not in names:
            continue
        t = time.time(); r = rgrad.est_grad(ms[nm], t0); g = r["grad"]
        rng = np.random.default_rng(372000 + IDX[nm]); U = rng.normal(size=(5, len(g))); U /= np.linalg.norm(U, axis=1, keepdims=True)
        U = np.vstack([U, g / np.linalg.norm(g)])
        delta = np.minimum(0.3 * r["est"] / np.abs(U @ g), 1.0)
        out = dict(model=nm, t0=t0, est=r["est"], grad=g.tolist(), C_walk=r["C_walk"], C_grad=r["C_grad"], n_end=r["n_end"],
                   n_tangent=r["n_tangent"], dirs=U.tolist(), delta=delta.tolist(), pred=(U @ g).tolist(),
                   pred_rel_change=(delta * np.abs(U @ g) / r["est"]).tolist(), w=rnn.flat(ms[nm]).tolist(), sec=time.time() - t)
        json.dump(out, open(os.path.join(HERE, "results", f"dirs_{nm}.json"), "w"))
        print(nm, "est %.4e |g| %.3e C_walk %.0f C_grad %.0f n_end %d" % (r["est"], np.linalg.norm(g), r["C_walk"], r["C_grad"], r["n_end"]),
              "g.u", np.round(U @ g / r["est"], 4), "delta", np.round(delta, 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
