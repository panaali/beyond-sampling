"""T386: run one or more line-rule configs on one model; save est, grad, cost, directional derivatives.
usage: python run_cfg.py <model> <cfg>[,<cfg>...]   cfg = "<v2spec>/<g12spec>", g12spec = gauss3 | qmid3x<Mg>
-> results/cfg_<model>_<v2spec>_<g12spec>.json"""
import json
import os
import sys
import time

import numpy as np

import common
import dealias


def g12spec(s):
    return ("gauss", int(s[5:])) if s.startswith("gauss") else ("qmid", tuple(map(int, s[4:].split("x"))))


def main(nm, cfgs):
    m, t0, U, D = common.case(nm)
    for cfg in cfgs:
        v2s, gs = cfg.split("/"); path = os.path.join(common.HERE, "results", f"cfg_{nm}_{v2s}_{gs}.json")
        if os.path.exists(path):
            continue
        t = time.time(); r = dealias.est_grad(m, t0, v2s, g12spec(gs))
        out = dict(model=nm, t0=t0, v2=v2s, g12=gs, est=r["est"], grad=r["grad"].tolist(), pred=(U @ r["grad"]).tolist(),
                   C_walk=r["C_walk"], C_grad=r["C_grad"], n_end=r["n_end"], sec=time.time() - t)
        json.dump(out, open(path, "w")); print(nm, cfg, f"est {r['est']:.4e}", "pred/P", np.round(U @ r["grad"] / r["est"], 1),
                                               f"C {r['C_walk']:.3g}+{r['C_grad']:.3g}", round(time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
