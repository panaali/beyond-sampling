"""T372 diagnostic (not pre-registered): is K3's P(W) flatter than the truth's? K3 estimate at W +- delta u_top for
ladder levels 0 and 1, against the CRN truths P+- of truth_fd2 at the same weights. -> results/k3_moved.json"""
import json
import os
import sys

import numpy as np

import k3src as K
import rnn

HERE = os.path.dirname(os.path.abspath(__file__))


def main(names):
    path = os.path.join(HERE, "results", "k3_moved.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    for nm in names:
        D = json.load(open(os.path.join(HERE, "results", f"dirs_{nm}.json"))); T = json.load(open(os.path.join(HERE, "results", f"truth_fd2_{nm}.json")))
        w0 = np.array(D["w"]); u = np.array(D["dirs"][5]); sh = [(16,), (16, 16), (10, 16)]; rows = []
        for lvl in (0, 1):
            d = T["delta"][5][lvl]
            for s, key in ((1, "P_plus"), (-1, "P_minus")):
                e, _, _ = K.T354.est_seg(rnn.unflat(w0 + s * d * u, sh), 3, t0s=(D["t0"],), t2s=(), tmax=6.0)
                rows.append(dict(level=lvl, sign=s, delta=d, K3=e[D["t0"]], truth=T[key][5][lvl], ratio=e[D["t0"]] / T[key][5][lvl]))
                print(nm, rows[-1], flush=True)
        out[nm] = dict(base_K3=D["est"], base_truth=T["P"], base_ratio=D["est"] / T["P"], rows=rows)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
