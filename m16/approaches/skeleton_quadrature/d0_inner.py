"""D0 (pre-registered test 3): gain g_in of a low-discrepancy inner rule over iid for q(S), the 7-D integral over the
low inputs given a top-3 skeleton. DIAGNOSIS ONLY (random skeletons, randomised scrambles to measure the error).
For each skeleton: R replicates of q_hat with n points, (a) iid truncnorm, (b) scrambled Sobol mapped by the inverse
conditional CDF (dimension j = j-th free position in time), (c) same Sobol with dimensions in REVERSED time order.
g_in(n) = E_S[Var_iid] / E_S[Var_rule]. End-to-end ceiling of any perfect skeleton rule = g_in / (1 - leverage).
usage: python m16/approaches/skeleton_quadrature/d0_inner.py MODEL [NS] [R]
"""
import json
import sys

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NL = (16, 64, 256)
model = rv.dev_model(name)
rng = np.random.default_rng(1001)
acc = {f"{k}{n}": [] for n in NL for k in ("iid", "sob", "sobrev")}
qs = []
for s in range(NS):
    pos, val = rv.skeletons(1, 3, rng)
    free = rv.free_positions(pos)
    lab = pos[0, 1]
    c3 = norm.cdf(val[0, 2])
    for n in NL:
        est = {"iid": [], "sob": [], "sobrev": []}
        blocks = []
        for r in range(R):
            blocks.append(("iid", rv.low_draws(val[:, 2], n, 7, rng)[0]))
            u = qmc.Sobol(7, scramble=True, seed=rng).random(n)
            x = norm.ppf(np.clip(u * c3, 1e-300, None))
            blocks.append(("sob", x))
            blocks.append(("sobrev", x[:, ::-1]))
        X = np.stack([b[1] for b in blocks])                     # (3R,n,7)
        xin = rv.assemble(np.repeat(pos, len(blocks), 0), np.repeat(val, len(blocks), 0),
                          np.repeat(free, len(blocks), 0), X)
        ok = (rv.run(model, xin.reshape(-1, rv.T))[0].argmax(1) == lab).reshape(len(blocks), n).mean(1)
        for (k, _), v in zip(blocks, ok):
            est[k].append(v)
        for k in est:
            acc[f"{k}{n}"].append(np.var(est[k], ddof=1))
        if n == NL[-1]:
            qs.append(np.mean(est["iid"]))
    if s % 100 == 0:
        print(s, flush=True)
out = dict(model=name, NS=NS, R=R, Eq1q=float(np.mean([q * (1 - q) for q in qs])))
for n in NL:
    vi = np.mean(acc[f"iid{n}"])
    out[f"n{n}"] = {k: dict(mean_var=float(np.mean(acc[f"{k}{n}"])), gain=float(vi / np.mean(acc[f"{k}{n}"])))
                    for k in ("iid", "sob", "sobrev")}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/d0_{name}.json", "w"), indent=1)
