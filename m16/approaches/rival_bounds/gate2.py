"""T289 gate 2 at (16,10): FLOP cost of the shared forward bounds over the 180 sectors (and the CROWN reference), plus the
pre-registered vacuity diagnostic (sampling only, describes the bounds): per sector, MC fraction of points where some rival's
upper bound < 0 (inner) and where all rivals' lower bounds > 0 (certified correct). DEV2 fused_s0/s1, torch_s0/s1. -> gate2.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import numpy as np, rb
z = np.load(sys.argv[1], allow_pickle=False); names = list(map(str, z["names"])); out = {}
rng = np.random.default_rng(2895)
for nm in ("fused_s0", "fused_s1", "torch_s0", "torch_s1"):
    i = names.index(nm); m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)); Wi, Wh, Wo = m
    r = {}
    for b in ("forward", "crown"):
        fl = rb.T279.Flops(); inner = corr = 0.0; ratio = []
        for R, lab, key in rb.T279.roots(10):
            Lk, Uk, riv = (rb.forward_bounds if b == "forward" else rb.crown_bounds)(m, R, lab, fl)
            lam = rng.dirichlet(np.ones(10), 2000); X = lam @ R; h = np.zeros((len(X), 16))
            for t in range(10): h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
            M = (h @ Wo.T)[:, [lab]] - (h @ Wo.T)[:, riv]; lo = lam @ Lk.T; up = lam @ Uk.T
            assert ((lo <= M + 1e-7) & (up >= M - 1e-7)).all()
            inner += (up < 0).any(1).mean() / 180; corr += (lo > 0).all(1).mean() / 180
            ratio.append(np.median(up - lo) / np.median(np.abs(M)))
        r[b] = dict(flops=fl.n, fpe=fl.n / 6000, gate_pass=bool(fl.n / 6000 <= 1000), frac_inner=float(inner), frac_cert_correct=float(corr),
                    median_width_over_margin=float(np.median(ratio)))
        print(nm, b, r[b], flush=True)
    out[nm] = r
json.dump(out, open(os.path.join(HERE, "gate2.json"), "w"), indent=1)
