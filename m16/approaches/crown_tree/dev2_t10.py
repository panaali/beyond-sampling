"""T279 add-on (DRI, independent check of T289): relaxation bounds on the 180 root sectors at (16,10), DEV2 fused_s0.
Per sector (i max, j 2nd, sign of the line coordinate) and rival k: width = hi - lo of the margin m_k over the
sector; margin = |m_k| at the sector's centroid direction (normalised sum of unit generators). FLOPs counted."""
import json
import numpy as np
from tree import roots, crown, crown_fwd, ibp, Flops

w = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
names = list(w["names"])
k0 = names.index("fused_s0")
model = (w["Wi"][k0].astype(float), w["Wh"][k0].astype(float), w["Wo"][k0].astype(float))
Wi, Wh, Wo = model


def logits(x):
    h = np.zeros(Wh.shape[0])
    for t in range(len(x)):
        h = np.maximum(Wh @ h + Wi * x[t], 0)
    return Wo @ h


out = {}
for name, fn in (("ibp", ibp), ("crown_fwd", crown_fwd), ("crown", crown)):
    fl = Flops()
    ratios, sound, cert = [], 0, 0
    for R, lab, key in roots(10):
        lo, hi = fn(model, R, lab, fl, return_bounds=True)
        c = R.sum(0); c /= np.linalg.norm(c)
        z = logits(c)
        mk = z[lab] - np.delete(z, lab)
        ratios += list((hi - lo) / np.maximum(np.abs(mk), 1e-12))
        sound += int(((mk < lo - 1e-9) | (mk > hi + 1e-9)).sum())
        cert += int((lo > 0).all() or (hi < 0).any())
    r = np.array(ratios)
    out[name] = {"median_width_over_margin": float(np.median(r)), "p10": float(np.percentile(r, 10)),
                 "p90": float(np.percentile(r, 90)), "sectors_decided": cert, "centroid_outside_bounds": sound,
                 "flops": fl.n, "FPE": fl.n / 6000.0}
    print(name, out[name], flush=True)
json.dump(out, open("results/dev2_fused_s0_t10.json", "w"), indent=1)
