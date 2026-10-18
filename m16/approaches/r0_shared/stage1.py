"""T300 stage 1 (DIAGNOSTIC sampling): per-triple R0 error masses with common value draws; sharing error of p1-shared
rules. usage: python stage1.py [nval] -> results/stage1.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nval = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
rng = np.random.default_rng(300_000)
_, v1, v2, v3 = r0.sample_skeleton(nval, rng)                 # common value draws (positions ignored)
ms = dict(r0.dev2()); names = [list(ms)[i] for i in r0.DEV_IDX]
TR = r0.TRIPLES
cls = {}                                                       # (p2,p3) -> indices of triples by p1
for i, (p1, p2, p3) in enumerate(TR):
    cls.setdefault((p2, p3), []).append(i)
out = {"nval": nval, "models": {}}
for nm in names:
    model = ms[nm]; E = np.zeros(len(TR))
    for i in range(len(TR)):
        tri = np.repeat(TR[i:i + 1], nval, 0)
        E[i] = 1 - r0.r0_correct(model, tri, v1, v2, v3).mean()
    res = {"err_R0": float(E.mean())}
    for rule in ("mid", "two", "orc"):
        d = []
        for (p2, p3), idx in cls.items():
            p1s = TR[idx, 0]; free = sorted(p1s)                 # the 8 possible max positions, ascending
            Ec = E[idx]; mean = Ec.mean()
            if rule == "mid":
                rep = Ec[list(p1s).index(free[3])]
            elif rule == "two":
                rep = 0.5 * (Ec[list(p1s).index(free[2])] + Ec[list(p1s).index(free[5])])
            else:
                rep = Ec[np.argmin(np.abs(Ec - mean))]
            d.append(rep - mean)
        res[f"share_{rule}_pp"] = float(100 * np.mean(d))
        res[f"share_{rule}_class_abs_pp"] = float(100 * np.mean(np.abs(d)))
    # spread of E across p1 within a class (how much the max's position matters)
    res["within_class_sd_pp"] = float(100 * np.mean([E[idx].std() for idx in cls.values()]))
    out["models"][nm] = res; print(nm, {k: round(v, 3) for k, v in res.items()}, flush=True)
for rule in ("mid", "two", "orc"):
    out[f"rms_{rule}_pp"] = float(np.sqrt(np.mean([m[f"share_{rule}_pp"] ** 2 for m in out["models"].values()])))
print({k: round(v, 3) for k, v in out.items() if k.startswith("rms")})
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "stage1.json"), "w"), indent=1)
