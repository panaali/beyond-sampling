"""T297 diagnostic: where does TB's integration error come from? TB's assumed-correct indicator vs R0's verdict on the
same sampled skeletons, by region (blind / band23 g23<.05 / band12 g12<.05 / other).
usage: python decomp.py VARIANT... -> results/decomp.json"""
import json
import os
import sys

import numpy as np

import r0
import tb

ms = dict(r0.dev2()); names = [list(ms)[i] for i in r0.DEV_IDX]
out = {}
for v in [(1, 1, 1), (2, 2, 2)]:
    agg = {}
    for nm in names:
        model = ms[nm]
        acc, _, P = tb.est(model, *v, return_parts=True)
        A = tb.anchors(v[0]); rng = np.random.default_rng(29_700)
        tri_i = rng.integers(720, size=1_000_000); tri = r0.TRIPLES[tri_i]
        _, v1, v2, v3 = r0.sample_skeleton(1_000_000, rng)      # values from the exact law, independent of positions
        c = r0.r0_correct(model, tri, v1, v2, v3)
        if v[0] == 1:
            t23 = P["t23"][0][tri_i]
        else:
            T = P["t23"][:, tri_i]                                     # (n23, N)
            w = np.clip((v2 - A[0]) / (A[1] - A[0]), 0, 1)              # n23 = 2: linear between anchors
            t23 = T[0] * (1 - w) + T[1] * w
        tbc = np.where(v2 > 0, (v2 - v3 >= t23) & (v1 - v2 >= P["t12"][tri_i]), P["B"][tri_i] > 0)
        reg = np.where(v2 <= 0, 0, np.where(v2 - v3 < .05, 1, np.where(v1 - v2 < .05, 2, 3)))
        for r, nmr in enumerate(("blind", "band23", "band12", "other")):
            s = reg == r
            d = agg.setdefault(nmr, {"tb_says_correct_r0_wrong": [], "tb_says_wrong_r0_correct": []})
            d["tb_says_correct_r0_wrong"].append(float((tbc & ~c & s).mean()))
            d["tb_says_wrong_r0_correct"].append(float((~tbc & c & s).mean()))
    out[str(v)] = {r: {k: round(100 * float(np.mean(x)), 3) for k, x in d.items()} for r, d in agg.items()}
    print(v, json.dumps(out[str(v)]), flush=True)
json.dump(out, open(os.path.join(r0.HERE, "results", "decomp.json"), "w"), indent=1)
