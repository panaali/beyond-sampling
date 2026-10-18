"""T270 diagnosis: which d = 6 mass rule fails? For the worst regions of subtree POS (largest |z| in
results/prefix6/mc6_POS.json, MC mass known), recompute the region mass under rule variants and compare with the MC:
 A default (|det| < 1e-2: sliver split cos >= .6 + ray rule with NTHIN^5 Duffy nodes; else nested Plackett)
 B ray rule with 2x NTHIN nodes   C sliver threshold cos >= .9   D B + C
 E Plackett down to |det| 1e-4 (thin rule below)   F Plackett for all
usage: python sx6.py POS [nreg]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass, orth6  # noqa: E402
pos = int(sys.argv[1]); nreg = int(sys.argv[2]) if len(sys.argv) > 2 else 8
w = json.load(open(os.path.join(Q.OUT, f"mc6_{pos}.json")))["worst"][:nreg]
model, front = Q.front_list(11); P._init(model, Q.K, True)
n, (hists, Rs, Is), o, s, sec = P._dfs(front[pos])
nt0 = orth6.NTHIN[0]; out = []
for r in w:
    R = Rs[Is == r["i"]]; I0 = np.zeros(len(R), np.int64); dt = np.abs(np.linalg.det(R)); res = {}
    for name, nt, thr, th6 in (("A", nt0, .6, 1e-2), ("B", 2 * nt0, .6, 1e-2), ("C", nt0, .9, 1e-2), ("D", 2 * nt0, .9, 1e-2),
                               ("E", nt0, .6, 1e-4), ("F", nt0, .6, 0.0)):
        orth6.NTHIN[0] = nt; gmass.THIN[6] = th6
        try:
            res[name] = float(gmass.grouped(R, I0, 1, "cuda", thr=thr, chunk=1024, tb=512)[0])
        except Exception as e:  # noqa: BLE001
            res[name] = repr(e)[:80]
    orth6.NTHIN[0] = nt0; gmass.THIN[6] = 1e-2
    out.append(dict(region=r["i"], mc=r["mc"], mc_se=(r["mc"] / 1e8) ** .5, simplices=len(R),
                    thin_1e2=int((dt < 1e-2).sum()), thin_1e4=int((dt < 1e-4).sum()), thin_1e6=int((dt < 1e-6).sum()), **res))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(os.path.join(Q.OUT, f"sx6_{pos}.json"), "w"), indent=1)
