"""H1/H3/H4 + learnability ceiling: held-out R^2 of compact model families for delta, |delta| and mass.
usage: python fit.py <model>   (train on first 2/3 of the v>0 lines, test on the rest)"""
import sys, json
import numpy as np
import feats as F

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"
R = F.load(nm); n = len(R["v"]); cut = 2 * n // 3
tr = {k: v[:cut] for k, v in R.items()}; te = {k: v[cut:] for k, v in R.items()}
Ttr, Tte = F.targets(tr), F.targets(te)
fam = [("const", "none", "const"), ("lookup(a,b)", "ab", "const"), ("lookup(a,b,p1)", "abp", "const"),
       ("(a,b,p1)-lookup + (a,b)-linear in v,max,sorted lower", None, None),
       ("per (a,b): linear in values by position", "ab", "raw"),
       ("per (a,b,p1): linear in values by position", "abp", "raw"),
       ("per (a,b): quadratic in values by position", "ab", "rawquad")]
out = {}
for tname in ["delta", "absdelta", "mass"]:
    ytr, yte = Ttr[tname], Tte[tname]
    for lab, ck, dk in fam:
        if ck is None:
            p1_, n1 = F.fit_cells(tr, ytr, tr, "abp", "const")
            p1t, _ = F.fit_cells(tr, ytr, te, "abp", "const")
            p2t, n2 = F.fit_cells(tr, ytr - p1_, te, "ab", "roles")
            p, npar = p1t + p2t, n1 + n2
        else:
            p, npar = F.fit_cells(tr, ytr, te, ck, dk)
        out[(tname, lab)] = (F.r2(p, yte), npar)
    p, npar = F.mlp(tr, ytr, te)
    out[(tname, "MLP 3x256 (not compact; ceiling)")] = (F.r2(p, yte), npar)
    # first-order mechanistic quantity at the tie (needs a forward+JVP pass per point)
    if tname == "delta":
        out[(tname, "delta_lin = -m/s at tie (1 pass/point)")] = (F.r2(np.clip(te["delta_lin"], -F.CL, F.CL), yte), 0)
    if tname == "absdelta":
        out[(tname, "|delta_lin| at tie (1 pass/point)")] = (F.r2(np.clip(np.abs(te["delta_lin"]), 0, F.CL), yte), 0)
print(f"model {nm}: train {cut}, test {n-cut}")
for (t, lab), (r, npar) in out.items():
    print(f"  {t:9s} {lab:58s} R2_test={r:+.3f}  params={npar}")
json.dump({f"{t}|{lab}": [r, npar] for (t, lab), (r, npar) in out.items()}, open(f"fit_{nm}.json", "w"), indent=1)
