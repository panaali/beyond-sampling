"""Phase B: encoding of each neuron's pre-activation by interpretable variables (held-out R^2).

Pooled fits for every step t and neuron j (all 16 at once), for the families in encode.FAMILIES.
Also: pooled vs within-pattern fits at t=10 for the 3 commonest step-9 gate patterns (the gates that drive the
last recurrence), and fits of the mode coordinates of final pattern 64975.
Usage: python phaseB.py [Ntrain] [Ntest]
"""
import json
import sys
import numpy as np
from nacommon import sample, Wi, Wh, Wo, T, H
from encode import FAMILIES, CellLinear, design, r2

Ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
Nte = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
xtr, xte = sample(Ntr, 241021), sample(Nte, 241022)
out = {"Ntrain": Ntr, "Ntest": Nte, "families": [f[0] for f in FAMILIES], "r2": {}, "unseen": {}}
htr, hte = np.zeros((Ntr, H)), np.zeros((Nte, H))
for t in range(1, T + 1):
    ptr = htr @ Wh.T + xtr[:, t - 1:t] * Wi[None]
    pte = hte @ Wh.T + xte[:, t - 1:t] * Wi[None]
    out["r2"][t], out["unseen"][t] = {}, {}
    for fam in FAMILIES:
        ctr, Vtr = design(fam, xtr, t)
        cte, Vte = design(fam, xte, t)
        mdl = CellLinear().fit(ctr, Vtr, ptr)
        yhat = mdl.predict(cte, Vte)
        out["r2"][t][fam[0]] = r2(pte, yhat).tolist()
        out["unseen"][t][fam[0]] = float(mdl.unseen_frac)
        del Vtr, Vte, mdl
    print(t, {f[0][:6]: np.round(np.median(out["r2"][t][f[0]]), 4) for f in FAMILIES}, flush=True)
    if t == T:
        # per-pattern fits (pattern = gates of h_9, i.e. which neurons feed the last step)
        g9tr = (htr > 0) @ (1 << np.arange(H))
        g9te = (hte > 0) @ (1 << np.arange(H))
        vals, cnt = np.unique(g9tr, return_counts=True)
        out["pattern"] = {}
        for g in vals[np.argsort(-cnt)[:3]]:
            itr, ite = g9tr == g, g9te == g
            rec = {"share": float(itr.mean())}
            for fam in [FAMILIES[0], FAMILIES[1], FAMILIES[3], FAMILIES[5]]:
                ctr, Vtr = design(fam, xtr, t)
                cte, Vte = design(fam, xte, t)
                pooled = CellLinear().fit(ctr, Vtr, ptr)
                within = CellLinear().fit(ctr[itr], Vtr[itr], ptr[itr])
                rec[fam[0]] = {"pooled_model": r2(pte[ite], pooled.predict(cte[ite], Vte[ite])).tolist(),
                               "within_pattern": r2(pte[ite], within.predict(cte[ite], Vte[ite])).tolist()}
            # exact reference: within the pattern, linear in the raw inputs x_0..x_9
            B = np.linalg.lstsq(xtr[itr], ptr[itr], rcond=None)[0]
            rec["raw x0..x9 linear"] = {"within_pattern": r2(pte[ite], xte[ite] @ B).tolist()}
            out["pattern"][int(g)] = rec
            print("pattern", int(g), rec["share"], {k: np.round(np.median(v["within_pattern"]), 3)
                                                   for k, v in rec.items() if k != "share"}, flush=True)
        # mode basis: final pattern 64975 (dominant), coordinates of h_10 in eigenmodes of D W
        g10tr = (ptr > 0) @ (1 << np.arange(H))
        g10te = (pte > 0) @ (1 << np.arange(H))
        g = 64975
        act = np.array([j for j in range(H) if g >> j & 1])
        lam, R = np.linalg.eig(Wh[np.ix_(act, act)])
        L = np.linalg.inv(R)
        # real coordinates: Re and Im of each complex coordinate (one per conjugate pair)
        keep, labels = [], []
        for k in np.argsort(-np.abs(lam)):
            if lam[k].imag < -1e-9:
                continue
            keep.append(k)
            labels.append(f"|l|={abs(lam[k]):.3f},period={2*np.pi/abs(np.angle(lam[k])):.2f}" if abs(lam[k].imag) > 1e-9
                          else f"|l|={abs(lam[k]):.3f},real")

        def coords(p):
            c = p[:, act] @ L[keep].T
            return np.concatenate([c.real, c.imag[:, [abs(lam[k].imag) > 1e-9 for k in keep]]], 1)
        itr, ite = g10tr == g, g10te == g
        Ytr, Yte = coords(ptr[itr]), coords(pte[ite])
        names = labels + [l + "(Im)" for k, l in zip(keep, labels) if abs(lam[k].imag) > 1e-9]
        rec = {"share": float(itr.mean()), "coords": names}
        for fam in FAMILIES:
            ctr, Vtr = design(fam, xtr[itr], t)
            cte, Vte = design(fam, xte[ite], t)
            rec[fam[0]] = r2(Yte, CellLinear().fit(ctr, Vtr, Ytr).predict(cte, Vte)).tolist()
        Bm = np.linalg.lstsq(xtr[itr], Ytr, rcond=None)[0]
        rec["raw x0..x9 linear"] = r2(Yte, xte[ite] @ Bm).tolist()
        out["modes64975"] = rec
        print("modes", rec["share"], {k: np.round(v, 3) for k, v in rec.items() if k not in ("share", "coords")})
    htr, hte = np.maximum(ptr, 0), np.maximum(pte, 0)
json.dump(out, open("results/phaseB.json", "w"), indent=1)
