"""D2 analysis: where does Z^2 E[F^2] (the iid-variance driver) live? Strata by face sign pattern, config, tau."""
import json
import numpy as np
import core as C
res = {}
for T in (4, 5):
  for s in range(5):
    D = np.load(f"../../../data/minimal_pair_d2_T{T}_s{s}.npz"); _, Z23, Z12 = C.tables(T)
    line = f"T{T} s{s}"
    tot_var = 0; parts = {}
    allF = []
    for kind, Z in (("23", Z23), ("12", Z12)):
        F = D[f"{kind}_F"]; y = D[f"{kind}_y"]; u = D[f"{kind}_u"]; cost = D[f"{kind}_cost"]
        kpos = (y > 0).sum(1)              # positives at the face point (tie counted twice)
        sgn = np.where(u > 0, "u+", "u-")
        # coarse model-free stratum: u sign x (#positive others) capped
        strat = np.array([f"{kind}:{sg}:k{min(k,3)}" for sg, k in zip(sgn, kpos)])
        EF2 = Z ** 2 * (F ** 2).mean(); tot_var += Z ** 2 * F.var()
        for st in np.unique(strat):
            w = strat == st
            parts[st] = dict(w=w.mean(), Zmean=Z * F[w].sum() / len(F), EF2share=Z**2 * (F[w]**2).sum() / len(F),
                             sd=F[w].std(), Z=Z, c=cost[w].mean(), meanF=F[w].mean())
        allF.append((kind, Z, F, cost, strat))
        # top1% share and where top faces sit
        top = F >= np.quantile(F, 0.99)
        line += f" | {kind}: top1% share {np.sort(F**2)[::-1][:len(F)//100].sum()/(F**2).sum():.2f}, top faces u<0 {np.mean(u[top]<0):.2f} k0 {np.mean(kpos[top]==0):.2f}"
    print(line)
    totF2 = sum(p["EF2share"] for p in parts.values())
    for st, p in sorted(parts.items(), key=lambda kv: -kv[1]["EF2share"])[:5]:
        print(f"    {st:12s} w={p['w']:.3f} err-contrib={p['Zmean']:.2e} E[F^2]-share={p['EF2share']/totF2:.2f} meanF={p['meanF']:.2e}")
    # theoretical gains (iid, per unit cost): unstratified vs Neyman over kinds (as sci_tao) vs Neyman over strata
    def neyman(groups):  # groups: list of (prob-weight W_h (incl Z), sd_h, cost_h)  -> variance*cost
        return sum(W * sd * np.sqrt(c) for W, sd, c in groups) ** 2
    g_kind = [(Z, F.std(), c.mean()) for _, Z, F, c, _ in allF]
    g_str = []
    for kind, Z, F, c, strat in allF:
        for st in np.unique(strat):
            w = strat == st; g_str.append((Z * w.mean(), F[w].std(), c[w].mean()))
    vk, vs = neyman(g_kind), neyman(g_str)
    err = sum(Z * F.mean() for _, Z, F, _, _ in allF); p = 1 - err
    res[f"{T}_{s}"] = dict(err=err, gain_kind=p*(1-p)/vk, gain_strat=p*(1-p)/vs)
    print(f"    iid gain (sampling var / est var at equal cost): Neyman-by-kind {p*(1-p)/vk:.1f}x, Neyman-by-sign-strata {p*(1-p)/vs:.1f}x")
json.dump(res, open("d2_analyse.json", "w"), indent=1)
