"""Gap-conditioned error profile (DRI/fable-review request): P(err | g23, g12, blind) and the error share in the
soft tail (non-blind inputs with gap >= 0.05 / 0.1 / 0.2), for every scored model. 2M fresh inputs per model
(seed 269300 + index). 'gap' = min(g12, g23) and, separately, g23 alone."""
import json
import numpy as np
from gfcommon import MODELS, make

E = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]
out = {}
for i, name in enumerate(MODELS):
    err, V = make(MODELS[name], 2_000_000, 269300 + i)
    g12, g23 = V["M1"] - V["M2"], V["M2"] - V["M3"]
    blind = V["M2"] <= 0
    nb = ~blind
    gmin = np.minimum(g12, g23)
    r = {"err": float(err.mean()), "blind_share_of_mass": float(blind.mean()),
         "P_err_blind": float(err[blind].mean()), "blind_share_of_err": float((err & blind).sum() / err.sum())}
    for gname, g in (("min_gap", gmin), ("g23", g23)):
        r[f"tail_{gname}"] = {thr: {"mass": float((nb & (g >= thr)).mean()),
                                    "P_err": float(err[nb & (g >= thr)].mean()),
                                    "share_of_err": float((err & nb & (g >= thr)).sum() / err.sum()),
                                    "err_pp": float(100 * (err & nb & (g >= thr)).mean())}
                              for thr in (0.05, 0.1, 0.2)}
    tab = np.zeros((len(E) - 1, len(E) - 1))
    cnt = np.zeros_like(tab)
    for a in range(len(E) - 1):
        for b in range(len(E) - 1):
            m = nb & (g23 >= E[a]) & (g23 < E[a + 1]) & (g12 >= E[b]) & (g12 < E[b + 1])
            cnt[a, b] = m.mean()
            tab[a, b] = err[m].mean() if m.any() else np.nan
    r["P_err_by_g23_rows_g12_cols"] = {"edges": [str(e) for e in E], "P_err": tab.round(4).tolist(), "mass": cnt.round(5).tolist()}
    out[name] = r
    t = r["tail_min_gap"]
    print(name, "err %.4f blind P %.3f share %.3f | tail(min gap>=.05/.1/.2) share %.3f %.3f %.3f  pp %.3f %.3f %.3f" % (
        r["err"], r["P_err_blind"], r["blind_share_of_err"], t[0.05]["share_of_err"], t[0.1]["share_of_err"],
        t[0.2]["share_of_err"], t[0.05]["err_pp"], t[0.1]["err_pp"], t[0.2]["err_pp"]), flush=True)
json.dump(out, open("results/profile.json", "w"), indent=1)
