"""Transfer of compact descriptions across models (same surface points, seed 2026, 30k lines each).
(i) refit each family on the model itself (2/3 train, 1/3 test): does the same FORM work?
(ii) apply the orig-fitted model unchanged: does the FIT transfer (noisy copies are 0.25-0.5% weight perturbations)?
(iii) zero-parameter structure-derived predictor: gates frozen at the canonical (expected-order-statistic) point of
     the full ordering, applied to the actual input. Compare E|delta| predicted vs true per model.
Prints held-out R^2 for clipped delta and mass, and E[mass] bias of the orig fit."""
import numpy as np, json
import feats as F, wtool as W
names = ["orig30k"] + [f"seed{s}" for s in range(5)] + [f"noisy0.0025_{k}" for k in range(3)] + [f"noisy0.005_{k}" for k in range(3)]
REF = np.sort(np.random.default_rng(0).standard_normal((400000, 10)), 1)[:, ::-1].mean(0)


def canon_delta(M, R, N=6000):
    y, a, b = R["y"][:N], R["a"][:N], R["b"][:N]; n = len(y); r = np.arange(n)
    d = np.zeros_like(y); d[r, a] = 1 / W.SQ2; d[r, b] = -1 / W.SQ2
    rk = np.argsort(np.argsort(-y + 1e-9 * (np.arange(10) == b[:, None]), 1), 1)
    yc = REF[rk]; mid = (yc[r, a] + yc[r, b]) / 2; yc[r, a] = mid; yc[r, b] = mid
    G = W.jac_at_tie(M, yc, a, b, +1)["G"]; Wi, Wh, Wo = M
    h = np.zeros((n, 16)); dh = np.zeros((n, 16))
    for t in range(10):
        h = np.where(G[:, t], h @ Wh.T + y[:, t:t + 1] * Wi[None], 0)
        dh = np.where(G[:, t], dh @ Wh.T + d[:, t:t + 1] * Wi[None], 0)
    L, dL = h @ Wo.T, dh @ Wo.T
    return -(L[r, a] - L[r, b]) / (dL[r, a] - dL[r, b])


ref = F.load("orig30k"); n = len(ref["v"]); cut = 2 * n // 3
split = lambda R: ({k: v[:cut] for k, v in R.items()}, {k: v[cut:] for k, v in R.items()})
rtr, _ = split(ref)
out = {}
print("model          | refit R2 delta: lookup(abp)  lin(abp) | refit R2 mass: lookup lin | orig-fit R2 delta, mass | E[mass] true, orig-fit pred | E|d| true, canon-gate pred, R2 canon")
for nm in names:
    R = F.load(nm); assert len(R["v"]) == n
    tr, te = split(R); Tt, Te = F.targets(tr), F.targets(te); To = F.targets(rtr)
    row = {}
    for t in ["delta", "mass"]:
        p0, _ = F.fit_cells(tr, Tt[t], te, "abp", "const"); p1, _ = F.fit_cells(tr, Tt[t], te, "abp", "raw")
        q1, _ = F.fit_cells(rtr, To[t], te, "abp", "raw")
        row[t] = (F.r2(p0, Te[t]), F.r2(p1, Te[t]), F.r2(q1, Te[t]), Te[t].mean(), q1.mean())
    M = W.get_model("orig" if nm == "orig30k" else nm)
    dc = np.clip(canon_delta(M, R), -.3, .3); dt = np.clip(R["delta"][:6000], -.3, .3)
    out[nm] = dict(row=row, canon=(np.abs(dt).mean(), np.abs(dc).mean(), F.r2(dc, dt)))
    print(f"{nm:14s} | {row['delta'][0]:+.3f} {row['delta'][1]:+.3f} | {row['mass'][0]:+.3f} {row['mass'][1]:+.3f} | "
          f"{row['delta'][2]:+.3f} {row['mass'][2]:+.3f} | {row['mass'][3]:.4f} {row['mass'][4]:.4f} | "
          f"{np.abs(dt).mean():.4f} {np.abs(dc).mean():.4f} {F.r2(dc, dt):+.2f}", flush=True)
json.dump(out, open("transfer.json", "w"), indent=1, default=float)
