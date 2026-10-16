"""H1 (clock) and H2 (gates at steps a, b) as explicit compact models; held-out R^2 on orig (60k data set).
 - additive clock model: one-hot a, b, p1, (b-a), (a-p1), (b-p1), (9-p1) ~ 70 params (ridge)
 - gate model: lookup(a,b,p1) + per-(a,b) linear in the 48 gate bits at steps a, b and 9 (at the tie point)
 - value model of matched size: lookup(a,b,p1) + per-(a,b) linear in the 10 values (for comparison)
 - delta_lin with the true tie gates: fraction exact; R^2 when gate flips inside the band are excluded."""
import numpy as np
import feats as F
R = F.load("orig"); n = len(R["v"]); cut = 2 * n // 3
G = np.unpackbits(R["G"], 1)[:, :160].reshape(n, 10, 16).astype(float)
r = np.arange(n)
R["gab"] = np.column_stack([G[r, R["a"]], G[r, R["b"]], G[:, 9]])
tr = {k: v[:cut] for k, v in R.items()}; te = {k: v[cut:] for k, v in R.items()}
E = np.eye(19)
def clockX(R):
    a, b, p = R["a"], R["b"], R["p1"]
    return np.column_stack([np.ones(len(a)), E[a], E[b], E[p], E[b - a], E[a - p + 9], E[b - p + 9], E[9 - p]])
def ridge(X, y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
def percell(Xtr, ytr, Xte, ctr, cte, lam=5.0):
    p = np.zeros(len(Xte))
    for c in np.unique(ctr):
        i, j = ctr == c, cte == c
        w = ridge(Xtr[i], ytr[i], lam)
        p[j] = Xte[j] @ w
    return p
for t in ["delta", "absdelta", "mass"]:
    ytr, yte = F.targets(tr)[t], F.targets(te)[t]
    w = ridge(clockX(tr), ytr); pc = clockX(te) @ w
    p0, _ = F.fit_cells(tr, ytr, tr, "abp", "const"); p0t, _ = F.fit_cells(tr, ytr, te, "abp", "const")
    cab_tr, cab_te = F.cell(tr, "ab"), F.cell(te, "ab")
    Xg_tr = np.column_stack([np.ones(cut), tr["gab"]]); Xg_te = np.column_stack([np.ones(n - cut), te["gab"]])
    pg = p0t + percell(Xg_tr, ytr - p0, Xg_te, cab_tr, cab_te)
    Xv_tr = np.column_stack([np.ones(cut), tr["y"]]); Xv_te = np.column_stack([np.ones(n - cut), te["y"]])
    pv = p0t + percell(Xv_tr, ytr - p0, Xv_te, cab_tr, cab_te)
    Xb_tr = np.column_stack([Xg_tr, tr["y"]]); Xb_te = np.column_stack([Xg_te, te["y"]])
    pb = p0t + percell(Xb_tr, ytr - p0, Xb_te, cab_tr, cab_te)
    print(f"{t:9s} clock-additive(~{clockX(tr).shape[1]}p) {F.r2(pc, yte):+.3f} | lookup(abp)+gates@a,b,9 ({360+45*49}p) {F.r2(pg, yte):+.3f}"
          f" | lookup(abp)+values ({360+45*11}p) {F.r2(pv, yte):+.3f} | both ({360+45*59}p) {F.r2(pb, yte):+.3f}")
d, dl = np.clip(te["delta"], -.3, .3), np.clip(te["delta_lin"], -.3, .3)
ex = np.abs(te["delta"] - te["delta_lin"]) < 1e-6
print("delta_lin exact on %.3f of lines; R2 overall %.3f; on exact lines trivially 1; rank corr %.3f" %
      (ex.mean(), F.r2(dl, d), np.corrcoef(np.argsort(np.argsort(dl)), np.argsort(np.argsort(d)))[0, 1]))
