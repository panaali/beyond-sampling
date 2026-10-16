"""Is the T=4->5 collapse structural or a quadrature-dimension effect?
For each (H,T) (best-of-5 zoo seed): MSE gain over sampling at equal cost of (a) the deterministic Sobol face
rule and (b) the same faces drawn i.i.d. (face-MC; no low-discrepancy benefit), each from R rule seeds, with a
chi-square 95% CI on the variance ratio. Also: tail share (top 1% of faces' share of E[F^2]) and the median
cancellation factor of the R23 tie margin. usage: python scan_HT.py n R H:T ..."""
import sys, json
import numpy as np
from scipy.stats import chi2
import core as C
from est import est_faces
from cancel_T import jac

n = int(sys.argv[1]); R = int(sys.argv[2])
try:
    out = json.load(open("scan_HT.json"))
except FileNotFoundError:
    out = {}


def faces_iid(M, n, seed):
    T = M[2].shape[0]; _, Z23, Z12 = C.tables(T); rng = np.random.default_rng(seed)
    err = 0; cost = 0; Fs = []
    for kind, Z in (("23", Z23), ("12", Z12)):
        if Z == 0: continue
        y, a, b, tau = C.face_points(T, kind, rng.random((n, T)))
        F, c, _ = C.walk_mass(M, y, a, b, tau, kind); err += Z * F.mean(); cost += c.sum(); Fs.append(F)
    return 1 - err, cost, Fs


for spec in sys.argv[3:]:
    if spec in out: continue
    H, T = map(int, spec.split(":")); M = C.zoo(H, T)
    row = {}
    for lab in ("sobol", "iid"):
        e, cs, tails = [], [], []
        for r in range(R):
            if lab == "sobol":
                a, c, o = est_faces(M, n, n, rule_seed=700 + r); Fs = list(o.values())
            else:
                a, c, Fs = faces_iid(M, n, 800 + r)
            e.append(a); cs.append(c)
            F = np.concatenate(Fs); s = np.sort(F)[::-1]
            tails.append((s[:max(1, len(s) // 100)] ** 2).sum() / max((s ** 2).sum(), 1e-300))
        e = np.array(e); p = e.mean(); v = e.var(ddof=1); c = np.mean(cs)
        g = p * (1 - p) / c / v
        lo, hi = g * chi2.ppf(0.025, R - 1) / (R - 1), g * chi2.ppf(0.975, R - 1) / (R - 1)
        row[lab] = dict(acc=p, sd=float(np.sqrt(v)), cost=c, gain=g, ci=[lo, hi], tail=float(np.mean(tails)))
    U = C.sobol(T, 2048, 9); y, a, b, _ = C.face_points(T, "23", U)
    k = y[np.arange(len(y)), a] > 0; y, a, b = y[k], a[k], b[k]; r_ = np.arange(len(y))
    J = jac(M, y + 1e-9 * (np.arange(T)[None] == a[:, None]))
    terms = (J[r_, a] - J[r_, b]) * y
    row["cancel"] = float(np.median(np.abs(terms).sum(1) / np.abs(terms.sum(1))))
    out[spec] = row; json.dump(out, open("scan_HT.json", "w"), indent=1)
    s, i = row["sobol"], row["iid"]
    print(f"H={H:2d} T={T:2d} acc={s['acc']:.4f} Sobol gain={s['gain']:.1f} [{s['ci'][0]:.1f},{s['ci'][1]:.1f}] "
          f"iid gain={i['gain']:.1f} [{i['ci'][0]:.1f},{i['ci'][1]:.1f}] tail={s['tail']:.2f} cancel={row['cancel']:.0f}", flush=True)
