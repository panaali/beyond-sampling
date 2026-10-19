"""D6 (T380 development, DEV models only): correlation of the linear proxy P with the exact radial integrand F per
sign stratum, and the two-phase (difference estimator, nested points) predicted gain over the frozen integrand:
    var*cost(two-phase) = (sqrt((VF - VD) c1) + sqrt(VD c2))^2,  D = F - P,  vs VF * c2 (frozen)."""
import sys
import numpy as np
sys.path.insert(0, "../..")
import leaderboard as LB
import core as C, estimator as E, hom, walk as W, proxy as P
N = 4096
for T in (4, 5):
    ms = LB.models(16, T, dev=True)
    for name in list(ms)[::4]:                     # 6 of the 24 DEV models per T
        m = ms[name]; tot_f = 0; tot_cv = 0; line = []
        for j, (kind, h) in enumerate(E.strata(T)):
            U = C.sobol(T, N, 999 + j)
            y, a, b, tau, wt = E.face_points_strat(T, kind, h, U)
            r = np.linalg.norm(y, axis=1); w = y / r[:, None]; tw = np.minimum(tau / r, 1e8)
            F, c2, _ = W.walk(m, w, a, b, tw, kind, cdf=hom.tcdf(T)); F = wt * F
            Pm, c1 = P.proxy_lin(m, w, a, b, tw, kind, cdf=hom.tcdf(T)); Pm = wt * Pm
            Mh = E.stratum_mass(T, kind, h)
            VF, VD = F.var(), (F - Pm).var(); c1m, c2m = c1.mean(), c2.mean() + c1.mean()
            rho = np.corrcoef(F, Pm)[0, 1] if F.std() > 0 and Pm.std() > 0 else np.nan
            tot_f += Mh * np.sqrt(VF * c2.mean())
            tot_cv += Mh * (np.sqrt(max(VF - VD, 0) * c1m) + np.sqrt(VD * c2m)) if VD < VF else Mh * np.sqrt(VF * c2.mean())
            line.append(f"{kind}{h}: rho {rho:.2f} meanP/meanF {Pm.mean()/max(F.mean(),1e-30):.2f}")
        print(f"T{T} {name}: predicted two-phase gain over frozen integrand {(tot_f/tot_cv)**2:.2f}x | " + "; ".join(line), flush=True)
