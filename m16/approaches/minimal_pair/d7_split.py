"""D7 (T380 development, DEV models only): split F = F_near (|s| < delta) + F_far. Each part is estimated on its own
node set (independent rules), so var*cost = (sqrt(Vn cn) + sqrt(Vf cf))^2 per stratum vs VF*c (frozen integrand).
Far-part cost = full walk cost - near cost + one first-cell (restart) cost, an upper bound on what an implementation pays."""
import sys
import numpy as np
sys.path.insert(0, "../..")
import leaderboard as LB
import core as C, estimator as E, hom, walk as W
N = 4096
DELTAS = (0.01, 0.03, 0.1, 0.3)
for T in (4, 5):
    ms = LB.models(16, T, dev=True)
    for name in list(ms)[::4]:
        m = ms[name]; base = 0; split = {dl: 0 for dl in DELTAS}; share = {dl: [] for dl in DELTAS}
        for j, (kind, h) in enumerate(E.strata(T)):
            U = C.sobol(T, N, 999 + j)
            y, a, b, tau, wt = E.face_points_strat(T, kind, h, U)
            r = np.linalg.norm(y, axis=1); w = y / r[:, None]; tw = np.minimum(tau / r, 1e8)
            F, c, _ = W.walk(m, w, a, b, tw, kind, cdf=hom.tcdf(T)); F = wt * F
            Mh = E.stratum_mass(T, kind, h); base += Mh * np.sqrt(F.var() * c.mean())
            for dl in DELTAS:
                # near part: walk on [-min(tau,dl), min(tau,dl)] (walk.walk integrates |s| < tau)
                Fn, cn, _ = W.walk(m, w, a, b, np.minimum(tw, dl), kind, cdf=hom.tcdf(T)); Fn = wt * Fn
                Ff = F - Fn; cf = c - cn + 2.0
                split[dl] += Mh * (np.sqrt(Fn.var() * cn.mean()) + np.sqrt(Ff.var() * cf.mean()))
                share[dl].append((Mh * Fn.mean(), Mh * F.mean(), cn.mean() / c.mean()))
        out = []
        for dl in DELTAS:
            sn = sum(x[0] for x in share[dl]) / sum(x[1] for x in share[dl])
            cr = np.mean([x[2] for x in share[dl]])
            out.append(f"d={dl}: gain {(base/split[dl])**2:.2f}x (near mass {sn:.2f}, near cost {cr:.2f})")
        print(f"T{T} {name}: " + " | ".join(out), flush=True)
