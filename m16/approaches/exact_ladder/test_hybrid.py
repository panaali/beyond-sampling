"""Checks of the hybrid's pieces (not scored): (1) mwalk with excl empty == frozen walk; (2) exact_part at tau=0 equals
ladder.exact_acc (T=3); (3) identity at T=3,4: exact wrong in E + walk(rest) ~ exact total, the walk restricted to E
~ exact wrong in E."""
import sys

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import hybrid as Hy  # noqa: E402
import mwalk  # noqa: E402
from ladder import exact_acc  # noqa: E402

FE = Hy.FE
for T in (3, 4, 5):
    m = L.models(16, T)["seed0"]
    U = FE.C.sobol(T, 512, 7)
    for kind, h in FE.strata(T):
        y, a, b, tau, wt = FE.face_points_strat(T, kind, h, U)
        r = np.linalg.norm(y, axis=1)
        args = (m, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind)
        f0 = Hy.FROZEN_WALK(*args, cdf=FE.hom.tcdf(T)); f1 = mwalk.walk(*args, cdf=FE.hom.tcdf(T))
        assert all(np.array_equal(u, v) for u, v in zip(f0, f1)), (T, kind, h)
print("1 ok: mwalk(excl=empty) == frozen walk, bitwise (T=3,4,5, all strata)")
m = L.models(16, 3)["seed0"]
a0, info = exact_acc(m)
ex = Hy.exact_part(m, 0.0)
print("2 T=3 tau=0: exact_acc", a0, "exact_part", 1 - ex["wrong"], "cells", info["cells"], len(ex["keys"]), "mass", ex["mass"])
for T, tau in ((3, 1e-3), (4, 1e-3)):
    m = L.models(16, T)["seed0"]
    a0, _ = exact_acc(m)
    ex = Hy.exact_part(m, tau)
    rest, used = Hy.walk_rest(m, 256000, ex["keys"])
    FE.W.walk = lambda *a, **k: mwalk.walk(*a, excl=frozenset(), **k)
    full = 1 - FE.est(m, 256000, None)[0]
    FE.W.walk = Hy.FROZEN_WALK
    print(f"3 T={T} tau={tau}: cells in E {len(ex['keys'])} mass {ex['mass']:.4f} wrong_E {ex['wrong']:.3e} "
          f"exact cost {ex['fpe']:.0f} FPE | exact total wrong {1 - a0:.6e}  hybrid {ex['wrong'] + rest:.6e}  "
          f"walk alone {full:.6e}  walk-rest {rest:.6e} vs exact rest {1 - a0 - ex['wrong']:.6e}")
