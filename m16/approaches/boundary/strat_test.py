import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import per_sample
M = (Wi, Wh, Wo)
x = np.random.default_rng(5).standard_normal((80000, 10))
r = per_sample(M, x); e = r["fA"] + r["fB"] + r["R"]; c = r["passes"]
np.savez("/tmp/claude-0/-home-user-clean-neurons/d6302e99-a019-5be1-9584-7e4c27ab3ffa/scratchpad/pilot.npz", x=x, e=e, c=c, fA=r["fA"], fB=r["fB"], R=r["R"])
s = -np.sort(-x, 1)
print("unstratified var*cost", e.var() * c.mean())
def neyman(lab):
    tot = 0; tc = 0; out = 0
    P = []; S = []; C = []
    for h in np.unique(lab):
        m = lab == h
        P.append(m.mean()); S.append(e[m].std()); C.append(c[m].mean())
    P, S, C = map(np.array, (P, S, C))
    # optimal: n_h ∝ P S / sqrt(C); var*budget = (sum P S sqrt C)^2
    return (P * S * np.sqrt(C)).sum() ** 2, (P * S ** 2).sum() * (P * C).sum()
for name, f in [("u=top3sum", s[:, :3].sum(1)), ("x4", s[:, 3])]:
    for nb in [4, 8, 16]:
        q = np.quantile(f, np.linspace(0, 1, nb + 1)[1:-1]); lab = np.searchsorted(q, f)
        print(name, nb, "neyman var*cost %.4f  proportional %.4f" % neyman(lab))
f1, f2 = s[:, :3].sum(1), s[:, 3]
for nb in [4, 8]:
    q1 = np.quantile(f1, np.linspace(0, 1, nb + 1)[1:-1]); q2 = np.quantile(f2, np.linspace(0, 1, nb + 1)[1:-1])
    lab = np.searchsorted(q1, f1) * 100 + np.searchsorted(q2, f2)
    print("joint", nb, "neyman %.4f prop %.4f" % neyman(lab))
