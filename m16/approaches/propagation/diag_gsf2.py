# DIAGNOSIS ONLY: per-step error of the propagated mixture mean of h (per (p1,p2)) vs Monte Carlo
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
import gsf2 as G
K = int(sys.argv[1]); eps = float(sys.argv[2])
x = np.random.default_rng(1).standard_normal((400_000, 10)); hs = run(x)
C = G.Counter(); M = np.zeros((16, 19)); M[:, :16] = Wh; M[:, 18] = Wi
c = G.Comp(mu=np.zeros((1, 18)), S=np.zeros((1, 18, 18)), w=np.ones(1), p1=np.array([-1]), p2=np.array([-1]), g=np.zeros((1, 16), bool))
for t in range(9):
    B = len(c)
    mu = np.concatenate([c.mu, np.zeros((B, 1))], 1)
    S = np.zeros((B, 19, 19)); S[:, :18, :18] = c.S; S[:, 18, 18] = 1
    aug = G.Comp(mu=mu, S=S, w=c.w, p1=c.p1, p2=c.p2, g=c.g)
    new = []
    for comp, sel in G.rank_split(aug, t, C):
        gc = G.enumerate_gates(comp, M, eps, C, [0.0])
        new.append(G.linear_step(gc, M, sel, C))
    c = G.merge(G.Comp.cat(new), K, C)
    o = np.argsort(-x[:, :t + 1], 1)
    errs, sds = [], []
    for p1 in range(t + 1):
        for p2 in range(t + 1):
            if t > 0 and p1 == p2: continue
            k = (c.p1 == p1) & ((c.p2 == p2) if t > 0 else True)
            w = c.w[k] / c.w[k].sum()
            m = (w[:, None] * c.mu[k, :16]).sum(0)
            sel = (o[:, 0] == p1) & ((o[:, 1] == p2) if t > 0 else True)
            errs.append(np.abs(m - hs[sel, t + 1].mean(0)).mean()); sds.append(hs[sel, t + 1].std(0).mean())
    print(t, 'ncomp', len(c), 'mean|err| of E[h|p1,p2]', round(np.mean(errs), 3), 'typical sd', round(np.mean(sds), 3), flush=True)
