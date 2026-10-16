# DIAGNOSIS ONLY: per (p1,p2) at step 9, compare predicted final-step accuracy (from MC-fitted Gaussian) with MC
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
import gsf2 as G
x = np.random.default_rng(2).standard_normal((500_000, 10)); hs = run(x)
ok = (hs[:, -1] @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]
o = np.argsort(-x[:, :9], 1); p1, p2 = o[:, 0], o[:, 1]
z = np.concatenate([hs[:, 9], x[np.arange(len(x)), p1][:, None], x[np.arange(len(x)), p2][:, None]], 1)
C = G.Counter(); M = np.zeros((16, 19)); M[:, :16] = Wh; M[:, 18] = Wi
for a, b in [(0, 1), (8, 7), (3, 5)]:
    sel = (p1 == a) & (p2 == b); zz = z[sel]
    mu = np.r_[zz.mean(0), 0][None]; S = np.zeros((1, 19, 19)); S[0, :18, :18] = np.cov(zz.T); S[0, 18, 18] = 1
    c = G.Comp(mu=mu, S=S, w=np.ones(1), p1=np.array([a]), p2=np.array([b]), g=np.zeros((1, 16), bool))
    names = ['rec', 'mid', 'low']
    for (comp, s_), nm in zip(G.rank_split(c, 9, C), names):
        gc = G.enumerate_gates(comp, M, 1e-6, C, [0.0])
        zc = G.final_prob(gc, M, Wo, C)
        x9 = x[sel, 9]; m1 = zz[:, 16]; m2 = zz[:, 17]
        ev = {'rec': x9 > m1, 'mid': (x9 < m1) & (x9 > m2), 'low': x9 < m2}[nm]
        print((a, b), nm, 'P pred', round(comp.w[0], 4), 'MC', round(ev.mean(), 4), '| acc pred', round((gc.w * zc).sum() / gc.w.sum(), 4), 'MC', round(ok[sel][ev].mean(), 4), 'npat', len(gc))
