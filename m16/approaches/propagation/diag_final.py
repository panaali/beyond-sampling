# DIAGNOSIS ONLY: feed the last step with Gaussians FITTED from Monte Carlo per key; isolates the
# representation error (Gaussian per key) + final step from the propagation error.
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
import gsf2 as G
x = np.random.default_rng(2).standard_normal((1_000_000, 10)); hs = run(x)
o = np.argsort(-x[:, :9], 1); p1, p2 = o[:, 0], o[:, 1]
z = np.concatenate([hs[:, 9], x[np.arange(len(x)), p1][:, None], x[np.arange(len(x)), p2][:, None]], 1)
C = G.Counter(); M = np.zeros((16, 19)); M[:, :16] = Wh; M[:, 18] = Wi
for keymode in ['p', 'pg']:
    g = np.packbits(hs[:, 9] > 0, axis=1).view(np.uint16)[:, 0].astype(np.int64)
    key = p1 * 10 + p2 + (g * 100 if keymode == 'pg' else 0)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True); inv = inv.ravel()
    mus, Ss, ws, P1, P2 = [], [], [], [], []
    for k in range(len(u)):
        if cnt[k] < 30: continue
        zz = z[inv == k]
        mus.append(zz.mean(0)); Ss.append(np.cov(zz.T) + 1e-9 * np.eye(18)); ws.append(cnt[k] / len(x))
        P1.append(u[k] % 100 // 10); P2.append(u[k] % 10)
    B = len(ws)
    mu = np.concatenate([np.array(mus), np.zeros((B, 1))], 1)
    S = np.zeros((B, 19, 19)); S[:, :18, :18] = np.array(Ss); S[:, 18, 18] = 1
    c = G.Comp(mu=mu, S=S, w=np.array(ws), p1=np.array(P1), p2=np.array(P2), g=np.zeros((B, 16), bool))
    tot, num = 0, 0
    for comp, sel in G.rank_split(c, 9, C):
        gc = G.enumerate_gates(comp, M, 1e-6, C, [0.0])
        zc = G.final_prob(gc, M, Wo, C)
        tot += gc.w.sum(); num += (gc.w * zc).sum()
    print(keymode, 'components', B, 'covered mass', sum(ws), 'acc', num / tot)
