# DIAGNOSIS ONLY: representation ceiling of 'Gaussian over (h_9, x_0..x_9) per key' for the final step.
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
import gsf2 as G
x = np.random.default_rng(2).standard_normal((1_000_000, 10)); hs = run(x)
ok = (hs[:, -1] @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]
o = np.argsort(-x[:, :9], 1); p1, p2 = o[:, 0], o[:, 1]
z = np.concatenate([hs[:, 9], x], 1)            # 26 dims; x9 is independent of the rest
D = 26; IX = 25
M = np.zeros((16, D)); M[:, :16] = Wh; M[:, IX] = Wi
C = G.Counter()
def final(mu, S, w, a, b):
    """rank split on x9 then gate enumeration then orthant, all in 26-dim."""
    tot = num = 0
    B = len(w)
    for out in ['rec', 'mid', 'low']:
        m_, S_ = mu.copy(), S.copy(); ww = w.copy()
        def con(v):
            nonlocal ww
            ww = ww * G.constrain(m_, S_, np.tile(v, (B, 1)), C)
        e1 = np.zeros(D); e1[IX] = 1; e1[16 + a] -= 1
        e2 = np.zeros(D); e2[IX] = 1; e2[16 + b] -= 1
        if out == 'rec': con(e1); tgt = a
        elif out == 'mid': con(-e1); con(e2); tgt = 9
        else: con(-e1); con(-e2); tgt = b
        c = G.Comp(mu=m_, S=S_, w=ww, p1=np.full(B, a), p2=np.full(B, tgt), g=np.zeros((B, 16), bool))
        gc = enum(c)
        L = np.einsum('jh,bh,hk->bjk', Wo, gc.g.astype(float), M)
        A = L[:, tgt][:, None] - L
        mu2, S2 = gc.mu.copy(), gc.S.copy(); zc = np.ones(len(gc))
        mm = np.einsum('bjk,bk->bj', A, mu2); vv = np.einsum('bjk,bkl,bjl->bj', A, S2, A)
        pr = mm / np.sqrt(np.maximum(vv, 1e-300)); pr[:, tgt] = np.inf
        ordr = np.argsort(pr, 1)
        for k in range(9):
            zc *= G.constrain(mu2, S2, A[np.arange(len(gc)), ordr[:, k]], C)
        tot += gc.w.sum(); num += (gc.w * zc).sum()
    return num, tot
# a generic-dim version of enumerate_gates: monkeypatch dims
import types
def enum(c):
    return G.enumerate_gates(c, M, 1e-6, C, [0.0])
for keymode in ['p', 'pg', 'pg8']:
    g9 = np.packbits(hs[:, 9] > 0, axis=1).view(np.uint16)[:, 0].astype(np.int64)
    g8 = np.packbits(hs[:, 8] > 0, axis=1).view(np.uint16)[:, 0].astype(np.int64)
    key = p1 * 10 + p2 + {'p': 0, 'pg': g9 * 100, 'pg8': g9 * 100 + g8 * 100 * 65536}[keymode]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True); inv = inv.ravel()
    order = np.argsort(inv); bounds = np.r_[0, np.cumsum(cnt)]
    NUM = TOT = 0; MCacc = 0; cov = 0
    for k in range(len(u)):
        if cnt[k] < 40: continue
        idx = order[bounds[k]:bounds[k + 1]]
        zz = z[idx]
        mu = zz.mean(0); S = np.cov(zz.T) + 1e-9 * np.eye(D)
        mu[IX] = 0; S[IX, :] = 0; S[:, IX] = 0; S[IX, IX] = 1
        w = cnt[k] / len(x)
        n_, t_ = final(mu[None], S[None], np.array([w]), int(u[k] % 100 // 10), int(u[k] % 10))
        NUM += n_; TOT += t_; MCacc += ok[idx].sum() / len(x); cov += w
    print(keymode, 'covered', round(cov, 4), 'pred acc', round(NUM / TOT, 4), 'MC acc on covered', round(MCacc / cov, 4), flush=True)
