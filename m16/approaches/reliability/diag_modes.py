"""Per-mode marginal P(g_k<0 | chamber): MC (diagnosis) vs FORM vs SORM, positive region."""
import sys, json, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/home/user/clean-neurons/m16')
from scipy.stats import norm
from arc_eval import noisy_models, Wi0, Wh0, Wo0
from form import *
from rosen import sorted_from_u
from sorm import x_grad, u_hessian, curvatures, sorm_pf
models = {'orig': (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s, seed=0)):
        models[f's{s}_{k}'] = m
name = sys.argv[1]; tm = tmodel(models[name]); pos = True
perms = torch.as_tensor(chambers(1, seed=0)); M = 1000; modes = (0, 2, 3, 4, 5, 6, 7, 8, 9)
G = np.zeros((720, 10))
with torch.no_grad():
    rng = torch.Generator().manual_seed(7)
    for c0 in range(0, 720, 40):
        P = perms[c0:c0+40].repeat_interleave(M, 0)
        u = torch.randn(len(P), 10, generator=rng)
        L = logits(tm, place(sorted_from_u(u, pos), P))
        Lr = L.gather(1, P)                     # logits by rank
        G[c0:c0+40] = (Lr < Lr[:, 1:2]).logical_not().double().view(-1, M, 10).mean(1).numpy()  # P(logit_k >= logit_p2)
np.save(f'mcg_{name}.npy', G)
Cn = 720; Pm = perms.repeat(len(modes), 1); ks = torch.as_tensor(np.repeat(modes, Cn))
b, a, u = design_points(tm, Pm, ks, pos, iters=6)
U = torch.as_tensor(u); w, _ = x_grad(tm, U, Pm, ks, pos); H, gu = u_hessian(U, w, pos); kap = curvatures(H, gu)
ps = sorm_pf(b, kap, 'hr').reshape(len(modes), Cn); pf = norm.cdf(-b).reshape(len(modes), Cn); bb = b.reshape(len(modes), Cn)
np.savez(f'dp_{name}.npz', b=b, a=a, u=u, kap=kap)
for i, k in enumerate(modes):
    mc = G[:, k]
    print(f'rank{k+1}: MC {mc.mean():.5f} FORM {pf[i].mean():.5f} SORM {ps[i].mean():.5f} notfound {np.mean(bb[i]>=7.99):.2f}'
          f' | MC mass in notfound chambers {mc[bb[i]>=7.99].sum()/720:.5f}  | negative-beta chambers {np.mean(bb[i]<0):.3f}')
