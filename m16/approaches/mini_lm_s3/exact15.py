"""Exact L = 15 truths for one dyck2 model: per-sequence max p_illegal (saved), B(tau) on TAU_GRID,
exact r(k) numerators/denominators, exact closer accuracy and illegal mass."""
import json
import sys
import time

import numpy as np

from s3lib import DATA, TAU_GRID, Dyck2, load, p_illegal, stack_info, logits_of
import torch

name = sys.argv[1]
L = 15
model, _ = load(name)
lang = Dyck2()
t0 = time.time()
yall, wall = lang.enumerate_small(L, as_int8=True)
N = len(yall)
mx = np.zeros(N, np.float32)
amx = np.zeros(N, np.int8)
K = L  # distances 1..14
R_TAUS = (0.5, 0.1, 0.01)
rnum = np.zeros((len(R_TAUS), K + 1)); rden = np.zeros(K + 1)  # exact counts of (pos, dist) pairs weighted
acc_num = acc_den = ill = 0.0
for i in range(0, N, 100000):
    y = yall[i:i + 100000].astype(np.int64); w = wall[i:i + 100000]
    lg = torch.from_numpy(logits_of(model, lang, y))
    p = torch.softmax(lg, -1).numpy()
    depth, top, opos = stack_info(y)
    wrong = np.where(top == 0, 3, 2); corr = np.where(top == 0, 2, 3)
    pw = np.take_along_axis(p, wrong[..., None], -1)[..., 0]
    pc = np.take_along_axis(p, corr[..., None], -1)[..., 0]
    pi = np.where(depth > 0, pw, p[..., 2] + p[..., 3])
    mx[i:i + len(y)] = pi.max(1); amx[i:i + len(y)] = pi.argmax(1)
    pos = depth > 0
    acc_num += float((w[:, None] * ((pc > pw) & pos)).sum()); acc_den += float((w[:, None] * pos).sum())
    ill += float((w[:, None] * pi).sum()) / L
    dist = np.where(pos, np.arange(L)[None, :] - opos, 0)
    for k in range(1, K + 1):
        c = (dist >= k)
        rden[k] += float((w[:, None] * c).sum())
        for j, tau in enumerate(R_TAUS):
            rnum[j, k] += float((w[:, None] * (c & (pi > tau))).sum())
np.save(f"{DATA}/{name}_L15_max.npy", mx); np.save(f"{DATA}/{name}_L15_amax.npy", amx)
B = {str(t): float(wall[mx > t].sum()) for t in TAU_GRID}
out = dict(name=name, L=L, n_seq=N, acc=acc_num / acc_den, illegal_mass=ill, B=B,
           r_taus=R_TAUS, r_num=rnum.tolist(), r_den=rden.tolist(), seconds=time.time() - t0)
json.dump(out, open(f"{DATA}/{name}_L15_exact.json", "w"))
print(name, f"{time.time()-t0:.0f}s", {k: f"{v:.2e}" for k, v in B.items()})
