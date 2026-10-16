"""Per-chamber sampled probabilities of each predicted rank (diagnosis only; original model)."""
import sys
import numpy as np, torch
torch.set_num_threads(1)
from form import *
from common import Wi, Wh, Wo
tm = tmodel((Wi, Wh, Wo))
n_arr = int(sys.argv[1]) if len(sys.argv) > 1 else 1
positive = (sys.argv[2] != 'neg') if len(sys.argv) > 2 else True
perms = torch.as_tensor(chambers(n_arr, seed=0)); Cn = len(perms)
M = 2000; rng = torch.Generator().manual_seed(1)
R = np.zeros((Cn, 10))
with torch.no_grad():
    for c0 in range(0, Cn, 30):
        P = perms[c0:c0+30]
        Pr = P.repeat_interleave(M, 0)
        u = torch.randn(len(Pr), 10, generator=rng)
        pred = logits(tm, place(sorted_from_u(u, positive), Pr)).argmax(1)
        rank = (Pr == pred[:, None]).long().argmax(1).view(-1, M)
        for k in range(10): R[c0:c0+len(P), k] = (rank == k).double().mean(1).numpy()
np.save(f'diag1_mc_{n_arr}_{int(positive)}.npy', R); print('done', R.mean(0).round(5))
