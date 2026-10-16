"""Diagnosis (uses random inputs): split the estimator error into chamber-design error and FORM/SORM error.
Chamber MC: 1000 Rosenblatt samples per chamber in each region -> 'what the estimator would give with exact
per-chamber probabilities'. Also per-mode MC vs SORM marginals."""
import sys, json, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import noisy_models, Wi0, Wh0, Wo0
from form import *
from rosen import sorted_from_u, G0
truth = json.load(open('/home/user/clean-neurons/results/leaderboard_truths.json'))
models = {'orig': (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s, seed=0)):
        models[f's{s}_{k}'] = m
perms = torch.as_tensor(chambers(1, seed=0)); M = 1000
for name in sys.argv[1:]:
    tm = tmodel(models[name]); rng = torch.Generator().manual_seed(7); fr = {}
    for pos in (True, False):
        R = np.zeros((720, 10))
        with torch.no_grad():
            for c0 in range(0, 720, 40):
                P = perms[c0:c0+40].repeat_interleave(M, 0)
                u = torch.randn(len(P), 10, generator=rng)
                pred = logits(tm, place(sorted_from_u(u, pos), P)).argmax(1)
                rank = (P == pred[:, None]).long().argmax(1).view(-1, M)
                for k in range(10): R[c0:c0+40, k] = (rank == k).double().mean(1).numpy()
        fr[pos] = R
        np.save(f'mc_{name}_{int(pos)}.npy', R)
    acc = (1 - G0) * fr[True][:, 1].mean() + G0 * fr[False][:, 1].mean()
    print(name, 'chamber-design accuracy (MC within chambers)', round(acc, 5), 'truth', truth[name]['acc'],
          'pos-fail by rank', fr[True].mean(0).round(4), 'blind acc', fr[False][:, 1].mean().round(3), flush=True)
