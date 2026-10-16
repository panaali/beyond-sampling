"""Multi-design-point variant: several starts per (chamber, mode), keep distinct design points (alpha dot < 0.9),
SORM each, union of all (mode, design point) half-spaces per chamber by Ditlevsen bounds (midpoint).
Shows how far 'mode explosion' must go to remove the bias, and what it costs."""
import sys, json, time, numpy as np, torch
from scipy.stats import norm
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import noisy_models, Wi0, Wh0, Wo0
from form import design_points, chambers, tmodel, C
from rosen import G0
from sorm import x_grad, u_hessian, curvatures, sorm_pf
from system import ditlevsen
from estimator import ROSEN_OVERHEAD, SORM_COST

torch.set_num_threads(1)
MODES = (0, 2, 3, 4, 5, 6, 7, 8, 9)


def starts(n):
    s = [None]
    for j, sg in ((1, -1), (2, 1), (0, -1), (1, 1), (0, 1), (2, -1), (3, 1)):
        e = torch.zeros(n, 10); e[:, j] = sg; s.append(e)
    return s


def region(model, positive, n_starts, thr=0.9):
    tm = tmodel(model); perms = torch.as_tensor(chambers(1, seed=0)); Cn = len(perms); m = len(MODES)
    P = perms.repeat(m, 1); ks = torch.as_tensor(np.repeat(MODES, Cn))
    Bs, As = [], []
    for d0 in starts(len(P))[:n_starts]:
        b, a, u = design_points(tm, P, ks, positive, iters=6, d0=d0)
        U = torch.as_tensor(u); w, _ = x_grad(tm, U, P, ks, positive); H, gu = u_hessian(U, w, positive)
        pf = sorm_pf(b, curvatures(H, gu), 'hr')
        bad = ~np.isfinite(pf) | np.isnan(a).any(1)
        pf[bad] = 0; a[bad] = 0
        Bs.append((-norm.ppf(np.clip(pf, 1e-15, 1 - 1e-15))).reshape(m, Cn)); As.append(a.reshape(m, Cn, 10))
    B = np.stack(Bs, 1).reshape(m * len(Bs), Cn); A = np.stack(As, 1).reshape(m * len(Bs), Cn, 10)
    B2 = np.full_like(B, 8.0); A2 = np.zeros_like(A); nd = []
    for c in range(Cn):
        kept = []
        for i in np.argsort(B[:, c]):
            if B[i, c] >= 7.99:
                continue
            if all(A[i, c] @ A[j, c] < thr for j in kept):
                kept.append(i)
        nd.append(len(kept))
        for r, i in enumerate(kept):
            B2[r, c] = B[i, c]; A2[r, c] = A[i, c]
    L = max(nd)
    lo, up = ditlevsen(B2[:L], A2[:L])
    extra = SORM_COST * len(P) * len(Bs)
    return float(np.mean(np.clip(0.5 * (lo + up), 0, 1))), extra, float(np.mean(nd))


if __name__ == '__main__':
    truth = json.load(open('/home/user/clean-neurons/results/leaderboard_truths.json'))
    models = {'orig': (Wi0, Wh0, Wo0)}
    for s in (0.0025, 0.005):
        for k, mm in enumerate(noisy_models(6, s, seed=0)):
            models[f's{s}_{k}'] = mm
    n_starts = int(sys.argv[1]); out = {}
    for name in sys.argv[2:]:
        t = time.time(); C.fwd = 0
        fp, e1, nd1 = region(models[name], True, n_starts)
        fn, e2, nd2 = region(models[name], False, n_starts)
        acc = 1 - ((1 - G0) * fp + G0 * fn); cost = C.fwd * ROSEN_OVERHEAD + e1 + e2
        out[name] = dict(est=acc, truth=truth[name]['acc'], cost=cost)
        print(f'starts {n_starts} {name} est {acc:.5f} truth {truth[name]["acc"]:.5f} err {100*(acc-truth[name]["acc"]):+.3f}pp '
              f'cost {cost:.0f} DPs/chamber {nd1:.1f}/{nd2:.1f} {time.time()-t:.0f}s', flush=True)
    json.dump(out, open(f'multistart_{n_starts}.json', 'w'), indent=1)
