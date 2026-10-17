"""Exact affine-envelope error attribution by predicted input rank on DEV ties."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr, ndtri
from probe import lp, lb, full_mass


def winning_mass(logits, slopes, origin, lo, hi):
    """Gaussian line mass where each affine logit is the first argmax."""
    c = logits[:, :, None]-logits[:, None, :]
    e = slopes[:, :, None]-slopes[:, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        root = origin[:, None, None]-c/e
    lower = np.maximum(lo[:, None], np.where(e > 0, root, -np.inf).max(2))
    upper = np.minimum(hi[:, None], np.where(e < 0, root, np.inf).min(2))
    k, j = np.arange(10)[:, None], np.arange(10)[None, :]
    impossible = ((e == 0) & ((c < 0) | ((c == 0) & (j < k)))).any(2)
    return np.where((upper > lower) & ~impossible, lp.W(lower, upper), 0)


def decompose(model, f, linear=False):
    n = len(f['hi'])
    rank_mass = np.zeros((n, 10))
    t = -f['hi'].copy()
    active = np.arange(n)
    # Order is fixed on each side; midpoint avoids ties and rank instabilities.
    ranks = []
    for sign in (-1, 1):
        order = np.argsort(-(f['x0']+sign*f['hi'][:, None]*f['d']/2), axis=1)
        ranks.append(np.argsort(order, axis=1))
    cells = 0
    for iteration in range(2000):
        if not len(active):
            break
        a = active
        origin = np.zeros(len(a)) if linear else t[a]+1e-10*(1+np.abs(t[a]))
        l, dl, p, dp = lp.fwd_jvp(model, f['x0'][a]+origin[:, None]*f['d'][a], f['d'][a])
        cells += len(a)
        if linear:
            end = f['hi'][a]
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                root = origin[:, None, None]-p/dp
            root = np.where(root > origin[:, None, None], root, np.inf)
            end = np.minimum(f['hi'][a], root.min((1, 2)))
        for side in (0, 1):
            lo = t[a] if side == 0 else np.maximum(t[a], 0)
            hi = np.minimum(end, 0) if side == 0 else end
            mass = winning_mass(l, dl, origin, lo, hi)
            mass[hi <= lo] = 0
            row = np.arange(len(a))[:, None]
            # All ranks are distinct, so this indexed addition has no repeated pairs.
            rank_mass[a[:, None], ranks[side][a]] += mass
        t[a] = end
        active = a[end < f['hi'][a]-1e-12]
    assert not len(active), 'unwalked line mass'
    return rank_mass, cells


def main():
    start = time.process_time()
    path = Path(__file__).parent
    old = json.loads((path/'results.json').read_text())['rows']
    state = torch.load('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth',
                       weights_only=True, map_location='cpu')
    base = (state['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
            state['rnn.weight_hh_l0'].numpy().astype(float), state['linear.weight'].numpy().astype(float))
    models = {f'dev_s{s}_0': lb._noisy(base, 6, s, seed=1)[0] for s in (.0025, .02)}
    rng = np.random.default_rng(138511)
    rows = []
    total_cells = 0
    for a in old:
        model = models[a['model']]
        cfg = np.array(a['config'])
        n = 2048
        u, mx = a['tie'], a['top']
        low = ndtri(rng.random((n, 7))*ndtr(u))
        f = lp.build23(np.repeat(cfg[None], n, 0), np.full(n, u), np.full(n, mx), low)
        exact, cells = decompose(model, f)
        linear, lc = decompose(model, f, linear=True)
        total_cells += cells+lc
        # Verify independent exact walker agrees on these same inputs.
        reference, rc = full_mass(model, f)
        total_cells += rc
        err = exact.sum(1)-exact[:, 1]
        assert np.max(np.abs(err-reference)) < 1e-6
        assert np.max(np.abs(exact.sum(1)-lp.W(-f['hi'], f['hi']))) < 1e-6
        perrank = exact.mean(0); perrank[1] = 0
        linrank = linear.mean(0); linrank[1] = 0
        # Rank2 is correct; others count as nonnegligible if >=1% of chamber error mass.
        nontrivial = np.flatnonzero(perrank >= .01*perrank.sum())+1
        row = dict(model=a['model'], case=a['case'], config=a['config'],
                   exact_error_by_rank=perrank.tolist(), linear_error_by_rank=linrank.tolist(),
                   total_exact=float(err.mean()), total_exact_se=float(err.std()/np.sqrt(n)),
                   total_linear=float(linrank.sum()), significant_competitor_ranks=nontrivial.tolist(),
                   max_reference_difference=float(np.max(np.abs(err-reference))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = dict(rows=rows, cpu_seconds=time.process_time()-start, cells=total_cells,
               recurrence_fpe_lower=2*total_cells,
               note='Additional 10x10 envelope comparisons charged only through CPU time; diagnostic, not estimator')
    (path/'envelope_results.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
