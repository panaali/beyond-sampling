"""Frozen DEV importance-proxy diagnostic; never constructs a normalized proposal."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'m16'))
import leaderboard as lb
spec = importlib.util.spec_from_file_location('proxy_walker', ROOT/'m16/approaches/sci_tao_v2/core.py')
walker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walker)


def linear_proxies(model, y, a, b, tau, kind):
    wi, wh, wo = model
    n, t = y.shape
    h = np.zeros((n, len(wi))); dh = np.zeros_like(h)
    d = np.zeros_like(y); ii = np.arange(n)
    d[ii, a] = 1/np.sqrt(2); d[ii, b] = -1/np.sqrt(2)
    for k in range(t):
        z = h@wh.T+y[:, k:k+1]*wi
        dz = dh@wh.T+d[:, k:k+1]*wi
        gate = z > 0
        h, dh = np.where(gate, z, 0), np.where(gate, dz, 0)
    l, dl = h@wo.T, dh@wo.T
    window = ndtr(tau)-ndtr(-tau)
    out = {}
    targets = (b, a) if kind == '23' else (a, b)
    for mode in ('pair', 'envelope'):
        error = np.zeros(n)
        for side, target in enumerate(targets):
            lo, hi = (-tau, np.zeros(n)) if side == 0 else (np.zeros(n), tau)
            c = l[ii, target, None]-l
            e = dl[ii, target, None]-dl
            with np.errstate(divide='ignore', invalid='ignore'):
                roots = -c/e
            lower = np.where(e > 0, roots, -np.inf)
            upper = np.where(e < 0, roots, np.inf)
            bad = (e == 0)&((c < 0)|((c == 0)&(np.arange(t)[None] < target[:, None])))
            lower[ii, target] = -np.inf; upper[ii, target] = np.inf; bad[ii, target] = False
            if mode == 'pair':
                rival = np.where(target == a, b, a)
                keep = np.arange(t)[None] == rival[:, None]
                lower = np.where(keep, lower, -np.inf)
                upper = np.where(keep, upper, np.inf)
                bad &= keep
            low = np.maximum(lo, lower.max(1)); high = np.minimum(hi, upper.min(1))
            good = np.where((high > low)&~bad.any(1), ndtr(high)-ndtr(low), 0)
            error += ndtr(hi)-ndtr(lo)-good
        out[mode] = np.clip(error, 0, window)
        for eta in (.01, .1):
            out[f'{mode}_floor{eta}'] = np.maximum(out[mode], eta*window)
    out['window'] = window
    hsize = len(wi)
    # Implemented dense value/tangent recurrence + all readouts + one envelope.
    recurrence = t*(4*hsize*hsize+7*hsize)
    readout = 2*t*(2*hsize-1)
    envelope = 2*(24*t+12+4*walker.CDF_FLOPS)
    window_flops = 2*walker.CDF_FLOPS+1
    flops = recurrence+readout+envelope+window_flops
    cost = {name: (window_flops if name == 'window' else flops+2)/walker.fpe_flops(hsize, t)
            for name in out}
    actual_all_flops = n*(recurrence+readout+2*envelope+window_flops+8)
    return out, cost, actual_all_flops


def weighted_top_share(value, weight, second, fraction=.01):
    order = np.argsort(value)[::-1]
    w, s = weight[order], second[order]
    target = fraction*w.sum()
    before = np.r_[0., np.cumsum(w)[:-1]]
    take = np.clip((target-before)/np.maximum(w, 1e-300), 0, 1)
    return float((s*take).sum()/s.sum()) if s.sum() else 0.


def metrics(f, g, exact_cost, proxy_cost):
    mu, qnorm = f.mean(), g.mean()
    base_var = float(np.var(f))
    base_cost = float(exact_cost.mean())
    tail = f >= np.quantile(f, .99)
    corr = lambda x,y: float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
    out = dict(mean_f=float(mu), mean_proxy=float(qnorm), base_variance=base_var,
               base_cost=base_cost, proxy_cost=proxy_cost,
               spearman=corr(f, g),
               tail_spearman=corr(f[tail], g[tail]),
               base_top1_second_share=weighted_top_share(f, np.ones(len(f)), f*f),
               proxy_zero_positive_f=int(((g == 0)&(f > 0)).sum()))
    if out['proxy_zero_positive_f'] or qnorm <= 0:
        out.update(importance_variance=None, variance_gain=0., cost_adjusted_gain=0.,
                   proposal_top1_second_share=None, expected_proposal_cost=None,
                   support_failure=True)
        return out
    ratio = np.divide(f, g, out=np.zeros_like(f), where=g > 0)
    second = f*ratio
    varq = max(float(qnorm*second.mean()-mu*mu), 0.)
    cq = float(np.mean(g*exact_cost)/qnorm+proxy_cost)
    out.update(importance_variance=varq, variance_gain=base_var/max(varq, 1e-300),
               cost_adjusted_gain=base_var*base_cost/max(varq*cq, 1e-300),
               proposal_top1_second_share=weighted_top_share(ratio, g, second),
               expected_proposal_cost=cq, support_failure=False)
    return out


def load_models(t):
    if t == 10:
        checkpoint = Path('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth')
    else:
        checkpoint = Path('/workspace/clean-neurons/work/checkpoints/H16_T5_seed0.pth')
    s = torch.load(checkpoint, map_location='cpu', weights_only=True)
    base = (s['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
            s['rnn.weight_hh_l0'].numpy().astype(float), s['linear.weight'].numpy().astype(float))
    return {f'dev_T{t}_s{sigma}_0': lb._noisy(base, 6, sigma, seed=1)[0]
            for sigma in (.0025, .02)}


def main():
    begin = time.process_time()
    folder = Path(__file__).parent
    rows, combined = [], []
    exact_work, proxy_flops = 0., 0
    # T5 remains gated by documented checkpoint/base verification.
    for t in (10,):
        _, z23, z12 = walker.tables(t)
        z = {'23':z23, '12':z12}
        for model_name, model in load_models(t).items():
            for split, seed in [('design',629031), ('heldout',629032)]:
                rng = np.random.default_rng(seed)
                family = {}
                for kind in ('23', '12'):
                    uniform = rng.random((16384, t))
                    y, a, b, tau = walker.face_points(t, kind, uniform)
                    f, cost, cells = walker.walk_mass(model, y, a, b, tau, kind)
                    proxies, costs, actual_flops = linear_proxies(model, y, a, b, tau, kind)
                    exact_work += cost.sum(); proxy_flops += actual_flops
                    report = {name: metrics(f, g, cost, costs[name]) for name, g in proxies.items()}
                    family[kind] = report
                    row = dict(model=model_name, split=split, kind=kind, n=len(f),
                               exact_fpe=float(cost.sum()), proxy_flops=actual_flops,
                               mean_cells=float(cells.mean()), proxies=report)
                    rows.append(row)
                    print(json.dumps(dict(model=model_name, split=split, kind=kind,
                                          gains={k:round(v['cost_adjusted_gain'],3) for k,v in report.items()})), flush=True)
                for name in family['23']:
                    v0 = sum(z[k]**2*family[k][name]['base_variance'] for k in ('23','12'))
                    c0 = sum(family[k][name]['base_cost'] for k in ('23','12'))
                    p = sum(z[k]*family[k][name]['mean_f'] for k in ('23','12'))
                    fail = any(family[k][name]['support_failure'] for k in ('23','12'))
                    vq = None if fail else sum(z[k]**2*family[k][name]['importance_variance'] for k in ('23','12'))
                    cq = None if fail else sum(family[k][name]['expected_proposal_cost'] for k in ('23','12'))
                    # Report-only posthoc oracle allocation; no node/proxy setting changes.
                    neyman0 = sum(z[k]*np.sqrt(family[k][name]['base_variance']*family[k][name]['base_cost'])
                                  for k in ('23','12'))**2
                    neymanq = None if fail else sum(z[k]*np.sqrt(family[k][name]['importance_variance']*
                                                               family[k][name]['expected_proposal_cost'])
                                                   for k in ('23','12'))**2
                    combined.append(dict(model=model_name, split=split, proxy=name, base_error=float(p),
                                         base_variance_cost=float(v0*c0),
                                         optimistic_proposal_variance_cost=None if fail else float(vq*cq),
                                         optimistic_gain=0. if fail else float(v0*c0/(vq*cq)),
                                         optimistic_vs_bernoulli=0. if fail else float(p*(1-p)/(vq*cq)),
                                         posthoc_oracle_neyman_gain=0. if fail else float(neyman0/neymanq),
                                         posthoc_oracle_neyman_vs_bernoulli=0. if fail else float(p*(1-p)/neymanq),
                                         posthoc_base_neyman_variance_cost=float(neyman0),
                                         posthoc_proposal_neyman_variance_cost=None if fail else float(neymanq)))
    out = dict(rows=rows, combined=combined, cpu_seconds=time.process_time()-begin,
               diagnostic_exact_fpe=exact_work, diagnostic_proxy_flops=proxy_flops,
               note='Variance/cost ceiling only; transport construction and normalizer unknown. No estimator.')
    (folder/'results.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
