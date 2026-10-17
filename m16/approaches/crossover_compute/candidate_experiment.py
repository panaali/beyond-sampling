"""Bounded, once-only evaluation of frozen T19 candidate; writes only own folder.

Run from repository root with one-thread BLAS, e.g.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/crossover_compute/candidate_experiment.py
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'm16'))
sys.path.insert(0, str(HERE))
import estimator as candidate
import leaderboard as lb


def main():
    import torch
    torch.set_num_threads(1)
    t0 = time.process_time()
    truth_path = ROOT / lb.truths_path(16, 10)
    truths = json.loads(truth_path.read_text())
    # Model-free sanity check: mixing exchangeable rank chambers recovers N(0,I) moments.
    for strata in candidate.LEVELS:
        ms = candidate.moments(10, strata)
        assert abs(np.mean([m.sum() for m, c in ms])) < 1e-8
        assert abs(np.mean([np.trace(c + np.outer(m,m)) for m,c in ms])-10) < 1e-8
        assert abs(np.mean([(c + np.outer(m,m)).sum() for m,c in ms])-10) < 1e-8
        assert min(np.linalg.eigvalsh(c).min() for m,c in ms) > 0
    # A zero-logit network must select index zero, hence accuracy 1/T, including ties.
    zero = (np.zeros(16), np.zeros((16,16)), np.zeros((10,16)))
    assert abs(candidate.est(zero, 1000, None)[0] - .1) < 1e-12
    unsupported = []
    for budget in (100,250,500):
        try:
            candidate.est(zero, budget, None)
        except ValueError as e:
            unsupported.append(dict(budget=budget, reason=str(e)))
        else:
            raise AssertionError('Low budget unexpectedly accepted')
    models = lb.models()
    raw = []
    for budget in (1000,2000,4000):
        for name, model in models.items():
            if name not in truths:
                raise ValueError(f'Missing cached truth: {name}; never generate new truth here')
            start = time.process_time()
            r = candidate.evaluate(model, budget)
            cpu = time.process_time()-start
            a, c = candidate.est(model, budget, np.random.default_rng(123))
            assert a == r['estimate'] and c == r['cost']
            raw.append(dict(name='crossover_compute', model=name, budget=budget,
                            est=a, cost=c, strata=r['strata'], union_gap=r['union_gap'],
                            cpu_s=cpu, truth=truths[name]['acc'], truth_se=truths[name]['se']))
        print(f'budget {budget} finished', flush=True)
    table = []
    rng = np.random.default_rng(0)
    for budget in (1000,2000,4000):
        for group in ('orig','seed','s0.0025','s0.005','s0.01','s0.02'):
            rows = [r for r in raw if r['budget']==budget and lb.group_of(r['model'])==group]
            error = np.array([(r['est']-r['truth'])**2-r['truth_se']**2 for r in rows])
            base = np.array([r['truth']*(1-r['truth'])/r['cost'] for r in rows])
            if len(rows)>1:
                idx = rng.integers(0,len(rows),(5000,len(rows)))
                ci = np.quantile(error[idx].mean(1)/base[idx].mean(1),[.025,.975]).tolist()
            else:
                ci = None
            model = models[rows[0]['model']]
            start = time.process_time()
            # CPU reference repeats amortize interpreter timer noise, at actual candidate cost.
            for rep in range(10):
                lb.est_sampling(model, int(rows[0]['cost']), np.random.default_rng(rep))
            sampling_cpu = (time.process_time()-start)/10
            table.append(dict(name='crossover_compute',cls='unreviewed',budget=budget,group=group,
                              cost=rows[0]['cost'],n=len(rows),ratio=float(error.mean()/base.mean()),ci=ci,
                              tuned=False,bias_pp=100*np.mean([r['est']-r['truth'] for r in rows]),
                              time_ratio=np.mean([r['cpu_s'] for r in rows])/sampling_cpu,
                              union_gap=np.mean([r['union_gap'] for r in rows]),
                              ci_note='paired model bootstrap, provisional' if ci else 'single model; no model-bootstrap CI'))
    meta = dict(cpu_seconds=time.process_time()-t0,
                estimator_sha256=hashlib.sha256((HERE/'estimator.py').read_bytes()).hexdigest(),
                truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                baseline='mean_i p_i(1-p_i)/actual_cost_i',unsupported=unsupported,
                precompute_flops=candidate.PRECOMPUTE_FLOPS,
                caution='Unreviewed candidate. Model-bootstrap CIs omit finite truth uncertainty; orig CI unavailable.')
    (HERE/'candidate_raw.json').write_text(json.dumps(dict(meta=meta,rows=raw),indent=2)+'\n')
    (HERE/'candidate_table.json').write_text(json.dumps(table,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    for r in table:
        print(r['budget'],r['group'],round(r['ratio'],3),r['ci'],round(r['bias_pp'],3),round(r['time_ratio'],2))


if __name__=='__main__':
    main()
