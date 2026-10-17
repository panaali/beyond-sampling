"""Contract the predeclared complete missing panels; retain two compute-allocation comparators."""
from pathlib import Path
import json,math,time
import numpy as np
from audit import contraction,bootstrap
HERE=Path(__file__).resolve().parent

def main():
    start=time.process_time();pilot=json.loads((HERE/'continuation_pilot.json').read_text());assert pilot['complete']
    outputs=[];shard_cpu=0.
    for seed in range(1,5):
        rows=[]
        for ns in range(20,24):
            d=json.loads((HERE/f'k30_s{seed}_noise{ns}.json').read_text());assert d['complete'] and len(d['rows'])==256
            assert d['seed']==seed and d['noise_seed']==ns
            rows.extend(d['rows']);shard_cpu+=d['cpu_seconds']
        assert [r['index'] for r in rows]==list(range(1024))
        assert sum(r.get('reused_pilot',False) for r in rows)==16
        for name in ('r85b','r112'):
            truth=np.array([r['saved_truth'] for r in rows]);estimate=np.array([r['recipes'][name]['value'] for r in rows]);ops=np.array([r['recipes'][name]['native_ops'] for r in rows]);err=estimate-truth
            assert np.all(ops>0) and np.all((truth>=0)&(truth<=1))
            stats=contraction(err.tolist(),truth.tolist(),float(ops.mean()))
            fixed=bootstrap(err,truth,float(ops.mean()))
            # Uniform sample budget set to mean native recipe cost versus budgets matched separately to each copy.
            pq=truth*(1-truth);uniform=float(np.mean(err*err)*np.mean(ops)/(72*np.mean(pq)))
            matched=float(np.mean(err*err)/np.mean(72*pq/ops))
            rng=np.random.default_rng(0);ids=rng.integers(0,len(rows),(1000,len(rows)))
            paired_uniform=np.mean((err*err)[ids],axis=1)*np.mean(ops[ids],axis=1)/(72*np.mean(pq[ids],axis=1))
            paired_matched=np.mean((err*err)[ids],axis=1)/np.mean((72*pq/ops)[ids],axis=1)
            q=lambda x:[float(v) for v in np.quantile(x,[.025,.975])]
            outputs.append(dict(seed=seed,recipe=name,n=len(rows),statistics=stats,native_mean_ops=float(ops.mean()),native_min_ops=int(ops.min()),native_max_ops=int(ops.max()),original_fixed_cost_bootstrap=fixed,paired_cost_uniform_budget_ci=q(paired_uniform),per_copy_budget_ratio=matched,paired_cost_per_copy_budget_ci=q(paired_matched),copy_ratio_ge1=int(np.sum(err*err>=72*pq/ops)),worst_copy_ratio=float(np.max(err*err/(72*pq/ops)))))
            assert math.isclose(uniform,stats['original72_ratio'],rel_tol=1e-14)
    clean=[]
    for r in pilot['rows']:
        if r['index']!=-1 or r['seed']==0:continue
        for name,z in r['recipes'].items():
            clean.append(dict(seed=r['seed'],recipe=name,statistics=contraction([z['value']-r['saved_truth']],[r['saved_truth']],z['native_ops']),native_ops=z['native_ops'],truth_difference=r['truth_difference']))
    out=dict(panels=outputs,clean=clean,pilot_cpu=pilot['cpu_seconds'],shard_cpu=shard_cpu,contraction_cpu=time.process_time()-start,scope='Original native72 convention only; pilot truth independently checked, remaining T101 truths reused. Uniform and per-copy budget allocations differ. Pointwise 95% bootstrap intervals; no full-cost, simultaneous or ladder certificate.',recipe_evaluations_new=2*(4*1024+5+4),independent_truth_evaluations=73)
    (HERE/'continuation_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(panels=len(outputs),max_ratio=max(r['statistics']['original72_ratio'] for r in outputs),max_upper=max(r['paired_cost_uniform_budget_ci'][1] for r in outputs),shard_cpu=shard_cpu)))
if __name__=='__main__':main()
