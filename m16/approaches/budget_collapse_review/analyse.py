"""Fixed-rule/truth uncertainty; retain unresolved MSE and budget deviations."""
import time
START=time.process_time()
import hashlib,json,resource
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TRUTH=Path('/workspace/clean-neurons/m16/approaches/ht_seeds/scan_seeds.json')
TRUTH_HASH='28a60e0e44f64d33d203807c1ae17d3d00f6865af040ab794ee1ad4d64520d36'


def interval(values):
    good=np.isfinite(values)&(values>0);bad=float(1-good.mean())
    # Any undefined denominator is preserved; do not condition on valid draws.
    # Replacing invalid draws by +infinity makes unresolved upper tails explicit.
    safe=np.where(good,values,np.inf)
    lo,hi=np.quantile(safe,[.025,.975],method='nearest')
    return dict(ci95=[float(lo) if np.isfinite(lo) else None,float(hi) if np.isfinite(hi) else None]
                if bad<=.025 else None,invalid_fraction=bad)


def main():
    assert hashlib.sha256(TRUTH.read_bytes()).hexdigest()==TRUTH_HASH
    truth=json.loads(TRUTH.read_text());rng=np.random.default_rng(44021);reps=2000
    idx=rng.integers(0,16,(reps,16));rows=[];draws={};hashes={};cycles=[]
    # Truth draw shared between the two budgets of each model, independent across models.
    for T in [4,5,6]:
        path=HERE/f'panel_T{T}.json';panel=json.loads(path.read_text());assert panel['complete']
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();cycles.extend(panel['cycles'])
        for model in panel['models']:
            sd=model['seed'];tr=truth[f'16:{T}:{sd}']['truth'];p,se=tr['acc'],tr['se']
            pt=np.clip(p+se*rng.standard_normal(reps),0,1)
            for cell in model['budgets']:
                assert cell['feasible'] and len(cell['rules'])==16
                B=cell['target_budget'];e=np.array([r['estimate'] for r in cell['rules']])
                cold=np.array([r['cold_fpe'] for r in cell['rules']]);warm=np.array([r['warm_fpe'] for r in cell['rules']])
                var=float(e.var(ddof=1));mse=float(np.mean((e-p)**2)-se**2)
                eb=e[idx];cb=cold[idx].mean(axis=1)
                denom={'variance':eb.var(axis=1,ddof=1),'mse':np.mean((eb-pt[:,None])**2,axis=1)-se**2}
                metrics={}
                for metric,d in [('variance',var),('mse',mse)]:
                    sampled=np.divide(pt*(1-pt)/cb,denom[metric],out=np.full(reps,np.nan),where=denom[metric]>0)
                    draws[T,sd,B,metric]=sampled
                    metrics[metric]=dict(gain=p*(1-p)/cold.mean()/d if d>0 else None,
                        denominator=d,**interval(sampled))
                row=dict(H=16,T=T,seed=sd,budget=B,faces_per_kind=cell['faces_per_kind'],
                    mean_estimate=float(e.mean()),truth=p,truth_se=se,truth_n=tr['n'],bias=float(e.mean()-p),
                    estimator_sd=float(np.sqrt(var)),truth_resolves_mse=bool(se<.5*np.sqrt(var)),
                    cold_fpe_mean=float(cold.mean()),cold_fpe_range=[float(cold.min()),float(cold.max())],
                    warm_fpe_mean=float(warm.mean()),budget_ratio=float(cold.mean()/B),
                    budget_mismatch=bool(abs(cold.mean()/B-1)>.1),metrics=metrics)
                rows.append(row)
    geometric=[];declines=[]
    for B in [16000,64000]:
        for metric in ['variance','mse']:
            for T in [4,5,6]:
                selected=[r for r in rows if r['T']==T and r['budget']==B]
                points=[r['metrics'][metric]['gain'] for r in selected]
                sampled=np.exp(np.mean(np.log(np.array([draws[T,sd,B,metric] for sd in range(5)])),axis=0))
                geometric.append(dict(T=T,budget=B,metric=metric,gain=float(np.exp(np.mean(np.log(points))))
                    if all(p is not None and p>0 for p in points) else None,**interval(sampled)))
            for Ta,Tb in [(4,5),(5,6)]:
                seedrows=[];boot=[]
                for sd in range(5):
                    a=next(r for r in rows if (r['T'],r['seed'],r['budget'])==(Ta,sd,B))['metrics'][metric]['gain']
                    b=next(r for r in rows if (r['T'],r['seed'],r['budget'])==(Tb,sd,B))['metrics'][metric]['gain']
                    sampled=draws[Ta,sd,B,metric]/draws[Tb,sd,B,metric];boot.append(sampled)
                    seedrows.append(dict(seed=sd,decline=a/b if a is not None and b is not None else None,**interval(sampled)))
                sampled=np.exp(np.mean(np.log(np.array(boot)),axis=0));points=[r['decline'] for r in seedrows]
                declines.append(dict(from_T=Ta,to_T=Tb,budget=B,metric=metric,per_seed=seedrows,
                    geometric_decline=float(np.exp(np.mean(np.log(points)))) if all(p is not None for p in points) else None,
                    **interval(sampled)))
    out=dict(rows=rows,geometric_gains=geometric,declines=declines,budget_mismatches=[
        dict(T=r['T'],seed=r['seed'],budget=r['budget'],ratio=r['budget_ratio']) for r in rows if r['budget_mismatch']],
        panel_hashes=hashes,truth_sha256=TRUTH_HASH,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_seed=44021,bootstrap_replicates=reps,production_cycles=cycles,
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        uncertainty='Fixed five-model panel; paired rule bootstrap and independent normal truth approximation; costs are declared proxies. Budget matching is approximate.')
    (HERE/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','declines']},indent=2))
    for r in declines:print(r['budget'],r['metric'],r['from_T'],r['to_T'],r['geometric_decline'],r['ci95'])


if __name__=='__main__':main()
