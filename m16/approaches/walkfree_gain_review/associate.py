"""Frozen association stage; features are already committed independently."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
TABLE=Path('/workspace/clean-neurons/m16/approaches/ht_seeds/table.json')
FEATURES=['gate_uncertainty','boundary_alignment']
METHODS=['sobol','iid']
B=2000


def centered_ranks(x,groups):
    out=np.zeros_like(x,dtype=float)
    for ids in groups:
        r=rankdata(x[...,ids],axis=-1,method='average')
        out[...,ids]=r-r.mean(axis=-1,keepdims=True)
    return out


def corr(x,y):
    denom=np.sqrt(np.sum(x*x,axis=-1)*np.sum(y*y,axis=-1))
    return np.divide(np.sum(x*y,axis=-1),denom,out=np.full_like(denom,np.nan,dtype=float),where=denom>0)


def group_indices(rows,by):
    keys=[tuple(r[k] for k in by) for r in rows]
    return [np.array([i for i,key in enumerate(keys) if key==group]) for group in sorted(set(keys))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-file',default='features.json')
    parser.add_argument('--feature-names',nargs='+',default=FEATURES)
    parser.add_argument('--output',default='associations.json')
    args=parser.parse_args();feature_names=args.feature_names
    feature_path=HERE/args.feature_file;features=json.loads(feature_path.read_text())
    table=json.loads(TABLE.read_text());lookup={(r['H'],r['T'],r['seed']):r for r in table}
    assert len(features['models'])==90 and len(lookup)==90
    joined=[]
    for row in features['models']:
        q=lookup[(row['H'],row['T'],row['seed'])]
        assert all(np.isfinite(q[m]['best']) and q[m]['best']>0 for m in METHODS)
        joined.append({k:row[k] for k in ['H','T','seed']+feature_names}|dict(accuracy=q['acc'],
            sobol_gain=q['sobol']['best'],iid_gain=q['iid']['best'],
            sobol_bias_resolved=q['sobol']['resolved'],iid_bias_resolved=q['iid']['resolved']))
    selections={'all90':joined,'exclude_T3':[r for r in joined if r['T']!=3],
                'accuracy_gt_097':[r for r in joined if r['accuracy']>.97]}
    rng=np.random.default_rng(45001);reports=[];total_rank_elements=0;degenerate=0
    for name,rows in selections.items():
        n=len(rows);groupsHT=group_indices(rows,['H','T']);groupsT=group_indices(rows,['T'])
        bootstrap=np.empty((B,n),dtype=int)
        for ids in groupsHT:bootstrap[:,ids]=rng.choice(ids,size=(B,len(ids)),replace=True)
        records=[]
        for strat,groups in [('global',[np.arange(n)]),('within_T',groupsT),('within_HT',groupsHT)]:
            for method in METHODS:
                y=np.log([r[method+'_gain'] for r in rows]);yb=centered_ranks(y[bootstrap],groups);yr=centered_ranks(y,groups)
                total_rank_elements+=(B+1)*n
                for feature in feature_names:
                    x=np.array([r[feature] for r in rows]);xb=centered_ranks(x[bootstrap],groups);xr=centered_ranks(x,groups)
                    total_rank_elements+=(B+1)*n
                    observed=float(corr(xr,yr));boot=corr(xb,yb);good=np.isfinite(boot)
                    degenerate+=int(np.sum(~good))
                    records.append(dict(feature=feature,gain=method,conditioning=strat,rho=observed,
                        ci95=list(map(float,np.quantile(boot[good],[.025,.975]))) if np.any(good) else None,
                        finite_bootstraps=int(good.sum()),degenerate_bootstraps=int(np.sum(~good))))
        baseline=[]
        for method in METHODS:
            x=np.array([-r['T'] for r in rows]);y=np.log([r[method+'_gain'] for r in rows])
            xr=centered_ranks(x,[np.arange(n)]);yr=centered_ranks(y,[np.arange(n)])
            xb=centered_ranks(x[bootstrap],[np.arange(n)]);yb=centered_ranks(y[bootstrap],[np.arange(n)])
            br=corr(xb,yb);good=np.isfinite(br);total_rank_elements+=2*(B+1)*n
            baseline.append(dict(predictor='negative_T',gain=method,rho=float(corr(xr,yr)),
                ci95=list(map(float,np.quantile(br[good],[.025,.975]))),
                note='Unchanged within each fixed-T group; no within-T identification.'))
        reports.append(dict(subset=name,n=n,HT_group_sizes=[len(q) for q in groupsHT],results=records,T_baseline=baseline,
            unresolved_sobol_gain_models=sum(not r['sobol_bias_resolved'] for r in rows),
            unresolved_iid_gain_models=sum(not r['iid_bias_resolved'] for r in rows)))
    out=dict(reports=reports,joined_model_rows=joined,feature_source_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        historical_gain_table_sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_seed=45001,bootstrap_replicates_per_subset=B,total_rank_elements=total_rank_elements,
        degenerate_bootstraps=degenerate,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        uncertainty='Stratified model bootstrap conditional on historical gains; excludes rule/truth/cost uncertainty; exploratory frozen-feature panel, no heldout forecast.')
    (HERE/args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['reports','joined_model_rows']},indent=2))
    for rep in reports:
        print(rep['subset'],rep['n'])
        for row in rep['results']:
            print(row['feature'],row['gain'],row['conditioning'],round(row['rho'],4),[round(x,4) for x in row['ci95']])
        print('T_baseline',rep['T_baseline'])


if __name__=='__main__':main()
