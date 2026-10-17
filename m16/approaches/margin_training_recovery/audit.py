"""Independent contractions of saved T115 rows; no estimator/model calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
import numpy as np
HERE=Path(__file__).parent

def rational(x):return F(float(x))
def close(a,b):return abs(a-b)<=5e-14*max(abs(a),abs(b),1e-25)

def contraction(errs,truths=None,ops=None):
    n=len(errs);e=list(map(rational,errs));mse=sum((v*v for v in e),F(0))/n;bias=sum(e,F(0))/n
    out=dict(n=n,mse=float(mse),rms=math.sqrt(float(mse)),bias=float(bias),max_abs=max(map(abs,errs)),exact_mse=str(mse),exact_bias=str(bias))
    if truths is not None:
        assert len(truths)==n;t=list(map(rational,truths));pq=sum((v*(1-v) for v in t),F(0))/n;ratio=mse*rational(ops)/(72*pq)
        out.update(mean_pq=float(pq),original72_ratio=float(ratio),exact_mean_pq=str(pq),exact_original72_ratio=str(ratio))
    return out

def bootstrap(errs,truths,ops):
    # Same frozen RNG/resampling law, independently written vectorized contraction.
    rng=np.random.default_rng(0);e=np.array(errs);p=np.array(truths);idx=rng.integers(0,len(e),(1000,len(e)))
    samples=(e[idx]**2).sum(1)/(p[idx]*(1-p[idx])).sum(1)*(ops/72)
    return np.quantile(samples,[.025,.975]).tolist()

def main():
    start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text())
    for name,r in manifest.items():
        if '/' not in name:assert hashlib.sha256((HERE/'frozen'/name).read_bytes()).hexdigest()==r['sha256']
    b=(HERE/'frozen/est75_t112.py').read_bytes();assert hashlib.sha256(b.split(b'\n',2)[2]).hexdigest()=='426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3'
    panels=[];clean=[];coverage=[]
    for k in (28,30):
        margin=json.loads((HERE/'frozen'/f'margin_k{k}.json').read_text());rung=json.loads((HERE/'frozen'/f'rung_k{k}.json').read_text())
        coverage.append(dict(k=k,saved_seeds=list(margin['seeds']),missing_seeds=[str(s) for s in range(5) if str(s) not in margin['seeds']]))
        for seed,row in margin['seeds'].items():
            tr=rung['seeds'][seed]['panel']['truths'];ctr=rung['seeds'][seed]['clean']['truth'];assert row['truth_spotcheck_maxdiff']==0
            for recipe,d in row['clean'].items():
                got=contraction([d['err']],[ctr],d['ops']);assert close(got['original72_ratio'],d['ratio']);clean.append(dict(k=k,seed=int(seed),recipe=recipe,truth=ctr,saved=d,recomputed=got))
            for panel in ('p20','p100'):
                for recipe,d in row.get(panel,{}).items():
                    got=contraction(d['errs'],tr if panel=='p20' else None,d['ops']);checks={key:close(got[key],d[key]) for key in ('rms','bias','max_abs')};assert all(checks.values())
                    r=dict(k=k,seed=int(seed),panel=panel,recipe=recipe,saved_summary={key:v for key,v in d.items() if key!='errs'},recomputed=got,checks=checks)
                    if panel=='p20':
                        checks['ratio']=close(got['original72_ratio'],d['ratio']);ci=bootstrap(d['errs'],tr,d['ops']);checks['bootstrap']=all(close(a,b) for a,b in zip(ci,d['ci']));assert all(checks.values());r['bootstrap_ci']=ci
                    else:
                        implied=got['mse']*d['ops']/(72*d['ratio']) if d['ratio'] else None
                        mean=row['p100_mean_truth'];bound=mean*(1-mean)
                        assert implied is None or 0<=implied<=bound+1e-15
                        r['implied_mean_pq_not_independent']=implied;r['jensen_upper_from_saved_mean_truth']=bound;r['missing']='Per-copy truth/counter arrays absent; ratio/CI and copy-loss counts not independently reconstructable.'
                    panels.append(r)
    # Basic stats controls could fail for swapped errors/truths or Jensen confusion.
    c=contraction([.25,-.25],[.25,.75],72);assert c['mse']==.0625 and c['bias']==0 and c['mean_pq']==.1875 and c['original72_ratio']==1/3
    assert .1875!=.5*(1-.5)
    out=dict(coverage=coverage,panels=panels,clean=clean,original72_counter_only=True,per_copy_counter_fields_not_reverified=True,missing_training_exponents=[29,27,26,25,24,23,22,21,20],cpu_seconds=time.process_time()-start,model_runs=0,scope='Saved-data arithmetic validation, not independent truth or full common-unit cost certification.')
    (HERE/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(coverage=coverage,panels=len(panels),clean=len(clean),p20_max_ratio=max(r['recomputed']['original72_ratio'] for r in panels if r['panel']=='p20'),p20_max_upper=max(r['bootstrap_ci'][1] for r in panels if r['panel']=='p20'),cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
