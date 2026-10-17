"""Frozen75 saved Hopf cases, numerical dependence-safe truth boxes and two native budget allocations."""
from pathlib import Path
from fractions import Fraction as Q
import json,hashlib,math,time
from scipy.special import betainc,betaincinv
HERE=Path(__file__).parent;N=10_000_000
def mean(xs):return sum(xs,Q(0))/len(xs)
def F(x):return Q(float(x))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))

def bounds(p):
    k=round(p*N);assert abs(p-k/N)<2e-16;tail=.05/(2*75)
    lo=float(betaincinv(k,N-k+1,tail));hi=float(betaincinv(k+1,N-k,1-tail))
    residual=max(abs(float(betainc(k,N-k+1,lo))-tail),abs(float(betainc(k+1,N-k,hi))-(1-tail)))
    assert residual<1e-10
    return dict(hits=k,n=N,lower=lo,upper=hi,beta_tail_residual=residual)

def group(rows,boxes,sample):
    p=[F(r['mc']) for r in rows];e=[F(r['h64']) for r in rows];ref=[F(r['ref']) for r in rows];ops=[Q(r['h64_ops']) for r in rows]
    pq=[v*(1-v) for v in p];mse=mean([(x-y)**2 for x,y in zip(e,p)]);rmse=mean([(x-y)**2 for x,y in zip(e,ref)]);avgops=mean(ops)
    uniform=sample*mean(pq)/avgops;matched=mean([sample*v/o for v,o in zip(pq,ops)]);adjusted=mse-mean(pq)/N
    result=dict(n=len(rows),mean_native_ops=float(avgops),native_sample=sample,raw_mse=float(mse),mean_pq=float(mean(pq)),finite_reference_mse=float(rmse),raw_uniform_ratio=float(mse/uniform),adjusted_uniform_ratio=float(max(Q(0),adjusted)/uniform),unclamped_adjusted_uniform_ratio=float(adjusted/uniform),finite_reference_uniform_ratio=float(rmse/uniform),raw_matched_ratio=float(mse/matched),adjusted_matched_ratio=float(max(Q(0),adjusted)/matched),finite_reference_matched_ratio=float(rmse/matched),exact_raw_uniform_ratio=str(mse/uniform),exact_raw_matched_ratio=str(mse/matched))
    errlo=[];errhi=[];pqlo=[];pqhi=[]
    for row,b in zip(rows,boxes):
        x=row['h64'];l,u=b['lower'],b['upper'];errlo.append(0. if l<=x<=u else min((x-l)**2,(x-u)**2));errhi.append(max((x-l)**2,(x-u)**2));pqlo.append(min(l*(1-l),u*(1-u)));pqhi.append(.25 if l<=.5<=u else max(l*(1-l),u*(1-u)))
    result['box_uniform']=[math.fsum(errlo)*float(avgops)/(sample*math.fsum(pqhi)),math.fsum(errhi)*float(avgops)/(sample*math.fsum(pqlo))]
    result['box_matched']=[math.fsum(errlo)/math.fsum(sample*q/float(o) for q,o in zip(pqhi,ops)),math.fsum(errhi)/math.fsum(sample*q/float(o) for q,o in zip(pqlo,ops))]
    return result

def main():
    start=time.process_time();manifest=json.loads((HERE/'wider_manifest.json').read_text());groups=[];boxes=[];coverage=[]
    for name,r in manifest.items():assert hashlib.sha256((HERE/'wider_frozen'/name).read_bytes()).hexdigest()==r['sha256']
    for h in (4,5,6):
        name=f'ev_{h}4.json';old=json.loads((HERE/'frozen'/name).read_text());data=json.loads((HERE/'wider_frozen'/name).read_text())
        assert set(data)==set(map(str,range(5)))
        for seed,r in old.items():assert canonical(r)==canonical(data[seed]),('Old row changed',h,seed)
        coverage.append(dict(H=h,T=4,old_seeds=sorted(old),new_seeds=sorted(data),old_rows=sum(len(r['rows']) for r in old.values()),new_rows=sum(len(r['rows']) for r in data.values()),old_rows_unchanged=True))
        for seed in range(5):
            rows=data[str(seed)]['rows'];assert len(rows)==5
            assert all(0<=r['h64']<=1 and 0<r['mc']<1 and r['h64_ops']>0 for r in rows)
            bs=[bounds(r['mc']) for r in rows];boxes.extend(dict(H=h,T=4,seed=seed,index=i,**b) for i,b in enumerate(bs))
            for pop,sl in [('pooled',slice(None)),('clean',slice(0,1)),('noise',slice(1,None))]:
                groups.append(dict(H=h,T=4,seed=seed,population=pop,**group(rows[sl],bs[sl],4*h*(h+2))))
    prior=json.loads((HERE/'raw_audit.json').read_text())
    for g in groups:
        if g['H']!=4:continue
        r=next(r for r in prior['groups'] if r['estimator']=='h64' and r['seed']==g['seed'] and r['population']==g['population'])
        assert math.isclose(g['raw_uniform_ratio'],r['statistics']['ratio_mc_raw'],rel_tol=1e-14)
        assert g['box_uniform'][1]>=r['simultaneous_truth_box']['ratio_upper']
    out=dict(coverage=coverage,groups=groups,mc_truth_boxes=boxes,nominal_joint_coverage=.95,family_size=75,legacy_sample_units=[96,140,192],costs_unaudited=True,numerical_beta_not_outward_rounded=True,scope='Saved-data finite-panel audit only, one75-case Bonferroni family permitting shared MC input streams; no new model calls, cost transfer, quadrature certificate or wider-rung acceptance.',cpu_seconds=time.process_time()-start,new_model_calls=0)
    (HERE/'wider_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(coverage=coverage,pooled=[{k:g[k] for k in ('H','seed','raw_uniform_ratio','adjusted_uniform_ratio','finite_reference_uniform_ratio','box_uniform')} for g in groups if g['population']=='pooled'],cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
