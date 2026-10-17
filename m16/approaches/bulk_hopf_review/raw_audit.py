"""Independent exact contractions and dependence-safe finite-MC truth boxes for frozen H4T4."""
from pathlib import Path
from fractions import Fraction as Q
import json,time,hashlib,math
import numpy as np
from scipy.special import betainc,betaincinv
HERE=Path(__file__).parent
N=10_000_000
def avg(x):return sum(x,Q(0))/len(x)
def f(x):return Q(float(x))
def contract(rows,key):
    p=[f(r['mc']) for r in rows];e=[f(r[key]) for r in rows];ref=[f(r['ref']) for r in rows];op=avg([Q(r[key+'_ops']) for r in rows]);pq=avg([x*(1-x) for x in p]);mse=avg([(x-y)**2 for x,y in zip(e,p)])
    factor=op/(96*pq)
    return dict(ratio_mc=float(max(Q(0),mse-pq/N)*factor),ratio_mc_raw=float(mse*factor),ratio_ref=float(avg([(x-y)**2 for x,y in zip(e,ref)])*factor),ops=float(op),acc=float(p[0]),mse_raw=float(mse),mc_variance_subtracted=float(pq/N),mean_pq=float(pq),unclamped_noise_adjusted_ratio=float((mse-pq/N)*factor),unbiased_marginal_mse_adjustment=float(mse-pq/(N-1)),exact_raw_ratio=str(mse*factor))

def interval(p):
    k=round(p*N);assert abs(p-k/N)<2e-16
    tail=.05/(25*2)
    lo=0. if k==0 else float(betaincinv(k,N-k+1,tail))
    hi=1. if k==N else float(betaincinv(k+1,N-k,1-tail))
    residual=max(abs(float(betainc(k,N-k+1,lo))-tail) if k else 0.,abs(float(betainc(k+1,N-k,hi))-(1-tail)) if k<N else 0.)
    assert residual<1e-10
    return dict(hits=k,n=N,lower=lo,upper=hi,beta_tail_residual=residual)

def box(rows,boxes,key):
    errlo=[];errhi=[];pqlo=[];pqhi=[]
    for row,b in zip(rows,boxes):
        e=row[key];l,u=b['lower'],b['upper'];errlo.append(0. if l<=e<=u else min((e-l)**2,(e-u)**2));errhi.append(max((e-l)**2,(e-u)**2));pqlo.append(min(l*(1-l),u*(1-u)));pqhi.append(.25 if l<=.5<=u else max(l*(1-l),u*(1-u)))
    ops=math.fsum(r[key+'_ops'] for r in rows)/len(rows)
    return dict(ratio_lower=math.fsum(errlo)*ops/(96*math.fsum(pqhi)),ratio_upper=math.fsum(errhi)*ops/(96*math.fsum(pqlo)),native_mean_ops=ops)

def main():
    start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text())
    for name,r in manifest.items():assert hashlib.sha256((HERE/'frozen'/name).read_bytes()).hexdigest()==r['sha256']
    data=json.loads((HERE/'frozen/ev_44.json').read_text());report=json.loads((HERE/'frozen/report.json').read_text())['44']
    predecessor=json.loads((HERE/'frozen/run87.json').read_text())
    assert set(data)==set(map(str,range(5))) and all(len(d['rows'])==5 for d in data.values())
    allboxes=[];groups=[];report_checks=[]
    for seed in range(5):
        rows=data[str(seed)]['rows'];boxes=[interval(r['mc']) for r in rows]
        for i,row in enumerate(rows):
            old=predecessor['clean'][str(seed)] if i==0 else predecessor['noisy'][str(seed)]['rows'][i-1]
            assert row['mc']==old['mc1e7' if i==0 else 'truth']
            assert row['recipe']==old['est'] and row['recipe_ops']==old['ops']
        allboxes.extend(dict(seed=seed,index=i,**b) for i,b in enumerate(boxes))
        for key in ('h64','recipe'):
            got=contract(rows,key);old=next(r for r in report[key] if r['seed']==seed)
            assert all(math.isclose(got[k],old[k],rel_tol=2e-13,abs_tol=2e-14) for k in old if k!='seed')
            report_checks.append(dict(seed=seed,estimator=key,exact_contraction=got,saved=old))
            for pop,sl in [('pooled',slice(None)),('clean',slice(0,1)),('noise',slice(1,None))]:
                groups.append(dict(seed=seed,estimator=key,population=pop,n=len(rows[sl]),statistics=contract(rows[sl],key),simultaneous_truth_box=box(rows[sl],boxes[sl],key)))
    out=dict(mc_truth_boxes=allboxes,report_checks=report_checks,groups=groups,N=N,nominal_joint_coverage=.95,predecessor_mc_recipe_cost_rows_matched=25,mc_model_dependence_allowed=True,beta_numerical_not_outward_rounded=True,source_cost_denominator=96,scope='25 fixed H4T4 models only. Exact-rational saved report contraction; numerical Clopper-Pearson inversion with Bonferroni union, valid regardless of cross-model dependence given binomial MC provenance. Native operation counts not independently certified. Larger-grid reference is not exact truth.',cpu_seconds=time.process_time()-start)
    (HERE/'raw_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(report_rows=len(report_checks),groups=len(groups),max_beta_residual=max(b['beta_tail_residual'] for b in allboxes),h64_pooled_bounds=[g['simultaneous_truth_box'] for g in groups if g['estimator']=='h64' and g['population']=='pooled'],cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
