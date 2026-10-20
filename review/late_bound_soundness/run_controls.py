"""Reproduce bounded checks. Producer file must be extracted from the frozen git pin."""
import hashlib,importlib.util,json,math,os,platform,resource,sys,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
import flint
from flint import arb,ctx
from local_verifier import mass_upper,positive_subregion_lower,exact_crown
resource.setrlimit(resource.RLIMIT_CPU,(500,510))
torch.set_num_threads(1)
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('producer',sys.argv[1]); producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
cases=json.loads((HERE/'controls.json').read_text())['cases']
tstart=time.process_time(); rows=[]
for case in cases:
    lo,hi=case['lo'],case['hi']; tl=torch.tensor([lo],dtype=torch.float64);th=torch.tensor([hi],dtype=torch.float64)
    old=float(producer.upper(tl,th,nr=4,ng=8)[0]); checks=[]
    for prec in [128,256]:
        pu=producer.upper_arb(lo,hi,nr=4,ng=8,prec=prec)
        ru,ball=mass_upper(lo,hi,prec=prec)
        lower=positive_subregion_lower(lo,hi,prec=prec)
        ctx.prec=prec
        checks.append(dict(prec=prec,producer_arb_float=pu,repair_upper=ru,
            repair_ball=str(ball),true_lower_ball=str(lower),
            float_below_repair_expression=bool(arb(old)<ball),
            actual_mass_underestimate=bool(arb(old)<lower),
            repair_above_lower=bool(arb(ru)>=lower),
            upward_conversion=bool(arb(ru)>=ball)))
    row=dict(name=case['name'],float_upper=old,checks=checks);rows.append(row)
    print(json.dumps(dict(name=case['name'],actual_failure=any(c['actual_mass_underestimate'] for c in checks))),flush=True)
first=next((c for c,r in zip(cases,rows) if any(k['actual_mass_underestimate'] for k in r['checks'])),next(c for c in cases if c['name']=='thin_a_20'))
fine=[]
for case in [cases[0],first]:
    lo,hi=case['lo'],case['hi']; old=float(producer.upper(torch.tensor([lo],dtype=torch.float64),torch.tensor([hi],dtype=torch.float64),nr=24,ng=48)[0])
    for prec in [128,256]:
        pu=producer.upper_arb(lo,hi,nr=24,ng=48,prec=prec)
        ru,ball=mass_upper(lo,hi,nr=24,ng=48,prec=prec)
        lower=positive_subregion_lower(lo,hi,prec=prec)
        fine.append(dict(name=case['name'],prec=prec,float_upper=old,producer_arb_float=pu,repair_upper=ru,true_lower_ball=str(lower),actual_mass_underestimate=bool(arb(old)<lower)))
# Independent exact 1-D piecewise network propagation, splitting on ALL activation roots.
def exact_minima(lo,hi,wi,wh,wo,A,c,target=0):
    wi=list(map(F,wi));wh=[list(map(F,r)) for r in wh];wo=[list(map(F,r)) for r in wo]
    regions=[(F(lo),F(hi),[(F(0),F(0)) for _ in wi])]
    for ar,ct in zip(A,c):
        new=[]
        for l,h,state in regions:
            pre=[(sum(F(wh[i][j])*state[j][0] for j in range(len(wi)))+wi[i]*F(ar[0]),
                  sum(F(wh[i][j])*state[j][1] for j in range(len(wi)))+wi[i]*F(ct)) for i in range(len(wi))]
            breaks=sorted(set([l,h]+[-b/a for a,b in pre if a and l < -b/a < h]))
            for left,right in ([(l,h)] if l==h else zip(breaks[:-1],breaks[1:])):
                mid=(left+right)/2
                new.append((left,right,[(a,b) if a*mid+b>0 else (F(0),F(0)) for a,b in pre]))
        regions=new
    minima=[]
    for out in range(len(wo)):
        if out==target:continue
        vals=[]
        for l,h,state in regions:
            a=sum((wo[out][j]-wo[target][j])*state[j][0] for j in range(len(wi)))
            b=sum((wo[out][j]-wo[target][j])*state[j][1] for j in range(len(wi)))
            vals.extend([a*l+b,a*h+b])
        minima.append(min(vals))
    return minima,len(regions)
rng=np.random.default_rng(1463);nets=[]
for i in range(16):
    def coeff(shape):return (rng.integers(-2,3,size=shape)/2).tolist()
    nets.append(dict(name='random_'+str(i),lo=[-1],hi=[1],wi=coeff(2),wh=coeff((2,2)),wo=coeff((2,2)),A=coeff((2,1)),c=coeff(2)))
# Exact zero cancellation and near-zero ReLU fixtures.
nets.extend([
 dict(name='cancellation_2p53',lo=[0],hi=[0],wi=[1,1],wh=[[0,0],[0,0]],wo=[[0,0],[2**53,-(2**53)]],A=[[0]],c=[1]),
 dict(name='relu_near_zero',lo=[-2.**-54],hi=[2.**-54],wi=[1,-1],wh=[[0,0],[0,0]],wo=[[0,0],[1,-1]],A=[[1]],c=[0]),
 dict(name='positive_prune',lo=[1],hi=[2],wi=[1,1],wh=[[0,0],[0,0]],wo=[[0,0],[1,1]],A=[[1]],c=[0])])
pruning=[]
for n in nets:
    kw={k:v for k,v in n.items() if k!='name'}
    bounds=exact_crown(**kw)
    mins,regions=exact_minima(n['lo'][0],n['hi'][0],n['wi'],n['wh'],n['wo'],n['A'],n['c'])
    pruning.append(dict(name=n['name'],lower=list(map(str,bounds)),exact_min=list(map(str,mins)),regions=regions,
                        lower_valid=all(a<=b for a,b in zip(bounds,mins)),pruned=any(v>0 for v in bounds),
                        prune_valid=not any(v>0 for v in bounds) or any(v>0 for v in mins)))
# Unsafe caller proposal must not drive pruning; exact verifier fallback detects zero-margin case.
unsafe_proposal=F(1,2**54); exact=exact_crown([0],[0],[1],[[0]],[[0],[1]],[[1]],[0])
unsafe=dict(proposed_lower=str(unsafe_proposal),verified_lower=list(map(str,exact)),rejected=not any(v>0 for v in exact))
invalid=[]
for name,args in [('negative_a',([-1,-1]+[0]*7,[1]*9)),('inverted',([1]+[0]*8,[0]+[1]*8)),('nan',([float('nan')]+[0]*8,[1]*9)),('wrong_dimension',([0],[1]))]:
    try:mass_upper(*args); passed=False
    except ValueError:passed=True
    invalid.append(dict(name=name,rejected=passed))
for k,v in [('nr',0),('ng',0),('prec',1)]:
    try:mass_upper(cases[0]['lo'],cases[0]['hi'],**{k:v});passed=False
    except ValueError:passed=True
    invalid.append(dict(name=k,rejected=passed))
result=dict(source_pin='4e29c33861a0983535cd9bc80d660cfd9f2dd412',manifest_sha256=hashlib.sha256((HERE/'controls.json').read_bytes()).hexdigest(),
 versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,flint=flint.__version__),mass=rows,fine_mass=fine,pruning=pruning,unsafe_proposal=unsafe,invalid_inputs=invalid,
 cpu_s=time.process_time()-tstart,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'pruning_controls.json').write_text(json.dumps(nets,indent=2)+'\n')
assert all(r['lower_valid'] and r['prune_valid'] for r in pruning)
assert all(v['rejected'] for v in invalid) and unsafe['rejected']
assert all(c['repair_above_lower'] and c['upward_conversion'] for r in rows for c in r['checks'])
print(json.dumps(dict(actual_mass_failures=sum(any(c['actual_mass_underestimate'] for c in r['checks']) for r in rows),cases=len(rows),cpu_s=result['cpu_s'],rss_kib=result['maxrss_kib'])),flush=True)
