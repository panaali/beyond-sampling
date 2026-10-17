import json,math,time
from pathlib import Path
from scipy.special import ndtr,owens_t
from .two_suffix import regions,integrate
from .rank import pair_rank

def main():
    start=time.process_time();path=Path(__file__).parent;m=json.loads((path/'results.json').read_text())['models'][0];wi,wh,wo=m['weights'];errors=[]
    for k in (.1,1.,3.,1e8):
        for t in (0.,1e-4,.1,1.,4.,8.):
            ref,_=pair_rank(k,t);q=math.atan(k)/(2*math.pi);g=(.5*float(ndtr(t))+float(owens_t(t,k))-q,.5*float(ndtr(-t))-float(owens_t(t,k)),0.);err=max(abs(a-b) for a,b in zip(ref,g));assert err<2e-12;errors.append(err)
    cells,setup=regions(wi,wh,wo);runs=[]
    for n in (8,16,32):
        v,s=integrate(wi,wh,wo,n);assert abs(s['event_mass_integrated']-s['event_mass_analytic'])<1e-14;runs.append(dict(estimate=v,stats=s))
    out=dict(complete=True,model_sha256=m['model_sha256'],owen_max_difference=max(errors),regions=[dict(**{k:v for k,v in c.items() if k not in ('lo','hi')},lo=str(c['lo']),hi=None if c['hi'] is None else str(c['hi'])) for c in cells],setup=setup,runs=runs,cpu_seconds=time.process_time()-start,random_model_inputs=0)
    assert out['cpu_seconds']<18
    (path/'two_suffix_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='regions'},indent=2))
if __name__=='__main__':main()
