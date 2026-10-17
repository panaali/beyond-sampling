"""Independent closed nu1/2/3 tail references for directed arithmetic."""
import importlib.util,json,math,time
from pathlib import Path
import mpmath as mp
start=time.process_time();mp.mp.dps=200
path=Path('/workspace/work/recover-small_h2_56/m16/approaches/directed_chebyshev/directed.py')
spec=importlib.util.spec_from_file_location('review_directed_tail',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=[(1.,1.,0),(1.,math.nextafter(1.,math.inf),3),('1e-200','4e-200',5),
       ('1','1e8',1000),('1','2',100000),('0.125','2',0)]
rows=[]
for a,b,degree in cases:
    ctx=module.Directed(50);aa=mp.mpf(a);bb=mp.mpf(b)
    s=mp.sqrt(aa*bb);alpha=(aa+bb)/(2*s);r=(bb-aa)/(aa+bb+2*s);N=degree+1
    refs={1:2*r**N/(s*(1-r)),
          2:2*r**N/s**2*((N+alpha)/(1-r)+r/(1-r)**2),
          3:r**N/(s**3*(1-r))*(N*N+3*alpha*N+3*alpha**2-1+(2*N+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)}
    for nu,reference in refs.items():
        bound=ctx.tail(a,b,nu,degree)
        assert mp.mpf(str(bound.lo))<=reference<=mp.mpf(str(bound.hi))
        row=dict(a=str(a),b=str(b),degree=degree,nu=nu,lo=str(bound.lo),hi=str(bound.hi),reference=mp.nstr(reference,80))
        try:
            lo,hi=bound.floats()
            assert mp.mpf(lo)<=reference<=mp.mpf(hi)
            row['binary64']=[lo,hi]
        except OverflowError:row['binary64']='explicit overflow rejection'
        rows.append(row)
out=dict(rows=rows,cpu_seconds=time.process_time()-start,
         scope='18 independent closed-tail containment checks including constant/adjacent/underflow/overflow; valid radial inputs assumed.')
Path(__file__).with_name('directed_tail_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),cpu_seconds=out['cpu_seconds']),indent=2))
