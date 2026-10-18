"""Independent analytic values and exact setup checks for counted primitive."""
import json,math,time,resource
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from .primitive import Rules,estimate,geometry,Counter
from ..radial_t4_reference.radial import determinant as exactdet
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();out=dict(analytic=[],exact_setup=[],unsupported=[]);rules=Rules()
    for a in (0.,.4,.9):
        for b in (0.,.4,.9):
            N=np.array([[1,0,0,0],[a,math.sqrt(1-a*a),0,0],[0,0,1,0],[0,0,b,math.sqrt(1-b*b)]]);R=np.linalg.inv(N).T;truth=(.25+math.asin(a)/(2*math.pi))*(.25+math.asin(b)/(2*math.pi))
            for transform,Y in [('base',R),('scale',R*8),('permute',R[:,[2,0,3,1]])]:
                for tol in (1e-7,1e-7/6347):
                    row=estimate(Y,tol,rules);row.update(a_correlation=a,b_correlation=b,transform=transform,tolerance=tol,truth=truth)
                    if 'value' in row:row.update(error=row['value']-truth,passes=abs(row['value']-truth)<=row['bound']+5e-14)
                    out['analytic'].append(row)
    rays=np.load(HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz')['rays']
    for i in (0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020):
        A=[[F(float(x)) for x in row] for row in rays[i]];c=[sum(row[j] for row in A) for j in range(4)];m=min(sum(x*y for x,y in zip(row,c)) for row in A);cc=sum(x*x for x in c);b=max(sum(x*x for x in row) for row in A);d=abs(exactdet(A));row=dict(index=i,exact_center_positive=m>0,exact_det=str(d),exact_b=str(b))
        try:
            af,bf,df=geometry(rays[i],Counter());a=m*m/cc;row.update(exact_a=str(a),float_a=af.v,float_b=bf.v,float_det=df.v,relative_a_error=float(F(af.v)/a-1),relative_b_error=float(F(bf.v)/b-1),relative_det_error=float(F(df.v)/d-1),absolute_det_error=float(F(df.v)-d))
        except Exception as exc:row['failure']=str(exc)
        out['exact_setup'].append(row)
    for name,R in [('zero',np.zeros((4,4))),('duplicate',np.ones((4,4))),('center_nonseparating',np.array([[1.,0,0,0],[-2,1,0,0],[0,0,1,0],[0,0,0,1]]))]:
        row=estimate(R,1e-7,rules);row['fixture']=name;out['unsupported'].append(row);assert row['mode']=='unresolved'
    out.update(complete=True,cpu=time.process_time()-start,returned_analytic=sum('value' in r for r in out['analytic']),unresolved_analytic=sum('value' not in r for r in out['analytic']),all_returned_analytic_pass=all(r.get('passes',True) for r in out['analytic']),cold_events=rules.events)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('analytic','exact_setup','unsupported','cold_events')},indent=2));assert out['all_returned_analytic_pass']
if __name__=='__main__':main()
