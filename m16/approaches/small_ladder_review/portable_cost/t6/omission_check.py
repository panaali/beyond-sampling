"""Independent positive quadrature of the old reference's omitted cones only."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi

HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56')

def rule(n):
    idx=np.array(np.unravel_index(np.arange(n**5),(n,)*5)).T
    rest=np.ones(n**5);w=np.ones(n**5);y=np.empty((n**5,6))
    for j in range(5):
        x,ww=roots_jacobi(n,4-j,0);u=(x[idx[:,j]]+1)/2
        y[:,j]=rest*u;rest*=1-u;w*=ww[idx[:,j]]/2**(5-j)
    y[:,5]=rest
    assert abs(w.sum()-1/120)<1e-16 and np.all(w>0)
    # Elementary simplex moments independently check Duffy/Jacobi normalization.
    assert np.max(np.abs(w@y-1/720))<1e-16
    return y,w

def main():
    start=time.process_time();out=dict(cases=[],complete=False,
        diagnostic_only=True,production_cost_unchanged=True,
        roundoff_certified=False,opaque_setup_kernels=['SciPy roots_jacobi','NumPy determinant'])
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'omission_results.json').write_text(json.dumps(out,indent=2)+'\n')
    original=AUTHOR/'t6_sigma0.0_cones.npz';retained=AUTHOR/'t6_retained_cones.npz'
    out['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,retained)}
    rays=np.load(original)['rays'];g=np.swapaxes(rays,1,2)@rays
    det=np.abs(np.linalg.det(rays));a=np.maximum((1-1e-12)/6,g.min(axis=(1,2))-1e-12)
    upper=det*a**-3/(120*math.pi**3);small=upper<=1e-4/len(rays)
    assert np.array_equal(rays[~small],np.load(retained)['rays'])
    out.update(original_cones=len(rays),omitted_cones=int(small.sum()),
               retained_cache_exact_match=True,omitted_upper_bound=float(upper[small].sum()))
    assert out['omitted_cones']==73853
    rays=rays[small];g=g[small];det=det[small]
    center=rays.sum(axis=2);center/=np.linalg.norm(center,axis=1)[:,None]
    supporting=np.min(np.einsum('bij,bi->bj',rays,center),axis=1)
    signed=np.sign(rays.sum(axis=2));signed/=np.linalg.norm(signed,axis=1)[:,None]
    orthant_support=np.min(np.einsum('bij,bi->bj',rays,signed),axis=1)
    a=np.maximum(g.min(axis=(1,2)),np.maximum(np.maximum(supporting,0)**2,np.maximum(orthant_support,0)**2))-1e-12
    b=np.diagonal(g,axis1=1,axis2=2).max(axis=1)+1e-12
    assert np.all(a>0)
    s=np.sqrt(a*b);alpha=(a+b)/(2*s);r=(np.sqrt(b)-np.sqrt(a))/(np.sqrt(b)+np.sqrt(a))
    for n in (4,6):
        y,w=rule(n);q=r/(1-r)
        tail=r**n/(s**3*(1-r))*(n*n+3*alpha*n+3*alpha*alpha-1+(2*n+3*alpha)*q+r*(1+r)/(1-r)**2)
        bounds=det/(60*math.pi**3)*tail
        row=dict(order=n,estimate=0.,completed_cones=0,nodes=0,
                 full_quadrature_bound=float(bounds.sum()),partial_quadrature_bound=0.)
        out['cases'].append(row);save()
        batch=max(1,65536//len(w))
        for first in range(0,len(rays),batch):
            if time.process_time()-start>105:
                out['failure']='CPU guard';save();return
            last=min(first+batch,len(rays));points=np.einsum('bij,nj->bni',rays[first:last],y)
            norm2=np.sum(points*points,axis=2)
            values=(1/(norm2*norm2*norm2))@w
            row['estimate']+=float(det[first:last]@values)/(math.pi**3)
            row['completed_cones']=last;row['nodes']+=(last-first)*len(w)
            row['partial_quadrature_bound']+=float(bounds[first:last].sum())
            if first%(batch*128)==0:save()
        row['complete']=True;save()
        if row['full_quadrature_bound']<=2e-7:break
    full=json.loads((HERE/'full_results.json').read_text())
    reference=json.loads((AUTHOR/'cycle13.json').read_text())
    revised=next(x for x in json.loads((AUTHOR/'cycle15.json').read_text())['rebounded_results'] if x['name']=='reference')
    reference_bound=revised['exact_coefficient_outer_bound']+revised['inner_positive_bound']+revised['inner_series_bound']
    row=out['cases'][-1];restored=reference['estimate']+row['estimate']
    out.update(complete=True,reference_midpoint=reference['estimate'],
               restored_reference_midpoint=restored,dd_estimate=full['partial_mass'],
               restored_midpoint_difference=abs(restored-full['partial_mass']),
               dd_conditional_bound=full['conditional_error_bound'],
               original_reference_integration_bound=reference['total_conditional_integration_bound'],
               reference_integration_bound=reference_bound,
               reference_bound_source='cycle15.json, unchanged estimate with tightened analytic coefficients',
               dd_plus_omission_bound=full['conditional_error_bound']+row['full_quadrature_bound'],
               combined_conditional_bound=full['conditional_error_bound']+row['full_quadrature_bound']+reference_bound)
    save();print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
