"""Independent MP scalar controls, analytic cones, and fixed prior sixteen rays."""
import hashlib,json,math,resource,time
from pathlib import Path
from collections import Counter
import numpy as np
import mpmath as mp
from .primitive import Rules,Ledger,positive_inner,integrate
from ..radial_t4_reference.radial import bounds,probability_bound
HERE=Path(__file__).parent
INDICES=(0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020)
def scalar_reference(P,D):
    with mp.workdps(80):
        pp=list(map(mp.mpf,map(float,P)));dd=list(map(mp.mpf,map(float,D)));a=mp.fdot(dd,dd);b=mp.fdot(pp,dd);c=mp.fdot(pp,pp)
        if not a:return 1/c**2
        delta=a*c-b*b;assert delta>0
        def F(z):q=a*z*z+2*b*z+c;return (a*z+b)/(2*delta*q)+a/(2*delta**mp.mpf('1.5'))*mp.atan((a*z+b)/mp.sqrt(delta))
        return F(1)-F(0)
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();out=dict(scalar=[],analytic=[],trained=[],complete=False,analytic_float_certificate=False);rules=Rules();_,_,_,_,x,w=rules.get(20)
    for slope,eps in [(s,e) for s in (-2.,-.5,0.,.1,2.,1e4) for e in (1e-8,1e-3,1.)]+[(0.,0.)]:
        for scale in (2.**-20,1.,2.**20):
            P=np.array([1.,0,0,0])*scale;D=np.array([slope,eps,0,0])*scale;L=Ledger();value,error,modes=positive_inner(P[None,:],D[None,:],20,x,w,L);ref=float(scalar_reference(P,D));pts=P+x[:,None]*D;q=np.sum(pts*pts,axis=1);direct=float(np.dot(w,1/q**2));analytic='fallback' not in modes;relative=abs(value[0]/ref-1);fallback_error=abs(value[0]/direct-1)
            row=dict(slope=slope,epsilon=eps,scale=scale,value=float(value[0]),reference=ref,relative_error=relative,direct_gl20=direct,fallback_parity_relative_error=fallback_error,series_error=float(error[0]),modes=modes,kernel_counts=L.dict(),passes=relative<3e-12 if analytic else fallback_error<3e-15);out['scalar'].append(row)
    for a in (0.,.4,.9):
        for b in (0.,.4,.9):
            N=np.array([[1,0,0,0],[a,math.sqrt(1-a*a),0,0],[0,0,1,0],[0,0,b,math.sqrt(1-b*b)]]);R=np.linalg.inv(N).T;truth=(.25+math.asin(a)/(2*math.pi))*(.25+math.asin(b)/(2*math.pi))
            for transform,Y in [('base',R),('scale',R*8),('permute',R[:,[2,0,3,1]])]:
                cert=bounds(Y)
                for n in (2,4,8,20):
                    r=integrate(Y,n,float(np.linalg.det(Y)),rules);bd=probability_bound(cert,n)['bound'];r.update(a_correlation=a,b_correlation=b,transform=transform,truth=truth,error=r['value']-truth,quadrature_bound=bd,passes=abs(r['value']-truth)<=bd+r['series_error_bound']+5e-14);out['analytic'].append(r)
    path=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()=='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581';rays=np.load(path)['rays'];ref=np.load(HERE.parent/'radial_t4_reference/combined_values.npz')
    for i in INDICES:
        R=rays[i];cert=bounds(R);r=integrate(R,20,float(np.linalg.det(R)),rules);bd=probability_bound(cert,20)['bound'];r.update(index=i,reference=float(ref['values'][i]),reference_bound=float(ref['bounds'][i]),quadrature_bound=bd,error=r['value']-float(ref['values'][i]),passes=abs(r['value']-ref['values'][i])<=bd+ref['bounds'][i]+r['series_error_bound']+5e-14);out['trained'].append(r)
    out.update(complete=True,cpu=time.process_time()-start,scalar_all_pass=all(r['passes'] for r in out['scalar']),analytic_all_pass=all(r['passes'] for r in out['analytic']),trained_all_pass=all(r['passes'] for r in out['trained']),scalar_modes=dict(sum((Counter(r['modes']) for r in out['scalar']),Counter())),cold_rules=rules.events)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('scalar','analytic','trained','cold_rules')},indent=2));assert out['scalar_all_pass'] and out['analytic_all_pass'] and out['trained_all_pass']
if __name__=='__main__':main()
