"""Distinct angular/radial check of fixed Gaussian wedge probabilities."""
import time
START=time.process_time()
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx,ndtr,roots_laguerre
HERE=Path(__file__).resolve().parent
RULES={n:roots_laguerre(n) for n in [32,64]}
COUNTS=dict(angular_evaluations=0,negative_laguerre_nodes=0,moderate_negative=0,nonnegative=0,negative_cauchy_residuals=[])

def radial(s,c,n=64):
    if s>=0:
        COUNTS['nonnegative']+=1;delta=c-s*s
        if delta<0:COUNTS['negative_cauchy_residuals'].append(delta)
        return math.exp(-delta/2)*(math.exp(-s*s/2)+s*math.sqrt(2*math.pi)*float(ndtr(s)))
    if s>=-5:
        COUNTS['moderate_negative']+=1
        return math.exp(-c/2)*(1+s*math.sqrt(math.pi/2)*float(erfcx(-s/math.sqrt(2))))
    z,w=RULES[n];COUNTS['negative_laguerre_nodes']+=n
    return math.exp(-c/2)*float(np.dot(w,z*np.exp(-z*z/(2*s*s))))/(s*s)

def wedge(mu,C,lo,hi,n=64):
    if hi<=lo:return dict(value=0.,error=0.,evaluations=0)
    inv=np.linalg.inv(C);v=inv@mu;c=float(mu@v);norm=2*math.pi*math.sqrt(float(np.linalg.det(C)))
    def f(theta):
        COUNTS['angular_evaluations']+=1;e=np.array([math.cos(theta),math.sin(theta)]);a=float(e@inv@e);s=float(e@v)/math.sqrt(a)
        return radial(s,c,n)/(a*norm)
    ans=quad(f,lo,hi,epsabs=1e-12,epsrel=1e-12,limit=300,full_output=1)
    assert len(ans)==3
    return dict(value=ans[0],error=ans[1],evaluations=ans[2]['neval'])

def main():
    controls=[]
    for s in [-20.,-5.01,-5.,-1.,0.,1.,20.]:
        c=s*s+1;p=radial(s,c)
        if s<0:
            f=lambda u:u*math.exp(-u*u/2+s*u)
            q,e=quad(f,0,12,epsabs=1e-14,epsrel=1e-13)
            q*=math.exp(-c/2);e*=math.exp(-c/2)
        else:
            f=lambda u:u*math.exp(-(u-s)**2/2-.5)
            cuts=sorted(set([0.,max(0.,s),max(12.,s+12.)]));q=e=0.
            for lo,hi in zip(cuts,cuts[1:]):
                x,er=quad(f,lo,hi,epsabs=1e-13,epsrel=1e-13);q+=x;e+=er
        rel=abs(p-q)/max(abs(q),1e-300);assert rel<1e-10
        controls.append(dict(s=s,value=p,independent_radial_quad=q,relative_difference=rel,radial_quad_error=e))
    narrow=[]
    for width in [1e-2,1e-4,1e-6,1e-7]:
        lo=.1;hi=lo+width;v=wedge(np.zeros(2),np.eye(2),lo,hi);truth=(hi-lo)/(2*math.pi)
        assert abs(v['value']-truth)<1e-15
        narrow.append(dict(width=width,truth=truth,**v))
    saved=json.loads((HERE/'support_conditioned.json').read_text());rows=[];totals={k:saved['constant_correct_mass'] for k in saved['accuracy']}
    for r in saved['rows']:
        mu=np.array(r['hidden_mean']);C=np.array(r['hidden_covariance']);methods={}
        for method,ref in r['methods'].items():
            bounds=[0,math.pi/2] if method=='orthant' else r['component_bounds']
            den=wedge(mu,C,*bounds);nums=[wedge(mu,C,*v['bounds']) for v in ref['fine']['rows']]
            ps=[v['value']/den['value'] for v in nums]
            coarse_den=wedge(mu,C,*bounds,32);coarse_nums=[wedge(mu,C,*v['bounds'],32) for v in ref['fine']['rows']]
            coarse=[v['value']/coarse_den['value'] for v in coarse_nums]
            methods[method]=dict(denominator=den,numerators=nums,q=ps,prior_difference=float(np.max(abs(np.array(ps)-ref['fine']['q']))),
                rule_difference=float(np.max(abs(np.array(ps)-coarse))),sum_residual=sum(ps)-1)
            totals[method]+=r['mass']*ps[r['target']]
        rows.append(dict(gate_cell=r['gate_cell'],order=r['order'],methods=methods))
    out=dict(complete=True,radial_controls=controls,narrow_controls=narrow,rows=rows,accuracy=totals,
        prior_accuracy=saved['accuracy'],counts=COUNTS,cpu_seconds=time.process_time()-START,
        max_prior_difference=max(x['prior_difference'] for r in rows for x in r['methods'].values()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((HERE/'support_conditioned.json').read_bytes()).hexdigest(),
        qualification='Independent coordinate formula for same approximate Gaussian laws. Adaptive quadrature and Laguerre sensitivity are not certified bounds; geometry/moments are shared inputs.')
    (HERE/'polar_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','radial_controls','narrow_controls']},indent=2))
if __name__=='__main__':main()
